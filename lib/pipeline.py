"""The pipeline steps, with progress callbacks. Shared by the command line
(make_video.py) and the control panel (studio.py) so there is one implementation."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import align, captions as cap, render, sheet as sheetlib, stock, tts
from . import voices as V

ROOT = Path(__file__).resolve().parent.parent

# One folder per project, everything for it inside:
#   projects/<pid>/sheets/   the human-editable production sheets
#   projects/<pid>/work/     picks, assets, per-language clips (intermediate)
#   projects/<pid>/out/      the finished mp4s, srt, description, approval
# Caches (stock footage, voice) stay shared at the top level, because they are
# reused ACROSS projects and duplicating them per project would waste gigabytes.
PROJECTS = ROOT / "projects"


def project_dir(pid: str) -> Path:
    return PROJECTS / pid


def sheets_dir(pid: str) -> Path:
    return PROJECTS / pid / "sheets"


def _pid_of_dir(sheet: Path) -> Path:
    """The project folder that owns a sheet, whatever depth it sits at.

    New layout: projects/<pid>/sheets/<pid>_MASTER…  -> parent is 'sheets',
    grandparent is the project folder. Written to survive either being handed
    the master or a narration file.
    """
    return sheet.resolve().parent.parent


def migrate_layout() -> dict:
    """Move any old flat files into the projects/<pid>/{sheets,work,out} layout.

    The pipeline used to scatter a project across three shared folders:
        sheets/<pid>_*        work/<pid>_*        out/<pid>_*
    This walks whatever is still sitting there and files it under one folder per
    project. It is safe to run repeatedly: it only touches the legacy folders,
    skips anything already migrated, and never crosses a project boundary. Caches
    (cache/stock, cache/voice) are shared and deliberately left alone.

    Returns a small report so callers can log what moved.
    """
    moved, projects = 0, set()
    old_sheets, old_work, old_out = ROOT / "sheets", ROOT / "work", ROOT / "out"

    # Learn the real project ids from the master sheets first — a pid may itself
    # contain underscores (it comes from the video title), so we can't just split
    # a filename on "_". We match work/out files against the longest known pid.
    known_pids: list[str] = []
    if old_sheets.is_dir():
        for m in old_sheets.glob("*_MASTER_production_sheet.md"):
            known_pids.append(project_id(m))
    known_pids.sort(key=len, reverse=True)          # longest prefix wins

    def _pid_from_name(name: str) -> str | None:
        for pid in known_pids:
            if name == pid or name.startswith(pid + "_"):
                return pid
        # No master seen (orphan file): fall back to the leading token.
        return name.split("_", 1)[0] if "_" in name else None

    # sheets/<pid>_*.md  ->  projects/<pid>/sheets/
    if old_sheets.is_dir():
        for f in old_sheets.glob("*.md"):
            pid = _pid_from_name(f.name)
            if not pid:
                continue
            dst = sheets_dir(pid)
            dst.mkdir(parents=True, exist_ok=True)
            target = dst / f.name
            if not target.exists():
                shutil.move(str(f), str(target))
                moved += 1
            projects.add(pid)

    # work/<pid>_*  ->  projects/<pid>/work/  (renamed to drop the pid prefix)
    if old_work.is_dir():
        for f in sorted(old_work.iterdir()):
            pid = _pid_from_name(f.name)
            if not pid:
                continue
            rest = f.name[len(pid) + 1:]            # strip "<pid>_"
            # picks.json / assets.json keep their canonical names; per-language
            # working dirs "<pid>_<lang>" become just "<lang>".
            dstdir = project_dir(pid) / "work"
            dstdir.mkdir(parents=True, exist_ok=True)
            target = dstdir / rest
            if not target.exists():
                shutil.move(str(f), str(target))
                moved += 1
            projects.add(pid)

    # out/<pid>_*  ->  projects/<pid>/out/  (mp4/srt keep the pid; approval and
    # meta lose it so they read cleanly inside the folder)
    if old_out.is_dir():
        for f in sorted(old_out.iterdir()):
            pid = _pid_from_name(f.name)
            if not pid:
                continue
            dstdir = project_dir(pid) / "out"
            dstdir.mkdir(parents=True, exist_ok=True)
            rest = f.name[len(pid) + 1:]
            if rest == "approval.html" or f.name.endswith("_approval.html"):
                newname = "approval.html"
            else:
                newname = f.name                    # keep pid on mp4/srt/meta
            target = dstdir / newname
            if not target.exists():
                shutil.move(str(f), str(target))
                moved += 1
            projects.add(pid)

    # Retire the empty legacy folders so we don't keep scanning them.
    for d in (old_sheets, old_work, old_out):
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass

    return {"moved": moved, "projects": sorted(projects)}


TAIL = 1.0        # seconds of held picture after each narration line
DISSOLVE = 0.6    # crossfade length between scenes

LANG_NAMES = {"en": "English", "de": "German", "es": "Spanish",
              "fr": "French", "it": "Italian", "pt": "Portuguese"}
# How per-language narration files are named, e.g. video04_GERMAN_narration.md.
# English is included now that any language can be the master: when English is
# NOT the structure language it gets its own narration file like the others.
LANG_FILE_WORDS = {"en": ("ENGLISH", "EN"), "de": ("GERMAN", "DE"),
                   "es": ("SPANISH", "ES"), "fr": ("FRENCH", "FR"),
                   "it": ("ITALIAN", "IT"), "pt": ("PORTUGUESE", "PT")}


def master_lang(sheet: Path) -> str:
    """Which language the master sheet's narration is in.

    Recorded as an HTML comment at the top of the sheet. Older sheets, written
    before projects could start in another language, have no marker and are
    English by definition — which is why "en" is the default.
    """
    try:
        head = sheet.read_text(encoding="utf-8", errors="ignore")[:400]
        import re as _re
        m = _re.search(r"master-lang:\s*([a-z]{2})", head)
        return m.group(1) if m else "en"
    except Exception:
        return "en"


class CaptionsSkipped(Exception):
    """The video rendered fine but captions could not be burned in.

    Carries the finished file so callers can report success-with-a-caveat rather
    than treating a cosmetic problem as a lost render.
    """

    def __init__(self, reason: str, video: Path, srt: Path):
        super().__init__(reason)
        self.reason, self.video, self.srt = reason, video, srt


def noop(*_a, **_k) -> None:
    pass


# --------------------------------------------------------------- discovery

def pretty_name(sheet: Path) -> str:
    """'video04_MASTER_production_sheet.md' -> 'video04 - Sharpest 80-Year-Olds'"""
    stem = project_id(sheet)
    title = ""
    for line in sheet.read_text(encoding="utf-8").splitlines()[:12]:
        if line.startswith("## "):
            title = line[3:].strip().strip('"').strip("'")
            break
    if len(title) > 46:
        title = title[:44].rstrip(" ,.") + "…"
    return f"{stem} — {title}" if title else stem


def project_id(sheet: Path) -> str:
    return (sheet.stem.replace("_MASTER_production_sheet", "")
            .replace("_MASTER", "").replace("_master", ""))


def find_project(pid: str) -> dict | None:
    """The one project with this id, or None. `proj["sheet"]` is a full path."""
    return next((p for p in find_projects() if p["id"] == pid), None)


def out_dir(pid: str) -> Path:
    """Where a project's finished files live: projects/<pid>/out/."""
    return PROJECTS / pid / "out"


def find_projects(_root: Path | None = None) -> list[dict]:
    """Every project under projects/, each master with its narration sheets.

    Scans projects/<pid>/sheets/. The old signature took a sheets directory;
    callers pass nothing now, but an argument is still accepted and ignored so
    nothing breaks mid-upgrade. `sheet` is the full path to the master, and
    `dir` is the project folder.
    """
    root = PROJECTS
    out = []
    if not root.exists():
        return out
    for sd in sorted(root.glob("*/sheets")):
        for f in sorted(sd.glob("*_MASTER_production_sheet.md")):
            # One unreadable project must not hide every other one. Anything that
            # goes wrong reading a single sheet skips just that project, so the
            # dashboard still loads the rest.
            try:
                pid = sd.parent.name
                mlang = master_lang(f)
                # The structure language reads from the master; no side file.
                langs = [{"code": mlang, "name": LANG_NAMES.get(mlang, mlang), "file": None}]
                narr = sorted(sd.glob(f"{pid}_*_narration.md"))
                for code, words in LANG_FILE_WORDS.items():
                    if code == mlang:
                        continue
                    hit = next((nf for nf in narr if words[0] in nf.stem.upper()), None)
                    if hit:
                        langs.append({"code": code, "name": LANG_NAMES.get(code, code),
                                      "file": hit.name})
                try:
                    n = len(sheetlib.parse_master(f))
                except SystemExit:
                    n = 0
                out.append({"id": pid, "sheet": str(f), "dir": str(sd.parent),
                            "label": pretty_name(f), "scenes": n, "languages": langs})
            except Exception:
                continue
    return out


def project_status(sheet: Path, langs: list[dict]) -> dict:
    """Which steps are finished, per language, judged from what's on disk.

    Deliberately derived rather than stored. A status file would drift the
    moment anyone deleted an MP4 or cleared a cache by hand — this way the
    dashboard can never claim something exists when it doesn't.

    Per language, four steps:
        sheets   the narration text exists (always true for en; needs a
                 translation file otherwise)
        visuals  stock footage has been sourced and assigned to scenes
        voice    every scene has a cached narration file
        render   the finished MP4 is on disk
    """
    pid = project_id(sheet)
    mlang = master_lang(sheet)          # the language the master itself is written in
    p_shared = paths_for(sheet, "en")
    n_scenes = 0
    try:
        n_scenes = len(sheetlib.parse_master(sheet))
    except Exception:
        pass

    # Visuals are shared across languages — sourced once, reused everywhere.
    assets_n = 0
    match_avg = None            # mean relevance across scored assets, 0..1
    weak_n = 0                  # how many matched only weakly
    if p_shared["assets"].exists():
        try:
            assets = json.loads(p_shared["assets"].read_text(encoding="utf-8"))
            assets_n = len(assets)
            clip_min = float(load_config().get("clip_min") or 0.45)
            scores = [a["score"] for a in assets.values()
                      if isinstance(a, dict) and a.get("score") is not None]
            if scores:
                match_avg = round(sum(scores) / len(scores), 3)
                weak_n = sum(1 for s in scores if s < clip_min)
        except Exception:
            assets_n = 0

    out = {"scenes": n_scenes, "assets": assets_n,
           "match": match_avg, "weak": weak_n, "languages": {}}
    for lg in langs:
        code = lg["code"]
        pl_ = paths_for(sheet, code)
        mp4 = pl_["out"]

        # Count cached narration for this language. The cache key includes the
        # reference clip and settings, so this counts only files that the
        # CURRENT voice choice would produce.
        voiced = 0
        try:
            scenes = load_scenes(sheet, code,
                                 translation_for(sheet.parent, pid, code))
            vp = tts.voice_paths(scenes, code, p_shared['voicecache'])
            voiced = sum(1 for v in vp
                         if v.exists() and v.stat().st_size > 1024)
        # SystemExit (not an Exception) is what sheet.load raises for a language
        # whose narration can't be found. Catch it here so one missing
        # translation degrades to "not voiced" instead of failing the dashboard.
        except (Exception, SystemExit):
            scenes = []

        # Which clip reads this language. The project page needs it to show and
        # change the voice in place, rather than sending you to another screen
        # to answer a question it just asked you.
        pref = V.pref_for(code)
        ref = pref.get("reference", "")
        ref_ok = False
        if ref:
            try:
                V.resolve(ref)
                ref_ok = True
            except FileNotFoundError:
                ref_ok = False

        out["languages"][code] = {
            "name": lg.get("name", code),
            "voice_ref": ref,
            "voice_label": V.label_for(ref) if ref else "",
            "voice_ok": ref_ok,
            "sheets": bool(lg.get("file")) or code == mlang,
            "visuals": assets_n > 0 and assets_n >= n_scenes,
            "visuals_n": assets_n,
            "voice_n": voiced,
            "voice": n_scenes > 0 and voiced >= n_scenes,
            "render": mp4.exists(),
            "mp4": mp4.name if mp4.exists() else None,
            "size_mb": round(mp4.stat().st_size / 1e6, 1) if mp4.exists() else None,
            "built": int(mp4.stat().st_mtime) if mp4.exists() else None,
            "srt": pl_["srt"].name if pl_["srt"].exists() else None,
        }
    return out


def deletable(sheet: Path, langs: list[dict]) -> dict:
    """Everything on disk belonging to one project, grouped by what it is.

    Nothing here touches cache/stock. Downloaded footage is content-addressed
    and shared between projects — deleting a clip because you finished with one
    video would silently break another that reuses it.
    """
    # Absolute from here on. A relative sheet path would make the safety check
    # below resolve against the current working directory instead of the
    # project, which is exactly the kind of subtlety that turns a delete button
    # into a bad afternoon.
    sheet = Path(sheet).resolve()
    pid = project_id(sheet)
    proj = _pid_of_dir(sheet)
    out: dict[str, list[Path]] = {"outputs": [], "voice": [], "visuals": [],
                                  "work": [], "sheets": []}

    outd = proj / "out"
    if outd.exists():
        for f in outd.glob("*"):
            out["outputs"].append(f)

    shared = paths_for(sheet, "en")
    for key in ("picks", "assets"):
        if shared[key].exists():
            out["visuals"].append(shared[key])
    ap = shared["approval"]
    if ap.exists():
        out["visuals"].append(ap)

    for lg in langs:
        code = lg["code"]
        base = paths_for(sheet, code)["base"]
        if base.exists():
            out["work"].append(base)
        try:
            scenes = load_scenes(sheet, code, translation_for(sheet.parent, pid, code))
            for f in tts.voice_paths(scenes, code, shared["voicecache"]):
                if f.exists():
                    out["voice"].append(f)
        except Exception:
            pass

    out["sheets"].append(sheet)
    for f in sheet.parent.glob(f"{pid}_*.md"):
        if f != sheet:
            out["sheets"].append(f)

    return out


def _inside(p: Path, root: Path) -> bool:
    """True only if p really sits under root, symlinks resolved.

    The guard that matters: a project id is user-supplied, and this function is
    the last thing standing between a stray '..' and someone's home directory.
    """
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def delete_project(sheet: Path, langs: list[dict], what: list[str]) -> dict:
    """Delete the chosen groups. Returns what actually went, and what didn't.

    Every path is re-checked against the project root immediately before
    unlinking. Belt and braces: the caller has already validated the project id
    against the known list, but this is destructive and irreversible, so it
    verifies rather than trusts.
    """
    import shutil

    groups = deletable(sheet, langs)
    removed, freed, refused = [], 0, []
    for key in what:
        for f in groups.get(key, []):
            if not _inside(f, ROOT):
                refused.append(str(f))       # never possible via the UI; still checked
                continue
            try:
                if f.is_dir():
                    freed += sum(x.stat().st_size for x in f.rglob("*") if x.is_file())
                    shutil.rmtree(f)
                elif f.exists():
                    freed += f.stat().st_size
                    f.unlink()
                else:
                    continue
                removed.append(f.name)
            except OSError as e:
                refused.append(f"{f.name}: {e}")
    # If the whole project was cleared out, drop its now-empty folder so the
    # projects/ directory doesn't accumulate hollow shells.
    proj = _pid_of_dir(sheet)
    if _inside(proj, PROJECTS) and proj.exists():
        for d in (proj / "sheets", proj / "work", proj / "out", proj):
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass
    return {"removed": removed, "count": len(removed),
            "freed_mb": round(freed / 1e6, 1), "refused": refused}


def narration_file(sheets_dir: Path, pid: str, lang: str) -> Path | None:
    """The per-language narration sheet for `lang`, or None if that language is
    the master's own (it reads from the master) or has no sheet yet."""
    master = sheets_dir / f"{pid}_MASTER_production_sheet.md"
    if master.exists() and master_lang(master) == lang:
        return None
    words = LANG_FILE_WORDS.get(lang, ())
    if not words:
        return None
    for nf in sorted(sheets_dir.glob(f"{pid}_*_narration.md")):
        if words[0] in nf.stem.upper():
            return nf
    return None


# Kept as the historical name; callers pass a language and get its sheet (or
# None for the master's own language). "Translation" is a misnomer now — the
# words are the user's, segmented, not machine-translated — but renaming every
# caller is churn for no behaviour change.
def translation_for(sheets_dir: Path, pid: str, lang: str) -> Path | None:
    return narration_file(sheets_dir, pid, lang)


# ------------------------------------------------------------------ paths

def paths_for(sheet: Path, lang: str) -> dict:
    pid = project_id(sheet)
    proj = _pid_of_dir(sheet)                    # projects/<pid>/
    work = proj / "work"
    outd = proj / "out"
    base = work / lang                           # per-language working dir
    return {
        "id": pid, "dir": proj, "base": base, "clips": base / "clips", "tmp": base / "tmp",
        "stockcache": ROOT / "cache" / "stock", "voicecache": ROOT / "cache" / "voice",
        "picks": work / "picks.json",            # shared by all languages
        "assets": work / "assets.json",          # shared by all languages
        "approval": outd / "approval.html",
        "out": outd / f"{pid}_{lang}.mp4",
        "srt": outd / f"{pid}_{lang}.srt",
        "meta": outd / f"{pid}_{lang}_meta.json",     # title/desc/tags
        "ass": base / "captions.ass",
    }


def load_config() -> dict:
    # Environment variables win over config.json, so keys can be kept out of
    # files entirely if you prefer.
    cfg = {"pexels_key": os.environ.get("PEXELS_API_KEY", ""),
           "pixabay_key": os.environ.get("PIXABAY_API_KEY", ""),
           "gemini_key": os.environ.get("GEMINI_API_KEY", ""),
           "gemini_model": os.environ.get("GEMINI_MODEL", "")}
    f = ROOT / "config.json"
    if f.exists():
        try:
            cfg.update({k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items()
                        if v and not k.startswith("_")})
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"config.json is not valid JSON ({e}).\n"
                f"Most often this is curly quotes from a word processor. Reopen "
                f"config.json in a plain text editor (Notepad, TextEdit) and "
                f"retype the quote marks."
            )
    return cfg


def _flag(v, default: bool = True) -> bool:
    """Read a config value as an on/off switch. 'auto'/'on'/True are on;
    'off'/'no'/'false'/'0'/False are off. Missing falls back to `default`."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("off", "false", "no", "0", "none", "")


def load_scenes(sheet: Path, lang: str, translation: Path | None):
    return sheetlib.load(sheet, lang, translation)


# --------------------------------------------------------- caption styling
# The look of the burned-in subtitles. Three levels: built-in presets (in
# lib/captions.py), the user's saved default and custom templates (captions.json,
# shared across machines), and a per-project override (projects/<pid>/subtitle.json).

CAPTIONS_FILE = ROOT / "captions.json"


def load_captions_config() -> dict:
    if CAPTIONS_FILE.exists():
        try:
            return json.loads(CAPTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_captions_config(data: dict) -> None:
    CAPTIONS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def global_caption_style():
    """The default style for every video, until a project overrides it."""
    return load_captions_config().get("default") or cap.DEFAULT_PRESET


def set_global_caption_style(spec) -> None:
    data = load_captions_config()
    data["default"] = spec
    save_captions_config(data)


def custom_caption_styles() -> dict:
    """The user's own saved templates, name -> style dict."""
    return load_captions_config().get("custom") or {}


def save_custom_caption_style(name: str, spec: dict) -> None:
    data = load_captions_config()
    data.setdefault("custom", {})[name] = spec
    save_captions_config(data)


def delete_custom_caption_style(name: str) -> None:
    data = load_captions_config()
    if name in (data.get("custom") or {}):
        del data["custom"][name]
        save_captions_config(data)


def _project_style_path(pid: str) -> Path:
    return project_dir(pid) / "subtitle.json"


def load_project_style(pid: str):
    f = _project_style_path(pid)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_project_style(pid: str, spec) -> None:
    """Set (or clear, with None) a project's own caption style."""
    f = _project_style_path(pid)
    if spec in (None, "", "default"):
        if f.exists():
            f.unlink()
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")


def effective_caption_style(pid: str | None = None):
    """The spec render should use: a project override if set, else the global
    default. Returned as a preset id or a style dict — render resolves it."""
    if pid:
        ov = load_project_style(pid)
        if ov not in (None, "", "default"):
            return ov
    return global_caption_style()


# --------------------------------------------------------- publish metadata
# Title, description and tags for a finished video, generated on demand and
# saved next to the render so they survive edits and reloads. One file per
# language, because each video is its own upload.

def load_metadata(sheet: Path, lang: str) -> dict | None:
    p = paths_for(sheet, lang)["meta"]
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def save_metadata(sheet: Path, lang: str, data: dict) -> dict:
    p = paths_for(sheet, lang)["meta"]
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        "title": (data.get("title") or "").strip(),
        "description": (data.get("description") or "").strip(),
        "tags": [str(t).strip().lstrip("#").strip()
                 for t in (data.get("tags") or []) if str(t).strip()],
        "lang": lang,
    }
    # ensure_ascii=False so German and Spanish read correctly in the file.
    p.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return clean


def build_metadata(sheet: Path, lang: str, cfg: dict) -> dict:
    """Generate metadata from the narration in this language, and save it."""
    from . import gemini as G, llm as LLM, sources as SRC   # on demand
    if not LLM.available(cfg):
        raise RuntimeError(
            "Writing a description needs a language model. Add a free Gemini key "
            "(gemini_key), or set llm=ollama with an ollama_model in config.json.")

    pid = project_id(sheet)
    tr = translation_for(sheet.parent, pid, lang)
    scenes = load_scenes(sheet, lang, tr)
    narration = " ".join(s.narration for s in scenes if s.narration).strip()
    if not narration:
        raise RuntimeError(
            f"No {LANG_NAMES.get(lang, lang)} narration to describe yet — "
            f"generate the sheets (and translation) for this language first.")

    # Canonical subjects, from the scene domains and the narration itself, so a
    # medical or historical video gets the right framing in its description.
    domains = " ".join(getattr(s, "domain", "") for s in scenes)
    topics = sorted(SRC.topics_in(domains, narration))
    data = G.generate_metadata(narration, LANG_NAMES.get(lang, lang), topics,
                               pid, LLM.key_for(cfg), LLM.model_for(cfg))
    return save_metadata(sheet, lang, data)


# ------------------------------------------------------------------ steps

def _expand_scene_queries(scenes, p, cfg: dict, on_progress=lambda *_: None) -> None:
    """Attach LLM-generated alternative image queries to each scene's fallbacks.

    Needs a Gemini key; a no-op without one, so it never blocks sourcing. Results
    are cached in the project's work folder keyed by the scene's own query, so a
    re-source spends no tokens and only genuinely new/changed scenes are sent.
    The extra phrases go to the END of `fallbacks`, keeping the human-written
    query and any existing fallbacks first.
    """
    from . import llm as LLM
    if not scenes or not LLM.available(cfg):
        return
    if not _flag(cfg.get("expand_queries", "auto")):
        return

    cache_f = p["base"].parent / "queries.json"      # work/queries.json (shared)
    cache: dict = {}
    if cache_f.exists():
        try:
            cache = json.loads(cache_f.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    need = [s for s in scenes
            if cache.get(str(s.n), {}).get("query") != (s.query or "")]
    if need:
        on_progress(f"expanding queries for {len(need)} scene(s)")
        from . import gemini as G
        got = G.expand_queries(
            [{"n": s.n, "query": s.query, "narration": s.narration} for s in need],
            LLM.key_for(cfg), LLM.model_for(cfg))
        for s in need:
            cache[str(s.n)] = {"query": s.query or "", "extra": got.get(s.n, [])}
        try:
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            cache_f.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        except Exception:
            pass

    # Merge cached expansions into each scene's ladder, de-duplved against what
    # the scene already carries so nothing is searched twice.
    for s in scenes:
        extra = cache.get(str(s.n), {}).get("extra") or []
        have = {(s.query or "").lower(), *(f.lower() for f in getattr(s, "fallbacks", []))}
        add = [q for q in extra if q.lower() not in have]
        if add:
            s.fallbacks = list(getattr(s, "fallbacks", [])) + add


def source_stock(scenes, sheet: Path, cfg: dict, redo: list[int] | None = None,
                 on_progress=noop) -> dict[int, dict]:
    """Fetch a visual per scene. Visuals are language-independent, so this is
    done once per project and reused by every language."""
    p = paths_for(sheet, "en")
    p["base"].parent.mkdir(parents=True, exist_ok=True)   # work/
    p["approval"].parent.mkdir(parents=True, exist_ok=True)  # out/

    picks: dict[int, int] = {}
    if p["picks"].exists():
        picks = {int(k): v for k, v in json.loads(p["picks"].read_text(encoding="utf-8")).items()}
    for n in (redo or []):
        picks[n] = picks.get(n, 0) + 1
    p["picks"].write_text(json.dumps(picks, indent=2), encoding="utf-8")

    assets: dict[int, dict] = {}
    if p["assets"].exists():
        assets = {int(k): v for k, v in json.loads(p["assets"].read_text(encoding="utf-8")).items()}

    todo = [s for s in scenes if redo is None or s.n in redo or s.n not in assets]

    # Give each scene a few concrete alternative queries (once, cached). They sit
    # at the BOTTOM of its query ladder, so a scene whose own query already finds
    # a strong match never uses them — they only rescue the weak ones, which is
    # what keeps the extra searches (and the one Gemini call) cheap.
    try:
        _expand_scene_queries(todo, p, cfg,
                              on_progress=lambda m: on_progress(0, len(todo), m))
    except Exception:
        pass

    # Delegated rather than reimplemented. stock.fetch_all owns the query
    # ladder, the routing to NASA/Smithsonian/stock, and the refusal to reuse
    # a clip already on screen — and this used to call stock.fetch directly,
    # which quietly meant none of that ran.
    keep = {n: a for n, a in assets.items() if n not in {s.n for s in todo}}
    fresh = stock.fetch_all(
        todo, p["stockcache"], cfg.get("pexels_key"), cfg.get("pixabay_key"),
        picks=picks, log=lambda m: None, cfg=cfg, already=keep,
        on_progress=on_progress)
    assets.update(fresh)

    p["assets"].write_text(
        json.dumps({str(k): v for k, v in assets.items()}, indent=2),
        encoding="utf-8")
    return assets


def generate_voice(scenes, lang: str, sheet: Path, voice: str | None = None,
                   on_progress=noop) -> list[Path]:
    """`voice` names a reference clip, overriding the one saved for this language."""
    p = paths_for(sheet, lang)
    done = [0]
    total = len(scenes)

    def log(msg: str) -> None:
        if isinstance(msg, str) and msg.lstrip().startswith("S"):
            done[0] += 1
            on_progress(done[0], total, msg.strip())

    return tts.synth(scenes, lang, p["voicecache"], voice=voice, log=log)


def _aligned_words(scenes, voices, vdurs, starts, lang, p, on_progress, n):
    """Per-scene word timings, in ABSOLUTE video time, for the karaoke captions.

    Each scene is aligned against its own audio and cached in the language's work
    folder keyed by the exact narration text, so a caption-only re-render (or a
    second language sharing nothing) never realigns a scene whose words haven't
    changed. Returns one word-list per scene: [{word, start, end}, ...].
    """
    cfg = load_config()
    cache_f = p["base"] / "words.json"
    cache: dict = {}
    if cache_f.exists():
        try:
            cache = json.loads(cache_f.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    out, changed = [], False
    for i, s in enumerate(scenes):
        rec = cache.get(str(s.n))
        if not (rec and rec.get("text") == s.narration and rec.get("words")):
            words = align.align_words(
                voices[i], s.narration, lang, cfg=cfg, dur=vdurs[i],
                log=lambda m: on_progress(n + 3, n + 4, m.strip()))
            cache[str(s.n)] = rec = {"text": s.narration, "words": words}
            changed = True
        # Relative -> absolute, so every scene's words sit at the right moment in
        # the finished audio.
        out.append([{"word": w["word"],
                     "start": round((w.get("start") or 0.0) + starts[i], 3),
                     "end": round((w.get("end") or 0.0) + starts[i], 3)}
                    for w in rec["words"]])

    if changed:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return out


def render_video(scenes, assets: dict[int, dict], voices: list[Path], sheet: Path,
                 lang: str, captions: bool = True, music: Path | None = None,
                 music_level: float = 0.20, zoom: bool = True,
                 caption_size: int = 58, style=None, master: bool = True,
                 on_progress=noop) -> Path:
    p = paths_for(sheet, lang)
    for d in (p["clips"], p["tmp"], p["out"].parent):
        d.mkdir(parents=True, exist_ok=True)

    missing = [s.n for s in scenes if s.n not in assets]
    if missing:
        raise RuntimeError(f"No visual for scenes {missing}. Re-run the visuals step.")

    n = len(scenes)
    clips, vdurs = [], []
    for i, s in enumerate(scenes):
        vd = render.duration_of(voices[i])
        vdurs.append(vd)
        src = Path(assets[s.n]["path"])
        out = p["clips"] / f"c{s.n:04d}.mp4"
        # Rebuild anything left over from before the 4:2:0 pin, so a cached clip
        # can't quietly drag the finished video back to an unplayable format.
        # yuvj420p is accepted: it is still 4:2:0 and plays everywhere - only the
        # colour range is flagged full rather than limited. Rejecting it would mean
        # re-encoding every cached clip for no visible gain.
        stale = out.exists() and render.pix_fmt_of(out) not in ("yuv420p", "yuvj420p")
        if not out.exists() or stale:
            if src.suffix.lower() in (".mp4", ".mov", ".webm"):
                render.make_video_clip(src, vd + TAIL, out)
            else:
                render.make_image_clip(src, vd + TAIL, out, zoom=zoom)
        clips.append((out, render.duration_of(out)))
        on_progress(i + 1, n + 4, f"scene {i + 1} of {n}")

    # The crossfade chain is the single most expensive step - tens of minutes for
    # 115 scenes. Reuse it when no clip has changed since it was built, so a retry
    # (say, for captions) is quick instead of another full pass.
    vid = p["base"] / "video_track.mp4"
    newest_clip = max((c.stat().st_mtime for c, _ in clips), default=0)
    reusable = (vid.exists() and vid.stat().st_mtime >= newest_clip
                and render.pix_fmt_of(vid) in ("yuv420p", "yuvj420p"))
    if reusable:
        on_progress(n + 1, n + 4, "reusing crossfaded video")
    else:
        on_progress(n + 1, n + 4, "crossfading scenes")
        render.dissolve_concat(clips, DISSOLVE, vid, p["tmp"], group=10)

    on_progress(n + 2, n + 4, "assembling narration")
    gaps = [max(0.0, cd - vd - DISSOLVE) for (_, cd), vd in zip(clips, vdurs)]
    aud = p["base"] / "audio_track.wav"
    starts = render.build_audio(voices, gaps, aud, p["tmp"], tail=DISSOLVE)

    acfg = load_config()
    if music:
        # Duck the bed under the narration unless explicitly turned off.
        duck = _flag(acfg.get("music_duck", True))
        mixed = p["base"] / "audio_mixed.wav"
        render.mix_music(aud, Path(music), mixed, level=music_level, duck=duck)
        aud = mixed

    # Master the final mix to broadcast loudness so the video plays back as loud
    # as everything else on YouTube. Never fatal: the audio is already fine, this
    # only polishes it, so a failure leaves the un-mastered track in place.
    if master and _flag(acfg.get("audio_master", "auto")):
        lufs = float(acfg.get("lufs_target") or -14.0)
        on_progress(n + 3, n + 4, f"mastering audio to {lufs:g} LUFS")
        try:
            mastered = p["base"] / "audio_master.wav"
            render.master_audio(aud, mastered, lufs=lufs)
            aud = mastered
        except Exception as e:
            on_progress(n + 3, n + 4, f"mastering skipped ({e})")

    on_progress(n + 3, n + 4, "muxing")
    silent = p["base"] / "muxed.mp4"
    render.mux(vid, aud, silent)

    texts = [s.narration for s in scenes]
    render.write_srt(texts, starts, vdurs, p["srt"])

    # Captions are the LAST step and the most fragile - they depend on how this
    # machine's ffmpeg was compiled. The video is already finished by now, so a
    # caption failure must never throw it away: save the film, report the problem,
    # and leave the .srt to upload alongside.
    if captions:
        st = cap.resolve_style(style)
        # Legacy callers passed only a pixel size; honour it when no style chosen.
        if style is None and caption_size:
            st = st.merged(size=caption_size)

        # Word-by-word timing. Aligned once per scene against its own audio and
        # cached, so a caption-only re-render doesn't realign 100+ clips.
        on_progress(n + 3, n + 4, "timing the words")
        scene_words = _aligned_words(scenes, voices, vdurs, starts, lang, p,
                                     on_progress, n)
        groups = cap.groups_from_scenes(scene_words, st)
        p["ass"].write_text(cap.build_ass(groups, st), encoding="utf-8")

        on_progress(n + 3, n + 4, "burning captions")
        try:
            method = render.burn_captions(silent, p["ass"], p["out"], texts=texts,
                                          starts=starts, durs=vdurs,
                                          size=st.size)
            if method == "drawtext":
                on_progress(n + 4, n + 4,
                            "done (captions burned without libass - plainer style)")
        except Exception as e:
            shutil.copy(silent, p["out"])
            on_progress(n + 4, n + 4, "done, but captions could not be burned")
            raise CaptionsSkipped(str(e), p["out"], p["srt"]) from None
    else:
        shutil.copy(silent, p["out"])

    on_progress(n + 4, n + 4, "done")
    return p["out"]
