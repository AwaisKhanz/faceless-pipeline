"""The pipeline steps, with progress callbacks. Shared by the command line
(make_video.py) and the control panel (studio.py) so there is one implementation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import align, captions as cap, render, sheet as sheetlib, stock, tts
from . import voices as V
from .chatterbox_engine import speech_text as _speech   # the one spoken-text cleaner

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

    Layout: projects/<pid>/sheets/<pid>_main_script.md  -> parent is 'sheets',
    grandparent is the project folder. Written to survive either being handed
    the main script or a narration file.
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

    # Learn the real project ids from the main scripts first — a pid may itself
    # contain underscores (it comes from the video title), so we can't just split
    # a filename on "_". We match work/out files against the longest known pid.
    known_pids: list[str] = []
    if old_sheets.is_dir():
        for m in old_sheets.glob("*_main_script.md"):
            known_pids.append(project_id(m))
    known_pids.sort(key=len, reverse=True)          # longest prefix wins

    def _pid_from_name(name: str) -> str | None:
        for pid in known_pids:
            if name == pid or name.startswith(pid + "_"):
                return pid
        # No main script seen (orphan file): fall back to the leading token.
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
# English is included now that any language can be the main script: when English
# is NOT the structure language it gets its own narration file like the others.
LANG_FILE_WORDS = {"en": ("ENGLISH", "EN"), "de": ("GERMAN", "DE"),
                   "es": ("SPANISH", "ES"), "fr": ("FRENCH", "FR"),
                   "it": ("ITALIAN", "IT"), "pt": ("PORTUGUESE", "PT")}
# Every accepted filename token -> its language code. Built from the table above
# so both spellings work: files are named with the language NAME for known langs
# (…_GERMAN_narration.md) but fall back to the CODE for others (…_EN_narration.md).
_TOKEN_TO_CODE = {tok.upper(): code
                  for code, toks in LANG_FILE_WORDS.items() for tok in toks}


def _narration_lang(pid: str, stem: str) -> str | None:
    """The language code a narration filename encodes, or None.

    Matches the token that sits EXACTLY between '{pid}_' and '_narration', so
    'EN' identifies English without a bare-substring 'EN' also matching 'FRENCH'.
    """
    if not (stem.startswith(pid + "_") and stem.endswith("_narration")):
        return None
    token = stem[len(pid) + 1: -len("_narration")].upper()
    return _TOKEN_TO_CODE.get(token)


def main_lang(sheet: Path) -> str:
    """Which language the main script's narration is in.

    Recorded as an HTML comment at the top of the sheet. Older sheets, written
    before projects could start in another language, have no marker and are
    English by definition — which is why "en" is the default.
    """
    try:
        head = sheet.read_text(encoding="utf-8", errors="ignore")[:400]
        import re as _re
        m = _re.search(r"main-lang:\s*([a-z]{2})", head)
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
    """'video04_main_script.md' -> 'video04 - Sharpest 80-Year-Olds'"""
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
    """The project id from a main-script path: '<pid>_main_script.md' -> '<pid>'."""
    return sheet.stem.replace("_main_script", "")


# The output shape a project is built at. 'short' is a vertical 9:16 reel;
# 'video' (default) is a standard 16:9 landscape video. This ONE choice drives
# the whole pipeline: the aspect images are generated at, and the frame size the
# render (clips + captions) is produced at.
_ORIENTATION = {
    "video": {"aspect": "16:9", "w": 1920, "h": 1080},
    "short": {"aspect": "9:16", "w": 1080, "h": 1920},
}


def project_format(sheet_or_pid) -> str:
    """A project's format: 'video' (16:9, default) or 'short' (9:16)."""
    pid = project_id(sheet_or_pid) if isinstance(sheet_or_pid, Path) else str(sheet_or_pid)
    try:
        from . import pflags as _pf
        return _pf.get(pid).get("format", "video")
    except Exception:
        return "video"


def orientation(sheet_or_pid) -> dict:
    """The output shape for a project: {format, aspect, w, h}."""
    fmt = project_format(sheet_or_pid)
    o = dict(_ORIENTATION.get(fmt, _ORIENTATION["video"]))
    o["format"] = fmt
    return o


def _cfg_with_aspect(sheet, cfg: dict | None) -> dict:
    """A copy of `cfg` carrying this project's aspect for the image/Veo engines.

    Aspect is NOT a user setting — it follows the project's format: a Video
    generates 16:9, a Short 9:16. Every generation path (source_stock,
    generate_scenes, Veo) runs `cfg` through here first, so image gen and the
    render always agree on the shape. `generate_aspect` is just the internal key
    the engines read; the pipeline is its only writer."""
    return {**(cfg or {}), "generate_aspect": orientation(sheet)["aspect"]}


_CLIP_VOICE_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")


def _is_clip_voice(v: str) -> bool:
    """A reference-CLIP voice (Chatterbox/Higgs) looks like a file — 'de/awais.mp3'
    — whereas a Chirp voice is a catalogue NAME like 'de-DE-Chirp3-HD-Enceladus'."""
    v = (v or "").lower()
    return "/" in v or v.endswith(_CLIP_VOICE_EXTS)


def channel_voice(sheet: Path, lang: str, cfg: dict | None = None) -> str:
    """The voice this project's CHANNEL narrates `lang` with — but ONLY when it fits
    the engine that will actually run, else '' (fall back to the per-language
    Settings default).

    A channel voice is a single string whose meaning depends on the engine: a Chirp
    catalogue name under Chirp, a reference clip under Chatterbox/Higgs. If the
    stored voice doesn't match the active engine — e.g. a Chirp channel voice while
    narrating locally with Chatterbox — using it would break the run, so we ignore
    it and let the language's own Settings voice speak instead. Every project in a
    channel narrates with the channel's voice; a run passing an explicit voice of
    its own still wins. Tolerates a missing channels file (returns '')."""
    try:
        from . import channels as _ch
        v = _ch.voice_for_project(project_id(sheet), lang)
    except Exception:
        return ""
    if not v:
        return ""
    try:
        chirp = tts.active_engine(cfg if cfg is not None else load_config()) == "chirp"
    except Exception:
        chirp = False
    # Compatible pairing only: Chirp name ↔ Chirp engine, clip ↔ clip engine.
    if chirp != _is_clip_voice(v):          # chirp AND name, or clip-engine AND clip
        return v
    return ""                                # mismatch → use the Settings default


def find_project(pid: str) -> dict | None:
    """The one project with this id, or None. `proj["sheet"]` is a full path."""
    return next((p for p in find_projects() if p["id"] == pid), None)


def out_dir(pid: str) -> Path:
    """Where a project's finished files live: projects/<pid>/out/."""
    return PROJECTS / pid / "out"


def find_projects(_root: Path | None = None) -> list[dict]:
    """Every project under projects/, each main script with its narration sheets.

    Scans projects/<pid>/sheets/. The old signature took a sheets directory;
    callers pass nothing now, but an argument is still accepted and ignored so
    nothing breaks mid-upgrade. `sheet` is the full path to the main script, and
    `dir` is the project folder.
    """
    root = PROJECTS
    out = []
    if not root.exists():
        return out
    for sd in sorted(root.glob("*/sheets")):
        for f in sorted(sd.glob("*_main_script.md")):
            # One unreadable project must not hide every other one. Anything that
            # goes wrong reading a single sheet skips just that project, so the
            # dashboard still loads the rest.
            try:
                pid = sd.parent.name
                mlang = main_lang(f)
                # The structure language reads from the main script; no side file.
                langs = [{"code": mlang, "name": LANG_NAMES.get(mlang, mlang), "file": None}]
                seen = {mlang}
                for nf in sorted(sd.glob(f"{pid}_*_narration.md")):
                    code = _narration_lang(pid, nf.stem)
                    if code and code not in seen:
                        seen.add(code)
                        langs.append({"code": code, "name": LANG_NAMES.get(code, code),
                                      "file": nf.name})
                try:
                    n = len(sheetlib.parse_main_script(f))
                except SystemExit:
                    n = 0
                out.append({"id": pid, "sheet": str(f), "dir": str(sd.parent),
                            "label": pretty_name(f), "scenes": n, "languages": langs})
            except Exception:
                continue
    return out


def _dur_safe(path: Path) -> float:
    """duration_of that never raises — a corrupt/half-written cached clip just
    reads as 0 so the caller rebuilds it instead of crashing the render."""
    try:
        return render.duration_of(path)
    except Exception:
        return 0.0


_SENT_END = re.compile(r"[.!?…]['\"’”)\]]*$")


def _flow_flags(scenes) -> list[bool]:
    """flow[i] is True when scene i runs MID-SENTENCE into the next one.

    A sentence is often split across several scenes so each shows its own
    picture ("… vitamin C than any other fruit," / "helping your immune system" /
    "and healthy skin."). Spoken as separate clips they'd each full-stop; this
    flags the joins so the render keeps almost no gap there — the fragments run
    together as one sentence — while real sentence ends still get the full pause.
    A scene ends a sentence when its cleaned narration ends with . ! ? or … .
    """
    n = len(scenes)
    out = []
    for i, s in enumerate(scenes):
        t = _speech(s.narration)
        ends = bool(_SENT_END.search(t)) or not t
        out.append(i < n - 1 and not ends)
    return out


def project_status(sheet: Path, langs: list[dict]) -> dict:
    """Which steps are finished, per language, judged from what's on disk.

    Deliberately derived rather than stored. A status file would drift the
    moment anyone deleted an MP4 or cleared a cache by hand — this way the
    dashboard can never claim something exists when it doesn't.

    Per language, four steps:
        sheets   the narration text exists (always true for the main script's
                 own language; needs a narration file otherwise)
        visuals  stock footage has been sourced and assigned to scenes
        voice    every scene has a cached narration file
        render   the finished MP4 is on disk
    """
    pid = project_id(sheet)
    mlang = main_lang(sheet)          # the language the main script is written in
    p_shared = paths_for(sheet, "en")
    n_scenes = 0
    try:
        n_scenes = len(sheetlib.parse_main_script(sheet))
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
    cfg_ps = load_config()
    # The engine that will ACTUALLY narrate (Chirp only if it's ready, else the
    # local fallback) — so the shown voice matches what a run would use.
    try:
        chirp_engine = tts.active_engine(cfg_ps) == "chirp"
    except Exception:
        chirp_engine = str(cfg_ps.get("voice_engine", "")).strip().lower() in (
            "chirp", "gtts", "google")
    for lg in langs:
        code = lg["code"]
        pl_ = paths_for(sheet, code)
        mp4 = pl_["out"]

        # Count cached narration for this language by the EXACT files the current
        # voice choice would produce — the cache key folds in the reference/voice,
        # engine, settings AND the scene's own text, so a clip only counts if it
        # was made for THIS project's words. (An earlier loose fallback matched by
        # language + scene number alone, which borrowed other projects' "de scene
        # 1" clips and falsely showed a never-voiced project as done — removed.)
        voiced = 0
        chv = channel_voice(sheet, code, cfg_ps)  # channel voice (engine-compatible only)
        try:
            scenes = load_scenes(sheet, code,
                                 narration_file(sheet.parent, pid, code))
            # Count against the voice this project would actually use — the
            # channel voice when set, else the per-language default — so the
            # dashboard reflects the clips a run would produce.
            vp = tts.voice_paths(scenes, code, p_shared['voicecache'], voice=chv or None)
            voiced = sum(1 for v in vp
                         if v.exists() and v.stat().st_size > 1024)
        # SystemExit (not an Exception) is what sheet.load raises for a language
        # whose narration can't be found. Catch it here so one missing
        # narration file degrades to "not voiced" instead of failing the dashboard.
        except (Exception, SystemExit):
            scenes = []

        # Which voice reads this language, for the project page. Show the voice
        # that will ACTUALLY narrate — the channel voice if the project's channel
        # set one, else the language's own default: a Google Chirp voice name when
        # Chirp is the engine, otherwise a reference clip. (Previously this always
        # showed the reference clip label, so a Chirp/channel voice was hidden
        # behind a stale clip name.)
        pref = V.pref_for(code)
        ref = pref.get("reference", "")
        gv = pref.get("google_voice", "")
        eff_voice = chv or (gv if chirp_engine else ref)   # chv computed above
        is_clip = bool(eff_voice) and (
            "/" in eff_voice
            or eff_voice.rsplit(".", 1)[-1].lower() in
            ("mp3", "wav", "m4a", "flac", "ogg", "aac"))
        if is_clip:
            voice_label = V.label_for(eff_voice)
            try:
                V.resolve(eff_voice)
                voice_ok = True
            except FileNotFoundError:
                voice_ok = False
        elif eff_voice:                          # a catalogue voice name
            voice_label = V.voice_display(eff_voice)   # nickname if set, else the name
            voice_ok = True
        else:
            voice_label, voice_ok = "", False

        out["languages"][code] = {
            "name": lg.get("name", code),
            "voice_ref": eff_voice,
            "voice_label": voice_label,
            "voice_ok": voice_ok,
            "sheets": bool(lg.get("file")) or code == mlang,
            "visuals": assets_n > 0 and assets_n >= n_scenes,
            "visuals_n": assets_n,
            "voice_n": voiced,
            # Voiced when this project's own clips exist, OR when it's already
            # rendered (the narration is baked into the finished video, so the
            # cached WAVs may have rotated / been made under an older voice).
            "voice": n_scenes > 0 and (voiced >= n_scenes or mp4.exists()),
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
            scenes = load_scenes(sheet, code, narration_file(sheet.parent, pid, code))
            chv = channel_voice(sheet, code) or None      # the voice actually in use
            for f in tts.voice_paths(scenes, code, shared["voicecache"], voice=chv):
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
    # Tidy up the project folder afterwards.
    proj = _pid_of_dir(sheet)
    if _inside(proj, PROJECTS) and proj.exists():
        if "sheets" in what:
            # Deleting the PROJECT itself: remove the whole folder, including any
            # stray files the groups above didn't cover (logs, .DS_Store, caches
            # written mid-run) — otherwise a hollow shell is left behind. rmtree,
            # not rmdir, so it never fails just because something remains.
            try:
                freed += sum(x.stat().st_size for x in proj.rglob("*") if x.is_file())
                shutil.rmtree(proj)
                removed.append(proj.name + "/")
            except OSError as e:
                refused.append(f"{proj.name}: {e}")
        else:
            # Partial delete: only drop any now-empty subfolders, leaving the
            # project in place.
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
    the main script's own (it reads from the main script) or has no sheet yet."""
    main_script = sheets_dir / f"{pid}_main_script.md"
    if main_script.exists() and main_lang(main_script) == lang:
        return None
    for nf in sorted(sheets_dir.glob(f"{pid}_*_narration.md")):
        if _narration_lang(pid, nf.stem) == lang:
            return nf
    return None


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


CONFIG_FILE = ROOT / "config.json"


def read_config_file() -> dict:
    """The raw config.json exactly as stored — every key, including empty/false
    values and the `_label` docs, and no environment merge. load_config() is for
    RUNNING the pipeline (it drops blanks and folds in env vars); this is for the
    Settings editor, which must see and round-trip the real file faithfully."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def write_config_file(data: dict) -> Path:
    """Save config.json atomically, keeping a timestamped backup first.

    Written to a temp file in the same folder then os.replace()'d in, so a crash
    mid-write can never leave a half-written config. The backup name matches the
    gitignored config.json.bak-* pattern, so it's never committed."""
    import os
    import shutil
    import tempfile
    import time
    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE,
                     CONFIG_FILE.with_name(f"config.json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
    # Temp file in the TARGET's own folder, so os.replace() is a same-filesystem
    # atomic rename (a temp on another mount would raise a cross-device error).
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_FILE.parent), prefix=".config.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, CONFIG_FILE)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return CONFIG_FILE


def detailed_log(cfg: dict | None = None) -> bool:
    """Whether the live Output should be verbose (config log_detail = full).
    One helper so every step decides the same way."""
    v = (cfg if cfg is not None else load_config()).get("log_detail", "normal")
    return str(v).strip().lower() in ("full", "all", "verbose", "on", "true")


def _flag(v, default: bool = True) -> bool:
    """Read a config value as an on/off switch. 'auto'/'on'/True are on;
    'off'/'no'/'false'/'0'/False are off. Missing falls back to `default`."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("off", "false", "no", "0", "none", "")


def load_scenes(sheet: Path, lang: str, narration: Path | None):
    return sheetlib.load(sheet, lang, narration)


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
    tr = narration_file(sheet.parent, pid, lang)
    scenes = load_scenes(sheet, lang, tr)
    narration = " ".join(s.narration for s in scenes if s.narration).strip()
    if not narration:
        raise RuntimeError(
            f"No {LANG_NAMES.get(lang, lang)} narration to describe yet — "
            f"generate the script for this language first.")

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

    Runs on whatever LLM is configured (Gemini, Vertex, Ollama, …); a no-op when
    none is, so it never blocks sourcing. Results are cached in the project's work
    folder keyed by the scene's own query, so a re-source spends no tokens and
    only genuinely new/changed scenes are sent.
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


def visuals_complete(sheet: Path) -> bool:
    """True when every scene already has a sourced visual on disk.

    Lets the hands-off 'Auto-process the rest' resume at voice + render instead
    of re-running the whole source step when the pictures were already found (by
    hand, or by an earlier run). Placeholders count as sourced — they render — so
    this asks 'is anything still to fetch?', not 'is every match perfect?'."""
    p = paths_for(sheet, "en")
    if not p["assets"].exists():
        return False
    try:
        assets = json.loads(p["assets"].read_text(encoding="utf-8"))
        have = {int(k) for k in assets}
        scenes = load_scenes(sheet, main_lang(sheet), None)
    except Exception:
        return False
    return bool(scenes) and all(s.n in have for s in scenes)


def source_stock(scenes, sheet: Path, cfg: dict, redo: list[int] | None = None,
                 on_progress=noop, log=noop, should_cancel=None) -> dict[int, dict]:
    """Fetch a visual per scene. Visuals are language-independent, so this is
    done once per project and reused by every language.

    `log` receives the detailed per-scene feedback (searches, scores, the pick);
    `on_progress` drives the progress bar. They are separate so the live Output
    can show real detail without the bar's bare labels cluttering it."""
    cfg = _cfg_with_aspect(sheet, cfg)           # generate 9:16 for a Short project
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

    # Persist the whole sourcing log to the project so it survives the run and
    # can be read later — the live Output panel only keeps the last few hundred
    # lines. Every line the pipeline logs is teed to work/sourcing.log verbatim.
    log_path = p["base"].parent / "sourcing.log"
    _logf = None
    try:
        _logf = log_path.open("w", encoding="utf-8")
        _logf.write(f"Sourcing log · {time.strftime('%Y-%m-%d %H:%M:%S')} · "
                    f"{len(todo)} scene(s)\n")
    except Exception:
        _logf = None
    user_log = log

    def log(msg):                       # tee: live panel + the saved file
        user_log(msg)
        if _logf is not None:
            try:
                _logf.write(str(msg).rstrip("\n") + "\n")
                _logf.flush()
            except Exception:
                pass

    try:
        rel_log = log_path.relative_to(ROOT)
    except ValueError:
        rel_log = log_path
    try:
        fresh = stock.fetch_all(
            todo, p["stockcache"], cfg.get("pexels_key"), cfg.get("pixabay_key"),
            picks=picks, log=log, cfg=cfg, already=keep,
            on_progress=on_progress, should_cancel=should_cancel)
    finally:
        if _logf is not None:
            _logf.write(f"\nSaved {rel_log}\n")
            _logf.close()
    log(f"Full step-by-step log saved to {rel_log}")
    assets.update(fresh)

    p["assets"].write_text(
        json.dumps({str(k): v for k, v in assets.items()}, indent=2),
        encoding="utf-8")
    return assets


def _gen_asset_name(pid: str, n: int, prefix: str, ext: str) -> str:
    """A collision-proof cache filename for a MANUALLY (re)generated scene asset.

    The stock cache directory is shared by every project. A name keyed only by
    scene number and take — gen_5_1.png — therefore collides ACROSS projects:
    regenerating scene 5 in one project overwrites the very file another
    project's scene 5 points to, so that project's review and render silently
    show the wrong (or a seemingly random) picture. This was the real cause of
    "regenerate changed other scenes / put a random image".

    Keying by a hash of the project id plus a fresh random token gives every
    generation its own file: no cross-project collision, and a new take never
    reuses (or serves a browser-cached) older filename. Scene number stays in
    the name for readability."""
    key = hashlib.sha1(str(pid).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{key}_{n}_{uuid.uuid4().hex[:8]}.{ext}"


UPLOAD_IMAGE_EXT = ("png", "jpg", "jpeg", "webp", "gif", "bmp")
UPLOAD_VIDEO_EXT = ("mp4", "mov", "webm", "m4v")


def set_scene_upload(sheet: Path, n: int, raw: bytes, ext: str) -> dict:
    """Replace scene `n`'s visual with a file the user uploaded (image OR video).

    The bytes are written to the shared cache under a project-namespaced, unique
    name (same scheme as generated assets, so it can never collide with another
    project or an earlier take), and ONLY that scene is repointed in assets.json —
    every other scene is left exactly as it was. Returns the new asset's basename
    and media kind. Raises ValueError for an unsupported type or an empty file, so
    the caller can report it without leaving the scene half-changed."""
    ext = (ext or "").lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    media = ("VIDEO" if ext in UPLOAD_VIDEO_EXT
             else "IMAGE" if ext in UPLOAD_IMAGE_EXT else "")
    if not media:
        raise ValueError("unsupported file type — use an image (JPG/PNG/WebP/GIF) "
                         "or a video (MP4/MOV/WebM)")
    if not raw:
        raise ValueError("the uploaded file is empty")

    p = paths_for(sheet, "en")                       # assets.json is shared
    p["stockcache"].mkdir(parents=True, exist_ok=True)
    fname = _gen_asset_name(p["id"], n, "up", ext)
    dest = p["stockcache"] / fname
    dest.write_bytes(raw)

    assets: dict[int, dict] = {}
    if p["assets"].exists():
        assets = {int(k): v for k, v in
                  json.loads(p["assets"].read_text(encoding="utf-8")).items()}
    prev = assets.get(n) or {}
    assets[n] = {"path": str(dest), "src": "upload", "query": prev.get("query", ""),
                 "media": media, "credit": "Uploaded by you", "page": "",
                 "license": "user-provided", "score": None, "generated": False,
                 "uploaded": True}
    p["assets"].write_text(
        json.dumps({str(k): v for k, v in assets.items()}, indent=2),
        encoding="utf-8")
    return {"n": n, "file": fname, "media": media, "path": str(dest)}


def generate_scenes(scenes, sheet: Path, cfg: dict, which: list[int],
                    on_progress=noop, log=noop, should_cancel=None) -> dict:
    """Generate a fresh AI image for each scene number in `which`, replacing its
    asset. This is the MANUAL, on-demand path from the review page — distinct
    from the automatic `generate` modes in source_stock.

    Each call makes a NEW take (the picks counter is bumped and put in the
    filename), so clicking 'generate' again gives a different image rather than
    the cached one. A scene whose generation is refused — a safety filter, or a
    named real person Imagen will not render — keeps its current picture and is
    reported, never left empty.
    """
    from . import imagen
    if not imagen.available(cfg):
        raise SystemExit(
            "Image generation needs \"vertex_project\" in config.json — the same "
            "Vertex setup the LLM uses. Add it, then try again.")
    cfg = _cfg_with_aspect(sheet, cfg)           # 9:16 for a Short project

    p = paths_for(sheet, "en")
    p["stockcache"].mkdir(parents=True, exist_ok=True)
    p["assets"].parent.mkdir(parents=True, exist_ok=True)     # work/
    assets: dict[int, dict] = {}
    if p["assets"].exists():
        assets = {int(k): v for k, v in
                  json.loads(p["assets"].read_text(encoding="utf-8")).items()}
    picks: dict[int, int] = {}
    if p["picks"].exists():
        picks = {int(k): v for k, v in
                 json.loads(p["picks"].read_text(encoding="utf-8")).items()}

    by_n = {s.n: s for s in scenes}
    want = [n for n in which if n in by_n]
    generated: list[int] = []
    failed: list[tuple] = []

    # Generate CONCURRENTLY — up to generate_workers images in flight at once,
    # instead of one-at-a-time. Each is an independent imagen.image() call, so the
    # Vertex region pool spreads them over separate regional backends / quota
    # buckets and the shared adaptive throttle still keeps them from stampeding.
    # A batch of N scenes now finishes in roughly N/workers of the old wall time.
    for n in want:                                    # bump takes up front (single-thread)
        picks[n] = picks.get(n, 0) + 1                # so each is a fresh, deterministic take
    workers = max(1, int(cfg.get("generate_workers") or 1))
    _lock = threading.Lock()                          # serialises the log callback only
    _done = [0]
    _stopped = threading.Event()

    def _safelog(msg: str) -> None:
        with _lock:
            log(msg)

    def _gen_one(n: int) -> tuple:
        """Runs on a worker thread: returns ('ok'|'fail'|'cancel', n, payload, err)."""
        if _stopped.is_set() or (should_cancel and should_cancel()):
            return ("cancel", n, None, None)
        s = by_n[n]
        prompt = imagen.prompt_for(s.query or getattr(s, "narration", "") or "", cfg)
        dest = p["stockcache"] / _gen_asset_name(p["id"], n, "gen", "png")
        det: dict = {}                                # filled with the exact model/region used
        try:
            eng = imagen.image(prompt, cfg, dest, log=_safelog,
                               should_cancel=should_cancel, detail=det)
        except imagen.Cancelled:                      # Stop pressed mid-wait
            _stopped.set()
            return ("cancel", n, None, None)
        except Exception as e:                        # GenError or anything else
            return ("fail", n, None, str(e))
        label = det.get("label") or imagen.engine_label(eng)
        rec = {"path": str(dest), "src": eng or "imagen", "query": s.query,
               "model": det.get("model", ""),         # e.g. gemini-2.5-flash-image@us-east4
               "media": "IMAGE",
               "credit": f"AI-generated ({label})",
               "page": "", "license": "AI-generated", "score": None,
               "generated": True}
        return ("ok", n, (label, rec), None)

    def _handle(res: tuple) -> None:                  # main thread only — no lock needed
        kind, n, payload, err = res
        _done[0] += 1
        on_progress(_done[0], len(want), f"S{n} generating")
        if kind == "ok":
            label, rec = payload
            assets[n] = rec
            generated.append(n)
            _safelog(f"✦ S{n:>3} · {label} · \"{(by_n[n].query or '')[:46]}\"")
        elif kind == "fail":
            failed.append((n, err))
            _safelog(f"✗ S{n:>3} · could not generate · {str(err)[:80]}")
        # 'cancel' → just counted; Stop is reported once after the loop

    if workers > 1 and len(want) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_gen_one, n): n for n in want}
            for fut in as_completed(futs):
                _handle(fut.result())
    else:
        for n in want:
            if _stopped.is_set() or (should_cancel and should_cancel()):
                break
            _handle(_gen_one(n))
    if _stopped.is_set():
        log("Stopped.")
    generated.sort()

    p["assets"].write_text(
        json.dumps({str(k): v for k, v in assets.items()}, indent=2),
        encoding="utf-8")
    p["picks"].write_text(
        json.dumps({str(k): v for k, v in picks.items()}, indent=2),
        encoding="utf-8")
    return {"generated": generated, "failed": failed, "assets": assets}


def generate_videos(scenes, sheet: Path, cfg: dict, which: list[int],
                    on_progress=noop, log=noop, should_cancel=None) -> dict:
    """Generate a Veo video clip for each scene in `which` — the MANUAL, capped,
    expensive path. Only ever called from the review page, one clip at a time,
    and never more than `veo_max` per run so a slip of the finger can't spend the
    whole credit. A refused clip keeps the scene's current picture and is
    reported. The renderer treats any .mp4 asset as a video, so nothing else in
    the build needs to change.
    """
    from . import veo
    if not veo.available(cfg):
        raise SystemExit(
            "Video generation needs \"vertex_project\" in config.json — the same "
            "Vertex setup the LLM uses. Add it, then try again.")
    cfg = _cfg_with_aspect(sheet, cfg)           # 9:16 Veo clips for a Short project

    cap = max(1, int(cfg.get("veo_max") or 2))
    p = paths_for(sheet, "en")
    p["stockcache"].mkdir(parents=True, exist_ok=True)
    p["assets"].parent.mkdir(parents=True, exist_ok=True)
    assets: dict[int, dict] = {}
    if p["assets"].exists():
        assets = {int(k): v for k, v in
                  json.loads(p["assets"].read_text(encoding="utf-8")).items()}
    picks: dict[int, int] = {}
    if p["picks"].exists():
        picks = {int(k): v for k, v in
                 json.loads(p["picks"].read_text(encoding="utf-8")).items()}

    by_n = {s.n: s for s in scenes}
    want = [n for n in which if n in by_n]
    skipped = want[cap:]                              # beyond the per-run cap
    want = want[:cap]
    generated: list[int] = []
    failed: list[tuple] = []

    # Craft a proper cinematic prompt per scene with the LLM: subject + a clear
    # ACTION that matches the narration + one camera move + documentary look, so
    # Veo animates the right thing instead of guessing from a noun-heavy search
    # phrase. Best effort — falls back to the plain prompt per scene.
    crafted: dict[int, str] = {}
    from . import llm as LLM
    if _flag(cfg.get("veo_smart_prompt", "auto")) and LLM.available(cfg):
        try:
            from . import gemini as G
            on_progress(0, len(want), "writing video prompts")
            crafted = G.video_prompts(
                [{"n": n, "query": by_n[n].query,
                  "narration": getattr(by_n[n], "narration", "")} for n in want],
                LLM.key_for(cfg), LLM.model_for(cfg))
        except Exception as e:
            log(f"  (couldn't craft smart prompts: {e}) — using plain prompts")

    for i, n in enumerate(want):
        if should_cancel and should_cancel():
            break
        s = by_n[n]
        on_progress(i + 1, len(want), f"S{n} rendering video (1–2 min)")
        picks[n] = picks.get(n, 0) + 1               # bump take (history/telemetry)
        prompt = crafted.get(n) or veo.prompt_for(
            s.query or getattr(s, "narration", "") or "", cfg)
        log(f"S{n:>3} video prompt: {prompt[:150]}")
        dest = p["stockcache"] / _gen_asset_name(p["id"], n, "veo", "mp4")
        try:
            veo.video(prompt, cfg, dest, should_cancel=should_cancel,
                      on_wait=lambda i=i: on_progress(i + 1, len(want),
                                                      f"S{want[i]} rendering video…"))
        except veo.Cancelled:                         # Stop pressed while rendering
            log(f"Stopped before S{n}.")
            break
        except Exception as e:
            failed.append((n, str(e)))
            log(f"✗ S{n:>3} · could not generate video · {str(e)[:80]}")
            continue
        assets[n] = {"path": str(dest), "src": "veo", "query": s.query,
                     "media": "VIDEO", "credit": "AI-generated (Veo)", "page": "",
                     "license": "AI-generated", "score": None, "generated": True}
        generated.append(n)
        log(f"✦ S{n:>3} video · generated · \"{(s.query or '')[:46]}\"")

    p["assets"].write_text(
        json.dumps({str(k): v for k, v in assets.items()}, indent=2),
        encoding="utf-8")
    p["picks"].write_text(
        json.dumps({str(k): v for k, v in picks.items()}, indent=2),
        encoding="utf-8")
    return {"generated": generated, "failed": failed, "skipped": skipped,
            "assets": assets}


_VOICE_SCENE_LINE = re.compile(r"S\s*\d+\s+(?:voiced|cached)", re.I)


def generate_voice(scenes, lang: str, sheet: Path, voice: str | None = None,
                   on_progress=noop) -> list[Path]:
    """`voice` overrides the voice for this language (a Google voice name under
    Chirp, else a reference clip). When the caller passes none, the project's
    CHANNEL voice is used if its channel set one — so every project in a channel
    narrates with the channel's voice — else the global per-language default."""
    explicit = bool(voice)                       # a voice the caller passed in
    if not voice:
        voice = channel_voice(sheet, lang) or None
    p = paths_for(sheet, lang)
    done = [0]
    total = len(scenes)

    def log(msg: str) -> None:
        # Forward EVERY line the engine emits to the live Output — the engine
        # header (which engine / model / device / voice), any fallback notice,
        # per-scene lines and the final summary — so the log is self-explanatory.
        # Only a real per-scene line ("S 3 voiced/cached …") advances the bar; a
        # line that merely starts with 'S' (e.g. "Switched to Chatterbox…") must
        # not be miscounted.
        line = msg if isinstance(msg, str) else str(msg)
        if _VOICE_SCENE_LINE.match(line.lstrip()):
            done[0] = min(total, done[0] + 1)
        on_progress(done[0], total, line.rstrip())

    # Where the voice came from, so Activity says WHY this voice is being used.
    if explicit:
        source = "this run's own choice"
    elif voice:                                  # came from the channel (compatible)
        cname = ""
        try:
            from . import channels as _ch
            cname = _ch.of(project_id(sheet))
        except Exception:
            cname = ""
        source = f'channel “{cname}”' if cname else "the project's channel"
    else:
        source = "Settings default (per-language)"
    log(f"Voice source · {source}")

    return tts.synth(scenes, lang, p["voicecache"], voice=voice, log=log)


def _aligned_words(scenes, voices, vdurs, starts, lang, p, on_progress, n,
                   lead: float = 0.0):
    """Per-scene word timings, in ABSOLUTE video time, for the karaoke captions.

    Each scene is aligned against its own audio and cached in the language's work
    folder keyed by the exact narration text, so a caption-only re-render (or a
    second language sharing nothing) never realigns a scene whose words haven't
    changed. Returns one word-list per scene: [{word, start, end}, ...].

    `lead` pulls every word a touch earlier so the highlight lands ON the word
    (or a hair before) rather than trailing it — the tiny anticipation that makes
    pro karaoke captions feel locked to the voice instead of lagging.
    """
    cfg = load_config()

    # Say up front how the words are being timed, so it's obvious in Activity
    # whether real forced alignment ran or it fell back to an estimate.
    capinfo = align.capability(cfg)
    if capinfo.get("ok"):
        on_progress(n + 3, n + 4,
                    f"timing words · {capinfo['engine']} ({capinfo.get('device', '-')})")
    else:
        on_progress(n + 3, n + 4,
                    f"timing words · estimated ({capinfo.get('reason', 'no aligner')}) "
                    f"— install torchaudio for exact word sync")

    cache_f = p["base"] / "words.json"
    cache: dict = {}
    if cache_f.exists():
        try:
            cache = json.loads(cache_f.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    out, changed = [], False
    for i, s in enumerate(scenes):
        # Align and DISPLAY the same cleaned text the engine actually spoke, so
        # the captions never show "#5 -" or "Hook:" the voice didn't say, and the
        # word highlight stays in step. Keyed on the cleaned text so a change to
        # the cleaner re-aligns instead of reusing stale word timings.
        clean = _speech(s.narration)
        rec = cache.get(str(s.n))
        if not (rec and rec.get("text") == clean and rec.get("words")):
            words = align.align_words(
                voices[i], clean, lang, cfg=cfg, dur=vdurs[i],
                log=lambda m: on_progress(n + 3, n + 4, m.strip()))
            cache[str(s.n)] = rec = {"text": clean, "words": words}
            changed = True
        # Relative -> absolute (and lead-shifted), so every scene's words sit at
        # the right moment in the finished audio, a hair ahead for a locked feel.
        out.append([{"word": w["word"],
                     "start": round(max(0.0, (w.get("start") or 0.0) + starts[i] - lead), 3),
                     "end": round(max(0.0, (w.get("end") or 0.0) + starts[i] - lead), 3)}
                    for w in rec["words"]])

    if changed:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return out


def _clip_fingerprint(src: Path, target: float, zoom: bool,
                      size: tuple | None = None) -> str:
    """Identity of the per-scene clip `c{n}.mp4` — everything that, if changed,
    means the cached clip no longer represents this scene. Crucially it includes
    the SOURCE picture (path + size + mtime), so swapping a scene's image in
    review (by search OR by AI) forces its clip to be rebuilt instead of the old
    one being silently reused. Target length, zoom AND the FRAME SIZE are folded
    in too, so a timing/effect change — or switching a project between video (16:9)
    and short (9:16) — also invalidates the cache and rebuilds at the new shape."""
    try:
        st = src.stat()
        sig = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        sig = "missing"
    dims = f"|{size[0]}x{size[1]}" if size else ""
    return f"{src}|{sig}|t={round(float(target), 3)}|z={int(bool(zoom))}{dims}"


def render_video(scenes, assets: dict[int, dict], voices: list[Path], sheet: Path,
                 lang: str, captions: bool = True, music: Path | None = None,
                 music_level: float = 0.20, zoom: bool = True,
                 caption_size: int = 58, style=None, master: bool = True,
                 on_progress=noop) -> Path:
    p = paths_for(sheet, lang)
    for d in (p["clips"], p["tmp"], p["out"].parent):
        d.mkdir(parents=True, exist_ok=True)

    # The output shape for THIS project (16:9 video or 9:16 short). Set it on the
    # render and caption modules for the whole run, so clips are built and captions
    # positioned at the right frame size. Thread-safe per render.
    _orient = orientation(sheet)
    _fw, _fh = _orient["w"], _orient["h"]
    render.set_frame(_fw, _fh)
    cap.set_frame(_fw, _fh)

    missing = [s.n for s in scenes if s.n not in assets]
    if missing:
        raise RuntimeError(f"No visual for scenes {missing}. Re-run the visuals step.")

    # Timing knobs, all overridable in config.json:
    #   scene_gap      silence between narration lines (breathing room)
    #   scene_dissolve crossfade length between pictures
    #   caption_lead   how far ahead of the spoken word the highlight sits
    # TAIL (held picture after each line) is derived so the audio gap and the
    # video hold stay in lockstep — the invariant the whole timing model rests on.
    rcfg = load_config()
    DISS = max(0.0, float(rcfg.get("scene_dissolve") or DISSOLVE))
    GAP = float(rcfg.get("scene_gap", 0.35) or 0.0)
    # A much smaller gap between fragments of ONE sentence, so a sentence split
    # across scenes flows instead of stopping at every cut. 0 = butt them right up.
    FLOW_GAP = max(0.0, float(rcfg.get("scene_flow_gap", 0.06) or 0.0))
    lead = max(0.0, float(rcfg.get("caption_lead", 0.12) or 0.0))
    verbose = detailed_log(rcfg)
    n = len(scenes)
    # Per-scene gap AFTER each line: tiny inside a sentence, full at a sentence end.
    flow = _flow_flags(scenes)
    gap_after = [FLOW_GAP if flow[i] else GAP for i in range(n)]
    TAIL_L = GAP + DISS      # kept for the fallback/last-scene hold
    if verbose:
        on_progress(0, n + 4,
                    f"timing · gap {GAP:.2f}s · dissolve {DISS:.2f}s · tail {TAIL_L:.2f}s "
                    f"· caption lead {lead:.2f}s · trim_silence "
                    f"{'on' if _flag(rcfg.get('trim_silence', True)) else 'off'}")

    # Strip the dead air each TTS clip carries at its head/tail, so the only
    # silence between lines is `scene_gap`. The trimmed clips are used for
    # EVERYTHING below — duration, audio, alignment — so captions and audio can't
    # drift apart. Off via "trim_silence": false for anyone who wants the raw takes.
    if _flag(rcfg.get("trim_silence", True)):
        on_progress(0, len(scenes) + 4, "tightening narration (trimming silence)")
        trimmed = []
        for i, v in enumerate(voices):
            t = p["tmp"] / f"voice_trim_{i:04d}.wav"
            trimmed.append(render.trim_silence(Path(v), t))
        voices = trimmed

    clips, vdurs = [], []
    for i, s in enumerate(scenes):
        vd = render.duration_of(voices[i])
        vdurs.append(vd)
        src = Path(assets[s.n]["path"])
        out = p["clips"] / f"c{s.n:04d}.mp4"
        fp_file = p["clips"] / f"c{s.n:04d}.src"
        # The picture holds voice + this scene's own gap + the dissolve. Because
        # the audio gap after this line is gap_after[i], the narration still lands
        # exactly on each clip start (the render's core invariant holds per scene).
        target = vd + gap_after[i] + DISS
        # Decide whether the cached clip can be reused, or must be rebuilt:
        #   • wrong pixel format — left over from before the 4:2:0 pin, and would
        #     drag the finished video back to an unplayable format. yuvj420p is
        #     fine (still 4:2:0, plays everywhere; only the colour range differs).
        #   • wrong length — trimming narration or changing a gap would otherwise
        #     silently reuse an old, too-long clip and re-open the gap.
        #   • DIFFERENT SOURCE — the fingerprint changed, i.e. the scene's picture
        #     was swapped in review (search or AI). Without this the render happily
        #     reuses the clip built from the OLD image, which is exactly the "I
        #     changed it but the video still shows the old one" bug.
        want_fp = _clip_fingerprint(src, target, zoom, (_fw, _fh))
        have_fp = fp_file.read_text(encoding="utf-8") if fp_file.exists() else ""
        stale = out.exists() and (
            have_fp != want_fp
            or render.pix_fmt_of(out) not in ("yuv420p", "yuvj420p")
            or abs(_dur_safe(out) - target) > (1.5 / render.FPS))
        was_built = (not out.exists()) or stale
        if was_built:
            if src.suffix.lower() in (".mp4", ".mov", ".webm"):
                render.make_video_clip(src, target, out)
            else:
                render.make_image_clip(src, target, out, zoom=zoom)
            fp_file.write_text(want_fp, encoding="utf-8")   # remember what we built
        clips.append((out, render.duration_of(out)))
        msg = f"scene {i + 1} of {n}"
        if verbose:
            kind = "video" if src.suffix.lower() in (".mp4", ".mov", ".webm") else "image"
            msg += (f" · {kind} · voice {vd:.1f}s → clip {clips[-1][1]:.1f}s"
                    f" · {'built' if was_built else 'cached'}")
        on_progress(i + 1, n + 4, msg)

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
        render.dissolve_concat(clips, DISS, vid, p["tmp"], group=10)

    on_progress(n + 2, n + 4, "assembling narration")
    gaps = [max(0.0, cd - vd - DISS) for (_, cd), vd in zip(clips, vdurs)]
    aud = p["base"] / "audio_track.wav"
    starts = render.build_audio(voices, gaps, aud, p["tmp"], tail=DISS)
    if verbose:
        on_progress(n + 2, n + 4,
                    f"audio track {_dur_safe(aud):.1f}s · {n} lines · "
                    f"{sum(vdurs):.1f}s speech + {sum(gaps):.1f}s gaps")

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
            info = render.master_audio(aud, mastered, lufs=lufs)
            aud = mastered
            if verbose:
                on_progress(n + 3, n + 4,
                            f"mastered · target {info.get('target_lufs')} LUFS · "
                            f"peak ceiling {info.get('tp')} dBTP · "
                            f"{'2-pass measured' if info.get('measured') else '1-pass'}")
        except Exception as e:
            on_progress(n + 3, n + 4, f"mastering skipped ({e})")

    on_progress(n + 3, n + 4, "muxing")
    silent = p["base"] / "muxed.mp4"
    render.mux(vid, aud, silent)

    # Captions show the cleaned, spoken text — matching the audio, not the raw
    # sheet line (which may carry "#5 -", "Hook:" etc.).
    texts = [_speech(s.narration) for s in scenes]
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
        # Shorts: punchy captions for a phone screen, lifted clear of the bottom
        # Shorts UI (progress bar, title, buttons) that covers ~12% of frame. The
        # frame is only 1080px wide (vs 1920), so keep lines SHORT — fewer words
        # per line and a modest size bump, not the wider frame's 5 big words, or a
        # long line runs off both edges. captions.py shrinks any phrase that would
        # still overflow (fit-to-width), so nothing ever clips regardless.
        if _orient["format"] == "short":
            st = st.merged(size=int(round(st.size * 1.2)),
                           max_words=min(st.max_words, 3),
                           margin_v=max(st.margin_v, int(round(_fh * 0.13))))

        # Word-by-word timing. Aligned once per scene against its own audio and
        # cached, so a caption-only re-render doesn't realign 100+ clips.
        on_progress(n + 3, n + 4, "timing the words")
        scene_words = _aligned_words(scenes, voices, vdurs, starts, lang, p,
                                     on_progress, n, lead=lead)
        groups = cap.groups_from_scenes(scene_words, st)
        p["ass"].write_text(cap.build_ass(groups, st), encoding="utf-8")
        if verbose:
            words = sum(len(g.get("words", [])) for g in groups)
            on_progress(n + 3, n + 4,
                        f"captions · style '{st.name}' · {len(groups)} phrases · "
                        f"{words} words · lead {lead:.2f}s")

        on_progress(n + 3, n + 4, "burning captions")
        try:
            method = render.burn_captions(silent, p["ass"], p["out"], texts=texts,
                                          starts=starts, durs=vdurs,
                                          size=st.size)
            if verbose:
                on_progress(n + 3, n + 4, f"caption method · {method}")
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
