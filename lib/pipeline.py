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

from . import render, sheet as sheetlib, stock, tts

ROOT = Path(__file__).resolve().parent.parent

TAIL = 1.0        # seconds of held picture after each narration line
DISSOLVE = 0.6    # crossfade length between scenes

LANG_NAMES = {"en": "English", "de": "German", "es": "Spanish",
              "fr": "French", "it": "Italian", "pt": "Portuguese"}
# How translation files are named, e.g. video04_GERMAN_narration.md
LANG_FILE_WORDS = {"de": ("GERMAN", "DE"), "es": ("SPANISH", "ES"),
                   "fr": ("FRENCH", "FR"), "it": ("ITALIAN", "IT"),
                   "pt": ("PORTUGUESE", "PT")}


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


def find_projects(sheets_dir: Path) -> list[dict]:
    """Every master sheet in sheets/, with the translations sitting next to it."""
    out = []
    for f in sorted(sheets_dir.glob("*.md")):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        if "- Narration:" not in txt:
            continue  # a translation file, not a master sheet
        pid = project_id(f)
        langs = [{"code": "en", "name": "English", "file": None}]
        for code, words in LANG_FILE_WORDS.items():
            for w in words:
                hits = [g for g in sheets_dir.glob(f"{pid}*{w}*.md")]
                if hits:
                    langs.append({"code": code, "name": LANG_NAMES.get(code, code),
                                  "file": hits[0].name})
                    break
        try:
            n = len(sheetlib.parse_master(f))
        except SystemExit:
            n = 0
        out.append({"id": pid, "sheet": f.name, "label": pretty_name(f),
                    "scenes": n, "languages": langs})
    return out


def translation_for(sheets_dir: Path, pid: str, lang: str) -> Path | None:
    if lang == "en":
        return None
    for w in LANG_FILE_WORDS.get(lang, ()):
        hits = list(sheets_dir.glob(f"{pid}*{w}*.md"))
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------------------ paths

def paths_for(sheet: Path, lang: str) -> dict:
    pid = project_id(sheet)
    base = ROOT / "work" / f"{pid}_{lang}"
    return {
        "id": pid, "base": base, "clips": base / "clips", "tmp": base / "tmp",
        "stockcache": ROOT / "cache" / "stock", "voicecache": ROOT / "cache" / "voice",
        "picks": ROOT / "work" / f"{pid}_picks.json",       # shared by all languages
        "assets": ROOT / "work" / f"{pid}_assets.json",     # shared by all languages
        "approval": ROOT / "out" / f"{pid}_approval.html",
        "out": ROOT / "out" / f"{pid}_{lang}.mp4",
        "srt": ROOT / "out" / f"{pid}_{lang}.srt",
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
            cfg.update({k: v for k, v in json.loads(f.read_text()).items()
                        if v and not k.startswith("_")})
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"config.json is not valid JSON ({e}).\n"
                f"Most often this is curly quotes from a word processor. Reopen "
                f"config.json in a plain text editor (Notepad, TextEdit) and "
                f"retype the quote marks."
            )
    return cfg


def load_scenes(sheet: Path, lang: str, translation: Path | None):
    return sheetlib.load(sheet, lang, translation)


# ------------------------------------------------------------------ steps

def source_stock(scenes, sheet: Path, cfg: dict, redo: list[int] | None = None,
                 on_progress=noop) -> dict[int, dict]:
    """Fetch a visual per scene. Visuals are language-independent, so this is
    done once per project and reused by every language."""
    p = paths_for(sheet, "en")
    p["base"].parent.mkdir(parents=True, exist_ok=True)

    picks: dict[int, int] = {}
    if p["picks"].exists():
        picks = {int(k): v for k, v in json.loads(p["picks"].read_text()).items()}
    for n in (redo or []):
        picks[n] = picks.get(n, 0) + 1
    p["picks"].write_text(json.dumps(picks, indent=2))

    assets: dict[int, dict] = {}
    if p["assets"].exists():
        assets = {int(k): v for k, v in json.loads(p["assets"].read_text()).items()}

    todo = [s for s in scenes if redo is None or s.n in redo or s.n not in assets]
    for i, s in enumerate(todo):
        try:
            assets[s.n] = stock.fetch(s.query, s.media, p["stockcache"],
                                      cfg.get("pexels_key"), cfg.get("pixabay_key"),
                                      picks.get(s.n, 0))
            on_progress(i + 1, len(todo), f"S{s.n} {s.media.lower()} sourced")
        except stock.StockError as e:
            on_progress(i + 1, len(todo), f"S{s.n} no match — {e}")

    p["assets"].write_text(json.dumps({str(k): v for k, v in assets.items()}, indent=2))
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


def render_video(scenes, assets: dict[int, dict], voices: list[Path], sheet: Path,
                 lang: str, captions: bool = True, music: Path | None = None,
                 music_level: float = 0.20, zoom: bool = True,
                 caption_size: int = 58, on_progress=noop) -> Path:
    p = paths_for(sheet, lang)
    for d in (p["clips"], p["tmp"], ROOT / "out"):
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

    if music:
        mixed = p["base"] / "audio_mixed.wav"
        render.mix_music(aud, Path(music), mixed, level=music_level)
        aud = mixed

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
        on_progress(n + 3, n + 4, "burning captions")
        render.write_ass(texts, starts, vdurs, p["ass"], size=caption_size)
        try:
            method = render.burn_captions(silent, p["ass"], p["out"], texts=texts,
                                          starts=starts, durs=vdurs,
                                          size=caption_size)
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
