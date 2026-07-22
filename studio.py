#!/usr/bin/env python3
"""Faceless Studio — a local control panel for the pipeline.

Double-click Start.bat (Windows) or Start.command (macOS),
or run:  python3 studio.py

Serves a small web app on 127.0.0.1 only. Nothing is uploaded anywhere; the
browser is just a nicer front end for the same pipeline the CLI uses.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _venv_python() -> Path | None:
    """Where this project's interpreter lives, on either platform."""
    venv = ROOT / ".venv"
    # Windows puts it in Scripts\python.exe; everything else in bin/python3.
    for rel in (("Scripts", "python.exe"), ("bin", "python3"), ("bin", "python")):
        p = venv.joinpath(*rel)
        if p.exists():
            return p
    return None


def _use_project_venv() -> None:
    """Re-run inside the project's .venv if we aren't already in it.

    Saves you from typing the full path to the venv interpreter every time.
    """
    # Are we already inside THIS project's venv? Compare prefixes, not
    # executables: on macOS .venv/bin/python3 is a symlink chain ending at the
    # same Homebrew binary you typed, so resolve() makes them look identical and
    # the handover silently never happens.
    if Path(sys.prefix) == (ROOT / ".venv"):
        return
    venv_py = _venv_python()
    if venv_py is None:
        return
    if os.environ.get("FACELESS_NO_REEXEC"):
        return
    os.environ["FACELESS_NO_REEXEC"] = "1"       # belt and braces against a loop
    args = [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]]
    if os.name == "nt":
        # os.execv on Windows lets the shell reclaim the console while the new
        # process is still running, which scrambles the output. Spawn and wait.
        import subprocess
        raise SystemExit(subprocess.run(args).returncode)
    os.execv(str(venv_py), args)




_use_project_venv()

from lib import compose, pipeline as pl  # noqa: E402
from lib import console  # noqa: E402

# Windows consoles default to a legacy codepage and die on box-drawing
# characters. Do this before anything is printed.
console.setup()
from lib import gemini as gem  # noqa: E402
from lib import render as R  # noqa: E402
from lib import tts, voices as vx  # noqa: E402
from lib import vision as VIS  # noqa: E402
from lib import captions as CAP, align as AL  # noqa: E402

PORT = 8765
UI = ROOT / "lib" / "ui.html"

# ------------------------------------------------------------------- job state

JOB = {
    "stage": "idle",      # idle | generate | stock | voice | render | done | error
    "label": "",
    "done": 0, "total": 0,
    "log": [],
    "error": "",
    "outputs": [],
    "project": None,
    "langs": [],
    "warnings": [],
    # Timing, so the interface can show real numbers instead of a spinner.
    "started": None,       # epoch seconds, whole job
    "step_started": None,  # epoch seconds, current step
    "steps": [],           # [{name, lang, seconds, items}] as each one finishes
    "eta": None,           # seconds remaining in this step, or None
    "rate": None,          # items per second in this step
    "lang": None,          # language currently being worked on
    "cancel": False,       # set by /api/cancel, checked between items
}
LOCK = threading.Lock()


def set_job(**kw) -> None:
    with LOCK:
        JOB.update(kw)


def begin_job(project: str, langs: list[str], stage: str) -> None:
    """Reset the board for a new run. Called once, before any work starts."""
    with LOCK:
        JOB.update(stage=stage, project=project, langs=langs, label="",
                   done=0, total=0, log=[], error="", outputs=[], warnings=[],
                   started=time.time(), step_started=time.time(),
                   steps=[], eta=None, rate=None, lang=None, cancel=False)


def begin_step(stage: str, lang: str | None = None) -> None:
    with LOCK:
        JOB.update(stage=stage, lang=lang, step_started=time.time(),
                   done=0, total=0, eta=None, rate=None, label="")


def end_step(items: int = 0) -> None:
    """Record how long the step took, so the UI can show a per-step breakdown."""
    with LOCK:
        t0 = JOB.get("step_started") or time.time()
        JOB["steps"].append({"name": JOB["stage"], "lang": JOB.get("lang"),
                             "seconds": round(time.time() - t0, 1),
                             "items": items or JOB.get("done", 0)})


def log(msg: str) -> None:
    with LOCK:
        JOB["log"].append({"t": time.time(), "text": str(msg)})
        del JOB["log"][:-600]


def progress(done: int, total: int, label: str = "") -> None:
    """Record progress and work out a rate and ETA from it.

    The estimate is based on elapsed time for THIS step only. Steps have wildly
    different per-item costs — sourcing a photo is a download, voicing a line is
    a GPU inference — so carrying a rate across them would produce a confidently
    wrong number, which is worse than no number.
    """
    with LOCK:
        JOB["done"], JOB["total"] = done, total
        if label:
            JOB["label"] = label
        t0 = JOB.get("step_started")
        if t0 and done > 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            JOB["rate"] = round(rate, 3) if rate else None
            remaining = max(0, (total or 0) - done)
            JOB["eta"] = round(remaining / rate) if rate > 0 and remaining else 0


RUNNING = ("generate", "stock", "voice", "render")
WORKER: list = [None]        # the one live worker thread, if any


def busy() -> bool:
    """Is a job genuinely running right now?

    Checks the thread as well as the recorded stage. If a worker died without
    tidying up, the stage alone would say 'running' forever and every later
    action would be refused with 'something is already running' — which is
    exactly what happened when a SystemExit escaped the old handler. Trusting
    the stage alone made a single crash permanent; this makes it self-healing.
    """
    with LOCK:
        stage = JOB["stage"]
    if stage not in RUNNING:
        return False
    t = WORKER[0]
    if t is not None and t.is_alive():
        return True
    # Stage says running, nothing is. Recover rather than stay wedged.
    set_job(stage="error",
            error=JOB.get("error") or
            "The last job stopped unexpectedly and left no message. "
            "It is safe to try again.")
    log("The previous job ended without reporting why — state reset.")
    return False


def _guarded(fn, *a) -> None:
    """Run a job and make sure the stage is never left mid-flight.

    Catches BaseException deliberately, not Exception. Library code raises
    SystemExit for user-facing problems (no reference clip chosen, malformed
    sheet), and SystemExit does NOT inherit from Exception — so 'except
    Exception' let the thread die silently with the job stuck at 'voice'.
    A background worker should never disappear without saying why.
    """
    try:
        fn(*a)
    except BaseException as e:                      # noqa: BLE001 - deliberate
        msg = str(e) or type(e).__name__
        set_job(stage="error", error=msg)
        log(f"ERROR: {msg}")
        traceback.print_exc()
    finally:
        with LOCK:
            if JOB["stage"] in RUNNING:
                JOB["stage"] = "done"               # never leave it hanging


def start_thread(fn, *a) -> bool:
    if busy():
        return False
    with LOCK:
        JOB["log"] = []
        JOB["error"] = ""
        JOB["cancel"] = False
    t = threading.Thread(target=_guarded, args=(fn, *a), daemon=True)
    WORKER[0] = t
    t.start()
    return True


def cancelled() -> bool:
    with LOCK:
        return bool(JOB.get("cancel"))


class Cancelled(Exception):
    """Raised inside a job when the user asks it to stop."""


# ------------------------------------------------------------------ the work

def run_generate(scripts: dict, pid: str, overwrite: bool) -> None:
    """Per-language scripts in → production sheets out. No translation: the
    structure language defines the scenes and visuals, and each other language's
    pasted script is segmented onto them. compose.py writes the file format, so
    it cannot come out malformed."""
    langs = [l for l in ("en", "de", "es") if (scripts.get(l) or "").strip()]
    langs += [l for l in scripts if l not in ("en", "de", "es")
              and (scripts.get(l) or "").strip()]
    try:
        begin_job(pid, langs, "generate")
        cfg = pl.load_config()
        key = cfg.get("gemini_key", "")
        if not key:
            raise RuntimeError(
                "No Gemini API key yet. Get a free one at "
                "https://aistudio.google.com/apikey and paste it into config.json "
                "as \"gemini_key\".")
        pid = re.sub(r"[^A-Za-z0-9_-]", "", pid).strip() or "video"
        set_job(stage="generate", label="reading the script", done=0, total=1,
                error="", outputs=[], project=pid)
        log(f"Generating sheets for '{pid}' — {', '.join(langs)}")

        res = compose.generate(
            scripts, pid, key,
            model=cfg.get("gemini_model", gem.DEFAULT_MODEL),
            on_progress=lambda d, t, m: (progress(d, t, m), log(f"  {m}")),
            on_warn=lambda m: log(f"  ⚠ {m}"))

        sdir = pl.sheets_dir(pid)
        written = compose.write_files(res, sdir, overwrite=overwrite)
        log(f"Wrote {len(written)} file(s): {', '.join(written)}")
        log(f"{len(res.scenes)} scenes · "
            f"{sum(1 for s in res.scenes if s.media == 'VIDEO')} video · "
            f"{sum(1 for s in res.scenes if s.hero)} hero")

        if res.warnings:
            log("")
            log(f"⚠ {len(res.warnings)} thing(s) to check:")
            for w in res.warnings:
                for line in w.splitlines():
                    log(f"    {line}")

        set_job(stage="generated", label="sheets written",
                outputs=[{"lang": "-", "name": n, "path": str(sdir / n),
                          "size_mb": 0} for n in written],
                warnings=res.warnings)
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_add_language(pid: str, lang: str, script: str, overwrite: bool) -> None:
    """Compose one more language onto an existing project's shared scenes."""
    try:
        begin_job(pid, [lang], "generate")
        cfg = pl.load_config()
        key = cfg.get("gemini_key", "")
        if not key:
            raise RuntimeError("No Gemini API key yet — add gemini_key to config.json.")
        proj = pl.find_project(pid)
        sheet = Path(proj["sheet"])
        sdir = sheet.parent
        set_job(stage="generate", label=f"adding {pl.LANG_NAMES.get(lang, lang)}",
                done=0, total=3, error="", outputs=[], project=pid)
        log(f"Adding {pl.LANG_NAMES.get(lang, lang)} to '{pid}' from your pasted script")

        res = compose.add_language(
            sheet, lang, script, key,
            model=cfg.get("gemini_model", gem.DEFAULT_MODEL),
            on_progress=lambda d, t, m: (progress(d, t, m), log(f"  {m}")),
            on_warn=lambda m: log(f"  ⚠ {m}"))

        written = compose.write_files(res, sdir, overwrite=overwrite)
        log(f"Wrote {', '.join(written) or '(nothing — file exists; use overwrite)'}")
        for w in res.warnings:
            for line in w.splitlines():
                log(f"    {line}")
        set_job(stage="generated", label=f"{pl.LANG_NAMES.get(lang, lang)} added",
                outputs=[{"lang": lang, "name": n, "path": str(sdir / n),
                          "size_mb": 0} for n in written],
                warnings=res.warnings)
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_sourcing(pid: str, redo: list[int] | None) -> None:
    try:
        proj = pl.find_project(pid)
        sheet = Path(proj["sheet"])
        cfg = pl.load_config()
        if not cfg.get("pexels_key") and not cfg.get("pixabay_key"):
            raise RuntimeError(
                "No stock API key yet. Open config.json and paste your free "
                "Pexels and Pixabay keys in, then try again.")
        scenes = pl.load_scenes(sheet, "en", None)
        begin_job(pid, ["en"], "stock")
        set_job(total=len(redo or scenes))
        log(f"Sourcing visuals for {proj['label']}")

        def onp(d, t, m):
            progress(d, t, m)
            log(f"  {m}")

        assets = pl.source_stock(scenes, sheet, cfg, redo=redo, on_progress=onp)
        end_step()

        # Be honest about the outcome. A scene with no asset at all breaks the
        # render; a placeholder builds but carries a generic background. Either
        # way the user needs to know before they hit render, not after it fails.
        missing = [s.n for s in scenes if s.n not in assets]
        placeheld = sorted(n for n, a in assets.items()
                           if isinstance(a, dict) and a.get("placeholder"))
        if missing:
            set_job(stage="approve", label=f"ready — {len(missing)} scene(s) still empty")
            log(f"⚠ {len(missing)} scene(s) still have NO picture: {missing}")
            log("  Reword their search line in the sheet, or swap them in review, "
                "before rendering.")
        elif placeheld:
            set_job(stage="approve",
                    label=f"ready — {len(placeheld)} placeholder(s)")
            log(f"⚠ {len(placeheld)} scene(s) got a neutral placeholder (no real "
                f"match found): {placeheld}")
            log("  The video will build. Swap these in review for a better shot.")
        else:
            set_job(stage="approve", label="ready for review")
            log("Visuals ready — review them below.")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_build(pid: str, langs: list[str], captions: bool, music: str | None,
              zoom: bool, voices: dict[str, str]) -> None:
    try:
        proj = pl.find_project(pid)
        sheet = Path(proj["sheet"])
        sdir = sheet.parent
        assets_f = pl.paths_for(sheet, "en")["assets"]
        assets = {int(k): v for k, v in json.loads(assets_f.read_text(encoding="utf-8")).items()}
        outputs = []
        begin_job(pid, langs, "voice")

        for li, lang in enumerate(langs):
            tag = f"[{li + 1}/{len(langs)}] {pl.LANG_NAMES.get(lang, lang)}"
            tr = pl.translation_for(sdir, pid, lang)
            scenes = pl.load_scenes(sheet, lang, tr)

            begin_step("voice", lang)
            set_job(label=f"{tag} — narration")
            log(f"{tag}: generating narration ({len(scenes)} lines)")
            t0 = time.time()
            vs = pl.generate_voice(
                scenes, lang, sheet, voice=voices.get(lang) or None,
                on_progress=lambda d, t, m: (progress(d, t, f"{tag} — voicing line {d} of {t}"),
                                             log(f"  {m}")))
            log(f"{tag}: narration done in {time.time() - t0:.0f}s")
            end_step(len(scenes))

            begin_step("render", lang)
            set_job(label=f"{tag} — building video")
            log(f"{tag}: rendering")
            t0 = time.time()
            try:
                out = pl.render_video(
                    scenes, assets, vs, sheet, lang, captions=captions,
                    music=Path(music) if music else None, zoom=zoom,
                    style=pl.effective_caption_style(pid),
                    on_progress=lambda d, t, m: progress(d, t, f"{tag} — {m}"))
            except pl.CaptionsSkipped as cs:
                out = cs.video
                log(f"{tag}: ⚠ video is finished, but captions were not burned in.")
                log(f"    {cs.reason}")
                log(f"    Upload {cs.srt.name} to YouTube instead — arguably better "
                    f"for search anyway.")
                log(f"    To fix burn-in: {R.ffmpeg_fix_hint()}")
            mins = (time.time() - t0) / 60
            log(f"{tag}: finished in {mins:.1f} min → {out.name}")
            end_step(len(scenes))
            outputs.append({"lang": lang, "name": out.name, "path": str(out),
                            "size_mb": round(out.stat().st_size / 1e6)})
            set_job(outputs=list(outputs))

        set_job(stage="done", label="all videos built")
        log("All done.")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_steps(pid: str, langs: list[str], steps: list[str], captions: bool,
              music: str | None, zoom: bool, voices: dict[str, str],
              force: bool = False) -> None:
    """Run a chosen subset of steps for chosen languages.

    `steps` is any of "voice" and "render". This is what the project view's
    per-step buttons call, so re-rendering does not silently redo narration
    (7 minutes) and vice versa.

    `force` deletes this language's cached narration first — the only way to
    genuinely redo voicing, since the cache is keyed by text and settings and
    would otherwise be reused.
    """
    try:
        proj = pl.find_project(pid)
        sheet = Path(proj["sheet"])
        sdir = sheet.parent
        begin_job(pid, langs, steps[0] if steps else "voice")

        assets = {}
        if "render" in steps:
            assets_f = pl.paths_for(sheet, "en")["assets"]
            if not assets_f.exists():
                raise RuntimeError(
                    "No visuals sourced yet. Run 'Find visuals' first.")
            assets = {int(k): v for k, v in
                      json.loads(assets_f.read_text(encoding="utf-8")).items()}

        # Fail before starting, not three minutes in. Voicing with no reference
        # clip chosen used to raise SystemExit from deep inside the worker.
        # Rendering needs narration, so it needs a chosen clip just as much as
        # voicing does — it simply generates it on the way through.
        if "voice" in steps or "render" in steps:
            missing = [l for l in langs
                       if not (voices.get(l) or vx.pref_for(l).get("reference"))]
            if missing:
                names = ", ".join(pl.LANG_NAMES.get(m, m) for m in missing)
                raise RuntimeError(
                    f"No reference clip chosen for {names}. Pick one in the "
                    f"Voices panel, then try again.")

        outputs = []
        for li, lang in enumerate(langs):
            if cancelled():
                raise Cancelled()
            tag = f"[{li + 1}/{len(langs)}] {pl.LANG_NAMES.get(lang, lang)}"
            tr = pl.translation_for(sdir, pid, lang)
            scenes = pl.load_scenes(sheet, lang, tr)
            vs = []

            if force and "voice" in steps:
                gone = 0
                for f in tts.voice_paths(scenes, lang,
                                         pl.paths_for(sheet, lang)["voicecache"]):
                    if f.exists():
                        f.unlink()
                        gone += 1
                log(f"{tag}: cleared {gone} cached narration file(s)")

            if "voice" in steps:
                begin_step("voice", lang)
                set_job(label=f"{tag} — narration")
                log(f"{tag}: generating narration ({len(scenes)} lines)")
                t0 = time.time()
                def on_voice(d, t, m, tag=tag):
                    progress(d, t, f"{tag} — voicing line {d} of {t}")
                    log(f"  {m}")
                    if cancelled():
                        raise Cancelled()

                vs = pl.generate_voice(
                    scenes, lang, sheet, voice=voices.get(lang) or None,
                    on_progress=on_voice)
                log(f"{tag}: narration done in {time.time() - t0:.0f}s")
                end_step(len(scenes))

            if "render" in steps:
                if not vs:
                    # Reuse what is already cached rather than regenerating.
                    vs = pl.generate_voice(scenes, lang, sheet,
                                           voice=voices.get(lang) or None)
                begin_step("render", lang)
                set_job(label=f"{tag} — building video")
                log(f"{tag}: rendering")
                t0 = time.time()
                try:
                    out = pl.render_video(
                        scenes, assets, vs, sheet, lang, captions=captions,
                        music=Path(music) if music else None, zoom=zoom,
                        style=pl.effective_caption_style(pid),
                        on_progress=lambda d, t, m: progress(d, t, f"{tag} — {m}"))
                except pl.CaptionsSkipped as cs:
                    out = cs.video
                    log(f"{tag}: WARNING video finished, captions not burned in.")
                    log(f"    {cs.reason}")
                    log(f"    Upload {cs.srt.name} to YouTube instead.")
                    log(f"    To fix burn-in: {R.ffmpeg_fix_hint()}")
                log(f"{tag}: finished in {(time.time() - t0) / 60:.1f} min "
                    f"-> {out.name}")
                end_step(len(scenes))
                outputs.append({"lang": lang, "name": out.name, "path": str(out),
                                "size_mb": round(out.stat().st_size / 1e6)})
                set_job(outputs=list(outputs))

        set_job(stage="done", label="finished")
        log("Done.")
    except Cancelled:
        set_job(stage="done", label="stopped")
        log("Stopped. Whatever was already generated is kept and reused.")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------- approval data

def approval_data(pid: str) -> dict:
    proj = pl.find_project(pid)
    sheet = Path(proj["sheet"])
    p = pl.paths_for(sheet, "en")
    scenes = pl.load_scenes(sheet, "en", None)
    assets = {}
    if p["assets"].exists():
        assets = {int(k): v for k, v in json.loads(p["assets"].read_text(encoding="utf-8")).items()}
    picks = {}
    if p["picks"].exists():
        picks = {int(k): v for k, v in json.loads(p["picks"].read_text(encoding="utf-8")).items()}

    items = []
    for s in scenes:
        a = assets.get(s.n)
        items.append({
            "n": s.n, "media": s.media, "hero": s.hero,
            "narration": s.narration, "query": s.query,
            "take": picks.get(s.n, 0) + 1,
            "src": (a or {}).get("src", ""),
            # How well this picture matched the line (0..1), or null if visual
            # matching was off when it was sourced.
            "score": (a or {}).get("score"),
            # True when nothing real was found and a neutral background was
            # dropped in so the render doesn't break — needs a manual swap.
            "placeholder": bool((a or {}).get("placeholder")),
            "url": f"/media/{Path(a['path']).name}" if a else "",
            "video": bool(a and Path(a["path"]).suffix.lower() in (".mp4", ".mov", ".webm")),
        })
    cfg = pl.load_config()
    return {"id": pid, "label": proj["label"], "items": items,
            "clip_min": float(cfg.get("clip_min") or 0.45),
            "clip_on": VIS.capability(cfg)["ok"],
            "missing": [i["n"] for i in items if not i["url"]]}


# ---------------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal clean
        pass

    def do_HEAD(self):
        """Some players and download managers probe with HEAD before GET.

        Answered by running the normal GET path with the body suppressed, so
        the headers can never drift out of step with what GET would send.
        """
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    def _serve_file(self, f: Path, root: Path) -> None:
        """Send a file from `root`, streamed, honouring HTTP Range requests.

        Range matters for video. Without it a browser cannot seek, and some
        will refuse to start playback at all — they ask for the first few bytes
        to read the container header, get the whole file instead, and give up.
        Streaming in chunks also means a 268 MB render is not loaded into
        memory in one go just to be handed to <video>.
        """
        f = f.resolve()
        if not str(f).startswith(str(root.resolve()) + os.sep) or not f.is_file():
            return self._send(404, b"not found", "text/plain")

        size = f.stat().st_size
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        partial = False

        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
        if m:
            a, b = m.group(1), m.group(2)
            if a:                       # bytes=500-  or  bytes=500-999
                start = int(a)
                end = int(b) if b else size - 1
            elif b:                     # bytes=-500  (the last 500 bytes)
                start = max(0, size - int(b))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            partial = True

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if getattr(self, "_head_only", False):
            return
        try:
            with f.open("rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    chunk = fh.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except ConnectionError:
            # The browser seeked away, closed the tab, or navigated to another
            # page mid-download. On macOS/Linux this is BrokenPipe/ConnectionReset;
            # on Windows it's ConnectionAbortedError (WinError 10053). All three
            # share the ConnectionError base and are completely normal here — the
            # client simply stopped listening. Nothing is wrong with the file or
            # the render, so swallow it rather than dumping a scary traceback.
            pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if getattr(self, "_head_only", False):
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path

        # Real URLs. The app uses the History API rather than hash fragments,
        # so /project/video05 is a genuine address you can type, bookmark,
        # reload or send to yourself — but it has to reach the server first,
        # and the server has to hand back the app rather than a 404.
        #
        # Anything that is NOT an asset or an API call is a navigation route.
        # Listing the asset prefixes rather than the app routes means adding a
        # new view needs no server change at all.
        if not path.startswith(("/api/", "/media/", "/out/", "/preview/")):
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/projects":
            projects = pl.find_projects()
            cfg = pl.load_config()
            music = sorted(f.name for f in (ROOT / "music").glob("*")
                           if f.suffix.lower() in (".mp3", ".m4a", ".wav", ".aac"))
            # Attach per-language status so the dashboard needs one request,
            # not one per project.
            for pr in projects:
                try:
                    pr["status"] = pl.project_status(
                        Path(pr["sheet"]), pr["languages"])
                except Exception as e:
                    pr["status"] = {"error": str(e), "scenes": pr.get("scenes", 0),
                                    "assets": 0, "languages": {}}
            return self._json({
                "projects": projects, "music": music,
                "has_keys": bool(cfg.get("pexels_key") or cfg.get("pixabay_key")),
                "has_gemini": bool(cfg.get("gemini_key")),
            })

        if path == "/api/deletable":
            pid = (q.get("id") or [""])[0]
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": f"no project called {pid!r}"}, 404)
            g = pl.deletable(Path(proj["sheet"]), proj["languages"])
            summary = {}
            for k, files in g.items():
                size = 0
                for f in files:
                    try:
                        size += (sum(x.stat().st_size for x in f.rglob("*") if x.is_file())
                                 if f.is_dir() else f.stat().st_size)
                    except OSError:
                        pass
                summary[k] = {"count": len(files), "mb": round(size / 1e6, 1),
                              "names": [f.name for f in files[:6]]}
            return self._json({"id": pid, "groups": summary})

        if path == "/api/doctor":
            # The same checks the `faceless check` command runs, for Settings.
            from lib import chatterbox_engine as CB
            import shutil as _sh
            caps = R.caption_method()
            dev = CB.device_info() if CB.installed() else {}
            langs = {}
            for lg in ("en", "de", "es"):
                try:
                    langs[lg] = vx.status(lg)
                except Exception:
                    langs[lg] = {}
            cfg = pl.load_config()
            return self._json({
                "python": sys.version.split()[0],
                "in_venv": Path(sys.prefix) == (ROOT / ".venv"),
                "ffmpeg": _sh.which("ffmpeg") or "",
                "ffprobe": _sh.which("ffprobe") or "",
                "captions": caps,
                "captions_ok": caps in ("ass", "subtitles"),
                "ffmpeg_hint": R.ffmpeg_fix_hint(),
                "chatterbox": CB.installed(),
                "device": dev,
                "gpu_ok": dev.get("device") in ("cuda", "mps"),
                "clip": VIS.capability(cfg),
                "align": AL.capability(cfg),

                "references": CB.list_references() if CB.installed() else [],
                "voices": langs,
                "keys": {"pexels": bool(cfg.get("pexels_key")),
                         "pixabay": bool(cfg.get("pixabay_key")),
                         "gemini": bool(cfg.get("gemini_key"))},
                "outputs": sorted(
                    ({"name": f.name,
                      "size_mb": round(f.stat().st_size / 1e6, 1),
                      "built": int(f.stat().st_mtime)}
                     for f in pl.PROJECTS.glob("*/out/*.mp4")),
                    key=lambda d: -d["built"]),
            })

        if path == "/api/status":
            with LOCK:
                return self._json(dict(JOB))

        if path == "/api/voices":
            lang = (q.get("lang") or ["en"])[0]
            pid = (q.get("id") or [""])[0]
            scenes = None
            if pid:
                try:
                    proj = pl.find_project(pid)
                    tr = pl.translation_for(pl.sheets_dir(pid), pid, lang)
                    scenes = pl.load_scenes(Path(proj["sheet"]), lang, tr)
                except Exception:
                    scenes = None
            vx.ensure_folders()
            refs = vx.references(lang)
            return self._json({
                "lang": lang,
                "lang_name": vx.LANGS.get(lang, lang),
                "languages": vx.LANGS,
                "status": vx.status(lang),
                # Only this language's clips, plus any left loose — a German
                # list full of English voices is noise, not choice.
                "references": [r for r in refs if r["lang"] == lang],
                "loose": [r for r in refs if not r["lang"]],
                "counts": {c: len(vx.references(c)) for c in ("en", "de", "es")},
                "folder": f"voices_refs/{lang}",
                "chosen": vx.pref_for(lang),
                "sample": vx.sample_line(lang, scenes),
            })

        if path.startswith("/preview/"):
            name = unquote(path[len("/preview/"):])
            f = (vx.PREVIEWS / name).resolve()
            if not str(f).startswith(str(vx.PREVIEWS.resolve())) or not f.exists():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, f.read_bytes(), "audio/mpeg")

        if path == "/api/approval":
            pid = (q.get("id") or [""])[0]
            try:
                return self._json(approval_data(pid))
            except StopIteration:
                return self._json({"error": "project not found"}, 404)

        if path == "/api/metadata":
            # Load whatever description/tags were last generated or edited for a
            # language. Empty is a valid answer (nothing generated yet).
            pid = (q.get("id") or [""])[0]
            lang = (q.get("lang") or ["en"])[0]
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": "project not found"}, 404)
            data = pl.load_metadata(Path(proj["sheet"]), lang)
            return self._json(data or {})

        if path == "/api/captions":
            # Everything the subtitle editor needs: presets, the user's saved
            # templates, the global default, and (if an id is given) this
            # project's override and what it would actually render with.
            pid = (q.get("id") or [""])[0] or None
            cfg = pl.load_config()
            default_spec = pl.global_caption_style()
            out = {
                "presets": CAP.preset_list(),
                "custom": pl.custom_caption_styles(),
                "default": default_spec,
                "default_resolved": CAP.resolve_style(default_spec).to_dict(),
                "align": AL.capability(cfg),
                "project": pid,
                "project_style": pl.load_project_style(pid) if pid else None,
            }
            if pid:
                out["effective_resolved"] = CAP.resolve_style(
                    pl.effective_caption_style(pid)).to_dict()
            return self._json(out)

        if path.startswith("/media/"):
            # Stock footage and photos, straight from the cache.
            name = unquote(path[len("/media/"):])
            return self._serve_file(ROOT / "cache" / "stock" / name,
                                    ROOT / "cache" / "stock")

        if path.startswith("/out/"):
            # Finished videos and subtitles now live per-project under
            # projects/<pid>/out/. The filename still carries the pid
            # (<pid>_<lang>.mp4), so find its folder by matching the name.
            name = unquote(path[len("/out/"):])
            if "/" in name or "\\" in name or ".." in name:
                return self._send(404, b"not found", "text/plain")
            hit = next((f for f in pl.PROJECTS.glob(f"*/out/{name}")
                        if f.name == name), None)
            if hit is None:
                return self._send(404, b"not found", "text/plain")
            return self._serve_file(hit, hit.parent)

        return self._send(404, b"not found", "text/plain")

    # --------------------------------------------------------------- POST
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            b = self._body()
        except json.JSONDecodeError:
            return self._json({"error": "bad request"}, 400)

        if path == "/api/preview":
            try:
                f = vx.preview(
                    b.get("text") or "", b.get("lang") or "en",
                    reference=b.get("reference") or "",
                    exaggeration=float(b.get("exaggeration", vx.DEFAULT_EXAGGERATION)),
                    cfg_weight=float(b.get("cfg_weight", vx.DEFAULT_CFG)))
                return self._json({"url": f"/preview/{f.name}"})
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/choose_voice":
            lang = b.get("lang") or "en"
            saved = vx.save_pref(
                lang,
                reference=b.get("reference"),
                exaggeration=b.get("exaggeration"),
                cfg_weight=b.get("cfg_weight"))
            log(f"Voice for {lang}: {tts.describe(lang)}")
            return self._json({"saved": saved})

        if path == "/api/generate":
            # `scripts` maps language -> that language's pasted script. A legacy
            # {script, langs} body is still accepted (all languages get the same
            # text), so nothing breaks mid-upgrade.
            scripts = b.get("scripts")
            if not isinstance(scripts, dict):
                one = (b.get("script") or "").strip()
                scripts = {l: one for l in (b.get("langs") or ["en"])} if one else {}
            scripts = {k: v for k, v in scripts.items() if (v or "").strip()}
            if not scripts:
                return self._json({"error": "Paste a script for at least one language."}, 400)
            ok = start_thread(run_generate, scripts, b.get("id") or "video",
                              bool(b.get("overwrite")))
            return self._json({"started": ok})

        if path == "/api/add_language":
            pid = b.get("id")
            lang = b.get("lang") or ""
            script = (b.get("script") or "").strip()
            if not (pid and lang and script):
                return self._json(
                    {"error": "Need a project, a language and a pasted script."}, 400)
            ok = start_thread(run_add_language, pid, lang, script,
                              bool(b.get("overwrite")))
            if not ok:
                return self._json({"error": "something is already running"}, 409)
            return self._json({"started": ok})

        if path == "/api/source":
            ok = start_thread(run_sourcing, b.get("id"), b.get("redo") or None)
            return self._json({"started": ok})

        if path == "/api/build":
            ok = start_thread(run_build, b.get("id"), b.get("langs") or ["en"],
                              bool(b.get("captions", True)),
                              b.get("music") and str(ROOT / "music" / b["music"]),
                              bool(b.get("zoom", True)), b.get("voices") or {})
            return self._json({"started": ok})

        if path == "/api/run":
            steps = [x for x in (b.get("steps") or ["voice", "render"])
                     if x in ("voice", "render")]
            if not steps:
                return self._json({"error": "nothing to run"}, 400)
            # "id" for consistency with every sibling endpoint; "project" is
            # accepted too so neither spelling is a silent no-op.
            pid = b.get("id") or b.get("project")
            if not pid:
                return self._json({"error": "which project?"}, 400)
            ok = start_thread(run_steps, pid,
                              b.get("langs") or ["en"], steps,
                              bool(b.get("captions")), b.get("music") or None,
                              b.get("zoom", True), b.get("voices") or {},
                              bool(b.get("force")))
            if not ok:
                return self._json({"error": "something is already running"}, 409)
            return self._json({"started": ok, "steps": steps})

        if path == "/api/delete":
            pid = b.get("id") or ""
            what = [x for x in (b.get("what") or [])
                    if x in ("outputs", "voice", "visuals", "work", "sheets")]
            if not what:
                return self._json({"error": "nothing selected to delete"}, 400)

            # The id is never used to build a path directly — it must match a
            # project we already found on disk. That, plus the per-file check
            # inside delete_project, is what keeps a crafted id harmless.
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": f"no project called {pid!r}"}, 404)

            # Removing the sheets removes the project itself, so make the caller
            # type its name. A misplaced click should not be able to do this.
            if "sheets" in what and b.get("confirm") != pid:
                return self._json(
                    {"error": "type the project name to confirm deleting it"}, 400)

            if busy():
                return self._json(
                    {"error": "something is running — wait for it to finish"}, 409)

            res = pl.delete_project(Path(proj["sheet"]), proj["languages"], what)
            log(f"Deleted {res['count']} file(s) from {pid} "
                f"({res['freed_mb']} MB freed)")
            return self._json(res)

        if path == "/api/metadata":
            # Generate title/description/tags for one language. Synchronous — it
            # is a single Gemini call, not a long job — so the UI just waits.
            pid = b.get("id")
            lang = b.get("lang") or "en"
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": "project not found"}, 404)
            try:
                data = pl.build_metadata(Path(proj["sheet"]), lang, pl.load_config())
                log(f"Wrote {lang} description for {pid}")
                return self._json(data)
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/metadata_save":
            # Persist the user's edits to the description/tags.
            pid = b.get("id")
            lang = b.get("lang") or "en"
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": "project not found"}, 404)
            try:
                saved = pl.save_metadata(Path(proj["sheet"]), lang, {
                    "title": b.get("title"), "description": b.get("description"),
                    "tags": b.get("tags") or []})
                return self._json({"saved": True, **saved})
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/captions":
            # One endpoint, a few actions, all about the subtitle style:
            #   set the global default, set/clear a project's override, or save
            #   and delete the user's own named templates.
            action = b.get("action") or "set"
            try:
                if action == "save":
                    name = (b.get("name") or "").strip()
                    if not name:
                        return self._json({"error": "name required"}, 400)
                    pl.save_custom_caption_style(name, b.get("style") or {})
                    return self._json({"saved": True, "custom": pl.custom_caption_styles()})
                if action == "delete":
                    pl.delete_custom_caption_style(b.get("name") or "")
                    return self._json({"deleted": True, "custom": pl.custom_caption_styles()})

                scope = b.get("scope") or "global"
                spec = b.get("style")           # preset id (str) or style dict, or None
                if scope == "project":
                    pid = b.get("id")
                    if pl.find_project(pid) is None:
                        return self._json({"error": "project not found"}, 404)
                    pl.save_project_style(pid, spec)     # None clears the override
                    return self._json({"saved": True, "scope": "project",
                                       "effective": CAP.resolve_style(
                                           pl.effective_caption_style(pid)).to_dict()})
                pl.set_global_caption_style(spec)
                return self._json({"saved": True, "scope": "global",
                                   "default_resolved": CAP.resolve_style(spec).to_dict()})
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/organise_voices":
            moved = vx.organise()
            return self._json({"moved": moved, "count": len(moved)})

        if path == "/api/cancel":
            if not busy():
                return self._json({"ok": True, "note": "nothing was running"})
            set_job(cancel=True)
            log("Stop requested — finishing the current item, then stopping.")
            return self._json({"ok": True})

        if path == "/api/reveal":
            target = pl.PROJECTS
            p = b.get("path")
            if p and Path(p).exists():
                target = Path(p)
            if sys.platform == "darwin":
                cmd = ["open", "-R", str(target)]
            elif os.name == "nt":
                cmd = ["explorer", "/select,", str(target)]
            else:
                cmd = ["xdg-open", str(target.parent)]
            subprocess.run(cmd, check=False)
            return self._json({"ok": True})

        return self._json({"error": "unknown endpoint"}, 404)


class QuietServer(ThreadingHTTPServer):
    """A threading server that doesn't shout when a browser hangs up mid-request.

    Every browser routinely opens and abandons connections — it cancels an image
    the moment you scroll past it, drops the video socket when you seek, closes
    the tab. The handler then writes to a socket nobody is reading, which raises
    a ConnectionError (BrokenPipe / ConnectionReset on macOS/Linux, WinError
    10053 'connection aborted' on Windows). The stock library prints a full
    traceback for each, which looks alarming and buries the real log. These are
    expected and harmless, so we swallow exactly that one family and let every
    other error surface normally.
    """
    def handle_error(self, request, client_address):
        import sys as _sys
        if isinstance(_sys.exc_info()[1], ConnectionError):
            return                      # client hung up; nothing to see here
        super().handle_error(request, client_address)


def main(open_browser: bool = True) -> None:
    vx.ensure_folders()
    for d in ("projects", "cache/stock", "cache/voice", "cache/refs", "music"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    # Fold any old flat sheets/work/out into projects/<pid>/… on startup. No-op
    # once everything has moved, so it is safe to leave in.
    rep = pl.migrate_layout()
    if rep["moved"]:
        log(f"Reorganised {rep['moved']} file(s) into projects/ "
            f"({len(rep['projects'])} project(s))")
    if not UI.exists():
        sys.exit(f"Missing {UI} — the app files are incomplete.")

    srv = QuietServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print()
    print("  ┌────────────────────────────────────────────┐")
    print("  │  Faceless Studio is running                │")
    print(f"  │  {url:<42}│")
    print("  │                                            │")
    print("  │  Leave this window open while you work.    │")
    print("  │  Press Ctrl+C here when you're finished.   │")
    print("  └────────────────────────────────────────────┘")
    print()
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped. Your work is saved in projects/ and cache/.\n")


if __name__ == "__main__":
    main()
