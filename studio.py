#!/usr/bin/env python3
"""Faceless Studio — a local control panel for the pipeline.

Double-click Start.bat (Windows) or Start.command (macOS),
or run:  python3 studio.py

Serves a small web app on 127.0.0.1 only. Nothing is uploaded anywhere; the
browser is just a nicer front end for the same pipeline the CLI uses.
"""
from __future__ import annotations

import inspect
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
from lib import llm as LLM  # noqa: E402
from lib import render as R  # noqa: E402
from lib import tts, voices as vx  # noqa: E402
from lib import vision as VIS  # noqa: E402
from lib import captions as CAP, align as AL  # noqa: E402

PORT = 8765
UI = ROOT / "lib" / "ui.html"

# ------------------------------------------------------------------- job state

# Every unit of background work is a persistent Job (lib/jobs.py). The studio is
# a thin layer over a JobStore + Scheduler: enqueue jobs and a small worker pool
# runs them in order — surviving a browser refresh AND a server restart. The
# existing workers are unchanged: set_job / log / progress report into the
# CURRENT job via a thread-local, so each worker thread updates its own Job.
from lib import jobs  # noqa: E402

STORE = jobs.JobStore(ROOT / "work" / "queue.json")
SCHED = None                       # jobs.Scheduler, built in main() after workers
_CUR = threading.local()           # per-thread current job id

RUNNING = ("generate", "stock", "voice", "render")     # studio 'stage' names
_STAGE_STATUS = {"generate": jobs.RUNNING, "stock": jobs.RUNNING,
                 "voice": jobs.RUNNING, "render": jobs.RUNNING,
                 "approve": jobs.APPROVE, "done": jobs.DONE, "error": jobs.ERROR,
                 # "generated" is a SUCCESS end-state (sheets written). Mapping it
                 # to DONE means a job that actually finished is marked done — not
                 # flipped to Canceled just because Stop was pressed a moment too
                 # late, after the work had already completed.
                 "generated": jobs.DONE}


def _cur_id():
    return getattr(_CUR, "job_id", None)


def set_job(**kw) -> None:
    """Update the CURRENT job. A studio 'stage' is translated to a queue status
    (and the elapsed clock frozen/released) so the queue and the old Activity
    screen stay in step."""
    jid = _cur_id()
    if jid is None:
        return
    if "stage" in kw:
        st = _STAGE_STATUS.get(kw["stage"])
        if st:
            kw["status"] = st
        job = STORE.get(jid)
        if kw["stage"] in RUNNING:
            kw["ended"] = None
        elif job and job.started and job.ended is None:
            kw["ended"] = time.time()
    STORE.update(jid, **kw)


def begin_job(project: str, langs: list[str], stage: str) -> None:
    """Start the CURRENT job: stamp it running and clear any last-run state."""
    jid = _cur_id()
    if jid is None:
        return
    args = dict(STORE.get(jid).args)
    args["langs"] = langs
    STORE.update(jid, project=project, args=args, stage=stage,
                 status=jobs.RUNNING, label="", done=0, total=0, log=[], error="",
                 outputs=[], steps=[], started=time.time(), ended=None,
                 step_started=time.time(), eta=None, rate=None, lang=None,
                 cancel=False, force_save=True)


def begin_step(stage: str, lang: str | None = None) -> None:
    jid = _cur_id()
    if jid is None:
        return
    STORE.update(jid, stage=stage, status=jobs.RUNNING, lang=lang,
                 step_started=time.time(), done=0, total=0, eta=None, rate=None,
                 label="", ended=None)


def end_step(items: int = 0) -> None:
    """Record how long the step took, for the per-step breakdown."""
    jid = _cur_id()
    if jid is None:
        return
    job = STORE.get(jid)
    t0 = job.step_started or time.time()
    steps = list(job.steps)
    steps.append({"name": job.stage, "lang": job.lang,
                  "seconds": round(time.time() - t0, 1),
                  "items": items or job.done})
    STORE.update(jid, steps=steps)


def _log_to(jid, msg: str) -> None:
    """Append a log line to a SPECIFIC job (thread-safe). Used both by the
    thread-local log() below and by callbacks bound to a captured job id, so
    lines emitted from the pipeline's own worker threads (parallel sourcing)
    still reach the right job's Output."""
    if jid is not None:
        STORE.append_log(jid, str(msg))


def _progress_to(jid, done: int, total: int, label: str = "") -> None:
    if jid is None:
        return
    job = STORE.get(jid)
    if job is None:
        return
    rate, eta = job.rate, job.eta
    t0 = job.step_started
    if t0 and done > 0:
        elapsed = time.time() - t0
        r = done / elapsed if elapsed > 0 else 0
        rate = round(r, 3) if r else None
        remaining = max(0, (total or 0) - done)
        eta = round(remaining / r) if r > 0 and remaining else 0
    STORE.update(jid, done=done, total=total, label=label or job.label,
                 rate=rate, eta=eta)


def log(msg: str) -> None:
    _log_to(_cur_id(), msg)


def progress(done: int, total: int, label: str = "") -> None:
    """Record progress and a rate/ETA for THIS step (per-item costs differ wildly
    between steps, so a rate is never carried across them)."""
    _progress_to(_cur_id(), done, total, label)


def bound_reporters():
    """A (log, progress) pair locked to the CURRENT job id, safe to call from any
    thread. Pass these to steps that fan work out across worker threads (parallel
    sourcing), where the thread-local job id isn't set and log()/progress() would
    otherwise silently drop everything."""
    jid = _cur_id()
    return (lambda m: _log_to(jid, m),
            lambda d, t, m="": _progress_to(jid, d, t, m))


def busy() -> bool:
    """Is any job running right now? No longer a gate — new work is queued."""
    return any(j.status == jobs.RUNNING for j in STORE.jobs())


def enqueue(project: str, kind: str, args: dict, auto: bool = False):
    """Add a job to the queue. The scheduler runs it when a worker slot frees."""
    return SCHED.enqueue(project or "", kind, args, auto=auto)


def cancelled() -> bool:
    """True when the running job has been asked to stop (checked between items)."""
    jid = _cur_id()
    if jid is None:
        return False
    job = STORE.get(jid)
    return bool(job and job.cancel)


def _stopper():
    """A Stop check that works from ANY thread. cancelled() reads a thread-local
    job id, so it's blind inside the pipeline's own worker threads (parallel
    sourcing, etc.); this captures the job id now and reads the live flag straight
    from the store, which is lock-protected and thread-safe."""
    jid = _cur_id()

    def stop() -> bool:
        job = STORE.get(jid)
        return bool(job and job.cancel)
    return stop


def _project_or_skip(pid: str):
    """The project for a job runner, or None after ending the job cleanly.

    A queued job whose project was deleted meanwhile is not a bug — end it with a
    plain 'no longer exists' message instead of a scary traceback."""
    proj = pl.find_project(pid)
    if proj is None:
        set_job(stage="error",
                error=f"Project '{pid}' no longer exists (it was deleted).")
        log(f"Project '{pid}' no longer exists — skipping this job.")
    return proj


class Cancelled(Exception):
    """Raised inside a job when the user asks it to stop."""


def _status_payload(job) -> dict:
    """The single active job the current Activity screen polls (backward-compat)."""
    if job is None:
        return {"stage": "idle", "status": "idle", "label": "", "done": 0,
                "total": 0, "log": [], "error": "", "outputs": [], "project": None,
                "langs": [], "started": None, "ended": None, "step_started": None,
                "eta": None, "rate": None, "steps": [], "id": None}
    d = job.to_dict()
    d["langs"] = job.args.get("langs", [])
    return d


def _active_job():
    """The job the Activity screen shows: the running one, else the most recently
    finished so its result and log stay visible."""
    js = STORE.jobs()
    running = [j for j in js if j.status == jobs.RUNNING]
    if running:
        return running[-1]
    shown = [j for j in js if j.status != jobs.QUEUED]
    return shown[-1] if shown else None


def _queue_payload() -> dict:
    """Everything the Queue view needs: every job, newest activity first."""
    return {"jobs": [{"id": j.id, "project": j.project, "kind": j.kind,
                      "status": j.status, "stage": j.stage, "label": j.label,
                      "done": j.done, "total": j.total, "auto": j.auto,
                      "error": j.error, "created": j.created,
                      "started": j.started, "ended": j.ended}
                     for j in STORE.jobs()],
            "max_concurrent": (SCHED.max_concurrent if SCHED else 1)}


def _worker_wrapper(fn):
    """Adapt a studio worker into a scheduler run-fn(job).

    Binds the thread-local 'current job' so the worker's own set_job/log/progress
    report into THIS job, then calls the worker with only the kwargs it declares —
    begin_job stashes extras like 'langs' into args, and a resumed job must not
    pass those to a worker that never accepted them.
    """
    params = set(inspect.signature(fn).parameters)

    def run(job) -> None:
        _CUR.job_id = job.id
        try:
            fn(**{k: v for k, v in job.args.items() if k in params})
        finally:
            _CUR.job_id = None
    return run


# ------------------------------------------------------------------ the work

def run_generate(scripts: dict, pid: str, overwrite: bool, channel: str = "") -> None:
    """Per-language scripts in → main script + narration files out. No
    translation: the structure language defines the scenes and visuals, and each
    other language's pasted script is segmented onto them. compose.py writes the
    file format, so it cannot come out malformed."""
    langs = [l for l in ("en", "de", "es") if (scripts.get(l) or "").strip()]
    langs += [l for l in scripts if l not in ("en", "de", "es")
              and (scripts.get(l) or "").strip()]
    try:
        begin_job(pid, langs, "generate")
        cfg = pl.load_config()
        if not LLM.available(cfg):
            raise RuntimeError(
                "No language model configured. Either add a free Gemini key "
                "(https://aistudio.google.com/apikey) as \"gemini_key\", or set "
                "\"llm\": \"ollama\" with an \"ollama_model\" to run locally.")
        pid = re.sub(r"[^A-Za-z0-9_-]", "", pid).strip() or "video"
        set_job(stage="generate", label="reading the script", done=0, total=1,
                error="", outputs=[], project=pid)
        log(f"Generating sheets for '{pid}' via {LLM.capability(cfg)['provider']} "
            f"— {', '.join(langs)}")

        def onp(d, t, m):
            if cancelled():
                raise Cancelled()
            progress(d, t, m)
            log(f"  {m}")

        res = compose.generate(
            scripts, pid, LLM.key_for(cfg),
            model=LLM.model_for(cfg),
            on_progress=onp,
            on_warn=lambda m: log(f"  ⚠ {m}"),
            name_people=pl._flag(cfg.get("name_real_people")),
            auto_split=pl._flag(cfg.get("auto_split", "auto")))

        sdir = pl.sheets_dir(pid)
        written = compose.write_files(res, sdir, overwrite=overwrite)
        log(f"Wrote {len(written)} file(s): {', '.join(written)}")
        # Assign the chosen channel now the project exists on disk (assigning
        # earlier would be pruned as a 'ghost' before its files were written).
        if (channel or "").strip():
            from lib import channels as _ch
            _ch.assign(pid, channel.strip())
            log(f"Added to channel: {channel.strip()}")
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
    except Cancelled:
        log("Stopped before the sheets were written.")
        return                               # scheduler marks it Stopped
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_add_language(pid: str, lang: str, script: str, overwrite: bool) -> None:
    """Compose one more language onto an existing project's shared scenes."""
    try:
        begin_job(pid, [lang], "generate")
        cfg = pl.load_config()
        if not LLM.available(cfg):
            raise RuntimeError(
                "No language model configured — add gemini_key, or set llm=ollama "
                "with an ollama_model in config.json.")
        proj = _project_or_skip(pid)
        if proj is None:
            return
        sheet = Path(proj["sheet"])
        sdir = sheet.parent
        set_job(stage="generate", label=f"adding {pl.LANG_NAMES.get(lang, lang)}",
                done=0, total=3, error="", outputs=[], project=pid)
        log(f"Adding {pl.LANG_NAMES.get(lang, lang)} to '{pid}' from your pasted script")

        res = compose.add_language(
            sheet, lang, script, LLM.key_for(cfg),
            model=LLM.model_for(cfg),
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


def run_sourcing(pid: str, redo: list[int] | None,
                 skip_review: bool = False) -> None:
    try:
        proj = _project_or_skip(pid)
        if proj is None:
            return
        sheet = Path(proj["sheet"])
        cfg = pl.load_config()
        if not cfg.get("pexels_key") and not cfg.get("pixabay_key"):
            raise RuntimeError(
                "No stock API key yet. Open config.json and paste your free "
                "Pexels and Pixabay keys in, then try again.")
        # Sourcing only needs the scene list and each scene's (English) search
        # query — never a specific language's narration. Read straight from the
        # main script in ITS OWN language, so a German- or Spanish-main project
        # sources without demanding an English narration file it doesn't have.
        mlang = pl.main_lang(sheet)
        scenes = pl.load_scenes(sheet, mlang, None)
        begin_job(pid, [mlang], "stock")
        set_job(total=len(redo or scenes))
        log(f"Sourcing visuals for {proj['label']}")

        # The bar shows "which scene, how far"; the detailed per-scene feedback
        # (searches, scores, the pick) comes through `log` from fetch_all. Keeping
        # them separate is what stops the Output being a wall of bare "S33 image".
        # fetch_all fans scenes out across worker threads (source_workers > 1),
        # where the thread-local job id isn't set — so use reporters BOUND to this
        # job id, or every per-scene line and the progress bar go nowhere.
        blog, bprog = bound_reporters()

        def onp(d, t, m):
            bprog(d, t, m)

        # Warm up the visual-matching model UP FRONT, with a visible note. The
        # very first source downloads it (SigLIP 2 is ~1.7 GB) and, until now,
        # that happened silently mid-scene and looked frozen. Loading it here
        # surfaces the status; the download's own progress bar is in the terminal.
        if pl._flag(cfg.get("clip", "auto")):
            cap = VIS.capability(cfg)
            if cap.get("ok"):
                log(f"Preparing visual matching — {cap['model'].split('/')[-1]} on "
                    f"{cap['device']}. The FIRST run downloads it once "
                    f"(watch the terminal window for the % bar)…")
                try:
                    VIS.get_scorer(cfg, log=lambda m: log(f"  {m.strip()}"))
                    log("  model ready — sourcing now.")
                except Exception as e:
                    log(f"  visual matching unavailable ({e}) — ranking by size only.")

        stop = _stopper()
        assets = pl.source_stock(scenes, sheet, cfg, redo=redo,
                                 on_progress=onp, log=blog, should_cancel=stop)
        end_step()

        # Stop was pressed mid-source: keep whatever was already found, and let
        # the scheduler mark the job Stopped (its cancel flag is set) rather than
        # parking it at 'Needs review'.
        if stop():
            log("Stopped — kept the visuals already sourced.")
            return

        # Be honest about the outcome. A scene with no asset at all breaks the
        # render; a placeholder builds but carries a generic background. Either
        # way the user needs to know before they hit render, not after it fails.
        missing = [s.n for s in scenes if s.n not in assets]
        placeheld = sorted(n for n, a in assets.items()
                           if isinstance(a, dict) and a.get("placeholder"))
        # In the hands-off chain there is no review stop, so this job finishes
        # 'done' and the pipeline moves straight on to voice+render. On its own
        # it stops at 'approve' (Needs review) so you can look before rendering.
        end_stage = "done" if skip_review else "approve"
        if missing:
            set_job(stage=end_stage, label=f"ready — {len(missing)} scene(s) still empty")
            log(f"⚠ {len(missing)} scene(s) still have NO picture: {missing}")
            log("  Reword their search line in the sheet, or swap them in review, "
                "before rendering.")
        elif placeheld:
            set_job(stage=end_stage,
                    label=f"ready — {len(placeheld)} placeholder(s)")
            log(f"⚠ {len(placeheld)} scene(s) got a neutral placeholder (no real "
                f"match found): {placeheld}")
            log("  The video will build. Swap these in review for a better shot.")
        else:
            set_job(stage=end_stage, label="ready for review")
            log("Visuals ready." if skip_review else "Visuals ready — review them below.")
        if skip_review:
            log("Auto-process: continuing to voice + render (no review stop).")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_regenerate(pid: str, which: list[int]) -> None:
    """Manually generate AI images for the scenes the user marked in review."""
    try:
        proj = _project_or_skip(pid)
        if proj is None:
            return
        sheet = Path(proj["sheet"])
        cfg = pl.load_config()
        mlang = pl.main_lang(sheet)
        scenes = pl.load_scenes(sheet, mlang, None)
        begin_job(pid, [mlang], "stock")
        set_job(total=len(which or []))
        log(f"Generating {len(which or [])} image(s) for {proj['label']}")

        def onp(d, t, m):
            progress(d, t, m)

        stop = _stopper()
        res = pl.generate_scenes(scenes, sheet, cfg, which or [],
                                 on_progress=onp, log=log, should_cancel=stop)
        end_step()
        if stop():
            log("Stopped — kept the images already generated.")
            return
        gen, failed = res["generated"], res["failed"]
        if failed:
            set_job(stage="approve",
                    label=f"generated {len(gen)}, {len(failed)} could not")
            log(f"⚠ {len(failed)} scene(s) could not be generated: "
                f"{[n for n, _ in failed]} — kept their current picture.")
            for n, reason in failed:               # the ACTUAL reason, per scene
                log(f"    S{n}: {str(reason)[:160]}")
        else:
            set_job(stage="approve", label=f"generated {len(gen)} image(s)")
            log("Done — the new images are below.")
    except SystemExit as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_regenerate_video(pid: str, which: list[int]) -> None:
    """Manually generate Veo video clips for the scenes marked in review. Capped,
    slow (a minute or two each), and expensive — so it is never automatic."""
    try:
        proj = _project_or_skip(pid)
        if proj is None:
            return
        sheet = Path(proj["sheet"])
        cfg = pl.load_config()
        mlang = pl.main_lang(sheet)
        scenes = pl.load_scenes(sheet, mlang, None)
        begin_job(pid, [mlang], "stock")
        set_job(total=len(which or []))
        log(f"Generating video for {len(which or [])} scene(s) — Veo takes a "
            f"minute or two each.")

        def onp(d, t, m):
            progress(d, t, m)

        stop = _stopper()
        res = pl.generate_videos(scenes, sheet, cfg, which or [],
                                 on_progress=onp, log=log, should_cancel=stop)
        end_step()
        if stop():
            log("Stopped — kept the clips already generated.")
            return
        gen, failed, skipped = res["generated"], res["failed"], res["skipped"]
        if skipped:
            log(f"Held back {len(skipped)} scene(s) past the per-run cap "
                f"(veo_max): {skipped}. Run again to do more.")
        if failed:
            set_job(stage="approve",
                    label=f"video: {len(gen)} done, {len(failed)} could not")
            log(f"⚠ {len(failed)} scene(s) could not be generated: "
                f"{[n for n, _ in failed]} — kept their current picture.")
            for n, reason in failed:               # the ACTUAL reason, per scene
                log(f"    S{n}: {str(reason)[:160]}")
        else:
            set_job(stage="approve", label=f"generated {len(gen)} video clip(s)")
            log("Done — the new clips are below.")
    except SystemExit as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_build(pid: str, langs: list[str], captions: bool, music: str | None,
              zoom: bool, voices: dict[str, str], master: bool = True) -> None:
    try:
        VIS.unload()          # free the sourcing model before the heavy voice work
        proj = _project_or_skip(pid)
        if proj is None:
            return
        sheet = Path(proj["sheet"])
        sdir = sheet.parent
        # No language chosen -> build the project's own main language, never a
        # hardcoded "en" (which a German- or Spanish-main project does not have).
        langs = langs or [pl.main_lang(sheet)]
        assets_f = pl.paths_for(sheet, "en")["assets"]   # assets.json is shared
        assets = {int(k): v for k, v in json.loads(assets_f.read_text(encoding="utf-8")).items()}
        outputs = []
        begin_job(pid, langs, "voice")

        for li, lang in enumerate(langs):
            if cancelled():
                raise Cancelled()
            tag = f"[{li + 1}/{len(langs)}] {pl.LANG_NAMES.get(lang, lang)}"
            tr = pl.narration_file(sdir, pid, lang)
            scenes = pl.load_scenes(sheet, lang, tr)

            begin_step("voice", lang)
            set_job(label=f"{tag} — narration")
            log(f"{tag}: generating narration ({len(scenes)} lines)")
            t0 = time.time()

            def on_voice(d, t, m, tag=tag):
                if cancelled():
                    raise Cancelled()
                progress(d, t, f"{tag} — voicing line {d} of {t}")
                log(f"  {m}")

            vs = pl.generate_voice(
                scenes, lang, sheet, voice=voices.get(lang) or None,
                on_progress=on_voice)
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
                    style=pl.effective_caption_style(pid), master=master,
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
    except Cancelled:
        log("Stopped — kept any videos already finished.")
        return                               # scheduler marks it Stopped
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_steps(pid: str, langs: list[str], steps: list[str], captions: bool,
              music: str | None, zoom: bool, voices: dict[str, str],
              force: bool = False, master: bool = True,
              skip_unvoiced: bool = False) -> None:
    """Run a chosen subset of steps for chosen languages.

    `steps` is any of "voice" and "render". This is what the project view's
    per-step buttons call, so re-rendering does not silently redo narration
    (7 minutes) and vice versa.

    `force` deletes this language's cached narration first — the only way to
    genuinely redo voicing, since the cache is keyed by text and settings and
    would otherwise be reused.
    """
    try:
        VIS.unload()          # free the sourcing model before voice/render
        proj = _project_or_skip(pid)
        if proj is None:
            return
        sheet = Path(proj["sheet"])
        sdir = sheet.parent
        # Default to the project's own main language, never a hardcoded "en".
        langs = langs or [pl.main_lang(sheet)]

        # A language needs a reference voice clip to be narrated or rendered.
        # Manually running a specific language with no clip is a mistake worth
        # stopping for; but the hands-off Auto-process just skips the ones that
        # aren't set up yet and builds the rest, so one un-voiced language never
        # sinks the whole run. Done before begin_job so the queue shows only the
        # languages actually being built.
        skipped: list[str] = []
        if "voice" in steps or "render" in steps:
            missing = [l for l in langs
                       if not (voices.get(l) or vx.pref_for(l).get("reference"))]
            if missing and skip_unvoiced:
                langs = [l for l in langs if l not in missing]
                skipped = missing
                if not langs:
                    raise RuntimeError(
                        "None of this project's languages have a reference voice "
                        "yet. Choose one in the Voices panel, then run again.")
            elif missing:
                names = ", ".join(pl.LANG_NAMES.get(m, m) for m in missing)
                raise RuntimeError(
                    f"No reference clip chosen for {names}. Pick one in the "
                    f"Voices panel, then try again.")

        begin_job(pid, langs, steps[0] if steps else "voice")
        if skipped:                              # after begin_job, which clears the log
            names = ", ".join(pl.LANG_NAMES.get(m, m) for m in skipped)
            log(f"Skipping {names}: no reference voice chosen yet "
                f"(set one in the Voices panel to include it next time).")

        assets = {}
        if "render" in steps:
            assets_f = pl.paths_for(sheet, "en")["assets"]
            if not assets_f.exists():
                raise RuntimeError(
                    "No visuals sourced yet. Run 'Find visuals' first.")
            assets = {int(k): v for k, v in
                      json.loads(assets_f.read_text(encoding="utf-8")).items()}

        outputs = []
        for li, lang in enumerate(langs):
            if cancelled():
                raise Cancelled()
            tag = f"[{li + 1}/{len(langs)}] {pl.LANG_NAMES.get(lang, lang)}"
            tr = pl.narration_file(sdir, pid, lang)
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
                        style=pl.effective_caption_style(pid), master=master,
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
        log("Stopped. Whatever was already generated is kept and reused.")
        return                               # scheduler marks it Stopped
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------- script view

def script_data(pid: str) -> dict:
    """The narration a project was built from, per scene and per language, with
    the image query beside each line — for a read-only 'View script' panel."""
    proj = pl.find_project(pid)
    if proj is None:
        return {"error": "project not found"}
    sheet = Path(proj["sheet"])
    langs = []
    for lg in proj["languages"]:
        code = lg["code"]
        try:
            tr = pl.narration_file(sheet.parent, pid, code)
            scenes = pl.load_scenes(sheet, code, tr)
        except Exception:
            scenes = []
        rows = [{"n": s.n, "narration": s.narration, "query": s.query,
                 "media": s.media, "hero": bool(getattr(s, "hero", False)),
                 "exact": bool(getattr(s, "exact", False))}
                for s in scenes]
        text = " ".join(r["narration"] for r in rows if r["narration"]).strip()
        langs.append({"code": code, "name": lg.get("name", code), "scenes": rows,
                      "text": text, "words": len(text.split())})
    return {"id": pid, "label": proj["label"], "languages": langs}


# ---------------------------------------------------------------- approval data

def approval_data(pid: str) -> dict:
    proj = pl.find_project(pid)
    sheet = Path(proj["sheet"])
    p = pl.paths_for(sheet, "en")        # assets/picks are shared, language-agnostic
    # Read scenes from the main script in its own language (English is not
    # guaranteed to exist); the review page shows structure + visuals, not
    # a specific language's narration.
    scenes = pl.load_scenes(sheet, pl.main_lang(sheet), None)
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
    try:
        from lib import imagen as _IM
        generate_on = _IM.available(cfg)          # image gen: true out of the box (Pollinations)
    except Exception:
        generate_on = False
    try:
        from lib import veo as _VEO               # video gen: only when Vertex is set up
        veo_on = _VEO.available(cfg)
    except Exception:
        veo_on = False
    return {"id": pid, "label": proj["label"], "items": items,
            "clip_min": float(cfg.get("clip_min") or 0.45),
            "clip_on": VIS.capability(cfg)["ok"],
            "generate_on": generate_on, "veo_on": veo_on,
            "engine": (cfg.get("generate_engine") or "pollinations"),
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

    MAX_UPLOAD = 200 * 1024 * 1024                    # 200 MB — comfortably fits a clip

    def _scene_upload(self) -> None:
        """Replace one scene's visual with an uploaded image/video. The file is the
        raw request body; the project id, scene number and filename come from the
        query string (POST /api/scene_upload?id=..&n=..&name=..)."""
        q = parse_qs(urlparse(self.path).query)
        pid = (q.get("id", [""])[0]).strip()
        n_raw = (q.get("n", [""])[0]).strip()
        fname = (q.get("name", [""])[0]).strip()

        proj = pl.find_project(pid) if pid else None
        if proj is None:
            return self._json({"error": f"no project called {pid!r}"}, 404)
        if not n_raw.isdigit():
            return self._json({"error": "missing scene number"}, 400)
        n = int(n_raw)
        sheet = Path(proj["sheet"])
        # Only accept a real scene, so an upload can never create an orphan asset.
        try:
            scene_ns = {s.n for s in pl.load_scenes(sheet, pl.main_lang(sheet), None)}
        except Exception:
            scene_ns = set()
        if scene_ns and n not in scene_ns:
            return self._json({"error": f"no scene {n} in this project"}, 400)

        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            return self._json({"error": "no file received"}, 400)
        if size > self.MAX_UPLOAD:
            return self._json({"error": "file is too large (max 200 MB)"}, 413)
        raw = self.rfile.read(size)

        ext = os.path.splitext(fname)[1].lstrip(".").lower()
        try:
            info = pl.set_scene_upload(sheet, n, raw, ext)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"n": info["n"], "media": info["media"],
                           "url": f"/media/{info['file']}"})

    def _guard(self, fn):
        """Run a handler so a crash becomes a 500 with a readable message, never
        a dead connection. Without this, one bad endpoint (a corrupt config.json,
        a malformed project file) closes the socket with no response — the
        browser reports ERR_EMPTY_RESPONSE and the whole UI says 'Failed to
        fetch' with no clue why. Catches SystemExit too, because load_config
        raises it on invalid JSON."""
        try:
            fn()
        except (Exception, SystemExit) as e:
            traceback.print_exc()
            try:
                self._json({"error": str(e) or e.__class__.__name__}, 500)
            except Exception:
                pass          # headers may already be on the wire — nothing to do

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        self._guard(self._get)

    def _get(self):
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
        if not path.startswith(("/api/", "/media/", "/out/", "/preview/", "/channelimg/")):
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/projects":
            projects = pl.find_projects()
            cfg = pl.load_config()
            music = sorted(f.name for f in (ROOT / "music").glob("*")
                           if f.suffix.lower() in (".mp3", ".m4a", ".wav", ".aac"))
            # Attach per-language status so the dashboard needs one request,
            # not one per project.
            for pr in projects:
                # SystemExit is not an Exception; library code raises it for
                # user-facing problems. Catch it too so one bad project shows an
                # error card instead of taking down the whole dashboard.
                try:
                    pr["status"] = pl.project_status(
                        Path(pr["sheet"]), pr["languages"])
                except (Exception, SystemExit) as e:
                    pr["status"] = {"error": str(e), "scenes": pr.get("scenes", 0),
                                    "assets": 0, "languages": {}}
            # Tag each project with its channel, and list the channels (pruning
            # any assignment whose project no longer exists) so the dashboard can
            # group by channel in one request.
            from lib import channels as _ch
            from lib import pflags as _pf
            chan = _ch.data(valid_pids=[pr["id"] for pr in projects])
            flags = _pf.data(valid_pids=[pr["id"] for pr in projects])
            for pr in projects:
                pr["channel"] = chan["assign"].get(pr["id"], "")
                pr["uploaded"] = bool(flags.get(pr["id"], {}).get("uploaded"))
            # Per-channel profile (image/email/description) for the dashboard.
            # The image is exposed as a ready-to-use URL (with a cache-buster).
            meta = {}
            for name in chan["channels"]:
                m = dict(chan.get("meta", {}).get(name, {}))
                img = m.get("image") or ""
                m["image_url"] = f"/channelimg/{img}" if img else ""
                meta[name] = {"email": m.get("email", ""),
                              "description": m.get("description", ""),
                              "image_url": m["image_url"]}
            return self._json({
                "projects": projects, "music": music,
                "channels": chan["channels"], "channel_meta": meta,
                "has_keys": bool(cfg.get("pexels_key") or cfg.get("pixabay_key")),
                "has_gemini": bool(cfg.get("gemini_key")),
            })

        if path == "/api/channels":
            from lib import channels as _ch
            pids = [p["id"] for p in pl.find_projects()]
            d = _ch.data(valid_pids=pids)
            counts = {}
            for c in d["assign"].values():
                counts[c] = counts.get(c, 0) + 1
            return self._json({"channels": d["channels"], "assign": d["assign"],
                               "counts": counts})

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

        if path == "/api/config":
            # The Settings editor: the schema (types, options, help, show-if) with
            # the CURRENT value folded into each field, read from the real file.
            from lib import config_schema as CS
            return self._json({"sections": CS.schema(pl.read_config_file())})

        if path == "/api/doctor":
            # The same checks the `faceless check` command runs, for Status.
            from lib import chatterbox_engine as CB
            import shutil as _sh
            caps = R.caption_method()
            dev = CB.device_info() if CB.installed() else {}
            # The GPU is a property of torch, not of the voice engine. If
            # Chatterbox can't import, don't let that masquerade as "no GPU" —
            # probe torch directly (the same real-kernel test vision/align use) so
            # the GPU line stays honest even when the voice engine is broken.
            if not dev.get("device"):
                try:
                    d2, vram = VIS._probe_device()
                    if d2 in ("cuda", "mps"):
                        dev = {"device": d2, "vram_gb": vram,
                               "name": "GPU (voice engine not loaded)"}
                except Exception:
                    pass
            langs = {}
            for lg in pl.LANG_NAMES:          # every narration-capable language
                try:
                    langs[lg] = vx.status(lg)
                except Exception:
                    langs[lg] = {}
            cfg = pl.load_config()

            # Voice engine — say clearly WHICH backend is narrating right now,
            # which model, and on what device, so Status isn't a guessing game.
            # engine_status() is the one place selection/readiness is decided.
            es = tts.engine_status(cfg)
            v_selected, v_active = es["selected"], es["active"]
            higgs_ok, higgs_present = es["higgs_usable"], es["higgs_installed"]
            google_ok = es["google_ready"]
            if v_active == "higgs":
                from lib import higgs_engine as HG
                v_model = str(cfg.get("higgs_model") or HG.DEFAULT_MODEL).split("/")[-1]
                v_devinfo = HG.device_info(cfg)
            elif v_active == "chirp":
                v_model = "Google Chirp 3 HD"
                v_devinfo = {"device": "cloud", "name": "Google Cloud TTS"}
            else:
                v_model = "Chatterbox Multilingual"
                v_devinfo = dev
            # Why the picked engine isn't the active one (fallback), and how to fix.
            v_hint = ""
            if v_selected == "chirp" and not google_ok:
                from lib import gtts_engine as GT
                v_hint = GT.install_hint()
            elif v_selected == "higgs" and not higgs_present:
                from lib import higgs_engine as HG
                v_hint = HG.install_hint()
            voice = {
                "selected": v_selected,
                "active": v_active,
                "chatterbox_installed": es["chatterbox_installed"],
                "higgs_installed": higgs_present,
                "higgs_usable": higgs_ok,
                "higgs_reason": es["higgs_reason"],
                "google_ready": google_ok,
                "google_reason": es["google_reason"],
                "model": v_model,
                "device": v_devinfo.get("device") or "cpu",
                "device_name": v_devinfo.get("name"),
                "fallback": v_active != v_selected,
                "install_hint": v_hint,
            }

            return self._json({
                "python": sys.version.split()[0],
                "in_venv": Path(sys.prefix) == (ROOT / ".venv"),
                "ffmpeg": _sh.which("ffmpeg") or "",
                "ffprobe": _sh.which("ffprobe") or "",
                "captions": caps,
                "captions_ok": caps in ("ass", "subtitles"),
                "ffmpeg_hint": R.ffmpeg_fix_hint(),
                "chatterbox": CB.installed(),
                "voice": voice,
                "device": dev,
                "gpu_ok": dev.get("device") in ("cuda", "mps"),
                "clip": VIS.capability(cfg),
                "align": AL.capability(cfg),
                "llm": LLM.capability(cfg),
                "audio": {"master": pl._flag(cfg.get("audio_master", "auto")),
                          "lufs": float(cfg.get("lufs_target") or -14.0),
                          "duck": pl._flag(cfg.get("music_duck", True))},

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
            # Backward-compatible single-active-job view the Activity screen polls.
            return self._json(_status_payload(_active_job()))

        if path == "/api/queue":
            return self._json(_queue_payload())

        if path == "/api/job":
            # One specific job by id — what the Activity screen shows when you
            # click "Watch" on a queue row (there can be several running at once,
            # so "the active job" is no longer a single thing).
            jid = (q.get("id") or [""])[0]
            return self._json(_status_payload(STORE.get(jid)))

        if path == "/api/voices":
            pid = (q.get("id") or [""])[0]
            scenes = None
            lang = (q.get("lang") or [""])[0]
            if pid:
                try:
                    proj = pl.find_project(pid)
                    sheet = Path(proj["sheet"])
                    # Default to the project's main language, not "en".
                    lang = lang or pl.main_lang(sheet)
                    tr = pl.narration_file(pl.sheets_dir(pid), pid, lang)
                    scenes = pl.load_scenes(sheet, lang, tr)
                # load_scenes raises SystemExit (not Exception) for a language
                # with no narration file — catch it so the panel still opens.
                except (Exception, SystemExit):
                    scenes = None
            lang = lang or "en"          # global (project-less) voices page
            vx.ensure_folders()
            refs = vx.references(lang)
            cfg_v = pl.load_config()
            es = tts.engine_status(cfg_v)      # single source of truth
            selected, google_ok = es["selected"], es["google_ready"]
            # Google Chirp: list the catalogue voices for this language so the
            # panel can offer them (only when Chirp is selected — it's a network
            # call, pointless otherwise), plus this language's starred favourites.
            google_voices: list = []
            google_error = ""
            google_favorites: list = []
            try:
                from lib import gtts_engine as _GT
                if selected == "chirp" and google_ok:
                    google_voices = _GT.voices(lang, cfg_v)
                    if not google_voices:
                        google_error = _GT.last_voice_error()
                elif selected == "chirp" and not google_ok:
                    google_error = _GT.install_hint()
                # Favourites for THIS language (voice names carry their locale, so
                # match on the language's first segment: en, de, …). Gender is
                # pulled from the catalogue when we have it.
                base = lang.split("-")[0]
                gender = {v["name"]: v.get("gender", "") for v in google_voices}
                google_favorites = [{"name": n, "gender": gender.get(n, ""),
                                     "favorite": True}
                                    for n in vx.favorites("google")
                                    if n.split("-")[0] == base]
            except Exception as e:
                google_voices, google_error = [], str(e)
            engine = {"selected": selected, "active": es["active"],
                      "higgs_installed": es["higgs_installed"],
                      "google_ready": google_ok}
            # The transcript actually in use for the chosen clip: the manual
            # override, else the auto-generated .txt on disk (display only).
            chosen_pref = vx.pref_for(lang)
            transcript = (chosen_pref.get("reference_text") or "").strip()
            if not transcript and chosen_pref.get("reference"):
                try:
                    from lib import higgs_engine as _HG2
                    transcript = _HG2.sibling_transcript(chosen_pref["reference"])
                except Exception:
                    transcript = ""
            return self._json({
                "lang": lang,
                "lang_name": vx.LANGS.get(lang, lang),
                "languages": vx.LANGS,
                "status": vx.status(lang),
                "engine": engine,
                "transcript": transcript,
                # Google Chirp: the catalogue for the panel dropdown + the chosen
                # voice name. Empty catalogue simply means Chirp isn't the active
                # engine (or creds aren't ready) — the panel hides the dropdown.
                "google_voices": google_voices,
                "google_voice": chosen_pref.get("google_voice", ""),
                "google_error": google_error,
                "google_favorites": google_favorites,
                # Only this language's clips, plus any left loose — a German
                # list full of English voices is noise, not choice.
                "references": [r for r in refs if r["lang"] == lang],
                "loose": [r for r in refs if not r["lang"]],
                "counts": {c: len(vx.references(c)) for c in pl.LANG_NAMES},
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

        if path == "/api/script":
            d = script_data((q.get("id") or [""])[0])
            return self._json(d, 404 if d.get("error") else 200)

        if path == "/api/metadata":
            # Load whatever description/tags were last generated or edited for a
            # language. Empty is a valid answer (nothing generated yet).
            pid = (q.get("id") or [""])[0]
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": "project not found"}, 404)
            lang = (q.get("lang") or [""])[0] or pl.main_lang(Path(proj["sheet"]))
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

        if path.startswith("/channelimg/"):
            # Uploaded channel avatars.
            from lib import channels as _ch
            name = unquote(path[len("/channelimg/"):]).split("?")[0]
            return self._serve_file(_ch.IMG_DIR / name, _ch.IMG_DIR)

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
        self._guard(self._post)

    def _post(self):
        path = urlparse(self.path).path
        # A file upload sends raw bytes, not JSON — handle it before the JSON body
        # is read, so a 100 MB video is streamed to disk rather than parsed.
        if path == "/api/scene_upload":
            return self._scene_upload()
        try:
            b = self._body()
        except json.JSONDecodeError:
            return self._json({"error": "bad request"}, 400)

        if path == "/api/preview":
            try:
                # Google Chirp: audition a catalogue voice (no reference clip).
                if b.get("google_voice"):
                    from lib import gtts_engine as GT
                    cfg_p = pl.load_config()
                    lang_p = b.get("lang") or "en"
                    f = GT.preview(
                        b.get("text") or vx.sample_line(lang_p),
                        lang_p, b.get("google_voice"), cfg=cfg_p,
                        out_dir=vx.PREVIEWS,
                        rate=float(cfg_p.get("google_tts_rate", 1.0) or 1.0),
                        log=lambda *_: None)
                    return self._json({"url": f"/preview/{f.name}"})
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
            fields = dict(reference=b.get("reference"),
                          exaggeration=b.get("exaggeration"),
                          cfg_weight=b.get("cfg_weight"))
            # Only touch reference_text (Higgs clone transcript) when the caller
            # actually sent it, so choosing a clip never wipes an existing one.
            if "reference_text" in b:
                fields["reference_text"] = b.get("reference_text")
            # Google Chirp voice, likewise only when sent, so picking a Chatterbox
            # clip never clears the Google choice and vice versa.
            if "google_voice" in b:
                fields["google_voice"] = b.get("google_voice")
            saved = vx.save_pref(lang, **fields)
            log(f"Voice for {lang}: {tts.describe(lang)}")
            return self._json({"saved": saved})

        if path == "/api/favorite_voice":
            # Star / unstar a Google voice so it stays at the top of the picker.
            name = (b.get("name") or "").strip()
            if not name:
                return self._json({"error": "no voice name"}, 400)
            favs, on = vx.toggle_favorite(b.get("engine") or "google", name,
                                          b.get("on"))
            return self._json({"favorites": favs, "favorite": on})

        if path == "/api/project_channel":
            # Move a project into a channel (or clear it with an empty string).
            # Create-on-assign: a new channel name is added automatically.
            from lib import channels as _ch
            pid = (b.get("id") or "").strip()
            if not pid or pl.find_project(pid) is None:
                return self._json({"error": f"no project called {pid!r}"}, 404)
            try:
                d = _ch.assign(pid, b.get("channel") or "")
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"channel": d["assign"].get(pid, ""),
                               "channels": d["channels"]})

        if path == "/api/project_upload":
            # Mark (or unmark) a project as uploaded to YouTube.
            from lib import pflags as _pf
            pid = (b.get("id") or "").strip()
            if not pid or pl.find_project(pid) is None:
                return self._json({"error": f"no project called {pid!r}"}, 404)
            f = _pf.set_uploaded(pid, bool(b.get("uploaded")))
            return self._json({"uploaded": f["uploaded"], "uploaded_at": f["uploaded_at"]})

        if path == "/api/channels":
            # Manage the channel list itself: create / rename / delete a channel,
            # or edit its profile (email / description). Deleting a channel only
            # drops the label — its projects are unassigned, never removed.
            from lib import channels as _ch
            action = (b.get("action") or "").strip().lower()
            try:
                if action == "create":
                    d = _ch.create(b.get("name") or "")
                elif action == "rename":
                    d = _ch.rename(b.get("old") or "", b.get("name") or "")
                elif action == "delete":
                    d = _ch.remove(b.get("name") or "")
                elif action == "meta":
                    d = _ch.set_meta(b.get("name") or "",
                                     email=b.get("email"),
                                     description=b.get("description"))
                else:
                    return self._json({"error": "unknown action"}, 400)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"channels": d["channels"]})

        if path == "/api/channel_image":
            # Upload a channel avatar. The image arrives as a data URL
            # ("data:image/png;base64,…"); decode and store it. Capped so a huge
            # paste can't fill the disk.
            from lib import channels as _ch
            name = (b.get("name") or "").strip()
            data_url = b.get("data") or ""
            if not name:
                return self._json({"error": "channel name is empty"}, 400)
            m = re.match(r"data:image/(png|jpe?g|webp|gif);base64,(.+)$",
                         data_url, re.I | re.S)
            if not m:
                return self._json({"error": "expected a PNG/JPG/WebP/GIF image"}, 400)
            try:
                import base64 as _b64
                raw = _b64.b64decode(m.group(2), validate=False)
            except Exception:
                return self._json({"error": "could not decode the image"}, 400)
            if len(raw) > 5 * 1024 * 1024:
                return self._json({"error": "image is too large (max 5 MB)"}, 400)
            ext = m.group(1).lower().replace("jpeg", "jpg")
            fname = _ch.set_image(name, raw, ext)
            return self._json({"image_url": f"/channelimg/{fname}"})

        if path == "/api/config":
            # Save settings from the editor. Every value is validated against the
            # schema; only known keys are written; the file is backed up and
            # replaced atomically. Nothing here is ever committed (gitignored).
            from lib import config_schema as CS
            updates = b.get("updates")
            if not isinstance(updates, dict):
                return self._json({"error": "expected an 'updates' object"}, 400)
            merged, errors = CS.validate_and_merge(pl.read_config_file(), updates)
            if errors:
                return self._json({"error": "Some values are invalid.",
                                   "errors": errors}, 400)
            pl.write_config_file(merged)
            return self._json({"saved": True})

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
            pid = b.get("id") or "video"
            job = enqueue(pid, "generate",
                          {"scripts": scripts, "pid": pid,
                           "overwrite": bool(b.get("overwrite")),
                           "channel": (b.get("channel") or "").strip()},
                          auto=bool(b.get("auto")))
            return self._json({"started": True, "job": job.id})

        if path == "/api/add_language":
            pid = b.get("id")
            lang = b.get("lang") or ""
            script = (b.get("script") or "").strip()
            if not (pid and lang and script):
                return self._json(
                    {"error": "Need a project, a language and a pasted script."}, 400)
            job = enqueue(pid, "add_language",
                          {"pid": pid, "lang": lang, "script": script,
                           "overwrite": bool(b.get("overwrite"))})
            return self._json({"started": True, "job": job.id})

        if path == "/api/source":
            pid = b.get("id")
            # auto=true chains straight on to voice+render (Auto-process the rest),
            # so it skips the review stop and finishes 'done' to trigger the chain.
            auto = bool(b.get("auto"))
            job = enqueue(pid, "source",
                          {"pid": pid, "redo": b.get("redo") or None,
                           "skip_review": auto}, auto=auto)
            return self._json({"started": True, "job": job.id})

        if path == "/api/regenerate":
            scenes = [int(n) for n in (b.get("scenes") or []) if str(n).isdigit()]
            if not scenes:
                return self._json({"error": "no scenes to generate"}, 400)
            job = enqueue(b.get("id"), "regenerate",
                          {"pid": b.get("id"), "which": scenes})
            return self._json({"started": True, "job": job.id})

        if path == "/api/regenerate_video":
            scenes = [int(n) for n in (b.get("scenes") or []) if str(n).isdigit()]
            if not scenes:
                return self._json({"error": "no scenes to generate"}, 400)
            job = enqueue(b.get("id"), "regenerate_video",
                          {"pid": b.get("id"), "which": scenes})
            return self._json({"started": True, "job": job.id})

        if path == "/api/build":
            pid = b.get("id")
            job = enqueue(pid, "build", {
                "pid": pid, "langs": b.get("langs") or [],
                "captions": bool(b.get("captions", True)),
                "music": b.get("music") and str(ROOT / "music" / b["music"]),
                "zoom": bool(b.get("zoom", True)), "voices": b.get("voices") or {},
                "master": bool(b.get("master", True))})
            return self._json({"started": True, "job": job.id})

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
            job = enqueue(pid, "run", {
                "pid": pid, "langs": b.get("langs") or [], "steps": steps,
                "captions": bool(b.get("captions")), "music": b.get("music") or None,
                "zoom": b.get("zoom", True), "voices": b.get("voices") or {},
                "force": bool(b.get("force")), "master": bool(b.get("master", True))})
            return self._json({"started": True, "job": job.id, "steps": steps})

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

            if any(j.project == pid and j.status in (jobs.RUNNING, jobs.QUEUED)
                   for j in STORE.jobs()):
                return self._json(
                    {"error": "this project has a job running or queued — "
                              "cancel it in the queue first"}, 409)

            res = pl.delete_project(Path(proj["sheet"]), proj["languages"], what)
            if "sheets" in what:                 # the project itself is gone
                from lib import channels as _ch
                from lib import pflags as _pf
                _ch.forget_project(pid)
                _pf.forget(pid)
            log(f"Deleted {res['count']} file(s) from {pid} "
                f"({res['freed_mb']} MB freed)")
            return self._json(res)

        if path == "/api/metadata":
            # Generate title/description/tags for one language. Synchronous — it
            # is a single Gemini call, not a long job — so the UI just waits.
            pid = b.get("id")
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": "project not found"}, 404)
            lang = b.get("lang") or pl.main_lang(Path(proj["sheet"]))
            try:
                data = pl.build_metadata(Path(proj["sheet"]), lang, pl.load_config())
                log(f"Wrote {lang} description for {pid}")
                return self._json(data)
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/metadata_save":
            # Persist the user's edits to the description/tags.
            pid = b.get("id")
            proj = pl.find_project(pid)
            if proj is None:
                return self._json({"error": "project not found"}, 404)
            lang = b.get("lang") or pl.main_lang(Path(proj["sheet"]))
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
            # Cancel a specific job by id, or (no id) whatever is running now.
            jid = b.get("job") or b.get("id")
            if jid and SCHED and STORE.get(jid):
                return self._json({"ok": SCHED.cancel(jid)})
            running = [j for j in STORE.jobs() if j.status == jobs.RUNNING]
            if not running:
                return self._json({"ok": True, "note": "nothing was running"})
            for j in running:
                SCHED.cancel(j.id)
            return self._json({"ok": True})

        if path == "/api/job/remove":
            jid = b.get("job") or b.get("id")
            job = STORE.get(jid) if jid else None
            if job is None:
                return self._json({"error": "no such job"}, 404)
            if job.status == jobs.RUNNING:
                return self._json({"error": "cancel it first, then remove"}, 409)
            return self._json({"ok": STORE.remove(jid)})

        if path == "/api/queue/clear":
            return self._json({"removed": STORE.clear_finished(), "ok": True})

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


# kind -> the worker that runs it. Each is wrapped so the scheduler can bind the
# thread-local current job and pass only the kwargs the worker declares.
RUN_MAP = {
    "generate": _worker_wrapper(run_generate),
    "add_language": _worker_wrapper(run_add_language),
    "source": _worker_wrapper(run_sourcing),
    "regenerate": _worker_wrapper(run_regenerate),
    "regenerate_video": _worker_wrapper(run_regenerate_video),
    "build": _worker_wrapper(run_build),
    "run": _worker_wrapper(run_steps),
}


# Heavy-on-the-GPU work: voicing (local TTS) and rendering (encode). Two of these
# at once fight over the card and get slower, not faster — so the scheduler never
# overlaps them. Everything else (script generation, finding visuals, AI image /
# video) is network-bound — those DO overlap a render, which is the whole point of
# the queue. If the language model runs locally (ollama), generation is GPU-heavy
# too, so it joins the exclusive set (handled in _start_scheduler).
GPU_KINDS = {"voice", "render", "build", "run"}


def _total_ram_gb() -> float | None:
    """Best-effort physical RAM in GB, cross-platform, no hard dependencies."""
    try:
        import psutil                                    # noqa: F401
        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 1e9
        except Exception:
            pass
    return None


def _auto_concurrency() -> int:
    """How many jobs to run at once, judged from the machine. The GPU is protected
    by GPU_KINDS regardless, so this really sizes how many network-bound jobs may
    overlap a render — a bigger machine can juggle more without feeling it."""
    cpu = os.cpu_count() or 2
    ram = _total_ram_gb()
    if ram is None:
        return 2 if cpu >= 8 else 1
    if ram >= 24 and cpu >= 8:
        return 3
    if ram >= 12 and cpu >= 4:
        return 2
    return 1


def _resolve_concurrency(cfg: dict) -> int:
    """config's max_concurrent_jobs wins if it's a real number; 'auto' (or unset,
    or junk) sizes it from the machine. Never below 1."""
    raw = cfg.get("max_concurrent_jobs")
    if isinstance(raw, bool):                            # True/False is not a count
        raw = None
    if isinstance(raw, (int, float)) and int(raw) >= 1:
        return int(raw)
    if isinstance(raw, str) and raw.strip().isdigit() and int(raw) >= 1:
        return int(raw)
    return _auto_concurrency()


# The hands-off pipeline: when a project is created with Auto-process on, each
# stage enqueues the next as it finishes, so a fresh script turns into a finished
# video with no clicks in between (option: review is skipped). The chain stops on
# error or a manual Stop — never runs past a stage that didn't truly succeed.
AUTO_CHAIN = {"generate": "source", "source": "run"}


def _auto_next(job) -> None:
    """Scheduler on_done hook: queue the next auto stage for this project."""
    # A stopped run reports 'done' (its partial work is kept) but carries the
    # cancel flag — never chain past a Stop, and never past an error.
    if not job.auto or job.cancel or job.error or SCHED is None:
        return
    nxt = AUTO_CHAIN.get(job.kind)
    if nxt == "source":
        SCHED.enqueue(job.project, "source",
                      {"pid": job.project, "redo": None, "skip_review": True},
                      auto=True)
    elif nxt == "run":
        # Build every language the project has, not just the main one — a project
        # with German and Spanish narration should come out the far end with all
        # three videos. Empty falls back to the main language inside the worker.
        proj = pl.find_project(job.project)
        langs = [l["code"] for l in (proj or {}).get("languages", [])]
        SCHED.enqueue(job.project, "run",
                      {"pid": job.project, "langs": langs, "steps": ["voice", "render"],
                       "captions": True, "music": None, "zoom": True, "voices": {},
                       "force": False, "master": True, "skip_unvoiced": True},
                      auto=True)


def _start_scheduler() -> None:
    """Load any saved queue and start running it, with resource-aware concurrency.
    A job left mid-run by a crashed server was reloaded as INTERRUPTED and is
    auto-resumed here — and because every pipeline step is cached, it resumes where
    it left off rather than 'starting from 0'."""
    global SCHED
    cfg = pl.load_config()
    cap = _resolve_concurrency(cfg)
    gpu = set(GPU_KINDS)
    if str(cfg.get("llm", "")).lower() == "ollama":
        gpu |= {"generate", "add_language"}              # LLM now runs on the card

    def gate(job, running) -> bool:
        # A GPU-heavy job waits until no other GPU-heavy job is running; anything
        # network-bound is free to overlap (up to the overall cap).
        if job.kind in gpu:
            return not any(r.kind in gpu for r in running)
        return True

    STORE.load()
    SCHED = jobs.Scheduler(STORE, RUN_MAP, max_concurrent=cap, resume=True,
                           gate=gate, on_done=_auto_next)
    print(f"  Queue ready — up to {cap} job(s) at once "
          f"(one GPU-heavy step at a time).")


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

    _start_scheduler()

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
