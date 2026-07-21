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
from lib import tts, voices as vx  # noqa: E402

PORT = 8765
UI = ROOT / "lib" / "ui.html"

# ------------------------------------------------------------------- job state

JOB = {
    "stage": "idle",      # idle | stock | voice | render | done | error
    "label": "",
    "done": 0, "total": 0,
    "log": [],
    "error": "",
    "outputs": [],
    "project": None,
    "langs": [],
    "warnings": [],
}
LOCK = threading.Lock()


def set_job(**kw) -> None:
    with LOCK:
        JOB.update(kw)


def log(msg: str) -> None:
    with LOCK:
        JOB["log"].append(msg)
        del JOB["log"][:-400]


def progress(done: int, total: int, label: str = "") -> None:
    with LOCK:
        JOB["done"], JOB["total"] = done, total
        if label:
            JOB["label"] = label


# ------------------------------------------------------------------ the work

def run_generate(script: str, pid: str, langs: list[str], overwrite: bool) -> None:
    """Script in → production sheets out. Gemini returns structured data only;
    the file format is written by compose.py, so it cannot come out malformed."""
    try:
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
            script, pid, langs, key,
            model=cfg.get("gemini_model", gem.DEFAULT_MODEL),
            on_progress=lambda d, t, m: (progress(d, t, m), log(f"  {m}")),
            on_warn=lambda m: log(f"  ⚠ {m}"))

        written = compose.write_files(res, ROOT / "sheets", overwrite=overwrite)
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
                outputs=[{"lang": "-", "name": n, "path": str(ROOT / "sheets" / n),
                          "size_mb": 0} for n in written],
                warnings=res.warnings)
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_sourcing(pid: str, redo: list[int] | None) -> None:
    try:
        sheets = ROOT / "sheets"
        proj = next(p for p in pl.find_projects(sheets) if p["id"] == pid)
        sheet = sheets / proj["sheet"]
        cfg = pl.load_config()
        if not cfg.get("pexels_key") and not cfg.get("pixabay_key"):
            raise RuntimeError(
                "No stock API key yet. Open config.json and paste your free "
                "Pexels and Pixabay keys in, then try again.")
        scenes = pl.load_scenes(sheet, "en", None)
        set_job(stage="stock", label="sourcing visuals", done=0,
                total=len(redo or scenes), error="", project=pid)
        log(f"Sourcing visuals for {proj['label']}")

        def onp(d, t, m):
            progress(d, t, m)
            log(f"  {m}")

        pl.source_stock(scenes, sheet, cfg, redo=redo, on_progress=onp)
        set_job(stage="approve", label="ready for review")
        log("Visuals ready — review them below.")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def run_build(pid: str, langs: list[str], captions: bool, music: str | None,
              zoom: bool, voices: dict[str, str]) -> None:
    try:
        sheets = ROOT / "sheets"
        proj = next(p for p in pl.find_projects(sheets) if p["id"] == pid)
        sheet = sheets / proj["sheet"]
        assets_f = pl.paths_for(sheet, "en")["assets"]
        assets = {int(k): v for k, v in json.loads(assets_f.read_text(encoding="utf-8")).items()}
        outputs = []
        set_job(stage="voice", error="", outputs=[], project=pid, langs=langs)

        for li, lang in enumerate(langs):
            tag = f"[{li + 1}/{len(langs)}] {pl.LANG_NAMES.get(lang, lang)}"
            tr = pl.translation_for(sheets, pid, lang)
            scenes = pl.load_scenes(sheet, lang, tr)

            set_job(stage="voice", label=f"{tag} — narration")
            log(f"{tag}: generating narration ({len(scenes)} lines)")
            t0 = time.time()
            vs = pl.generate_voice(
                scenes, lang, sheet, voice=voices.get(lang) or None,
                on_progress=lambda d, t, m: (progress(d, t, f"{tag} — voicing line {d} of {t}"),
                                             log(f"  {m}")))
            log(f"{tag}: narration done in {time.time() - t0:.0f}s")

            set_job(stage="render", label=f"{tag} — building video")
            log(f"{tag}: rendering")
            t0 = time.time()
            try:
                out = pl.render_video(
                    scenes, assets, vs, sheet, lang, captions=captions,
                    music=Path(music) if music else None, zoom=zoom,
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
            outputs.append({"lang": lang, "name": out.name, "path": str(out),
                            "size_mb": round(out.stat().st_size / 1e6)})
            set_job(outputs=list(outputs))

        set_job(stage="done", label="all videos built")
        log("All done.")
    except Exception as e:
        set_job(stage="error", error=str(e))
        log(f"ERROR: {e}")
        traceback.print_exc()


def start_thread(fn, *a) -> bool:
    with LOCK:
        if JOB["stage"] in ("stock", "voice", "render"):
            return False
        JOB["log"] = []
        JOB["error"] = ""
    threading.Thread(target=fn, args=a, daemon=True).start()
    return True


# ---------------------------------------------------------------- approval data

def approval_data(pid: str) -> dict:
    sheets = ROOT / "sheets"
    proj = next(p for p in pl.find_projects(sheets) if p["id"] == pid)
    sheet = sheets / proj["sheet"]
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
            "url": f"/media/{Path(a['path']).name}" if a else "",
            "video": bool(a and Path(a["path"]).suffix.lower() in (".mp4", ".mov", ".webm")),
        })
    return {"id": pid, "label": proj["label"], "items": items,
            "missing": [i["n"] for i in items if not i["url"]]}


# ---------------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal clean
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
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

        if path in ("/", "/index.html"):
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/projects":
            projects = pl.find_projects(ROOT / "sheets")
            cfg = pl.load_config()
            music = sorted(f.name for f in (ROOT / "music").glob("*")
                           if f.suffix.lower() in (".mp3", ".m4a", ".wav", ".aac"))
            return self._json({
                "projects": projects, "music": music,
                "has_keys": bool(cfg.get("pexels_key") or cfg.get("pixabay_key")),
                "has_gemini": bool(cfg.get("gemini_key")),
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
                    sheets = ROOT / "sheets"
                    proj = next(p for p in pl.find_projects(sheets) if p["id"] == pid)
                    tr = pl.translation_for(sheets, pid, lang)
                    scenes = pl.load_scenes(sheets / proj["sheet"], lang, tr)
                except Exception:
                    scenes = None
            return self._json({
                "lang": lang,
                "languages": vx.LANGS,
                "status": vx.status(lang),
                "references": vx.references(),
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

        if path.startswith("/media/"):
            name = unquote(path[len("/media/"):])
            f = (ROOT / "cache" / "stock" / name).resolve()
            # never serve outside the cache folder
            if not str(f).startswith(str((ROOT / "cache" / "stock").resolve())) \
                    or not f.exists():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            return self._send(200, f.read_bytes(), ctype)

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
            script = (b.get("script") or "").strip()
            if len(script.split()) < 120:
                return self._json(
                    {"error": "That script looks too short — paste the whole thing."}, 400)
            ok = start_thread(run_generate, script, b.get("id") or "video",
                              b.get("langs") or ["en"], bool(b.get("overwrite")))
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

        if path == "/api/reveal":
            target = ROOT / "out"
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


def main(open_browser: bool = True) -> None:
    for d in ("sheets", "cache/stock", "cache/voice", "work", "out", "music"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    if not UI.exists():
        sys.exit(f"Missing {UI} — the app files are incomplete.")

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
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
        print("\n  Stopped. Your work is saved in out/ and cache/.\n")


if __name__ == "__main__":
    main()
