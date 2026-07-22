#!/usr/bin/env python3
"""Command-line front end. Most people should just double-click Start.bat
(Windows) or Start.command (macOS) instead — this exists for scripting,
automation and troubleshooting.

  python3 make_video.py studio                       # open the control panel
  python3 make_video.py stock  --sheet projects/video04/sheets/video04_MASTER_production_sheet.md
  python3 make_video.py voice  --sheet projects/video04/sheets/video04_MASTER_production_sheet.md --lang en
  python3 make_video.py render --sheet projects/video04/sheets/video04_MASTER_production_sheet.md --lang en --captions
  python3 make_video.py all    --sheet projects/video04/sheets/video04_MASTER_production_sheet.md --lang en --yes

Non-English needs a narration file (found automatically if it sits next to the
master in projects/<id>/sheets/ and is named like video04_GERMAN_narration.md):

  python3 make_video.py all --sheet projects/video04/sheets/video04_MASTER_production_sheet.md --lang de --yes

Tip: `make_video.py list` prints the exact --sheet path for every project.

Everything is cached, so re-running is cheap and picks up where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

from lib import approve, pipeline as pl, tts  # noqa: E402
from lib import console  # noqa: E402

# Windows consoles default to a legacy codepage and die on box-drawing
# characters. Do this before anything is printed.
console.setup()


def banner(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m")


def bar(done: int, total: int, label: str) -> None:
    if not total:
        return
    w = 28
    f = int(w * done / total)
    pct = int(100 * done / total)
    print(f"\r  [{'█' * f}{'░' * (w - f)}] {pct:>3}%  {label[:44]:<44}",
          end="", flush=True)
    if done >= total:
        print()


def resolve(a) -> tuple[Path, Path | None, list]:
    sheet = Path(a.sheet)
    if not sheet.exists():
        raise SystemExit(f"Sheet not found: {sheet}")
    pid = pl.project_id(sheet)
    tr = Path(a.translation) if a.translation else pl.translation_for(
        sheet.parent, pid, a.lang)
    if a.lang != "en" and tr is None:
        raise SystemExit(
            f"Language '{a.lang}' needs a translation file.\n"
            f"Put it in {sheet.parent}/ named like {pid}_GERMAN_narration.md, "
            f"or pass --translation.")
    if tr:
        print(f"Translation: {tr.name}")
    return sheet, tr, pl.load_scenes(sheet, a.lang, tr)


# ---------------------------------------------------------------------- steps

def step_stock(a) -> None:
    sheet, _, scenes = resolve(a)
    cfg = pl.load_config()
    if not cfg.get("pexels_key") and not cfg.get("pixabay_key"):
        raise SystemExit(
            "No stock API key found.\n"
            "Get free keys at https://www.pexels.com/api/ and "
            "https://pixabay.com/api/docs/, then paste them into config.json.")

    redo = [int(x) for x in a.redo.split(",") if x.strip()] if a.redo else None
    if redo:
        print(f"Pulling the next take for scenes: {redo}")

    banner(f"Sourcing visuals for {len(scenes)} scenes")
    assets = pl.source_stock(scenes, sheet, cfg, redo=redo, on_progress=bar)

    p = pl.paths_for(sheet, "en")
    picks = json.loads(p["picks"].read_text(encoding="utf-8")) if p["picks"].exists() else {}
    approve.build(scenes, assets, p["approval"], a.sheet, "en",
                  {int(k): v for k, v in picks.items()})
    banner("Approval sheet ready")
    print(f"  open {p['approval']}")
    missing = [s.n for s in scenes if s.n not in assets]
    if missing:
        print(f"  {len(missing)} scene(s) found nothing: {missing}")
        print("  Edit those 'ALT / search:' lines and re-run this step.")


def step_voice(a) -> None:
    sheet, _, scenes = resolve(a)
    banner(f"Narration — {len(scenes)} lines, {pl.LANG_NAMES.get(a.lang, a.lang)}")
    t0 = time.time()
    pl.generate_voice(scenes, a.lang, sheet, voice=a.voice, on_progress=bar)
    print(f"  done in {time.time() - t0:.0f}s")


def step_render(a) -> None:
    sheet, _, scenes = resolve(a)
    p = pl.paths_for(sheet, a.lang)
    af = pl.paths_for(sheet, "en")["assets"]
    if not af.exists():
        raise SystemExit("Run the 'stock' step first.")
    assets = {int(k): v for k, v in json.loads(af.read_text(encoding="utf-8")).items()}

    voices = pl.generate_voice(scenes, a.lang, sheet, voice=a.voice)
    banner(f"Building {len(scenes)} scenes")
    t0 = time.time()
    warn_caps = ""
    try:
        out = pl.render_video(scenes, assets, voices, sheet, a.lang,
                              captions=a.captions,
                              music=Path(a.music) if a.music else None,
                              music_level=a.music_level, zoom=not a.no_zoom,
                              caption_size=a.caption_size,
                              style=pl.effective_caption_style(pl.project_id(sheet)),
                              on_progress=bar)
    except pl.CaptionsSkipped as cs:
        out, warn_caps = cs.video, cs.reason
    from lib import render as R
    dur = R.duration_of(out)
    banner("Done")
    print(f"  {out}")
    print(f"  {int(dur // 60)}m {dur % 60:04.1f}s · {out.stat().st_size / 1e6:.0f} MB "
          f"· built in {(time.time() - t0) / 60:.1f} min")
    print(f"  subtitles: {p['srt']}")
    if warn_caps:
        print(f"\n  ⚠ Captions were NOT burned in — the video itself is fine.")
        print(f"    {warn_caps}")
        print(f"    Upload the .srt to YouTube instead, or run: {R.ffmpeg_fix_hint()}")


def step_generate(a) -> None:
    from lib import compose
    if not a.script:
        raise SystemExit("--script path/to/script.txt is required")
    src = Path(a.script)
    if not src.exists():
        raise SystemExit(f"Script not found: {src}")
    pid = a.id or src.stem
    cfg = pl.load_config()
    if not cfg.get("gemini_key"):
        raise SystemExit(
            "No Gemini key. Get a free one at https://aistudio.google.com/apikey "
            "and add it to config.json as \"gemini_key\".")

    # One script file = one language now (no translation). --lang says which.
    # If the project already has a master in another language, this ADDS the new
    # language onto its shared scenes; otherwise it creates the master.
    lang = (a.lang or (a.langs or "en").split(",")[0]).strip() or "en"
    text = src.read_text(encoding="utf-8")
    model = cfg.get("gemini_model", "gemini-2.5-flash")
    sdir = pl.sheets_dir(pid)
    master = sdir / f"{pid}_MASTER_production_sheet.md"
    if master.exists() and pl.master_lang(master) != lang:
        banner(f"Adding {pl.LANG_NAMES.get(lang, lang)} to '{pid}'")
        res = compose.add_language(master, lang, text, cfg["gemini_key"],
                                   model=model, on_progress=bar,
                                   on_warn=lambda m: print(f"\n  ⚠ {m}"))
    else:
        banner(f"Generating the {pl.LANG_NAMES.get(lang, lang)} sheet for '{pid}'")
        res = compose.generate({lang: text}, pid, cfg["gemini_key"],
                               model=model, on_progress=bar,
                               on_warn=lambda m: print(f"\n  ⚠ {m}"))
    written = compose.write_files(res, sdir, overwrite=a.overwrite)
    banner("Written")
    rel = sdir.relative_to(ROOT)
    for w in written:
        print(f"  {rel}/{w}")
    print(f"\n  {len(res.scenes)} scenes · "
          f"{sum(1 for s in res.scenes if s.media == 'VIDEO')} video · "
          f"{sum(1 for s in res.scenes if s.hero)} hero")
    if res.warnings:
        banner(f"{len(res.warnings)} thing(s) to check")
        for w in res.warnings:
            print("  " + w.replace("\n", "\n  "))


def step_all(a) -> None:
    step_stock(a)
    if not a.yes:
        banner("Review before rendering")
        print("  Open the approval sheet above. When you're happy:")
        print(f"  python3 make_video.py render --sheet {a.sheet} --lang {a.lang}")
        return
    step_voice(a)
    step_render(a)


# ------------------------------------------------------------------------ cli

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Faceless video pipeline (command line)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("step", choices=["studio", "doctor", "sources", "benchtts",
                                     "generate",
                                     "models", "stock", "voice", "render", "all",
                                     "voices", "list"])
    ap.add_argument("--no-browser", action="store_true",
                    help="studio: don't open a browser window automatically")
    ap.add_argument("--sheet")
    ap.add_argument("--script", help="script file to generate sheets from")
    ap.add_argument("--id", help="project name, e.g. video05")
    ap.add_argument("--langs", help="comma list for generate, default en,de,es")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="show full Python tracebacks")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--translation")
    ap.add_argument("--redo", help="comma-separated scene numbers to re-source")
    ap.add_argument("--voice", help="reference clip to use, e.g. english-narrator-1.mp3")
    ap.add_argument("--music")
    ap.add_argument("--music-level", type=float, default=0.20)
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--caption-size", type=int, default=58)
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--yes", action="store_true", help="'all' skips the approval pause")
    a = ap.parse_args()

    # Fold any old flat sheets/work/out into projects/<pid>/… before doing
    # anything. A no-op once migrated, so it's cheap to leave here for the CLI
    # too — otherwise a --sheet path from before the move would 404.
    try:
        rep = pl.migrate_layout()
        if rep["moved"]:
            print(f"  reorganised {rep['moved']} file(s) into projects/")
    except Exception:
        pass

    if a.step == "studio":
        # Same thing Start.bat / Start.command do, without leaving the terminal.
        import studio
        studio.main(open_browser=not a.no_browser)
        return

    if a.step == "sources":
        # I could not reach these APIs from where this was written, so nothing
        # about them is assumed to work. This calls each one for real and says
        # plainly what came back.
        from lib import sources as SRC
        cfg = pl.load_config()
        have = SRC.usable(cfg)
        banner("Picture sources")
        print(f"  usable right now: {', '.join(sorted(have)) or 'none'}\n")

        failed: list[str] = []
        # Built from the registry, not written out here. A hardcoded list means
        # adding a source and forgetting to test it — and an untested adapter
        # looks exactly like a working one until a real run needs it. Each
        # source declares its own probe subject, so registering it is enough.
        probes = [(name, src.probe, media)
                  for name, src in sorted(SRC.REGISTRY.items())
                  if src.probe and src.search
                  for media in src.media]
        for name, q, media in probes:
            src = SRC.REGISTRY.get(name)
            if not src or not src.can(media):
                continue
            if name not in have:
                need = src.needs_key or "no key needed"
                print(f"  -- {name:<12} {media:<6} not configured ({need})")
                continue
            try:
                hits = SRC.search(name, q, media, 3, cfg)
            except Exception as e:
                print(f"  !! {name:<12} {media:<6} {type(e).__name__}: {str(e)[:56]}")
                failed.append(name)
                continue
            if not hits:
                print(f"  !! {name:<12} {media:<6} reachable, but 0 results for {q!r}")
                continue
            print(f"  ok {name:<12} {media:<6} {len(hits)} hit(s) for {q!r}")
            print(f"       {hits[0].license}")
            print(f"       {hits[0].url[:82]}")

        if failed:
            banner("Why those failed")
            print("  A reset looks the same whatever caused it, so this tries the")
            print("  same host several ways. Read the cause off the results.\n")
            for name in sorted(set(failed)):
                print(f"  {name}:")
                for label, ok, detail in SRC.diagnose(name, cfg):
                    print(f"    {'ok' if ok else '!!'}  {label:<26}{detail}")
                print()
            print("  homepage fails too       -> your network cannot reach this host")
            print("  homepage ok, api resets  -> the API is refusing this client")
            print("  browser agent works      -> our User-Agent is the problem")
            print("  everything resets        -> TLS, proxy, or something in between")

        banner("Routing")
        print(f"  {len(SRC.TOPICS)} topics, {len(SRC._WORD2TOPIC)} words recognised.")
        print(f"  Scene tags are free text — anything sensible works.\n")
        for dom, q, media in (
                ("astrophysics", "spiral galaxy in deep space", "IMAGE"),
                ("ancient rome", "stone aqueduct arches", "IMAGE"),
                ("dinosaurs", "fossil skeleton in a museum", "IMAGE"),
                ("modern medicine", "surgeon in an operating room", "IMAGE"),
                ("daily life", "older woman making tea at home", "IMAGE"),
                ("sport", "runners crossing a finish line", "IMAGE"),
                # The two that show motion routing: modern goes to stock, the
                # archival one earns Internet Archive the third slot.
                ("modern life", "friends laughing in a cafe", "VIDEO"),
                ("wartime", "1930s newsreel of a city street", "VIDEO"),
                ("(anything unknown)", "wibble flurb", "IMAGE")):
            print(f"  {dom:<18}{media:<6} {SRC.explain(dom, media, have, q)}")
        print(f"\n  Routing is frozen by tools/test_routing.py — run that after")
        print(f"  changing the vocabulary. It is offline and free.")
        print(f"\n  Sheets carry a 'Domain:' line per scene; blank routes to stock.")
        print(f"  Only CC0 and public-domain material is accepted — nothing here")
        print(f"  needs crediting or restricts commercial use.")
        return

    if a.step == "voices":
        from lib import voices as V
        cur = V.pref_for(a.lang)
        banner(f"Reference clips for {pl.LANG_NAMES.get(a.lang, a.lang)}")
        if a.voice:                                   # set it and leave
            saved = V.save_pref(a.lang, reference=a.voice)
            print(f"  saved: {saved['reference']}")
            return
        refs = V.references()
        if not refs:
            print(f"  None yet. Put a clip in voices_refs/ — 30s of clean speech.")
            print(f"  It must be audio you hold rights to: your own voice, a CC0")
            print(f"  clip from Mozilla Common Voice, or public-domain LibriVox.")
            return
        for r in refs:
            mark = "  ← in use" if r["name"] == cur["reference"] else \
                   ("  under 8s, clones poorly" if r["short"] else "")
            print(f"  {r['name']:<34} {r['seconds']:>6}s{mark}")
        print(f"\n  Currently: {cur['reference'] or '(none set)'}")
        print(f"  Pick one:  python3 make_video.py voices --lang {a.lang} "
              f"--voice CLIP-NAME")
        return

    if a.step == "benchtts":
        from lib import chatterbox_engine as CB
        banner("Chatterbox benchmark")
        if not CB.installed():
            print("  " + CB.install_hint().replace("\n", "\n  "))
            return

        refs = CB.list_references()
        if not refs:
            print(f"  No reference clips found in {CB.REFS}/")
            print("  Put one there first — 30s of clean speech works best.")
            print("  It must be audio you hold rights to: your own voice, a CC0")
            print("  clip from Mozilla Common Voice, or public-domain LibriVox.")
            return
        ref = Path(a.voice) if a.voice else Path(refs[0]["path"])
        print(f"  reference: {ref.name}")
        if any(r["short"] and r["path"] == str(ref) for r in refs):
            print("  ⚠ under 8 seconds — clones noticeably better with 30s+")

        prepared = CB.prepare_reference(ref)
        print(f"  prepared:  {prepared.name}")

        # Benchmark on real lines from a real sheet, not toy sentences.
        lines = []
        projs = pl.find_projects()
        if projs:
            p0 = next((p for p in projs if p["scenes"] > 20), projs[0])
            tr = pl.translation_for(pl.sheets_dir(p0["id"]), p0["id"], a.lang)
            sc = pl.load_scenes(Path(p0["sheet"]), a.lang, tr)
            mid = [s.narration for s in sc if 12 <= len(s.narration.split()) <= 30]
            lines = mid[:5] or [s.narration for s in sc[:5]]
            print(f"  lines:     5 real ones from {p0['id']} ({a.lang})")
        if not lines:
            lines = ["This is a test of the narration engine, read at an "
                     "unhurried pace."] * 3

        try:
            r = CB.benchmark(lines, prepared, a.lang)
        except Exception as e:
            print(f"\n  Benchmark failed:")
            print("  " + str(e).replace("\n", "\n  "))
            inner = getattr(e, "tracebacks", None)
            if a.verbose and inner:
                for dev, _err, tb in inner:
                    print(f"\n  ── original traceback on {dev} "
                          f"{'─' * max(0, 46 - len(dev))}")
                    print("  " + tb.strip().replace("\n", "\n  "))
            elif a.verbose:
                import traceback
                traceback.print_exc()
            else:
                print("\n  For the traceback from inside the library, add --verbose")
            return

        # Extrapolate to the real job.
        n = 115
        one = r["per_line_s"] * n / 60
        three = one * 3
        banner("What that means")
        print(f"  device                {r['device']}")
        print(f"  model load (once)     {r['load_s']:.0f}s")
        if r.get("warmup_s"):
            print(f"  first line (once)     {r['warmup_s']:.0f}s  "
                  f"— GPU warm-up, not counted below")
        print(f"  per line (median)     {r['per_line_s']:.1f}s")
        print(f"  115-scene video       {one:.0f} min  (one language)")
        print(f"  all three languages   {three:.0f} min")
        print()
        if r.get("degrading"):
            py = "python" if os.name == "nt" else "python3"
            print(f"  ⚠ The later lines ran {r['drift']:.1f}x slower than the "
                  f"earlier ones.")
            print("    Line length does not explain it, so the machine is running "
                  "out of")
            print("    memory as it goes — the estimate above is optimistic. Close "
                  "other")
            print("    apps and re-run. If it still degrades, voice one language "
                  "at a time:")
            print(f"      {py} make_video.py voice --sheet sheets\\YOUR_SHEET.md "
                  f"--lang en")
            print()
        if three <= 25:
            print("  → Comfortably practical. Build all three languages in one go.")
        elif three <= 60:
            print("  → Usable. Fine to run while you do something else.")
        elif three <= 150:
            print("  → Slow. Do one language at a time, or leave it running")
            print("    overnight. Voicing is cached, so it never repeats work.")
        else:
            print("  → Very slow on this machine. Consider voicing one language")
            print("    per day rather than all three in a sitting.")
        print(f"\n  Voicing is cached — a re-render costs nothing.")
        return

    if a.step == "doctor":
        import shutil as _sh
        import subprocess as _sp
        from lib import render as R
        banner("Checking this machine")
        venv = (ROOT / ".venv" / "bin" / "python3")
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        pyv = f"{sys.version_info.major}.{sys.version_info.minor}"
        # The ML stack trails new Python releases; 3.14 in particular installs
        # cleanly and then fails at runtime with obscure errors.
        pyok = (3, 10) <= sys.version_info[:2] <= (3, 13)
        print(f"  {'ok ' if pyok else '!! '}python {pyv:<8}{sys.executable}")
        if not pyok:
            print(f"                 Python {pyv} is ahead of what the voice engine")
            print("                 supports. Install Python 3.12 and rebuild .venv.")
            print(f"                 then: rm -rf .venv && bash setup.sh")
        print(f"  {'ok ' if in_venv else '!! '}environment    "
              f"{'project .venv' if in_venv else 'system Python'}")
        if not in_venv and venv.exists():
            print(f"                 .venv exists but isn't being used — run:")
            print(f"                 .venv/bin/python3 make_video.py doctor")
        elif not in_venv:
            print(f"                 run 'bash setup.sh' to create one "
                  f"(fixes pip's PEP 668 error)")

        ff = _sh.which("ffmpeg")
        ff_hint = ("winget install Gyan.FFmpeg" if os.name == "nt"
                   else "brew install ffmpeg")
        print(f"  ffmpeg         {ff or f'NOT FOUND — run: {ff_hint}'}")
        if ff:
            v = _sp.run(["ffmpeg", "-version"], capture_output=True, text=True)
            print(f"                 {v.stdout.splitlines()[0][:70]}")
            cfg_line = next((l for l in v.stdout.splitlines()
                             if "configuration:" in l), "")
            for lib in ("libass", "libfreetype", "libharfbuzz", "libfribidi"):
                got = f"--enable-{lib}" in cfg_line
                print(f"  {'ok ' if got else '!! '}build {lib:<10} "
                      f"{'compiled in' if got else 'NOT compiled in'}")
            m = R.caption_method()
            note = {"ass": "libass present — full styled captions",
                    "subtitles": "libass present (srt path)",
                    "drawtext": "NO libass — plainer captions via drawtext",
                    "none": "NO subtitle filter at all — captions cannot be burned"}[m]
            flag = "ok " if m in ("ass", "subtitles") else "!! "
            print(f"  {flag}captions    {m}: {note}")
            if m != "ass":
                print(f"                 fix with:  {ff_hint}")
            for f in ("xfade", "zoompan", "amix", "concat"):
                print(f"  {'ok ' if R.has_filter(f) else '!! '}filter {f:<9} "
                      f"{'present' if R.has_filter(f) else 'MISSING'}")
        print(f"  ffprobe        {_sh.which('ffprobe') or 'NOT FOUND'}")
        from lib import chatterbox_engine as CB
        from lib import voices as V
        if CB.installed():
            di = CB.device_info()
            desc = di["name"] or di["device"]
            if di["vram_gb"]:
                desc += f" · {di['vram_gb']} GB VRAM"
            if di["note"]:
                desc += f" · {di['note']}"
            flag = "ok " if di["device"] in ("cuda", "mps") else "!! "
            print(f"  {flag}chatterbox  installed · {desc}")
        else:
            print("  !! chatterbox  missing — run: bash setup.sh")

        from lib import vision as VIS
        cap = VIS.capability(pl.load_config())
        if cap["ok"]:
            where = cap["device"] + (f" {cap['vram_gb']}GB" if cap["vram_gb"] else "")
            print(f"  ok visual match {cap['model'].split('/')[-1]} on {where} "
                  f"(scores pictures against the scene)")
        else:
            print(f"  -- visual match off ({cap['reason']}) "
                  f"— ranking by size and aspect only")

        # perth (Chatterbox's watermarker) imports pkg_resources, which
        # setuptools 81 removed. When it is gone the watermarker class is
        # silently None and Chatterbox dies with a useless TypeError.
        try:
            import pkg_resources  # noqa: F401
            has_pr = True
        except Exception:
            has_pr = False
        try:
            import perth
            perth_ok = getattr(perth, "PerthImplicitWatermarker", None) is not None
        except Exception:
            perth_ok = False
        if CB.installed():
            pr_note = ("available" if has_pr else
                       "MISSING - run: pip install \"setuptools<81\"")
            wm_note = ("ready" if perth_ok else
                       "not loadable - see pkg_resources above")
            print(f"  {'ok ' if has_pr else '!! '}pkg_resources {pr_note}")
            print(f"  {'ok ' if perth_ok else '!! '}watermarker   {wm_note}")

            # Prove we can actually WRITE a wav. torchaudio.save now routes
            # through TorchCodec, which nothing installs; we bypass it, and
            # this checks the bypass rather than assuming it.
            try:
                import tempfile
                import numpy as np
                probe = Path(tempfile.gettempdir()) / "faceless_write_probe.wav"
                CB._save_wav(np.zeros(2400, dtype="float32"), 24000, probe)
                writer = "soundfile"
                try:
                    import soundfile  # noqa: F401
                except Exception:
                    writer = "wave (stdlib fallback)"
                probe.unlink(missing_ok=True)
                print(f"  ok audio write via {writer}")
            except Exception as e:
                print(f"  !! audio write cannot save wav files: {e}")
        refs = V.references()
        print(f"  {'ok ' if refs else '!! '}references   "
              f"{len(refs)} clip(s) in voices_refs/")
        for lang in ("en", "de", "es"):
            st = V.status(lang)
            mark = "ok " if st["reference_ok"] else "-- "
            print(f"  {mark}voice {lang}     {st['reference'] or 'none chosen'}")
        cfg = pl.load_config()
        for k, label in (("pexels_key", "Pexels"), ("pixabay_key", "Pixabay"),
                         ("gemini_key", "Gemini")):
            print(f"  {'ok ' if cfg.get(k) else '-- '}{label:<13}"
                  f"{'set' if cfg.get(k) else 'not set'}")
        print(f"\n  projects: {len(pl.find_projects())} in projects/")
        return
    if a.step == "models":
        from lib import gemini as gem
        cfg = pl.load_config()
        if not cfg.get("gemini_key"):
            raise SystemExit("No gemini_key in config.json (or GEMINI_API_KEY).")
        ms = gem.list_models(cfg["gemini_key"])
        chosen = gem.resolve_model(cfg["gemini_key"], "")
        banner(f"{len(ms)} model(s) your key can call")
        for m in sorted(ms, key=lambda x: x["name"]):
            n = m["name"].split("/", 1)[-1]
            mark = "  ←  will be used" if n == chosen else ""
            print(f"  {n:<44}{mark}")
        print(f"\n  Auto-selected: {chosen}")
        print("  Pin a different one with \"gemini_model\" in config.json.")
        return
    if a.step == "generate":
        return step_generate(a)

    if a.step == "list":
        # Was advertised in the help and the docs but never implemented — it
        # fell through to the --sheet check below and then raised KeyError.
        projects = pl.find_projects()
        if not projects:
            print("\n  No projects in projects/.")
            print("  Make one with:  make_video.py generate --script my.txt "
                  "--id video06")
            return
        banner(f"{len(projects)} project(s) in projects/")
        for p in projects:
            langs = ", ".join(
                lg["code"] + ("" if lg["file"] or lg["code"] == "en" else "?")
                for lg in p["languages"])
            print(f"\n  {p['id']}  ·  {p['scenes']} scenes  ·  {langs}")
            print(f"    {p['label']}")
            print(f"    --sheet {Path(p['sheet']).relative_to(ROOT)}")
        print("\n  A '?' marks a language with no translation file yet.")
        return

    if not a.sheet:
        raise SystemExit("--sheet is required")

    {"stock": step_stock, "voice": step_voice,
     "render": step_render, "all": step_all}[a.step](a)


if __name__ == "__main__":
    main()
