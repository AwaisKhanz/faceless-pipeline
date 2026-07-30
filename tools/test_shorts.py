#!/usr/bin/env python3
"""Shorts: a project can be a 16:9 video or a 9:16 short, and that ONE choice
drives image-generation aspect AND the render frame size (clips + captions).

    python3 tools/test_shorts.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pflags as pf        # noqa: E402
from lib import pipeline as pl      # noqa: E402
from lib import render as R         # noqa: E402
from lib import captions as C       # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<60}{'' if ok else repr(got)}")
        bad += not ok

    pf.FILE = Path(tempfile.mkdtemp()) / "project_flags.json"

    print("\n  per-project format store (default video; preserved across flags):")
    check("default format is video", pf.get("p1")["format"], "video")
    pf.set_format("p1", "short")
    check("set to short", pf.get("p1")["format"], "short")
    pf.set_uploaded("p1", True)
    check("marking uploaded keeps the format", pf.get("p1")["format"], "short")
    check("uploaded flag also set", pf.get("p1")["uploaded"], True)
    pf.set_format("p1", "video")
    check("back to video (default) clears it", pf.get("p1")["format"], "video")
    check("but the uploaded flag survives", pf.get("p1")["uploaded"], True)
    check("a bad format falls back to video", pf.set_format("p1", "portrait")["format"], "video")

    print("\n  orientation: format -> aspect + frame size:")
    sheet = Path("/x/projects/Demo/sheets/Demo_main_script.md")
    pf.set_format(pl.project_id(sheet), "short")
    o = pl.orientation(sheet)
    check("short -> 9:16", o["aspect"], "9:16")
    check("short -> 1080x1920", (o["w"], o["h"]), (1080, 1920))
    pf.set_format(pl.project_id(sheet), "video")
    o = pl.orientation(sheet)
    check("video -> 16:9 / 1920x1080", (o["aspect"], o["w"], o["h"]), ("16:9", 1920, 1080))

    print("\n  the project's aspect is forced onto generation cfg:")
    pf.set_format(pl.project_id(sheet), "short")
    cfg2 = pl._cfg_with_aspect(sheet, {"generate_aspect": "16:9", "vertex_project": "p"})
    check("a Short overrides the global 16:9 with 9:16", cfg2["generate_aspect"], "9:16")
    check("other cfg keys are preserved", cfg2["vertex_project"], "p")

    print("\n  clip fingerprint folds in the frame size (switching format rebuilds):")
    img = Path(tempfile.mkdtemp()) / "s.png"
    img.write_bytes(b"IMG")
    check("16:9 and 9:16 fingerprints differ",
          pl._clip_fingerprint(img, 3.0, True, (1920, 1080))
          != pl._clip_fingerprint(img, 3.0, True, (1080, 1920)), True)

    print("\n  render builds clips at the CURRENT frame size (real ffmpeg args):")
    cmds = []
    R.run = lambda cmd, **k: cmds.append(cmd)          # capture, don't run ffmpeg
    R.set_frame(1080, 1920)
    R.make_image_clip(Path("a.png"), 3.0, Path("o.mp4"), zoom=False)
    vf = " ".join(cmds[-1])
    check("still clip scaled to 1080x1920", "scale=1080:1920" in vf and "crop=1080:1920" in vf)
    R.make_video_clip(Path("a.mp4"), 3.0, Path("o.mp4"))
    check("video clip scaled to 1080x1920", "1080:1920" in " ".join(cmds[-1]))
    R.set_frame(1920, 1080)
    R.make_image_clip(Path("a.png"), 3.0, Path("o.mp4"), zoom=True)
    check("back to 16:9: zoompan targets 1920x1080", "s=1920x1080" in " ".join(cmds[-1]))

    print("\n  captions are positioned in the Short's frame:")
    C.set_frame(1080, 1920)
    hdr = C.ass_header(C.resolve_style(None))
    check("ASS PlayRes matches the Short frame",
          "PlayResX: 1080" in hdr and "PlayResY: 1920" in hdr)
    C.set_frame(1920, 1080)
    check("ASS PlayRes matches a 16:9 frame",
          "PlayResX: 1920" in C.ass_header(C.resolve_style(None)))

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
