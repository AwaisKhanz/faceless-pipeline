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

    print("\n  captions never bleed off a narrow 9:16 Short (fit-to-width):")
    import re as _re
    long_words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.38}
                  for i, w in enumerate("Der Arbeitspreis ist der Betrag".split())]
    st_short = C.PRESETS["reference"].merged(size=79, max_words=5)  # the pre-fix overflow case

    def widest_center_extent(ass_text, frame_w):
        """Half-width the widest phrase would need, from the estimated glyph run
        of each Layer-1 event (text is centred with \\an5\\pos, so it spreads
        symmetrically about the centre)."""
        worst = 0
        for ln in ass_text.splitlines():
            if not ln.startswith("Dialogue: 1"):
                continue
            m = _re.search(r"\\fs(\d+)", ln)
            size = int(m.group(1)) if m else st_short.size
            body = ln[ln.index("{"):]                     # from the first override
            text = _re.sub(r"\{[^}]*\}", "", body).strip()  # drop all ASS tags
            worst = max(worst, C._glyph_w(text, size, st_short.bold))
        return worst

    C.set_frame(1080, 1920)
    ass_s = C.build_ass(C.chunk_words(long_words, st_short), st_short)
    half = widest_center_extent(ass_s, 1080) / 2
    check("the long German line fits inside the 1080 frame",
          (1080 // 2 - half) >= 0 and (1080 // 2 + half) <= 1080, True)
    check("an overflowing phrase carries a shrink \\fs tag",
          any(l.startswith("Dialogue: 1") and "\\fs" in l for l in ass_s.splitlines()), True)

    C.set_frame(1920, 1080)
    ass_v = C.build_ass(C.chunk_words(long_words, st_short), st_short)
    check("the same line on a roomy 16:9 frame is left untouched (no \\fs)",
          any(l.startswith("Dialogue: 1") and "\\fs" in l for l in ass_v.splitlines()), False)
    C.set_frame(1920, 1080)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
