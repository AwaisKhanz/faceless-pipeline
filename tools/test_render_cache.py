#!/usr/bin/env python3
"""Per-scene clip cache: a swapped picture must invalidate its cached clip.

The render reuses `c{n}.mp4` between runs so a caption-only retry is fast. The
bug this guards against: the reuse test ignored the SOURCE picture, so swapping
a scene's image in review (by search or by AI) left the render happily reusing
the clip built from the OLD image — "I changed it but the video still shows the
old one". `_clip_fingerprint` folds the source (path + size + mtime), the target
length and the zoom flag into one identity; render rebuilds whenever it changes.

    python3 tools/test_render_cache.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pipeline as pl   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<56}{'' if ok else repr(got)}")
        bad += not ok

    fp = pl._clip_fingerprint
    d = Path(tempfile.mkdtemp())
    old = d / "gen_5_1.png"
    old.write_bytes(b"OLD-IMAGE-BYTES")

    print("\n  same source, same timing -> identical fingerprint (clip reused):")
    a = fp(old, 3.20, True)
    check("stable across calls", fp(old, 3.20, True), a)

    print("\n  swap the picture -> fingerprint changes (clip rebuilt):")
    newpath = d / "gen_5_2.png"                      # AI regenerate: new filename
    newpath.write_bytes(b"NEW-IMAGE-BYTES")
    check("different path (AI take / search pick) differs", fp(newpath, 3.20, True) != a)

    print("\n  same path but new bytes -> fingerprint changes:")
    time.sleep(0.01)
    os.utime(old, (old.stat().st_atime, old.stat().st_mtime + 5))  # touch: new mtime
    check("same path, newer mtime differs", fp(old, 3.20, True) != a)
    same_name = d / "search_pick.jpg"
    same_name.write_bytes(b"AAAA")
    fp_small = fp(same_name, 3.2, True)
    same_name.write_bytes(b"AAAABBBBCCCC")            # different size, same path
    check("same path, different size differs", fp(same_name, 3.2, True) != fp_small)

    print("\n  timing / effect changes also invalidate:")
    base = fp(newpath, 3.20, True)
    check("different target length differs", fp(newpath, 4.00, True) != base)
    check("zoom on vs off differs", fp(newpath, 3.20, False) != base)

    print("\n  a vanished source is a distinct, non-crashing fingerprint:")
    gone = d / "not_there.png"
    check("missing file -> 'missing' marker, no exception", "missing" in fp(gone, 3.2, True))

    print("\n  generated-asset names never collide across projects:")
    name = pl._gen_asset_name
    a1 = name("ProjectAlpha", 13, "gen", "png")
    b1 = name("ProjectBeta", 13, "gen", "png")     # same scene+take, other project
    check("same scene, different project -> different file", a1 != b1)
    check("two takes in one project -> different files",
          name("ProjectAlpha", 13, "gen", "png") != a1)
    check("scene number kept in the name", "_13_" in a1)
    check("project key is stable for a given id",
          name("ProjectAlpha", 1, "gen", "png").split("_")[1]
          == a1.split("_")[1])
    check("veo prefix + extension honoured",
          name("P", 4, "veo", "mp4").startswith("veo_") and name("P", 4, "veo", "mp4").endswith(".mp4"))

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
