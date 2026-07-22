#!/usr/bin/env python3
"""Freeze how sourcing ranks candidates, so "pick the best" stays picked.

    python3 tools/test_scoring.py

The scorer reads only the dimensions a search API already returned, so this runs
offline and costs nothing. It checks the two things that matter: a 16:9 1080p+
frame wins, and a source that reports no dimensions is never demoted below one
that does purely for being un-measured (that would quietly undo the routing).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import stock as S  # noqa: E402


def cand(w, h, tag):
    return {"width": w, "height": h, "src": tag, "page": tag}


def main() -> int:
    bad = 0

    def check(label, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<52}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    print("\n  a single score is sane:")
    s_1080 = S._score(cand(1920, 1080, "hd"))
    s_4k = S._score(cand(3840, 2160, "4k"))
    s_720 = S._score(cand(1280, 720, "sd"))
    s_43 = S._score(cand(1200, 900, "43"))
    s_port = S._score(cand(1080, 1920, "portrait"))
    s_none = S._score(cand(0, 0, "unknown"))
    check("4K 16:9 beats 1080p 16:9", s_4k > s_1080, True)
    check("1080p beats 720p at the same aspect", s_1080 > s_720, True)
    check("16:9 beats 4:3", s_1080 > s_43, True)
    check("portrait scores negative (unusable on 16:9)", s_port < 0, True)
    check("unknown dimensions are neutral (0.0)", s_none, 0.0)
    check("portrait ranks below unknown", s_port < s_none, True)

    print("\n  ranking a mixed pool (best first):")
    pool = [cand(1080, 1920, "portrait"), cand(1280, 720, "sd"),
            cand(3840, 2160, "4k"), cand(1200, 900, "43"),
            cand(1920, 1080, "hd")]
    order = [c["src"] for c in sorted(pool, key=S._score, reverse=True)]
    check("best 16:9 high-res first, portrait last",
          order, ["4k", "hd", "sd", "43", "portrait"])

    print("\n  routing is not disturbed by scoring:")
    # An all-archive pool reports no dimensions — every score ties at 0, so a
    # STABLE sort must return it untouched, or archives would shuffle for no
    # reason and the routed order would be lost.
    archive = [cand(0, 0, "nasa-1"), cand(0, 0, "nasa-2"), cand(0, 0, "nasa-3")]
    order = [c["src"] for c in sorted(archive, key=S._score, reverse=True)]
    check("all-unknown pool keeps its routed order",
          order, ["nasa-1", "nasa-2", "nasa-3"])

    # Two 16:9 images of different resolution: the bigger one wins, and that is
    # the only reason the order changes.
    same_ar = [cand(1280, 720, "small"), cand(1920, 1080, "big")]
    order = [c["src"] for c in sorted(same_ar, key=S._score, reverse=True)]
    check("equal aspect ranks by resolution", order, ["big", "small"])

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
