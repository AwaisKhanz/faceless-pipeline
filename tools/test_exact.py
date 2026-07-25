#!/usr/bin/env python3
"""The `exact` flag survives the whole chain: an AI-flagged scene is written to
the sheet, parsed back, and (when generation is on) drives generation.

    python3 tools/test_exact.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import compose as C   # noqa: E402
from lib import sheet as S     # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<54}{'' if ok else repr(got)}")
        bad += not ok

    print("\n  sheet round-trip: exact is written and parsed back:")
    scenes = [
        C.Scene(n=1, narration="Camu Camu, a tiny Amazon berry.", media="IMAGE",
                query="camu camu berries", topic="nature", exact=True),
        C.Scene(n=2, narration="A calm forest at dawn.", media="IMAGE",
                query="forest at dawn", topic="nature", exact=False),
    ]
    md = C.render_main_script({"title_en": "T"}, scenes, "vid", "en")
    check("exact scene writes an 'Exact:' line", "- Exact: yes" in md, True)
    check("non-exact scene writes no 'Exact:' line", md.count("- Exact:"), 1)

    tmp = pathlib.Path(tempfile.mkdtemp()) / "vid_main_script.md"
    tmp.write_text(md, encoding="utf-8")
    parsed = S.parse_main_script(tmp)
    by_n = {s.n: s for s in parsed}
    check("S1 parses back as exact", by_n[1].exact, True)
    check("S2 parses back as not exact", by_n[2].exact, False)

    print("\n  a split of an exact scene keeps exact on every beat:")
    parts = C._validate_split(
        {"narration": "Walnuts and flax and fish.", "query": "walnuts",
         "hero": False, "note": "", "exact": True},
        [{"narration": "Walnuts", "query": "walnuts on table"},
         {"narration": "and flax", "query": "flax seeds"},
         {"narration": "and fish.", "query": "salmon fillet"}])
    if parts is None:
        check("split produced parts", False)
    else:
        check("all split beats inherit exact", all(p.get("exact") for p in parts), True)

    print("\n  older sheets without the line still parse (exact defaults False):")
    old = md.replace("- Exact: yes\n", "")
    tmp.write_text(old, encoding="utf-8")
    check("no Exact line -> exact False",
          all(not s.exact for s in S.parse_main_script(tmp)), True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
