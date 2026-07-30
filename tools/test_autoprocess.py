#!/usr/bin/env python3
"""'Auto-process the rest' must resume at voice+render when the visuals were
already found — not re-run the whole source step.

    python3 tools/test_autoprocess.py

pl.visuals_complete(sheet) is the gate the hands-off sourcing path checks: it is
True only when every scene already has an asset on disk, so re-sourcing is
skipped and the chain continues straight to voice + render.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pipeline as pl        # noqa: E402

SHEET = """<!-- main-lang: en -->
# Demo
_3 scenes · language: en_

---

**S1 ⬜** · IMAGE ⚑ hook
- Narration: "One."
- ALT / search: `a`

**S2 ⬜** · IMAGE
- Narration: "Two."
- ALT / search: `b`

**S3 ⬜** · IMAGE
- Narration: "Three."
- ALT / search: `c`
"""


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<54}{'' if ok else repr(got)}")
        bad += not ok

    proj = Path(tempfile.mkdtemp()) / "DemoProj"
    (proj / "sheets").mkdir(parents=True)
    (proj / "work").mkdir(parents=True)
    sheet = proj / "sheets" / "DemoProj_main_script.md"
    sheet.write_text(SHEET, encoding="utf-8")
    assets = proj / "work" / "assets.json"

    check("the demo sheet parses to 3 scenes",
          len(pl.load_scenes(sheet, pl.main_lang(sheet), None)), 3)

    print("\n  no assets yet -> not complete (Auto-process must source):")
    check("no assets.json at all", pl.visuals_complete(sheet), False)

    print("\n  a partial source is still not complete (keep sourcing):")
    assets.write_text(json.dumps({"1": {"path": "x"}, "2": {"path": "y"}}), encoding="utf-8")
    check("2 of 3 scenes sourced", pl.visuals_complete(sheet), False)

    print("\n  every scene sourced -> complete (skip straight to voice+render):")
    assets.write_text(json.dumps({"1": {"path": "x"}, "2": {"path": "y"}, "3": {"path": "z"}}),
                      encoding="utf-8")
    check("3 of 3 scenes sourced", pl.visuals_complete(sheet), True)

    print("\n  a placeholder still counts as sourced (it renders):")
    assets.write_text(json.dumps({"1": {"placeholder": True}, "2": {"path": "y"}, "3": {"path": "z"}}),
                      encoding="utf-8")
    check("placeholder counts as a picture", pl.visuals_complete(sheet), True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
