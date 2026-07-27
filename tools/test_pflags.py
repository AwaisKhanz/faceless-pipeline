#!/usr/bin/env python3
"""Per-project flags: set/clear 'uploaded', prune stale, forget on delete, and
tolerate a missing or corrupt file.

    python3 tools/test_pflags.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pflags as PF   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<52}{'' if ok else repr(got)}")
        bad += not ok

    PF.FILE = pathlib.Path(tempfile.mkdtemp()) / "project_flags.json"

    print("\n  set / get / clear:")
    check("unset project -> not uploaded", PF.get("v1")["uploaded"], False)
    r = PF.set_uploaded("v1", True)
    check("mark uploaded", r["uploaded"], True)
    check("stamps a time", bool(r["uploaded_at"]), True)
    check("persists across reads", PF.get("v1")["uploaded"], True)
    PF.set_uploaded("v1", False)
    check("unmarking clears it", PF.get("v1")["uploaded"], False)

    print("\n  prune to real projects + forget on delete:")
    PF.set_uploaded("v1", True)
    PF.set_uploaded("ghost", True)
    d = PF.data(valid_pids=["v1"])            # 'ghost' is gone
    check("stale entry pruned", "ghost" not in d, True)
    check("real entry kept", d["v1"]["uploaded"], True)
    PF.forget("v1")
    check("forget drops the flag", PF.get("v1")["uploaded"], False)

    print("\n  tolerates a corrupt file:")
    PF.FILE.write_text("{ not json", encoding="utf-8")
    check("corrupt file -> not uploaded, no crash", PF.get("v1")["uploaded"], False)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
