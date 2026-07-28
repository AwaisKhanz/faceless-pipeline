#!/usr/bin/env python3
"""Favourite voices can carry a friendly nickname.

Catalogue voice names are cryptic (de-DE-Chirp3-HD-Charon). A nickname is a
display-only label the user sets; the voice NAME stays the identity that TTS is
called with. Stored in voices.json under a reserved key, so it never masquerades
as a per-language preference.

    python3 tools/test_voice_nicknames.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import voices as vx   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<58}{'' if ok else repr(got)}")
        bad += not ok

    # Never touch the real voices.json.
    vx.PREFS = Path(tempfile.mkdtemp()) / "voices.json"
    n = "de-DE-Chirp3-HD-Charon"

    print("\n  default display is the raw voice name:")
    check("no nickname -> shows the name", vx.voice_display(n), n)
    check("no nicknames map yet", vx.voice_nicknames(), {})

    print("\n  set / read a nickname:")
    vx.toggle_favorite("google", n, True)
    check("set returns the effective display", vx.set_voice_nickname(n, "  Deep   narrator "),
          "Deep narrator")                              # whitespace collapsed
    check("display now the nickname", vx.voice_display(n), "Deep narrator")
    check("nicknames map has it", vx.voice_nicknames(), {n: "Deep narrator"})
    check("renaming does NOT unstar the voice", n in vx.favorites("google"))

    print("\n  the reserved key never leaks into per-language prefs:")
    check("__voice_names__ hidden from all_prefs", "__voice_names__" not in vx.all_prefs())

    print("\n  clear a nickname (revert to the raw name):")
    check("clearing returns the raw name", vx.set_voice_nickname(n, ""), n)
    check("display back to the name", vx.voice_display(n), n)
    check("map empty again", vx.voice_nicknames(), {})
    check("still a favourite after reset", n in vx.favorites("google"))

    print("\n  guards:")
    try:
        vx.set_voice_nickname("", "x")
        check("blank voice name rejected", False)
    except ValueError:
        check("blank voice name rejected", True)
    long = vx.set_voice_nickname(n, "x" * 200)
    check("nickname is length-capped", len(long) <= 80)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
