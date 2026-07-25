#!/usr/bin/env python3
"""Higgs engine wiring: text normalisation, a stable/separate cache key, the
transcript lookup for cloning, and the tts router's safe fallback.

    python3 tools/test_higgs.py

The heavy model is NEVER loaded here (boson_multimodal isn't installed in this
environment); these tests cover only the glue that must be correct BEFORE the
model runs on the user's GPU.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import higgs_engine as HG   # noqa: E402
from lib import tts, voices as V     # noqa: E402


class _S:
    def __init__(self, n, t):
        self.n, self.narration = n, t


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<54}{'' if ok else repr(got)}")
        bad += not ok

    print("\n  Higgs text normalisation (clean, terminated, no symbols):")
    check("markdown stripped + terminal punctuation added",
          HG._norm("#4 - Matterhorn steht "), "4 - Matterhorn steht.")
    check("degree units are spoken, not symbols",
          HG._norm("bei -40°C"), "bei -40 degrees Celsius.")
    check("already-terminated line is left as-is",
          HG._norm("Fertig!"), "Fertig!")

    print("\n  cache key: separate from Chatterbox, sensitive to the voice knob:")
    k = HG._key("hallo", "de/a.mp3", "de", "m", 0.7)
    check("temperature changes the key", HG._key("hallo", "de/a.mp3", "de", "m", 0.6) != k, True)
    check("model changes the key", HG._key("hallo", "de/a.mp3", "de", "m2", 0.7) != k, True)
    paths = HG.expected_paths([_S(1, "Eins."), _S(2, "Zwei")], "de", "de/a.mp3", cfg={"higgs_model": "m"})
    check("expected_paths use the hg_ prefix (never mixes with cb_)",
          all(p.name.startswith("hg_de_") for p in paths), True)

    print("\n  reference transcript lookup (needed for cloning):")
    tmp = pathlib.Path(tempfile.mkdtemp())
    clip = tmp / "awais.wav"
    clip.write_bytes(b"RIFF")
    (tmp / "awais.txt").write_text("Dies ist meine Stimme.", encoding="utf-8")
    _orig_resolve, _orig_pref = V.resolve, V.pref_for
    V.resolve = lambda name: clip
    V.pref_for = lambda lang: {"reference_text": ""}
    try:
        check("finds the sibling .txt next to the clip",
              HG._transcript_for("de", "awais.wav"), "Dies ist meine Stimme.")
        V.pref_for = lambda lang: {"reference_text": "aus voices.json"}
        check("voices.json reference_text wins when set",
              HG._transcript_for("de", "awais.wav"), "aus voices.json")
    finally:
        V.resolve, V.pref_for = _orig_resolve, _orig_pref

    print("\n  tts router falls back safely:")
    check("higgs selected but not installed -> not used (falls back to Chatterbox)",
          tts._use_higgs({"voice_engine": "higgs"}), False)
    check("default engine is not higgs", tts._use_higgs({}), False)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
