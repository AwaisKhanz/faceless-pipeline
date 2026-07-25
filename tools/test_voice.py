#!/usr/bin/env python3
"""Chatterbox artifact guard: retry-on-glitch, best-of, and a stable cache key.

    python3 tools/test_voice.py

Runs offline — torch and the 3 GB model are stubbed, so this only exercises the
CONTROL logic (which take we keep, and that control knobs never change the cache
key), never real audio generation.
"""
from __future__ import annotations

import logging
import pathlib
import sys
import tempfile
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# A minimal fake torch so the engine's no_grad / seeding runs without the real stack.
_ft = types.ModuleType("torch")


class _NG:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_ft.no_grad = lambda: _NG()
_ft.manual_seed = lambda s: None
_ft.cuda = types.SimpleNamespace(is_available=lambda: False,
                                 manual_seed_all=lambda s: None, empty_cache=lambda: None)
sys.modules.setdefault("torch", _ft)

from lib import chatterbox_engine as CB   # noqa: E402

SR = 24000
ALOG = "chatterbox.models.t3.inference.alignment_stream_analyzer"


def _good(sec=0.66, amp=0.1):
    return (np.random.RandomState(0).randn(int(SR * sec)) * amp).astype("float32")


def _silent(sec=0.66):
    return np.zeros(int(SR * sec), dtype="float32")


def _tiny():
    return (np.random.RandomState(1).randn(500) * 0.1).astype("float32")


class _Model:
    sr = SR

    def __init__(self, takes, warn_on=()):
        self.takes, self.i, self.warn_on = list(takes), -1, set(warn_on)

    def generate(self, text, **kw):
        self.i += 1
        if self.i in self.warn_on:
            logging.getLogger(ALOG).warning("Detected 2x repetition of token 6324")
        return self.takes[self.i]


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<52}{'' if ok else repr(got)}")
        bad += not ok

    saved = {}
    CB.installed = lambda: True
    CB.lang_id = lambda lang: lang
    CB._free_device_memory = lambda: None
    CB._save_wav = lambda samples, sr, out: saved.__setitem__("s", samples) or out

    def run(model, retries=2, best_of=1):
        CB.load_model = lambda **k: model
        CB.synth_one("hallo welt", pathlib.Path("ref.wav"), "de",
                     pathlib.Path(tempfile.mktemp(suffix=".wav")),
                     {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.7,
                      "retries": retries, "best_of": best_of})
        return saved["s"]

    print("\n  Chatterbox artifact guard:")
    s = run(_Model([_silent(), _good(), _good()]))
    check("a silent take is retried and a good one kept", float(np.abs(s).mean()) > 0.01)
    s = run(_Model([_tiny(), _good()]))
    check("a cut-off (too-short) take is retried", len(s) > 5000)
    s = run(_Model([_good(), _good()], warn_on={0}))
    check("a repetition warning forces a retry, no crash", s is not None)
    s = run(_Model([_silent(), _silent(), _silent()]))
    check("all-bad still returns the best effort (no crash)", s is not None)
    s = run(_Model([_good(amp=0.02), _good(amp=0.2)]), retries=0, best_of=2)
    check("best_of=2 generates two and picks one", s is not None)

    print("\n  the cache key ignores control knobs (no cache invalidation):")
    k_plain = CB._key("hallo", "ref", "de", CB._voice_opts(
        {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.7}))
    k_ctrl = CB._key("hallo", "ref", "de", CB._voice_opts(
        {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.7,
         "retries": 9, "best_of": 5, "seed": 123}))
    check("retries/best_of/seed do not change the key", k_plain, k_ctrl)
    k_temp = CB._key("hallo", "ref", "de", CB._voice_opts(
        {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.6}))
    check("temperature (a real voice knob) DOES change the key", k_temp != k_plain, True)

    print("\n  speech_text strips speaking artifacts, keeps real words/numbers:")
    S = CB.speech_text
    check("countdown marker '#5 - ' removed", S("#5 - Camu Camu"), "Camu Camu")
    check("plain '5. ' list number removed", S("5. Jabuticaba"), "Jabuticaba")
    check("trailing 'Hook:' label removed",
          S("5 Superfruits You've Never Tried Hook:"), "5 Superfruits You've Never Tried")
    check("leading 'Hook:' label removed", S("Hook: Forget apples."), "Forget apples.")
    check("dangling trailing dash removed",
          S("packed with health benefits-"), "packed with health benefits")
    check("slash becomes a space", S("Pitanga/Surinam Cherry"), "Pitanga Surinam Cherry")
    check("a real leading number is KEPT", S("5 Superfruits"), "5 Superfruits")
    check("an ordinary hyphenated word is KEPT", S("cold-pressed juice"), "cold-pressed juice")
    check("markdown/emoji still stripped", S("## **Big** 🍎 news"), "Big news")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
