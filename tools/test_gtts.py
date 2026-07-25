#!/usr/bin/env python3
"""Google Chirp 3 HD engine wiring: text normalisation, a separate cache key,
the voice catalogue parse, the synth request/caching, and the tts router.

    python3 tools/test_gtts.py

No real network is touched: urllib is stubbed so these tests cover only the glue
that must be correct BEFORE the cloud call runs on the user's project.
"""
from __future__ import annotations

import base64
import io
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import gtts_engine as GT   # noqa: E402
from lib import tts                 # noqa: E402


class _S:
    def __init__(self, n, t):
        self.n, self.narration = n, t


class _Resp(io.BytesIO):
    """A minimal stand-in for urlopen()'s context-manager response."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<58}{'' if ok else repr(got)}")
        bad += not ok

    # Never mint a real token / hit the network, and don't depend on google-auth
    # being installed here: readiness = a project is set (the real check is at the
    # call site). usable() still layers the sticky-failure latch on top.
    GT.llm._vertex_token = lambda sa: "fake-token"
    GT.available = lambda c=None: bool((c or {}).get("vertex_project"))
    cfg = {"vertex_project": "proj-x"}

    print("\n  normalisation + locale mapping:")
    check("markdown stripped + terminal punctuation",
          GT._norm("## Hello **world** "), "Hello world.")
    check("locale default for German", GT.locale_for("de"), "de-DE")
    check("locale default for Mandarin is cmn-CN", GT.locale_for("zh"), "cmn-CN")
    check("locale override wins",
          GT.locale_for("en", {"google_tts_locale": {"en": "en-GB"}}), "en-GB")

    print("\n  cache key: separate prefix, sensitive to voice + rate:")
    k = GT._key("hi", "en-US-Chirp3-HD-Kore", "en", 1.0)
    check("different voice changes the key",
          GT._key("hi", "en-US-Chirp3-HD-Charon", "en", 1.0) != k, True)
    check("different rate changes the key",
          GT._key("hi", "en-US-Chirp3-HD-Kore", "en", 0.9) != k, True)
    paths = GT.expected_paths([_S(1, "One."), _S(2, "Two")], "en",
                              "en-US-Chirp3-HD-Kore", cache=pathlib.Path("/tmp"))
    check("expected_paths use the gc_ prefix (never mixes with cb_/hg_)",
          all(p.name.startswith("gc_en_") for p in paths), True)

    print("\n  voice catalogue parse (filters to Chirp3-HD):")
    catalogue = {"voices": [
        {"name": "en-US-Chirp3-HD-Kore", "languageCodes": ["en-US"], "ssmlGender": "FEMALE"},
        {"name": "en-US-Chirp3-HD-Charon", "languageCodes": ["en-US"], "ssmlGender": "MALE"},
        {"name": "en-US-Standard-A", "languageCodes": ["en-US"], "ssmlGender": "FEMALE"},
    ]}
    GT._VOICE_CACHE.clear()
    GT.urllib.request.urlopen = lambda req, timeout=0: _Resp(json.dumps(catalogue).encode())
    got = GT.voices("en", cfg)
    check("only Chirp3-HD voices are returned", [v["name"] for v in got],
          ["en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Kore"])
    check("gender is carried through", got[0]["gender"], "Male")
    check("default_voice is the first catalogue voice",
          GT.default_voice("en", cfg), "en-US-Chirp3-HD-Charon")

    print("\n  empty catalogue records WHY (API-not-enabled surfaces to the UI):")
    GT._VOICE_CACHE.clear()
    import urllib.error
    err_body = json.dumps({"error": {"message":
        "Cloud Text-to-Speech API has not been used in project 123 before or it is disabled."}}).encode()

    def raise_403(req, timeout=0):
        raise urllib.error.HTTPError("http://x", 403, "Forbidden", {}, io.BytesIO(err_body))

    GT.urllib.request.urlopen = raise_403
    check("403 -> empty list", GT.voices("de", cfg), [])
    check("reason captured for the UI",
          "has not been used" in GT.last_voice_error(), True)

    print("\n  synth: writes gc_ wavs, caches, skips on re-run:")
    tmp = pathlib.Path(tempfile.mkdtemp())
    calls = {"n": 0}
    wav = base64.b64encode(b"RIFF" + b"\x00" * 4000).decode()

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        return _Resp(json.dumps({"audioContent": wav}).encode())

    GT.urllib.request.urlopen = fake_urlopen
    scenes = [_S(1, "One."), _S(2, "Two.")]
    out = GT.synth(scenes, "en", None, tmp, {"speaking_rate": 1.0}, log=lambda *_: None,
                   cfg=cfg, reference="en-US-Chirp3-HD-Kore")
    check("one file per scene", len(out), 2)
    check("both files written and non-trivial",
          all(p.exists() and p.stat().st_size > 1024 for p in out), True)
    check("made exactly 2 network calls", calls["n"], 2)
    GT.synth(scenes, "en", None, tmp, {"speaking_rate": 1.0}, log=lambda *_: None,
             cfg=cfg, reference="en-US-Chirp3-HD-Kore")
    check("re-run adds no calls (served from cache)", calls["n"], 2)
    check("synth paths match expected_paths",
          [p.name for p in out],
          [p.name for p in GT.expected_paths(scenes, "en", "en-US-Chirp3-HD-Kore",
                                             cache=tmp, opts={"speaking_rate": 1.0})])

    print("\n  tts router:")
    check("chirp selected + creds -> routes to Google",
          tts._use_gtts({"voice_engine": "chirp", "vertex_project": "p"}), True)
    check("chirp with no project -> not usable (falls back)",
          tts._use_gtts({"voice_engine": "chirp"}), False)
    check("default engine is not chirp", tts._use_gtts({}), False)
    GT.mark_unusable("boom")
    check("after a failure, usable() latches off",
          tts._use_gtts({"voice_engine": "chirp", "vertex_project": "p"}), False)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
