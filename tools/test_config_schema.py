#!/usr/bin/env python3
"""Config schema: coercion, validation, and — most important — that a save only
touches known value keys and preserves every label/section/unknown key.

    python3 tools/test_config_schema.py
"""
from __future__ import annotations

import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import config_schema as CS   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<56}{'' if ok else repr(got)}")
        bad += not ok

    print("\n  schema shape:")
    secs = CS.schema()
    check("sections returned", len(secs) == len(CS.SECTIONS), True)
    check("every field has a type",
          all("type" in f for s in secs for f in s["fields"]), True)
    check("help is pulled from the example",
          any(f["help"] for s in secs for f in s["fields"]), True)

    print("\n  coercion:")
    F = {f["key"]: f for f in CS._FIELDS}
    check("bool from 'true'", CS._coerce(F["music_duck"], "true"), True)
    check("bool from false", CS._coerce(F["music_duck"], False), False)
    check("number clamps to range (caption_lead max .5)",
          CS._coerce(F["caption_lead"], 9), 0.5)
    check("int number rounds", CS._coerce(F["source_workers"], "3"), 3)
    check("select passes a valid option", CS._coerce(F["voice_engine"], "chirp"), "chirp")
    check("auto_or_int keeps auto", CS._coerce(F["max_concurrent_jobs"], "auto"), "auto")
    check("auto_or_int coerces a number", CS._coerce(F["max_concurrent_jobs"], "3"), 3)
    check("multiselect filters to known sources",
          CS._coerce(F["disable_sources"], ["openverse", "bogus"]), ["openverse"])
    check("dict from JSON string", CS._coerce(F["google_tts_locale"], '{"en":"en-GB"}'),
          {"en": "en-GB"})

    print("\n  allow_custom selects accept any model id (open set):")
    check("a brand-new model id is accepted",
          CS._coerce(F["generate_model"], "gemini-3.1-flash-lite-image"),
          "gemini-3.1-flash-lite-image")
    check("empty custom value is rejected",
          "generate_model" in CS.validate_and_merge({}, {"generate_model": "  "})[1], True)

    print("\n  invalid values are rejected, not silently written:")
    _, errs = CS.validate_and_merge({}, {"voice_engine": "nope"})
    check("bad select (closed) -> error", "voice_engine" in errs, True)
    _, errs = CS.validate_and_merge({}, {"google_tts_locale": "{not json}"})
    check("bad JSON dict -> error", "google_tts_locale" in errs, True)

    print("\n  merge only touches known keys; labels/sections/secrets preserved:")
    current = collections.OrderedDict([
        ("_section_1", "Keys"),
        ("_pexels_key", "doc"),
        ("pexels_key", "SECRET-STAYS"),
        ("voice_engine", "higgs"),
        ("_mystery", "some future label"),
        ("mystery_key", "keep me"),
    ])
    merged, errs = CS.validate_and_merge(current, {"voice_engine": "chirp",
                                                   "injected_evil": "x"})
    check("no errors", errs, {})
    check("known key updated", merged["voice_engine"], "chirp")
    check("untouched secret preserved", merged["pexels_key"], "SECRET-STAYS")
    check("label docs preserved", merged["_pexels_key"], "doc")
    check("section header preserved", merged["_section_1"], "Keys")
    check("unknown existing key preserved", merged["mystery_key"], "keep me")
    check("UI cannot inject an unknown key", "injected_evil" not in merged, True)
    check("key order preserved", list(merged.keys())[:3],
          ["_section_1", "_pexels_key", "pexels_key"])

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
