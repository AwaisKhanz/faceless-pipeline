#!/usr/bin/env python3
"""Freeze the scene-granularity behaviour: long, list-like sentences get split
into one beat per visual, and the retry loop refuses under-split output.

    python3 tools/test_split.py

Runs offline — Gemini is faked. It checks:
  - _under_split flags real compound sentences (the user's history script) and
    leaves genuinely single-picture sentences alone;
  - split_into_scenes retries when the model returns one long scene, and keeps
    the finer version;
  - when the model splits finely on the first try, it does NOT retry (no wasted
    LLM calls);
  - the word-for-word guarantee still holds after refining.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import compose, gemini as G  # noqa: E402

bad = 0


def check(label, got, want=True):
    global bad
    ok = got == want
    bad += not ok
    print(f"  {'ok' if ok else '!!'}  {label:<54}{got}"
          f"{'' if ok else f'  (wanted {want})'}")


# The six scenes the user showed — every one is a compound sentence that should
# become several beats. All must be flagged as under-split.
HISTORY = [
    "Die Geschichte der Menschheit ist eine Reise voller großer Veränderungen, "
    "erstaunlicher Entdeckungen und entscheidender Wendepunkte.",
    "Von den ersten Hochkulturen in Ägypten und Mesopotamien bis hin zu "
    "mächtigen Reichen haben Menschen die Welt immer wieder neu geprägt.",
    "Kriege, Erfindungen und wissenschaftliche Fortschritte haben das Leben von "
    "Generation zu Generation verändert.",
    "Mit der industriellen Revolution kamen Maschinen und Fabriken auf - und die "
    "Gesellschaft wandelte sich grundlegend.",
    "Das 20. Jahrhundert wurde von Weltkriegen, politischen Umbrüchen und "
    "bahnbrechenden Technologien geprägt.",
    "Wenn wir heute in die Vergangenheit blicken, können wir besser verstehen, "
    "woher wir kommen und was wir aus der Geschichte lernen können.",
]

# Genuinely single-picture sentences (or tiny beats) — must NOT be flagged.
SINGLE = [
    "It was people.",
    "The old fisherman rowed his small wooden boat across the calm harbour at dawn",
    "apples, bananas and oranges",          # one bowl of fruit, short list
    "senior man studying an old manuscript in a warm quiet library",
]


def main() -> int:
    print("\n  _under_split flags every compound sentence in the history script:")
    for i, s in enumerate(HISTORY, 1):
        check(f"S{i} flagged as under-split", compose._under_split(s))

    print("\n  _under_split leaves single-picture sentences alone:")
    for s in SINGLE:
        check(f'not flagged: "{s[:34]}…"', compose._under_split(s), False)

    # ---- retry loop refines an under-split section -------------------------
    print("\n  split_into_scenes retries a coarse section and keeps the fine cut:")
    SEC = ("The wars raged, inventions appeared and science advanced across the "
           "whole world")
    fine = [
        {"narration": "The wars raged,", "media": "IMAGE", "query": "a battlefield"},
        {"narration": "inventions appeared and", "media": "IMAGE",
         "query": "an inventor workshop"},
        {"narration": "science advanced across the whole world", "media": "VIDEO",
         "query": "a science laboratory"},
    ]
    calls = {"n": 0}

    def fake_coarse_then_fine(sec, plan, key, model, fb=""):
        calls["n"] += 1
        if calls["n"] == 1:                       # first try: one long scene
            return [{"narration": SEC, "media": "IMAGE", "query": "history montage"}]
        return fine                               # after feedback: split finely

    G.scenes_for_section = fake_coarse_then_fine
    res = compose.Result()
    scenes = compose.split_into_scenes(SEC, {}, "k", "m", res,
                                       lambda *_: None, lambda *_: None)
    check("retried at least once", calls["n"] >= 2)
    check("kept the finer, 3-scene cut", len(scenes), 3)
    check("scene numbers are 1..3", [s.n for s in scenes], [1, 2, 3])
    check("words still reproduce the section exactly",
          G.words(" ".join(s.narration for s in scenes)) == G.words(SEC))
    check("no under-split scene survived",
          any(compose._under_split(s.narration) for s in scenes), False)

    # ---- a first-try fine split does NOT retry (no wasted calls) -----------
    print("\n  a clean first split costs exactly one call:")
    calls["n"] = 0

    def fake_fine_first(sec, plan, key, model, fb=""):
        calls["n"] += 1
        return fine

    G.scenes_for_section = fake_fine_first
    res2 = compose.Result()
    scenes2 = compose.split_into_scenes(SEC, {}, "k", "m", res2,
                                        lambda *_: None, lambda *_: None)
    check("exactly one LLM call", calls["n"], 1)
    check("three scenes", len(scenes2), 3)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
