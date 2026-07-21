#!/usr/bin/env python3
"""Freeze the routing behaviour, so extending the vocabulary cannot break it.

    python3 tools/test_routing.py

The topic vocabulary is meant to grow — that is the whole point of the design —
and every addition risks pulling an existing subject somewhere silly. A word
added to `culture` could quietly send modern hospitals to a museum. These cases
are the contract: they run offline, cost nothing, and fail loudly.

Cases deliberately include subjects that were never in the original eight
domains, because "does it handle things nobody listed" is the only question
that matters for a design claiming to be open-ended.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import sources as S  # noqa: E402

ALL = {"nasa", "smithsonian", "pexels", "pixabay"}
STOCK = ("pexels", "pixabay")

# (domain tag, query, what must come first, why it matters)
CASES = [
    # ── the archives genuinely win ─────────────────────────────────────────
    ("astrophysics", "spiral galaxy against black space", "nasa",
     "space reaches NASA even without the word 'space'"),
    ("", "apollo lunar module on the moon surface", "nasa",
     "the query alone is enough when there is no tag"),
    ("history", "roman stone aqueduct arches", "smithsonian",
     "historical subjects go to the museum"),
    ("dinosaurs", "fossil skeleton in a museum hall", "smithsonian",
     "fossils are museum material, NOT NASA (this was a real bug)"),
    ("geology", "layered rock strata in a canyon wall", "smithsonian",
     "geology is museum material"),
    ("victorian", "surgical instruments in a wooden case", "smithsonian",
     "historical medicine is a museum subject"),
    ("mythology", "greek temple ruins at dawn", "smithsonian", "antiquity"),
    ("pottery", "ancient clay vessel on a plinth", "smithsonian", "artefacts"),

    # ── stock is genuinely right ───────────────────────────────────────────
    ("people", "older woman making tea in a bright kitchen", STOCK,
     "no free archive holds modern domestic life"),
    ("modern medicine", "surgeon operating under bright lights", STOCK,
     "a modern operating room is not a museum piece (this was a real bug)"),
    ("sport", "marathon runners crossing a finish line", STOCK, "modern life"),
    ("food", "bread dough kneaded on a wooden table", STOCK, "modern life"),
    ("banking", "trading floor with screens of numbers", STOCK, "modern life"),
    ("weather", "lightning over a dark plain", STOCK,
     "weather is stock, not NASA satellite imagery"),
    ("mining", "deep mine shaft with headlamps", STOCK, "industry"),
    ("office", "people in a meeting room", STOCK, "modern life"),

    # ── nothing recognised still has to work ───────────────────────────────
    ("", "", STOCK, "no signal at all falls back to stock"),
    ("zzzqqq", "wibble flurb", STOCK, "unknown words never dead-end"),
]


def first_of(route: list, want) -> bool:
    if not route:
        return False
    return route[0] in want if isinstance(want, tuple) else route[0] == want


def main() -> int:
    print(f"\n  vocabulary: {len(S.TOPICS)} topics, {len(S._WORD2TOPIC)} words\n")
    bad = 0

    for domain, query, want, why in CASES:
        route = S.route(domain, "IMAGE", ALL, query)
        ok = first_of(route, want)
        bad += not ok
        label = domain or "(no tag)"
        print(f"  {'ok' if ok else '!!'}  {label:<16}{str(route):<42}{why}")
        if not ok:
            print(f"      wanted {want} first, topics found: "
                  f"{sorted(S.topics_in(domain, query))}")

    print("\n  invariants across every case:")
    checks = [
        ("VIDEO always starts with stock",
         all(first_of(S.route(d, "VIDEO", ALL, q), STOCK) for d, q, _, _ in CASES)),
        ("Smithsonian never asked for VIDEO",
         all("smithsonian" not in S.route(d, "VIDEO", ALL, q) for d, q, _, _ in CASES)),
        (f"never more than {S.MAX_SOURCES} sources",
         all(len(S.route(d, m, ALL, q)) <= S.MAX_SOURCES
             for d, q, _, _ in CASES for m in ("IMAGE", "VIDEO"))),
        ("always returns at least one source",
         all(S.route(d, m, ALL, q) for d, q, _, _ in CASES for m in ("IMAGE", "VIDEO"))),
        ("degrades cleanly when nothing is configured",
         S.route("space", "IMAGE", set(), "galaxy") == []),
        ("works with stock keys only",
         all(S.route(d, "IMAGE", {"pexels", "pixabay"}, q) for d, q, _, _ in CASES)),
    ]
    for label, passed in checks:
        bad += not passed
        print(f"    {'ok' if passed else '!!'}  {label}")

    # ── the module's own surface ───────────────────────────────────────────
    # A section replacement once deleted usable() and search() while every
    # routing case still passed, because none of them call those. stock.py
    # calls both, so sourcing would have crashed at runtime. Check they exist.
    print("\n  module surface (stock.py depends on all of these):")
    for name in ("route", "explain", "usable", "search", "topics_in",
                 "REGISTRY", "TOPICS", "MAX_SOURCES", "MIN_WIDTH"):
        present = hasattr(S, name)
        bad += not present
        print(f"    {'ok' if present else '!!'}  sources.{name}")

    print("\n  every registered source is well-formed:")
    for name, src in sorted(S.REGISTRY.items()):
        problems = []
        if src.media not in (("IMAGE",), ("IMAGE", "VIDEO"), ("VIDEO",)):
            problems.append(f"odd media {src.media}")
        if src.search is None and name not in ("pexels", "pixabay"):
            problems.append("no search function")
        if not src.covers and not src.generalist:
            problems.append("covers nothing and is not a generalist")
        if not src.note:
            problems.append("undocumented")
        bad += bool(problems)
        detail = "; ".join(problems) or f"{len(src.covers)} topics, {'/'.join(src.media)}"
        print(f"    {'ok' if not problems else '!!'}  {name:<14}{detail}")

    print("\n  no-key install still finds pictures:")
    bare = S.usable({})
    bad += not bare
    print(f"    {'ok' if bare else '!!'}  {sorted(bare)}")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
