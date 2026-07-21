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
    print(f"\n  vocabulary: {len(S.TOPICS)} topics, {len(S._WORD2TOPIC)} words")
    multi = {w: t for w, t in S._WORD2TOPIC.items() if len(t) > 1}
    print(f"  {len(multi)} word(s) mean more than one thing, and now count "
          f"for all of them\n")
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

    # ── the full archive set ───────────────────────────────────────────────
    # The cases above run against four sources, which is what a bare install
    # has. With every archive configured, five of them share the `history`
    # topic and tie on subject — so these check the tie-break puts the right
    # one first, and that adding archives never displaces stock for modern
    # subjects. Before source.reliability existed the winner of those ties was
    # whichever name sorted first, which is not a reason.
    print("\n  with every archive configured:")
    S.reset_failures()
    WIDE = {"nasa", "smithsonian", "openverse", "loc", "wikimedia",
            "europeana", "ia", "pexels", "pixabay"}
    # (domain, query, media, what must come first, why)
    WIDE_CASES = [
        ("history", "roman stone aqueduct arches", "IMAGE", "smithsonian",
         "the museum still leads on antiquity, not whoever sorts first"),
        ("space", "spiral galaxy in deep space", "IMAGE", "nasa",
         "a source that holds the subject beats one that is merely reliable"),
        ("wartime", "farm family outside a wooden shack", "IMAGE", "loc",
         "history + people is exactly what LoC documentary photography is"),
        ("medieval europe", "illuminated manuscript page", "IMAGE", "europeana",
         "the European institutions lead where the American ones thin out"),
        ("people", "older woman making tea in a bright kitchen", "IMAGE", STOCK,
         "LoC covering `people` must NOT pull modern domestic life away"),
        ("sport", "marathon runners crossing a finish line", "IMAGE", STOCK,
         "six archives configured still cannot beat stock at modern life"),
        # Archival VIDEO is the one place IA belongs: stock leads because it is
        # motion, but IA earns the third slot for footage stock cannot hold.
        ("wartime", "1930s newsreel of a city street", "VIDEO", STOCK,
         "archival video still opens with stock — nothing free beats it there"),
    ]
    for domain, query, media, want, why in WIDE_CASES:
        route = S.route(domain, media, WIDE, query)
        ok = first_of(route, want)
        bad += not ok
        print(f"    {'ok' if ok else '!!'}  {domain+'/'+media:<22}"
              f"{str(route):<46}{why}")
        if not ok:
            print(f"        wanted {want} first, topics: "
                  f"{sorted(S.topics_in(domain, query))}")

    # IA belongs on archival video and nowhere else. These two pin both edges.
    ia_archival = "ia" in S.route("wartime", "VIDEO", WIDE,
                                  "1930s newsreel of a city street")
    ia_not_modern = "ia" not in S.route("people", "VIDEO", WIDE,
                                        "friends laughing in a modern cafe")
    for label, passed in (
        ("IA is offered for archival video", ia_archival),
        ("IA is NOT offered for modern video (it holds nothing there)",
         ia_not_modern)):
        bad += not passed
        print(f"    {'ok' if passed else '!!'}  {label}")

    # Every configured source must win some scene, across BOTH media — a
    # VIDEO-only source like IA never appears in an IMAGE route, so checking
    # images alone would wrongly flag it as dead weight.
    def wins_somewhere(name: str) -> bool:
        pool = WIDE_CASES + [(d, q, "IMAGE", w, y) for d, q, w, y in CASES]
        return any(name in S.route(d, m, WIDE, q) for d, q, m, _, _ in pool)
    every_source_reachable = all(wins_somewhere(n) for n in WIDE)
    bad += not every_source_reachable
    print(f"    {'ok' if every_source_reachable else '!!'}  "
          f"every configured source wins some scene "
          f"(a source nothing ever routes to is dead weight)")

    print("\n  circuit breaker (a source the network cannot reach):")
    S.reset_failures()
    full = {"nasa", "smithsonian", "openverse", "wikimedia", "pexels", "pixabay"}
    before = S.route("ancient rome", "IMAGE", full, "aqueduct")
    for _ in range(S.FAIL_LIMIT):
        S.note_failure("wikimedia")
    after = S.route("ancient rome", "IMAGE", full, "aqueduct")
    dropped = "wikimedia" in before and "wikimedia" not in after
    bad += not dropped
    print(f"    {'ok' if dropped else '!!'}  dropped after {S.FAIL_LIMIT} failures")
    print(f"        {before}  ->  {after}")
    S.note_success("wikimedia")
    back = "wikimedia" in S.route("ancient rome", "IMAGE", full, "aqueduct")
    bad += not back
    print(f"    {'ok' if back else '!!'}  restored by a single success "
          f"(a network coming back needs no intervention)")
    still = bool(S.route("ancient rome", "IMAGE", full, "aqueduct"))
    bad += not still
    print(f"    {'ok' if still else '!!'}  routing survives losing a source")
    S.reset_failures()

    print("\n  no-key install still finds pictures:")
    bare = S.usable({})
    bad += not bare
    print(f"    {'ok' if bare else '!!'}  {sorted(bare)}")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
