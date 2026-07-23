#!/usr/bin/env python3
"""Freeze the visual-accuracy upgrades.

    python3 tools/test_sourcing.py

No network, no CLIP, no ffmpeg: everything the accuracy work changed is control
flow, and that is what this locks —

  - fetch() now pools candidates from EVERY routed source and lets relevance pick
    the best across all of them (the old code stopped at the first source);
  - route() offers Openverse as extra breadth on image scenes;
  - _expand_scene_queries attaches Gemini's alternative queries to a scene's
    ladder, cached so a re-source spends no tokens;
  - the scoring calibration moved (version bump, more templates, gentler junk).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import stock, sources as SRC, vision, pipeline as pl  # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<56}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    # ── multi-source pooling ────────────────────────────────────────────────
    print("\n  fetch pools every routed source, not just the first:")
    cache = Path(tempfile.mkdtemp())

    def fake_pexels(q, media, key, want):
        return [{"url": "px1", "ext": ".jpg", "width": 1920, "height": 1080,
                 "thumb": "px1t", "src": "pexels", "credit": "", "page": ""}]

    def fake_search(name, q, media, want, cfg):
        # Openverse holds a much better match for this scene.
        return [SimpleNamespace(url="ov1", ext=".jpg", width=1920, height=1080,
                                thumb="ov1t", src="openverse", credit="",
                                page="", license="cc0")]

    stock._pexels = fake_pexels
    stock._SRC.search = fake_search
    stock._relevance = lambda pool, q, media, cfg: {"px1": 0.30, "ov1": 0.82}
    stock._fetch_bytes = lambda url: b"imgbytes"
    stock._pixel_width = lambda f: 1920           # above the floor

    meta = stock.fetch("a calm scene", "IMAGE", cache, "PXKEY", None, index=0,
                       sources=["pexels", "openverse"], cfg={})
    check("winner is the best across sources (openverse)", meta["src"], "openverse")
    check("its relevance is recorded", meta["score"], 0.82)

    print("\n  a swap (index 1) returns the runner-up, still cross-source:")
    meta2 = stock.fetch("a calm scene", "IMAGE", cache, "PXKEY", None, index=1,
                        sources=["pexels", "openverse"], cfg={})
    check("second pick is the lower-scored candidate", meta2["src"], "pexels")

    # ── routing offers Openverse on image scenes ────────────────────────────
    print("\n  route offers Openverse as breadth for image scenes:")
    avail = {"pexels", "pixabay", "openverse", "wikimedia", "loc"}
    modern = SRC.route("", "IMAGE", avail, "senior lying awake in bed at night")
    check("a modern image scene now includes openverse", "openverse" in modern)
    check("stock still leads it", modern.index("pexels") < modern.index("openverse"))
    vid = SRC.route("", "VIDEO", {"pexels", "pixabay"}, "calm water")
    check("video scenes stay on stock only", vid, ["pexels", "pixabay"])

    # ── LLM query expansion is attached + cached ────────────────────────────
    print("\n  Gemini expansions join the ladder and cache:")
    import lib.gemini as G
    calls = {"n": 0}

    def fake_expand(items, key, model="auto"):
        calls["n"] += 1
        return {s["n"]: ["person asleep in bed", "dark bedroom night"] for s in items}
    G.expand_queries = fake_expand

    tmp = Path(tempfile.mkdtemp())
    p = {"base": tmp / "work" / "en"}
    scenes = [SimpleNamespace(n=1, query="hypnogram", narration="your sleep cycle",
                              fallbacks=["sleep graph"]),
              SimpleNamespace(n=2, query="clock", narration="3am", fallbacks=[])]
    pl._expand_scene_queries(scenes, p, {"gemini_key": "K"})
    check("expansions appended to the ladder",
          scenes[0].fallbacks, ["sleep graph", "person asleep in bed", "dark bedroom night"])
    check("the human query/fallbacks stay first", scenes[0].fallbacks[0], "sleep graph")
    check("one Gemini call for the batch", calls["n"], 1)
    check("cache file written", (tmp / "work" / "queries.json").exists())

    # Re-source: same queries -> cache hit, no second Gemini call.
    scenes2 = [SimpleNamespace(n=1, query="hypnogram", narration="your sleep cycle",
                               fallbacks=["sleep graph"]),
               SimpleNamespace(n=2, query="clock", narration="3am", fallbacks=[])]
    pl._expand_scene_queries(scenes2, p, {"gemini_key": "K"})
    check("a re-source spends no tokens (cache hit)", calls["n"], 1)
    check("expansions still applied from cache",
          "person asleep in bed" in scenes2[0].fallbacks)

    print("\n  no Gemini key -> expansion is a silent no-op:")
    s3 = [SimpleNamespace(n=1, query="x", narration="y", fallbacks=[])]
    pl._expand_scene_queries(s3, {"base": Path(tempfile.mkdtemp()) / "w" / "en"}, {})
    check("nothing added without a key", s3[0].fallbacks, [])

    # ── no-dimension archive still gets a fair shot ─────────────────────────
    print("\n  a source that reports no dimensions is still scored and can win:")
    # 20 dimensioned stock hits would, under the old technical-sort-first code,
    # push the single no-dimension NASA hit outside the scored window. _fair_pool
    # round-robins across sources so the archive is always looked at.
    many = ([{"url": f"px{i}", "src": "pexels", "width": 1920, "height": 1080}
             for i in range(20)]
            + [{"url": "nasa1", "src": "nasa", "width": 0, "height": 0}])
    pool = stock._fair_pool(many, 4)
    check("fair pool includes the no-dimension source",
          "nasa1" in [h["url"] for h in pool])

    cache3 = Path(tempfile.mkdtemp())
    stock._pexels = lambda q, m, k, w: [
        {"url": "pxD", "ext": ".jpg", "width": 1920, "height": 1080,
         "thumb": "pxDt", "src": "pexels", "credit": "", "page": ""}]
    stock._SRC.search = lambda name, q, m, w, cfg: [
        SimpleNamespace(url="nasaD", ext=".jpg", width=0, height=0, thumb="",
                        src="nasa", credit="", page="", license="pd")]
    stock._relevance = lambda pool, q, media, cfg: {"pxD": 0.40, "nasaD": 0.88}
    stock._fetch_bytes = lambda url: b"imgbytes"
    stock._pixel_width = lambda f: 1920
    meta3 = stock.fetch("moon surface", "IMAGE", cache3, "PXKEY", None, index=0,
                        sources=["nasa", "pexels"], cfg={})
    check("no-dimension archive wins when it is the better match", meta3["src"], "nasa")

    # ── scoring calibration moved ───────────────────────────────────────────
    print("\n  scoring recalibration is in place:")
    check("score version bumped (re-source recomputes)", vision.SCORE_VERSION, 6)
    check("prompt ensemble has four templates", len(vision.TEMPLATES), 4)
    check("clip band top raised so matches stop pegging at 100%",
          vision._band_of(vision.BASE32), (0.15, 0.35))
    check("siglip has its own band", vision._family_of(vision.SIGLIP_SO400M), "siglip")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
