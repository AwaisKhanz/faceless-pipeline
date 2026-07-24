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

    # ── detailed per-scene live feedback ────────────────────────────────────
    print("\n  fetch_all logs rich per-scene feedback and leaks no telemetry:")
    fcache = Path(tempfile.mkdtemp())

    def fx(query, media, cache, pk, xk, index, sources=None, cfg=None):
        return {"path": f"/x/{query[:5]}_{index}", "src": "nasa", "query": query,
                "media": media, "score": 0.9, "credit": "", "page": "", "license": "",
                "_detail": {"sources": sources or ["nasa"],
                            "counts": {"nasa": 9, "pexels": 3},
                            "pooled": 12, "scored": 8,
                            "ranked": [("nasa", 0.9), ("pexels", 0.5)]}}
    stock.fetch = fx
    stock.vision.get_scorer = lambda cfg, log=None: object()   # scoring "on"
    stock._SRC.usable = lambda cfg: {"nasa", "pexels"}
    stock._SRC.route = lambda *a, **k: ["nasa", "pexels"]
    stock._SRC.down_sources = lambda: []
    logs: list = []
    a = stock.fetch_all(
        [SimpleNamespace(n=1, media="IMAGE", query="moon surface",
                         fallbacks=[], domain="space", topic="space")],
        fcache, "PK", "XK", log=logs.append, cfg={})
    joined = "\n".join(logs)
    check("result line shows source + score", "nasa" in joined and "90%" in joined)
    check("detail line shows searches + scoring",
          "searched" in joined and "scored" in joined and "top" in joined)
    check("detail line shows per-source counts (hides nothing)",
          "nasa 9" in joined and "pexels 3" in joined)
    check("telemetry stripped from the stored asset", "_detail" not in a[1])

    # A fallback prints a ladder line; the candidate list caps by default and is
    # exhaustive under source_log: "full".
    def fx2(query, media, cache, pk, xk, index, sources=None, cfg=None):
        weak = "sunset" in query
        return {"path": f"/y/{query[:5]}_{index}",
                "src": "wikimedia" if weak else "openverse", "query": query,
                "media": media, "score": 0.38 if weak else 0.74,
                "credit": "", "page": "", "license": "",
                "_detail": {"sources": sources or [],
                            "counts": {"openverse": 4, "loc": 2, "smithsonian": 1},
                            "pooled": 10,
                            "scored": 8, "ranked": [("openverse", 0.74),
                            ("wikimedia", 0.63), ("loc", 0.55), ("smithsonian", 0.51),
                            ("pexels", 0.48), ("wikimedia", 0.44), ("openverse", 0.41),
                            ("loc", 0.37)]}}
    stock.fetch = fx2
    stock._SRC.route = lambda *a, **k: ["smithsonian", "openverse", "loc", "wikimedia"]
    scn = SimpleNamespace(n=5, media="IMAGE", query="roman aqueduct at sunset",
                          fallbacks=["roman aqueduct arches"], domain="history",
                          topic="history")
    j2 = []
    stock.fetch_all([scn], Path(tempfile.mkdtemp()), "PK", "XK",
                    log=j2.append, cfg={"clip_min": 0.45})
    j2s = "\n".join(j2)
    check("a fallback prints a ladder line", "ladder:" in j2s and "→" in j2s)
    check("a source asked but empty reads as 0, not hidden", "wikimedia 0" in j2s)
    check("default view caps the candidate list", "more)" in j2s)
    j3 = []
    stock.fetch_all([scn], Path(tempfile.mkdtemp()), "PK", "XK",
                    log=j3.append, cfg={"clip_min": 0.45, "source_log": "full"})
    j3s = "\n".join(j3)
    check("source_log 'full' lists every candidate", "all:" in j3s and "more)" not in j3s)
    check("full view opens a per-scene header", "Scene 5" in j3s)
    check("full view narrates each query step by step",
          "search:" in j3s and "fallback 1:" in j3s)
    check("full view lists every source with its count per query",
          j3s.count("sources:") >= 2)

    # ── biography mode drops stock on people scenes ─────────────────────────
    print("\n  biography mode drops stock on people scenes (real person wins):")
    seen = {}

    def cap_fetch(query, media, cache, pk, xk, index, sources=None, cfg=None):
        seen["s"] = list(sources or [])
        return {"path": f"/z/{index}", "src": (sources or ["x"])[0], "query": query,
                "media": media, "score": None, "credit": "", "page": "", "license": ""}
    stock.fetch = cap_fetch
    stock.vision.get_scorer = lambda cfg, log=None: None      # scoring off: first match wins
    stock._SRC.usable = lambda cfg: {"pexels", "pixabay", "openverse", "wikimedia"}
    stock._SRC.route = lambda *a, **k: ["pexels", "pixabay", "openverse", "wikimedia"]
    stock._SRC.down_sources = lambda: []
    ppl = SimpleNamespace(n=1, media="IMAGE", query="Elon Musk speaking",
                          fallbacks=[], domain="biography", topic="people")
    stock.fetch_all([ppl], Path(tempfile.mkdtemp()), "PK", "XK",
                    cfg={"name_real_people": True}, log=lambda *_: None)
    check("stock dropped for a people scene in biography mode",
          [s for s in seen["s"] if s in ("pexels", "pixabay")], [])
    check("archives kept", "openverse" in seen["s"] and "wikimedia" in seen["s"])
    seen.clear()
    stock.fetch_all([ppl], Path(tempfile.mkdtemp()), "PK", "XK",
                    cfg={}, log=lambda *_: None)
    check("stock kept when biography mode is off", "pexels" in seen["s"])

    # ── a disabled source is announced the moment it happens ────────────────
    print("\n  the circuit breaker announces a source the instant it is disabled:")
    stock._SRC._FAILS.clear()
    stock._SRC._JUST_DOWN.clear()
    stock.vision.get_scorer = lambda cfg, log=None: None
    stock._SRC.usable = lambda cfg: {"pexels", "wikimedia"}
    stock._SRC.route = lambda *a, **k: ["pexels", "wikimedia"]
    stock._SRC.down_sources = lambda: ["wikimedia"]

    def failing_fetch(query, media, cache, pk, xk, index, sources=None, cfg=None):
        # wikimedia crosses the fail limit during this scene's ladder
        for _ in range(stock._SRC.FAIL_LIMIT):
            stock._SRC.note_failure("wikimedia")
        return {"path": f"/d/{index}", "src": "pexels", "query": query,
                "media": media, "score": None, "credit": "", "page": "", "license": ""}
    stock.fetch = failing_fetch
    dl = []
    stock.fetch_all([SimpleNamespace(n=1, media="IMAGE", query="anything",
                                     fallbacks=[], domain="x", topic="tech")],
                    Path(tempfile.mkdtemp()), "PK", "XK", log=dl.append, cfg={})
    dls = "\n".join(dl)
    check("names the disabled source", "wikimedia" in dls and "disabled" in dls)
    check("explains it will be skipped for the run", "rest of this run" in dls)
    stock._SRC._FAILS.clear()
    stock._SRC._JUST_DOWN.clear()

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
