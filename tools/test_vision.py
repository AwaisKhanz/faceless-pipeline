#!/usr/bin/env python3
"""Freeze the relevance ladder — how sourcing searches harder for a real match.

    python3 tools/test_vision.py

The CLIP model needs a GPU and a download, so it cannot run in CI. But the logic
around it — pick the model for the machine, rank by relevance, escalate down the
query ladder until a match clears the bar, flag a weak best, and fall straight
back to the old behaviour when scoring is off — is pure control flow, and that
is what breaks. This drives it with a fake scorer and a fake fetch so every
branch runs offline in milliseconds.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import stock, vision  # noqa: E402


def scene(n, query, fallbacks=(), media="IMAGE"):
    return SimpleNamespace(n=n, query=query, fallbacks=list(fallbacks),
                           media=media, domain="")


def main() -> int:
    bad = 0

    def check(label, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<50}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    print("\n  the machine picks a model it can run:")
    check("RTX-class GPU -> SigLIP 2 so400m", vision._pick_model("cuda", 16), vision.SIGLIP_SO400M)
    check("mid GPU -> SigLIP 2 large", vision._pick_model("cuda", 8), vision.SIGLIP_L)
    check("small GPU -> reliable CLIP", vision._pick_model("cuda", 6), vision.BASE16)
    check("Apple MPS -> reliable CLIP", vision._pick_model("mps", None), vision.BASE16)
    check("cpu-only laptop -> base32", vision._pick_model("cpu", None), vision.BASE32)
    check("SigLIP picks a CLIP fallback", vision._family_of(
        vision._clip_fallback("cuda", 16)), "clip")
    check("family detection: siglip", vision._family_of(vision.SIGLIP_SO400M), "siglip")
    check("family detection: clip", vision._family_of(vision.BASE32), "clip")
    check("config can turn it off", vision.capability({"clip": "off"})["ok"], False)

    # Relevance each query "returns", keyed by query text. The fake fetch reads
    # this to stand in for a real CLIP score on a real download.
    SCORES = {
        "strong": 0.82, "weakA": 0.30,
        "wb1": 0.31, "strongB": 0.71, "wb2": 0.20,
        "c1": 0.20, "c2": 0.30, "c3": 0.38,      # all below the 0.45 bar
        "vidq": None,                            # a video with nothing to score
    }
    calls = []

    def fake_fetch(query, media, cache, pk, xk, index=0, sources=None, cfg=None):
        calls.append((query, index))
        if query == "dry":                       # a query free stock can't fill
            raise stock.StockError("no result for 'dry'")
        return {"path": f"/fake/{query}", "credit": "", "page": query,
                "src": "pexels", "query": query, "media": media,
                "index": index, "score": SCORES.get(query)}

    real_fetch, real_get = stock.fetch, vision.get_scorer
    stock.fetch = fake_fetch

    # ---- scoring ON --------------------------------------------------------
    vision.get_scorer = lambda cfg=None, log=lambda *a: None: object()  # truthy
    out = stock.fetch_all(
        [scene(1, "strong", ["weakA"]),                    # primary already good
         scene(2, "wb1", ["strongB", "wb2"]),              # escalates to fallback
         scene(3, "c1", ["c2", "c3"]),                     # all weak -> best + flag
         scene(4, "vidq", media="VIDEO")],                 # unscorable -> take it
        Path("/tmp"), "pk", "xk", log=lambda *a: None)

    print("\n  with scoring on:")
    check("a strong primary is used as-is", out[1]["query"], "strong")
    check("stops at the primary, does not escalate",
          ("weakA", 0) not in calls, True)
    check("escalates past a weak primary to a strong fallback",
          out[2]["query"], "strongB")
    check("when every rung is weak, keeps the best of them",
          out[3]["query"], "c3")
    check("an unscorable video is taken without escalating",
          out[4]["query"], "vidq")

    # ---- scoring OFF (graceful) --------------------------------------------
    calls.clear()
    vision.get_scorer = lambda cfg=None, log=lambda *a: None: None
    out = stock.fetch_all(
        [scene(1, "wb1", ["strongB", "wb2"])],             # would escalate if on
        Path("/tmp"), "pk", "xk", log=lambda *a: None)

    print("\n  with scoring off (old behaviour, exactly):")
    check("takes the first usable match, no escalation", out[1]["query"], "wb1")
    check("never even tries the fallbacks", ("strongB", 0) in calls, False)

    # ---- safety net: a scene the ladder can't fill is never left empty -------
    calls.clear()
    vision.get_scorer = lambda cfg=None, log=lambda *a: None: None
    out = stock.fetch_all(
        [scene(1, "dry", ["dry"])],                        # nothing real exists
        Path("/tmp"), "pk", "xk", log=lambda *a: None)

    print("\n  when a scene finds nothing at all:")
    check("the scene is still filled, not dropped", 1 in out, True)
    check("filled from a generic safety query",
          out[1]["query"] in stock._SAFETY_QUERIES, True)
    check("flagged as a placeholder", out[1].get("placeholder"), True)
    check("carries no score (needs a real swap)", out[1].get("score"), None)

    stock.fetch, vision.get_scorer = real_fetch, real_get

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
