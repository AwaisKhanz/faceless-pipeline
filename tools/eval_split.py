#!/usr/bin/env python3
"""Score the scene splitter against a gold set.

Without this, changing the prompt is guesswork and "did that help?" gets
answered by vibes. This runs the real splitter over scripts a human has already
decided the answer for, and reports what changed.

    python3 tools/eval_split.py                  # the whole gold set
    python3 tools/eval_split.py --case space     # one case, verbose
    python3 tools/eval_split.py --sheets         # also re-split real sheets
    python3 tools/eval_split.py --save before    # record a baseline
    python3 tools/eval_split.py --against before # compare to it

WHAT IS SCORED, AND WHY IT IS WEIGHTED THIS WAY

  fidelity   Does the output reproduce the script word for word? PASS/FAIL,
             no partial credit. A splitter that improves a word is broken no
             matter how well it cut.

  count      Scene count against the human's. Scored within a tolerance,
             because where a scene ends is genuinely arguable and pretending
             otherwise would tune the prompt towards one person's taste.

  boundaries How many of the human's cut points the model also chose. Fraction,
             not pass/fail, for the same reason.

  queries    Mechanical checks only — three queries present, English, sane
             length, no abstractions from a known-bad list, no duplicates
             within a script. Whether a query returns *good* footage cannot be
             judged without looking, so this does not pretend to.

Every request costs Gemini quota. The full set is about 15 calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import gemini as G, pipeline as pl, sheet as sheetlib  # noqa: E402

GOLD = ROOT / "tools" / "gold" / "split_cases.json"
RUNS = ROOT / "tools" / "gold" / "runs"

# Words that mean the query describes an idea rather than a picture. Stock
# libraries return clip-art or nothing for these.
ABSTRACT = re.compile(
    r"\b(concept|abstract|feeling|emotion|sense of|idea of|symboli[sz]|"
    r"metaphor|representing|essence|spirit of|journey of|power of)\b", re.I)


def norm(s: str) -> str:
    """Compare text the way a listener would: words only, case-folded."""
    return " ".join(re.findall(r"[\w']+", (s or "").lower()))


# ─────────────────────────────────────────────────────────── scoring

def score_case(case: dict, scenes: list[dict]) -> dict:
    want_n = case.get("expect_scenes", 0)
    got_n = len(scenes)

    joined = " ".join(s.get("narration", "") for s in scenes)
    fidelity = norm(joined) == norm(case["narration"])

    # Count: exact is best, ±1 acceptable, a hard ceiling where the case sets one.
    ceiling = case.get("max_scenes")
    if ceiling and got_n > ceiling:
        count_ok, count_note = False, f"over the {ceiling}-scene ceiling"
    elif got_n == want_n:
        count_ok, count_note = True, "exact"
    elif abs(got_n - want_n) <= 1:
        count_ok, count_note = True, "within 1"
    else:
        count_ok, count_note = False, f"off by {got_n - want_n:+d}"

    # Boundaries: did it cut where the human cut?
    want_cuts = case.get("cuts_after") or []
    hit = 0
    if want_cuts:
        ends = {norm(s.get("narration", "")).split()[-1]
                for s in scenes if norm(s.get("narration", ""))}
        for c in want_cuts:
            w = norm(c).split()
            if w and w[-1] in ends:
                hit += 1

    # Queries.
    qissues: list[str] = []
    shapes: dict[frozenset, int] = {}
    for i, s in enumerate(scenes, 1):
        q = (s.get("query") or "").strip()
        fb = (s.get("fallback_query") or "").strip()
        sf = (s.get("safety_query") or "").strip()
        if not q:
            qissues.append(f"S{i} no query")
            continue
        if not fb or not sf:
            qissues.append(f"S{i} incomplete ladder")
        for label, text in (("query", q), ("fallback", fb), ("safety", sf)):
            if not text:
                continue
            n = len(text.split())
            # The rungs are meant to differ in specificity. A safety query
            # SHOULD be short and plain — "night sky", "ocean waves" are
            # exactly right — so only the primary is held to a real minimum.
            # Getting this wrong would push the prompt to pad the safety net,
            # defeating the point of having one.
            floor = 3 if label == "query" else 2
            if n < floor:
                qissues.append(f"S{i} {label} too short ({n}w)")
            elif n > 14:
                qissues.append(f"S{i} {label} too long ({n}w)")
            if ABSTRACT.search(text):
                qissues.append(f"S{i} {label} abstract: {text[:34]!r}")
            if not re.search(r"[a-z]", text):
                qissues.append(f"S{i} {label} not English?")
        # Each rung should be no more specific than the one above it.
        if fb and sf and len(sf.split()) > len(q.split()):
            qissues.append(f"S{i} safety is wordier than the primary — "
                           f"the ladder does not loosen")

        key = frozenset(w for w in re.findall(r"[a-z]+", q.lower())
                        if w not in {"a", "an", "the", "of", "in", "on", "at", "and"})
        if key and key in shapes:
            qissues.append(f"S{i} repeats S{shapes[key]}")
        elif key:
            shapes[key] = i

    return {
        "id": case["id"], "register": case.get("register", ""),
        "want": want_n, "got": got_n,
        "fidelity": fidelity,
        "count_ok": count_ok, "count_note": count_note,
        "boundaries": (hit, len(want_cuts)),
        "media_video": sum(1 for s in scenes if (s.get("media") or "") == "VIDEO"),
        "hero": sum(1 for s in scenes if s.get("hero")),
        "longest": max((len((s.get("narration") or "").split()) for s in scenes),
                       default=0),
        "qissues": qissues,
    }


def run_case(case: dict, key: str, model: str) -> tuple[list[dict], str]:
    try:
        scenes = G.split_section(
            case["narration"], {"title": case["id"], "spine_phrase": ""},
            key, model)
        return scenes, ""
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def report(rows: list[dict], baseline: dict | None) -> int:
    print()
    hdr = (f"  {'case':<22}{'reg':<13}{'scenes':<10}{'fidelity':<10}"
           f"{'bounds':<9}{'long':<6}queries")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    bad = 0
    for r in rows:
        if r.get("error"):
            print(f"  {r['id']:<22}{'':<13}ERROR  {r['error'][:44]}")
            bad += 1
            continue
        fid = "ok" if r["fidelity"] else "CHANGED WORDS"
        cnt = f"{r['got']}/{r['want']}"
        if not r["count_ok"]:
            cnt += " !!"
        b = r["boundaries"]
        bs = f"{b[0]}/{b[1]}" if b[1] else "-"
        qs = "ok" if not r["qissues"] else f"{len(r['qissues'])} issue(s)"
        print(f"  {r['id']:<22}{r['register']:<13}{cnt:<10}{fid:<10}"
              f"{bs:<9}{r['longest']:<6}{qs}")
        bad += (not r["fidelity"]) or (not r["count_ok"])

    for r in rows:
        if r.get("qissues"):
            print(f"\n  {r['id']}:")
            for q in r["qissues"][:8]:
                print(f"      {q}")

    ok_fid = sum(1 for r in rows if r.get("fidelity"))
    ok_cnt = sum(1 for r in rows if r.get("count_ok"))
    tb = sum(r["boundaries"][1] for r in rows if "boundaries" in r)
    hb = sum(r["boundaries"][0] for r in rows if "boundaries" in r)
    n = len(rows)
    print(f"\n  fidelity   {ok_fid}/{n} reproduced the script exactly")
    print(f"  count      {ok_cnt}/{n} within tolerance")
    print(f"  boundaries {hb}/{tb} of the human's cut points matched"
          f"  ({100 * hb / tb:.0f}%)" if tb else "")
    print(f"  queries    {sum(len(r.get('qissues', [])) for r in rows)} issue(s) total")

    if baseline:
        print("\n  against the baseline:")
        old = {r["id"]: r for r in baseline.get("rows", [])}
        for r in rows:
            o = old.get(r["id"])
            if not o or r.get("error") or o.get("error"):
                continue
            deltas = []
            if r["got"] != o["got"]:
                deltas.append(f"scenes {o['got']}->{r['got']}")
            if r["boundaries"] != tuple(o["boundaries"]):
                deltas.append(f"bounds {o['boundaries'][0]}->{r['boundaries'][0]}")
            if len(r["qissues"]) != len(o["qissues"]):
                deltas.append(f"query issues {len(o['qissues'])}->{len(r['qissues'])}")
            if deltas:
                print(f"    {r['id']:<22}{', '.join(deltas)}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="run a single case by id, and show its scenes")
    ap.add_argument("--sheets", action="store_true",
                    help="also re-split the opening of each real sheet")
    ap.add_argument("--save", help="save this run as a named baseline")
    ap.add_argument("--against", help="compare against a saved baseline")
    a = ap.parse_args()

    cfg = pl.load_config()
    key = cfg.get("gemini_key", "")
    if not key:
        print("No Gemini key in config.json — nothing to evaluate.")
        return 1
    model = cfg.get("gemini_model", "auto")
    if model in ("", "auto"):
        model = G.resolve_model(key, "")
    print(f"  model: {model}")

    cases = json.loads(GOLD.read_text(encoding="utf-8"))["cases"]
    if a.case:
        cases = [c for c in cases if c["id"] == a.case] or cases[:0]
        if not cases:
            print(f"No case called {a.case!r}.")
            return 1

    if a.sheets and not a.case:
        # Real narrative writing, so short samples cannot quietly become the
        # only thing the prompt is good at.
        for f in sorted((ROOT / "sheets").glob("*MASTER*.md"))[:2]:
            try:
                sc = sheetlib.parse_master(f)[:8]
                cases.append({
                    "id": f"real:{pl.project_id(f)}",
                    "register": "people-led",
                    "expect_scenes": len(sc),
                    "narration": " ".join(s.narration for s in sc),
                })
            except Exception:
                pass

    rows = []
    for c in cases:
        scenes, err = run_case(c, key, model)
        if err:
            rows.append({"id": c["id"], "error": err})
        else:
            rows.append(score_case(c, scenes))
        if a.case:
            print(f"\n  {len(scenes)} scene(s) for {c['id']}:\n")
            for i, s in enumerate(scenes, 1):
                print(f"   S{i} [{s.get('media')}]"
                      f"{'  HERO' if s.get('hero') else ''}")
                print(f"      {s.get('narration')}")
                print(f"      1 {s.get('query')}")
                print(f"      2 {s.get('fallback_query')}")
                print(f"      3 {s.get('safety_query')}\n")
        time.sleep(0.4)          # be polite to the free tier

    baseline = None
    if a.against:
        f = RUNS / f"{a.against}.json"
        if f.exists():
            baseline = json.loads(f.read_text(encoding="utf-8"))
        else:
            print(f"  (no baseline called {a.against!r})")

    bad = report(rows, baseline)

    if a.save:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{a.save}.json").write_text(
            json.dumps({"model": model, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\n  saved as tools/gold/runs/{a.save}.json")

    print()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
