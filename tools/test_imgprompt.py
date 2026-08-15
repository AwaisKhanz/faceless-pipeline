#!/usr/bin/env python3
"""AI image-prompt quality + consistency:
  - the project's Visual-style line is read back from the sheet,
  - imagen.prompt_for folds it into every prompt (and is unchanged without it),
  - gemini.image_prompts writes a detailed prompt per scene, sharing a project
    style and a recurring-subject bible so a set stays consistent — driven here
    by a FAKE llm so no network/model is needed.

    python3 tools/test_imgprompt.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import sheet as SH          # noqa: E402
from lib import imagen as IM         # noqa: E402
from lib import gemini as G          # noqa: E402

SHEET = """<!-- main-lang: en -->
# Demo
_2 scenes · language: en_

**Visual style:** Warm slate and soft amber palette, calm cinematic daylight.

---

**S1 ⬜** · IMAGE
- Narration: "An older woman checks her phone."
- ALT / search: `senior woman looking at smartphone`

**S2 ⬜** · IMAGE
- Narration: "The same woman frowns."
- ALT / search: `senior woman concerned expression`
"""


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<58}{'' if ok else repr(got)}")
        bad += not ok

    print("\n  sheet.visual_style reads the header line:")
    d = Path(tempfile.mkdtemp())
    sheet = d / "Demo_main_script.md"
    sheet.write_text(SHEET, encoding="utf-8")
    check("parses the Visual style line",
          SH.visual_style(sheet), "Warm slate and soft amber palette, calm cinematic daylight.")
    nostyle = d / "n.md"
    nostyle.write_text("# X\n\n**S1 ⬜** · IMAGE\n- Narration: \"Hi.\"\n", encoding="utf-8")
    check("no style line -> ''", SH.visual_style(nostyle), "")

    print("\n  prompt_for folds in the project style (and is unchanged without it):")
    plain = IM.prompt_for("a red sofa in a living room", {})
    styled = IM.prompt_for("a red sofa in a living room", {}, style="Warm amber palette")
    check("no style reproduces the classic prompt",
          plain.startswith("A real photograph of a red sofa in a living room."), True)
    check("style is injected right after the subject", "Warm amber palette" in styled, True)
    check("style sits before the photoreal look",
          styled.index("Warm amber palette") < styled.index("photorealistic"), True)
    check("an explicit generate_style override still wins (no auto style)",
          IM.prompt_for("x", {"generate_style": "cel shaded"}, style="Warm amber"),
          "x. cel shaded.")

    print("\n  gemini.image_prompts: bible built once, reused; one prompt per scene:")
    seen = {"bible_calls": 0, "img_calls": 0, "ctx_had_style": 0, "ctx_had_bible": 0}

    def fake_call(prompt, schema, key, model=None, system="", **kw):
        if system == G._BIBLE_SYSTEM:
            seen["bible_calls"] += 1
            return {"subjects": [{"name": "the woman",
                                  "description": "early 70s, silver bob, navy cardigan"}]}
        # image chunk: echo one prompt per numbered scene, and record the context
        seen["img_calls"] += 1
        if "SHARED STYLE" in prompt:
            seen["ctx_had_style"] += 1
        if "CONSISTENCY BIBLE" in prompt and "silver bob" in prompt:
            seen["ctx_had_bible"] += 1
        ns = [int(m) for m in re.findall(r"^(\d+)\. line:", prompt, re.M)]
        return {"scenes": [{"scene": n, "prompt": f"photoreal image for scene {n}"} for n in ns]}

    orig = G.call
    G.call = fake_call
    try:
        scenes = [{"n": 1, "query": "senior woman looking at smartphone",
                   "narration": "An older woman checks her phone."},
                  {"n": 2, "query": "senior woman concerned expression",
                   "narration": "The same woman frowns."}]
        out = G.image_prompts(scenes, key="k", model="m",
                              style="Warm slate and soft amber palette")
    finally:
        G.call = orig

    check("a prompt for every scene", sorted(out), [1, 2])
    check("prompts are non-empty", all(out.values()), True)
    check("the bible was built exactly once", seen["bible_calls"], 1)
    check("the shared style was passed into the crafting", seen["ctx_had_style"] >= 1, True)
    check("the bible was passed into the crafting", seen["ctx_had_bible"] >= 1, True)

    print("\n  image_prompts degrades safely when the LLM fails:")
    def boom(*a, **k):
        raise RuntimeError("no llm")
    G.call = boom
    try:
        check("any failure -> {} (caller uses the plain prompt)",
              G.image_prompts([{"n": 1, "query": "x", "narration": "y"}], "k", "m"), {})
    finally:
        G.call = orig

    print("\n  plan() casts to the audience, not a hardcoded senior default:")
    cap = {}

    def cap_call(prompt, schema, key, model=None, system="", **kw):
        cap["p"] = prompt
        return {"title_en": "x"}

    save = G.call
    G.call = cap_call
    try:
        G.plan("a general script about kitchen gadgets", "k", "m")
        check("default -> a general adult audience", "a general adult audience" in cap["p"], True)
        check("default -> the old '60+' bias is gone", "60+" not in cap["p"], True)
        flat = re.sub(r"\s+", " ", cap["p"]).lower()
        check("a 'do not default to elderly' directive is present",
              "not default to elderly or senior" in flat, True)
        G.plan("s", "k", "m", audience="young tech enthusiasts, energetic")
        check("a custom audience is threaded into the plan prompt",
              "young tech enthusiasts, energetic" in cap["p"], True)
    finally:
        G.call = save

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
