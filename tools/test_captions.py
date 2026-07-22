#!/usr/bin/env python3
"""Freeze the subtitle styling + word-timing logic.

    python3 tools/test_captions.py

No GPU, no model, no ffmpeg needed: this drives the pure logic — colour maths,
chunking words into on-screen lines, the ASS the renderer will burn, the
estimated-timing fallback, and the style store (presets, custom, per-project).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import captions as C, align as A  # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<54}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    print("\n  colours convert to ASS correctly:")
    check("white, opaque -> AABBGGRR", C.hex_to_ass("#FFFFFF", 1.0), "&H00FFFFFF")
    check("purple active word", C.hex_to_ass("#B57BFF", 1.0), "&H00FF7BB5")
    check("inline \\c form is 6-digit BGR + &", C._rgb_only("#B57BFF"), "&HFF7BB5&")
    check("55% opacity -> alpha byte", C._alpha_only(0.55), "&H73&")

    print("\n  words chunk into short on-screen lines:")
    words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.38}
             for i, w in enumerate("one two three four five six seven eight".split())]
    st = C.PRESETS["reference"].merged(max_words=5)
    groups = C.chunk_words(words, st)
    check("splits 8 words at max 5", [len(g["words"]) for g in groups], [5, 3])
    check("group start/end track the words", (groups[0]["start"], round(groups[1]["end"], 2)),
          (0.0, 3.18))

    print("\n  a sentence end breaks the line early:")
    sw = [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.28}
          for i, w in enumerate("calm night. deep water".split())]
    g2 = C.chunk_words(sw, st)
    check("breaks after the full stop", [[x["word"] for x in g["words"]] for g in g2],
          [["calm", "night."], ["deep", "water"]])

    print("\n  estimated timing (no aligner) still covers the whole line:")
    hw = C.heuristic_words("the real thief of sleep", 10.0, 2.5)
    check("every word kept", [w["word"] for w in hw],
          ["the", "real", "thief", "of", "sleep"])
    check("starts at the offset", hw[0]["start"], 10.0)
    check("ends within the line", hw[-1]["end"] <= 12.5 + 0.01, True)

    print("\n  the ASS the renderer burns:")
    ass = C.build_ass(groups, st)
    lines = ass.splitlines()
    l0 = [x for x in lines if x.startswith("Dialogue: 0")]
    l1 = [x for x in lines if x.startswith("Dialogue: 1")]
    check("has a PlayRes header", "PlayResX: 1920" in ass)
    check("one bar per group (Layer 0)", len(l0), 2)
    check("one event per word (Layer 1)", len(l1), 8)
    check("active word carries the accent colour",
          C._rgb_only(st.active_color) in l1[0])
    check("bar uses colour + separate alpha",
          C._rgb_only(st.bar_color) in l0[0] and C._alpha_only(st.bar_opacity) in l0[0])

    print("\n  bar off removes the bar layer; karaoke off collapses the highlight:")
    nobar = st.merged(bar=False)
    a2 = C.build_ass(groups, nobar)
    check("no Layer-0 events when the bar is off",
          any(x.startswith("Dialogue: 0") for x in a2.splitlines()), False)
    plain = st.merged(karaoke=False)
    a3 = C.build_ass(C.chunk_words(words, plain), plain)
    check("karaoke off -> one event per group",
          sum(x.startswith("Dialogue: 1") for x in a3.splitlines()), 2)

    print("\n  styles resolve from id / dict / none:")
    check("a preset id", C.resolve_style("bold_yellow").name, "Bold Yellow")
    check("a dict override merges onto its base",
          C.resolve_style({"template": "reference", "active_color": "#00FF88"}).active_color,
          "#00FF88")
    check("None -> the default preset", C.resolve_style(None).name,
          C.PRESETS[C.DEFAULT_PRESET].name)
    check("round-trips through a dict",
          C.Style.from_dict(st.to_dict()).to_dict() == st.to_dict(), True)

    print("\n  alignment degrades gracefully with no model installed:")
    capd = A.capability({})
    check("reports estimated timing, not a crash", capd["engine"], "heuristic")
    check("align_words still returns words",
          [w["word"] for w in A.align_words("/none.wav", "one two three", "en",
                                            cfg={}, dur=1.5)],
          ["one", "two", "three"])
    gaps = A._fill_gaps([{"word": "a", "start": 0.0, "end": 0.5},
                         {"word": "b", "start": None, "end": None},
                         {"word": "c", "start": 1.5, "end": 2.0}], 2.0)
    check("a missing word is interpolated between neighbours",
          (gaps[1]["start"], gaps[1]["end"]), (0.5, 1.0))

    print("\n  the style store: presets + custom + per-project override:")
    import lib.pipeline as pl
    tmp = Path(tempfile.mkdtemp())
    pl.ROOT, pl.PROJECTS = tmp, tmp / "projects"
    pl.CAPTIONS_FILE = tmp / "captions.json"
    pl.set_global_caption_style("night_cyan")
    check("global default persists", pl.global_caption_style(), "night_cyan")
    pl.save_custom_caption_style("Mine", {"template": "reference", "active_color": "#FF0"})
    check("custom template saved", "Mine" in pl.custom_caption_styles())
    (pl.PROJECTS / "vid").mkdir(parents=True)
    pl.save_project_style("vid", {"template": "minimal", "size": 80})
    check("project override wins over global",
          C.resolve_style(pl.effective_caption_style("vid")).size, 80)
    pl.save_project_style("vid", None)
    check("clearing it falls back to global",
          pl.effective_caption_style("vid"), "night_cyan")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
