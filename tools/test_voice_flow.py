#!/usr/bin/env python3
"""Sentence-flow: grouping, cache-key consistency, the audio cut boundaries, and
the safe fallback — all with fake engine/alignment/ffmpeg so no models or ffmpeg
are needed.

    python3 tools/test_voice_flow.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import voice_flow as VF   # noqa: E402


def _sc(n, text):
    return types.SimpleNamespace(n=n, narration=text)


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<56}{'' if ok else repr(got)}")
        bad += not ok

    scenes = [
        _sc(1, "Camu Camu is a tiny Amazon berry"),                 # continues
        _sc(2, "packed with more vitamin C than any other fruit,"), # continues
        _sc(3, "helping your immune system"),                       # continues
        _sc(4, "and healthy skin."),                                # ends
        _sc(5, "This is a whole separate sentence."),               # ends (alone)
    ]

    print("\n  grouping: consecutive scenes -> sentence utterances:")
    gs = VF.groups(scenes)
    check("first sentence groups S1-S4, then S5", gs, [[0, 1, 2, 3], [4]])

    print("\n  expected_paths: scene order, fl_ prefix, stable keys:")
    cache = pathlib.Path(tempfile.mkdtemp())
    paths = VF.expected_paths(scenes, "en", cache, "voiceA", "chatterbox")
    check("one path per scene", len(paths), 5)
    check("all use the fl_ prefix", all(p.name.startswith("fl_en_") for p in paths), True)
    check("keys are deterministic (recompute matches)",
          [p.name for p in paths],
          [p.name for p in VF.expected_paths(scenes, "en", cache, "voiceA", "chatterbox")])
    check("a different voice changes the keys",
          paths[0].name != VF.expected_paths(scenes, "en", cache, "voiceB", "chatterbox")[0].name,
          True)

    print("\n  boundary math: cuts fall in the gaps between members' words:")
    # joined S1-S4 has 6+8+4+3 = 21 words; fake even 0.5s words with 0.1s gaps
    group = [0, 1, 2, 3]
    joined = " ".join(VF._clean(scenes[j].narration) for j in group)
    nwords = len(joined.split())
    words = []
    t = 0.0
    for _ in range(nwords):
        words.append({"word": "x", "start": round(t, 3), "end": round(t + 0.5, 3)})
        t += 0.6
    dur = t
    spans = VF._boundaries(group, scenes, words, dur)
    check("one span per member", len(spans or []), 4)
    check("spans are contiguous and forward",
          all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
          and all(en > st for st, en in spans), True)
    counts = [len(VF._clean(scenes[j].narration).split()) for j in group]
    # boundary after member 0 must sit between word 5's end and word 6's start
    b0 = spans[0][1]
    check("first cut lands in the silence between members",
          words[counts[0] - 1]["end"] <= b0 <= words[counts[0]]["start"], True)
    check("word-count mismatch -> None (caller falls back)",
          VF._boundaries(group, scenes, words[:-1], dur), None)

    print("\n  synth: joins a sentence, splits it, caches; single scene passes through:")
    calls = {"raw": 0, "slice": 0}
    made_src = pathlib.Path(tempfile.mkdtemp())

    def raw_synth(text):
        calls["raw"] += 1
        f = made_src / f"src_{calls['raw']}.wav"
        f.write_bytes(b"RIFF" + b"\0" * 4000)
        return f

    def align_words(wav, text):
        n = len(text.split())
        out, tt = [], 0.0
        for _ in range(n):
            out.append({"word": "x", "start": round(tt, 3), "end": round(tt + 0.4, 3)})
            tt += 0.5
        return out

    def duration_of(wav):
        return 30.0

    def slice_audio(src, start, end, out):
        calls["slice"] += 1
        pathlib.Path(out).write_bytes(b"RIFF" + b"\0" * 4000)
        return out

    out = VF.synth(scenes, "en", cache, "voiceA", "chatterbox",
                   raw_synth=raw_synth, align_words=align_words,
                   duration_of=duration_of, slice_audio=slice_audio, log=lambda *_: None)
    check("returns a clip per scene", len(out), 5)
    check("output paths == expected_paths", [p.name for p in out],
          [p.name for p in VF.expected_paths(scenes, "en", cache, "voiceA", "chatterbox")])
    check("every clip written", all(p.exists() for p in out), True)
    check("the 4-scene sentence was sliced 4 times", calls["slice"], 4)
    check("raw_synth called twice (1 joined group + 1 single scene)", calls["raw"], 2)

    before = dict(calls)
    VF.synth(scenes, "en", cache, "voiceA", "chatterbox",
             raw_synth=raw_synth, align_words=align_words,
             duration_of=duration_of, slice_audio=slice_audio, log=lambda *_: None)
    check("re-run is fully cached (no new synth/slice)",
          calls == before, True)

    print("\n  progress: one countable 'S N voiced/cached' line per scene:")
    # The Activity bar counts lines matching this exact pattern; flow works in
    # sentence GROUPS, so without a per-scene line the bar sat at 0/N. Guard it.
    from lib.pipeline import _VOICE_SCENE_LINE      # the very regex the bar uses
    cacheP = pathlib.Path(tempfile.mkdtemp())
    lines: list[str] = []
    VF.synth(scenes, "en", cacheP, "voiceA", "chatterbox",
             raw_synth=raw_synth, align_words=align_words,
             duration_of=duration_of, slice_audio=slice_audio, log=lines.append)
    counted = sum(1 for m in lines if _VOICE_SCENE_LINE.match(m.lstrip()))
    check("fresh run emits one progress line per scene (bar reaches N)", counted, len(scenes))
    lines2: list[str] = []
    VF.synth(scenes, "en", cacheP, "voiceA", "chatterbox",   # everything cached now
             raw_synth=raw_synth, align_words=align_words,
             duration_of=duration_of, slice_audio=slice_audio, log=lines2.append)
    counted2 = sum(1 for m in lines2 if _VOICE_SCENE_LINE.match(m.lstrip()))
    check("a fully-cached re-run still reports N (bar isn't stuck)", counted2, len(scenes))

    print("\n  proportional split by text length (used when no aligner):")
    ps = VF._proportional_boundaries([0, 1, 2, 3], scenes, 30.0)
    check("one span per member", len(ps or []), 4)
    check("spans are contiguous and cover the whole take",
          ps[0][0] == 0.0 and ps[-1][1] == 30.0
          and all(ps[i][1] == ps[i + 1][0] for i in range(len(ps) - 1)), True)
    check("a longer fragment gets a longer slice",
          ps[1][1] - ps[1][0] > ps[3][1] - ps[3][0], True)   # S2 (longer) > S4 (short)
    check("zero-duration take -> None (caller falls back)",
          VF._proportional_boundaries([0, 1], scenes, 0.0), None)

    print("\n  alignment mismatch still JOINS — splits the one take proportionally:")
    cache2 = pathlib.Path(tempfile.mkdtemp())
    calls2 = {"raw": 0, "slice": 0}

    def raw2(text):
        calls2["raw"] += 1
        f = made_src / f"b_{calls2['raw']}.wav"
        f.write_bytes(b"RIFF" + b"\0" * 4000)
        return f

    def bad_align(wav, text):
        return [{"word": "x", "start": 0.0, "end": 0.4}]      # wrong count -> mismatch

    def slice2(src, start, end, out):
        calls2["slice"] += 1
        pathlib.Path(out).write_bytes(b"RIFF" + b"\0" * 4000)
        return out

    out2 = VF.synth(scenes, "en", cache2, "voiceA", "chatterbox",
                    raw_synth=raw2, align_words=bad_align,
                    duration_of=duration_of, slice_audio=slice2, log=lambda *_: None)
    check("one clip per scene", len(out2), 5)
    check("paths match expected", [p.name for p in out2],
          [p.name for p in VF.expected_paths(scenes, "en", cache2, "voiceA", "chatterbox")])
    check("the 4-scene sentence is still sliced (proportional)", calls2["slice"], 4)
    # 1 joined take + 1 single scene — NOT a synth per fragment.
    check("only 2 raw synths (joined take + single scene)", calls2["raw"], 2)

    print("\n  no local aligner (can_join=False): still joins via proportional split:")
    cache3 = pathlib.Path(tempfile.mkdtemp())
    calls3 = {"raw": 0, "slice": 0}

    def raw3(text):
        calls3["raw"] += 1
        f = made_src / f"c_{calls3['raw']}.wav"
        f.write_bytes(b"RIFF" + b"\0" * 4000)
        return f

    def slice3(src, start, end, out):
        calls3["slice"] += 1
        pathlib.Path(out).write_bytes(b"RIFF" + b"\0" * 4000)
        return out

    out3 = VF.synth(scenes, "en", cache3, "voiceA", "chatterbox",
                    raw_synth=raw3, align_words=align_words,
                    duration_of=duration_of, slice_audio=slice3,
                    can_join=False, log=lambda *_: None)
    check("one clip per scene", len(out3), 5)
    check("sentence sliced without an aligner (proportional)", calls3["slice"], 4)
    check("one joined take + one single scene (2 synths)", calls3["raw"], 2)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
