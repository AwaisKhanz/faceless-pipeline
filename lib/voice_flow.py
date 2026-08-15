"""Sentence-flow narration: speak a whole sentence as ONE utterance, then split
the audio back into per-scene clips by word alignment.

A sentence is often spread over several scenes so each shows its own picture
("… vitamin C than any other fruit," / "helping your immune system" / "and
healthy skin."). Voiced as separate clips, each one carries its own sentence-
final intonation, so even with no gap the pitch resets at every cut. This module
instead speaks the WHOLE sentence in one take (natural, unbroken prosody) and
cuts it back into per-scene pieces at the silences between words — so each scene
still gets its own clip, but they are pieces of one flowing sentence.

Design (deliberately engine-agnostic and cache-consistent):

  * It never talks to a TTS engine directly. The caller passes `raw_synth(text)`
    — "speak this text with the active engine, return a wav" — and `align_words`
    — "word timings for this wav against this text". So Chatterbox, Higgs and
    Chirp all work unchanged, and tests can drive it with fakes.
  * Output paths use a distinct `fl_` prefix and a key derived from the JOINED
    group text + the member's position + the voice + the engine. `synth()` and
    `expected_paths()` compute the grouping the same way from the same scenes, so
    the status/render lookups always match what was written.
  * It is SAFE: a sentence is always spoken as ONE take. If forced alignment
    can't run (no local model — the common cloud-voice case), the take is split
    proportionally by text length instead; only if the take itself fails to
    synthesise does a group fall back to per-scene clips. Either way the audio is
    never worse than before — usually much smoother.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from .chatterbox_engine import speech_text

# A scene ends a sentence when its cleaned text ends with . ! ? or … (optionally
# followed by a closing quote/bracket). Everything else runs on into the next.
_SENT_END = re.compile(r"[.!?…]['\"’”)\]]*$")

# Guardrails so one runaway "sentence" can't become a minute-long utterance that
# stresses the engine or the aligner.
MAX_SCENES = 6
MAX_CHARS = 320


def _clean(text: str) -> str:
    return speech_text(text or "")


def _ends_sentence(text: str) -> bool:
    t = _clean(text)
    return (not t) or bool(_SENT_END.search(t))


def groups(scenes) -> list[list[int]]:
    """Partition scene indices into consecutive sentence utterances.

    A group closes at a scene that ends a sentence, or when it would grow past
    the size guardrails. Groups are consecutive and cover every scene in order,
    so flattening them preserves scene order.
    """
    out: list[list[int]] = []
    cur: list[int] = []
    for i, s in enumerate(scenes):
        cur.append(i)
        chars = sum(len(_clean(scenes[j].narration)) for j in cur)
        last = i == len(scenes) - 1
        if _ends_sentence(s.narration) or len(cur) >= MAX_SCENES or chars >= MAX_CHARS or last:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _key(joined: str, member: int, voice: str, lang: str, engine: str) -> str:
    blob = f"flow|{engine}|{voice}|{lang}|{member}|{joined}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _paths_for_group(group: list[int], scenes, lang: str, cache: Path,
                     voice: str, engine: str) -> tuple[str, list[Path]]:
    joined = " ".join(_clean(scenes[j].narration) for j in group)
    dests = [cache / f"fl_{lang}_{scenes[j].n:03d}_"
             f"{_key(joined, k, voice, lang, engine)}.wav"
             for k, j in enumerate(group)]
    return joined, dests


def expected_paths(scenes, lang: str, cache: Path, voice: str,
                   engine: str) -> list[Path]:
    """Where each scene's flowed clip is (or would be) cached — in scene order.
    Generates nothing; must stay in lock-step with what synth() writes."""
    out: list[Path] = []
    for g in groups(scenes):
        _, dests = _paths_for_group(g, scenes, lang, cache, voice, engine)
        out.extend(dests)
    return out


def _boundaries(group: list[int], scenes, words: list[dict], dur: float):
    """Per-member (start, end) cut times inside the group audio, or None if the
    alignment can't be trusted to line up with the scene boundaries.

    The joined text is the members' cleaned narrations space-joined, so word i of
    the joined text belongs to a specific member by cumulative word count. Each
    internal boundary is placed in the SILENCE between the last word of one member
    and the first word of the next (their midpoint), so a cut never lands inside a
    word.
    """
    counts = [len(_clean(scenes[j].narration).split()) for j in group]
    if any(c == 0 for c in counts) or sum(counts) != len(words):
        return None                       # mismatch → caller falls back per-scene

    def t(i: int, which: str) -> float:
        v = words[i].get(which)
        return float(v) if v is not None else (0.0 if which == "start" else dur)

    edges = [0.0]
    idx = 0
    for c in counts[:-1]:
        idx += c
        left = t(idx - 1, "end")          # end of this member's last word
        right = t(idx, "start")           # start of the next member's first word
        edges.append(max(left, min(right, (left + right) / 2.0)))
    edges.append(dur)

    spans = [(edges[k], edges[k + 1]) for k in range(len(group))]
    # Every piece must be a sane, forward span; otherwise don't risk the cut.
    if any(en - st < 0.05 for st, en in spans):
        return None
    return spans


def _proportional_boundaries(group, scenes, dur: float):
    """Cut the joined take by each member's share of the sentence, measured in
    characters (a decent proxy for how long a fragment takes to say).

    Used when real word alignment isn't available (e.g. a cloud voice with no
    local aligner installed). A single natural take split a little imprecisely is
    far smoother than voicing each fragment on its own — the pieces play back to
    back, so the sentence's prosody carries across the cuts instead of resetting.
    """
    if dur <= 0:
        return None
    weights = [max(1, len(_clean(scenes[j].narration))) for j in group]
    total = float(sum(weights)) or 1.0
    spans, acc = [], 0.0
    for k, w in enumerate(weights):
        st = acc
        acc = dur if k == len(weights) - 1 else acc + dur * (w / total)
        spans.append((st, acc))
    if any(en - st < 0.02 for st, en in spans):
        return None                        # too tiny to cut safely → caller falls back
    return spans


def synth(scenes, lang: str, cache: Path, voice: str, engine: str,
          raw_synth, align_words, duration_of, slice_audio,
          can_join: bool = True, log=print) -> list[Path]:
    """Produce one flowed clip per scene (scene order).

    Injected callables keep this engine- and ffmpeg-agnostic (and testable):
      raw_synth(text) -> Path     speak text with the active engine
      align_words(wav, text)      -> [{word,start,end}] (relative to wav)
      duration_of(wav) -> float
      slice_audio(src, start, end, out) -> Path

    `can_join` selects the CUT method, not whether to join: True uses real forced
    alignment to place cuts in the silences between words; False (no local
    aligner) splits the one take proportionally by text length. Either way the
    sentence is spoken once, so its intonation never resets mid-sentence.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    made = joined_groups = 0

    # One "S N voiced/cached" line per SCENE (every path), so the progress bar —
    # which counts those lines — advances even though flow works in sentence
    # GROUPS. Without this the Activity counter sat at 0/N through the whole run
    # while the log clearly showed it progressing.
    def _done(idx: int, made_now: bool) -> None:
        log(f"S{scenes[idx].n:>3} {'voiced' if made_now else 'cached'}")

    for g in groups(scenes):
        joined, dests = _paths_for_group(g, scenes, lang, cache, voice, engine)
        if all(d.exists() and d.stat().st_size > 1024 for d in dests):
            out.extend(dests)
            for j in g:
                _done(j, False)
            continue

        if len(g) == 1:                                   # nothing to join
            src = raw_synth(_clean(scenes[g[0]].narration))
            _place(src, dests[0])
            out.append(dests[0])
            made += 1
            _done(g[0], True)
            continue

        # Speak the WHOLE sentence in one take (unbroken prosody), then cut it
        # back into per-scene pieces. Prefer real word alignment for the cut
        # points; when that isn't available (a cloud voice with no local aligner),
        # split the SAME take proportionally by text length — still one natural
        # take, far smoother than voicing each fragment on its own.
        grp_wav = spans = None
        try:
            grp_wav = raw_synth(joined)
            dur = duration_of(grp_wav)
            if can_join:
                try:
                    spans = _boundaries(g, scenes, align_words(grp_wav, joined), dur)
                except Exception as e:                    # noqa: BLE001
                    log(f"  flow: alignment failed for S{scenes[g[0]].n}-"
                        f"S{scenes[g[-1]].n} ({e}); splitting proportionally")
            if spans is None:                             # no aligner, or it declined
                spans = _proportional_boundaries(g, scenes, dur)
        except Exception as e:                            # noqa: BLE001
            log(f"  flow: couldn't voice S{scenes[g[0]].n}-S{scenes[g[-1]].n} as "
                f"one take ({e}); voicing them separately")

        if grp_wav is not None and spans is not None:
            for (st, en), d in zip(spans, dests):
                slice_audio(grp_wav, st, en, d)
            joined_groups += 1
            made += len(g)
            log(f"  flow: joined S{scenes[g[0]].n}-S{scenes[g[-1]].n} as one "
                f"sentence, split into {len(g)} clips")
            for j in g:
                _done(j, True)
        else:
            # Last resort — the joined take itself failed: voice each member.
            for k, j in enumerate(g):
                _place(raw_synth(_clean(scenes[j].narration)), dests[k])
                _done(j, True)
            made += len(g)
        out.extend(dests)

    log(f"Flow: {made} clip(s), {joined_groups} sentence(s) joined "
        f"({len(out)} scenes total).")
    return out


def _place(src: Path, dest: Path) -> None:
    """Move a freshly-synthesised clip onto its cache path (copy, not rename, so a
    shared source cache isn't disturbed)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(src) != Path(dest):
        shutil.copy(src, dest)
