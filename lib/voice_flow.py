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
  * It is SAFE: if alignment can't run, or the word counts don't line up, that
    group silently falls back to per-scene synthesis (written to the same fl_
    paths), so the audio is never worse than today — only, at best, smoother.
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


def synth(scenes, lang: str, cache: Path, voice: str, engine: str,
          raw_synth, align_words, duration_of, slice_audio,
          can_join: bool = True, log=print) -> list[Path]:
    """Produce one flowed clip per scene (scene order).

    Injected callables keep this engine- and ffmpeg-agnostic (and testable):
      raw_synth(text) -> Path     speak text with the active engine
      align_words(wav, text)      -> [{word,start,end}] (relative to wav)
      duration_of(wav) -> float
      slice_audio(src, start, end, out) -> Path

    `can_join` is False when REAL forced alignment isn't available (only a
    proportional estimate would be) — then every group is voiced per scene into
    its fl_ path, never cut, so a guessed boundary can't clip a word.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    made = joined_groups = 0

    for g in groups(scenes):
        joined, dests = _paths_for_group(g, scenes, lang, cache, voice, engine)
        if all(d.exists() and d.stat().st_size > 1024 for d in dests):
            out.extend(dests)
            continue

        if len(g) == 1:                                   # nothing to join
            src = raw_synth(_clean(scenes[g[0]].narration))
            _place(src, dests[0])
            out.append(dests[0])
            made += 1
            continue

        # Speak the whole sentence once, then try to cut it at the word joins.
        # Only attempt this when real alignment is available (can_join).
        spans = grp_wav = None
        if can_join:
            try:
                grp_wav = raw_synth(joined)
                words = align_words(grp_wav, joined)
                spans = _boundaries(g, scenes, words, duration_of(grp_wav))
            except Exception as e:                        # noqa: BLE001
                log(f"  flow: couldn't join S{scenes[g[0]].n}-S{scenes[g[-1]].n} "
                    f"({e}); voicing them separately")

        if spans is not None:
            for (st, en), d in zip(spans, dests):
                slice_audio(grp_wav, st, en, d)
            joined_groups += 1
            made += len(g)
            log(f"  flow: joined S{scenes[g[0]].n}-S{scenes[g[-1]].n} as one "
                f"sentence, split into {len(g)} clips")
        else:
            # Fallback: voice each member on its own, into the same fl_ paths.
            for k, j in enumerate(g):
                _place(raw_synth(_clean(scenes[j].narration)), dests[k])
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
