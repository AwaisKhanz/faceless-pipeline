"""Engine-agnostic helpers shared by every voice backend (Chatterbox, Higgs, …).

Kept tiny and pure (numpy only) so both engines guard against the same failure
modes — a stutter, silence/noise, or a clip cut short or rambling — in exactly
the same way, and so the "retry a bad take, keep the cleanest" policy lives in
ONE place instead of being copied per engine.
"""
from __future__ import annotations

import hashlib


def seed_all(seed: int) -> None:
    """Make one take reproducible AND make the next take different, so a retry
    explores a new sample instead of reproducing the same glitch."""
    try:
        import random

        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def take_quality(samples, sr: int, text: str, repetition: bool = False):
    """Judge one generated take. Returns (ok, badness); lower badness is better.

    Catches the three ways neural TTS fails on short narration lines: a
    repeated/forced-EOS stutter (flagged by the engine), near-silence/noise (no
    real voice), and a clip far too short (a cut-off word) or far too long."""
    import numpy as np
    n = 0 if samples is None else int(getattr(samples, "size", 0) or len(samples))
    dur = n / float(sr or 24000)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if n else 0.0
    words = max(1, len(text.split()))
    expected = max(0.35, words * 0.33)              # ~seconds of narration
    silent = rms < 0.005                            # produced silence / faint noise
    too_short = dur < max(0.18, 0.30 * expected)    # a word got cut off
    too_long = dur > expected * 5 + 4               # it rambled past the end
    bad = bool(repetition or silent or too_short or too_long)
    badness = (0.0 if not bad else 100.0) + abs(dur - expected) + (1000.0 if silent else 0.0)
    return (not bad, badness)


def seed_for(text: str) -> int:
    """A stable per-line seed so re-voicing the same text is deterministic."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def best_take(generate, text: str, best_of: int = 1, retries: int = 2,
              seed0: int | None = None, log=None):
    """Run `generate(i) -> (samples, sr, repetition)` up to best_of+retries times,
    keeping the cleanest take.

    `best_of` takes are always produced (and the best kept); `retries` adds more
    ONLY while the best so far still looks broken. Returns (samples, sr, ok, taken).
    The caller's `generate` owns any per-attempt tweak (e.g. nudging temperature
    down on a retry) — this function only decides how many to run and which to
    keep, so the policy is identical across engines."""
    needed = max(1, int(best_of))
    max_takes = needed + max(0, int(retries))
    if seed0 is None:
        seed0 = seed_for(text)
    best = None                                     # (ok, badness, samples, sr)
    taken = 0
    for i in range(max_takes):
        seed_all((seed0 + i) % (2 ** 31))
        samples, sr, rep = generate(i)
        ok, badness = take_quality(samples, sr, text, rep)
        taken += 1
        if best is None or badness < best[1]:
            best = (ok, badness, samples, sr)
        if ok and taken >= needed:                  # good enough, quota met
            break
    ok, _, samples, sr = best
    if log and (taken > 1 or not ok):
        log("      · " + (f"{taken} takes, kept the cleanest"
                          if ok else f"{taken} takes, still imperfect — kept the best"))
    return samples, sr, ok, taken
