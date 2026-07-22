"""Word-level forced alignment: *when* is each word spoken.

Captions light up each word as it's said, which needs the start time of every
word. We already have each scene's audio and its exact text, so this is forced
alignment (fit known words to the audio) — tighter than transcribing blind.

IT ADDS NOTHING TO INSTALL. Alignment runs on torchaudio, which the voice engine
(Chatterbox) already pulls in, using torchaudio's built-in multilingual
forced-alignment model (MMS_FA). No extra package, so it can never fight the
carefully-pinned voice/vision stack — an earlier WhisperX-based version dragged
in conflicting torch/numpy/transformers pins and broke Chatterbox; this cannot.

Adaptive, like lib/vision.py:
  - uses the GPU only if a real kernel launches on it (a Blackwell card with the
    wrong torch says "available", then fails — we test for real);
  - loads the alignment model once and reuses it;
  - ALWAYS degrades to a proportional estimate (lib/captions.heuristic_words), so
    a machine that can't load the model still gets word-by-word captions, just
    with looser sync. Nothing here is imported at module load, so importing this
    file is free and safe even where torch/torchaudio are absent.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from . import captions as _cap

_OFF = ("off", "false", "no", "0", "none")
_SR = 16000                     # MMS_FA works at 16 kHz


def _cfg_get(cfg: dict | None, key: str, default=None):
    v = (cfg or {}).get(key)
    return default if v is None or v == "" else v


def capability(cfg: dict | None = None) -> dict:
    """What alignment can do here, without loading anything.

    {ok, engine, device, reason}. ok False means the caller uses estimated
    timing; engine is 'heuristic' in that case so the UI can say so plainly.
    """
    cfg = cfg or {}
    if str(_cfg_get(cfg, "align", "auto")).lower() in _OFF:
        return {"ok": False, "engine": "heuristic", "device": "-",
                "reason": "turned off in config (align: off) — estimated timing"}
    try:
        import torch          # noqa: F401
        import torchaudio
    except Exception:
        return {"ok": False, "engine": "heuristic", "device": "cpu",
                "reason": "torch/torchaudio not installed — estimated word timing"}
    # The forced-alignment bundle arrived in torchaudio 2.1. Older builds simply
    # fall back to estimated timing rather than erroring.
    if not hasattr(getattr(torchaudio, "pipelines", None), "MMS_FA"):
        return {"ok": False, "engine": "heuristic", "device": "cpu",
                "reason": "torchaudio too old for forced alignment — estimated timing"}
    device, vram = _probe_device()
    return {"ok": True, "engine": "torchaudio MMS", "device": device,
            "vram_gb": vram, "model": "MMS_FA (multilingual)", "reason": "ready"}


_HW: tuple | None = None


def _device_runs(torch, dev: str) -> bool:
    """Does a real kernel actually launch? Same trap as the voice/vision code: a
    Blackwell GPU on the wrong torch reports available, then fails the first op."""
    try:
        x = torch.randn(16, 16, device=dev)
        _ = (x @ x).sum().item()
        return True
    except Exception:
        return False


def _probe_device() -> tuple:
    """cuda if it genuinely computes, else cpu (a Mac aligns on CPU — fine for a
    few minutes of narration). Cached for the process."""
    global _HW
    if _HW is not None:
        return _HW
    device, vram = "cpu", None
    try:
        import torch
        if torch.cuda.is_available() and _device_runs(torch, "cuda"):
            device = "cuda"
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        device, vram = "cpu", None
    _HW = (device, vram)
    return _HW


def _norm_token(word: str) -> str:
    """A word reduced to what the MMS alignment dictionary understands: lower
    case, diacritics folded to ASCII (ä→a, ñ→n), ß→ss, letters only. Used ONLY to
    place the word in time — the caption still shows the original spelling."""
    w = word.replace("ß", "ss")
    w = unicodedata.normalize("NFKD", w)
    w = "".join(c for c in w if not unicodedata.combining(c))
    w = re.sub(r"[^a-z]", "", w.lower())
    return w


def _fill_gaps(words: list[dict], dur: float) -> list[dict]:
    """Backfill any word the aligner left without a timestamp — a number, a token
    the dictionary doesn't hold. Spread the unplaced words evenly between their
    nearest timed neighbours so the highlight never desyncs."""
    if not words:
        return words
    n = len(words)
    if words[0].get("start") is None:
        words[0]["start"] = 0.0
    if words[-1].get("end") is None:
        words[-1]["end"] = dur
    i = 0
    while i < n:
        if words[i].get("start") is not None and words[i].get("end") is not None:
            i += 1
            continue
        j = i
        while j < n and (words[j].get("start") is None or words[j].get("end") is None):
            j += 1
        left = words[i - 1]["end"] if i > 0 else 0.0
        right = words[j]["start"] if j < n else dur
        step = max(0.0, right - left) / (j - i + 1)
        for k in range(i, j):
            words[k]["start"] = round(left + step * (k - i), 3)
            words[k]["end"] = round(left + step * (k - i + 1), 3)
        i = j
    return words


class Aligner:
    """Holds torchaudio's forced-alignment model for the life of the process."""

    def __init__(self, device: str):
        self.device = device
        self._model = None
        self._tokenizer = None
        self._aligner = None

    def _load(self):
        if self._model is not None:
            return
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        self._model = bundle.get_model().to(self.device).eval()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()

    def words(self, wav: Path, text: str, lang: str) -> list[dict]:
        """Per-word [{word,start,end}] for `text` as spoken in `wav`, relative to
        the clip start. Words the dictionary can't place are left un-timed for
        _fill_gaps to interpolate."""
        import torch
        import torchaudio
        self._load()

        wave, sr = torchaudio.load(str(wav))
        if sr != _SR:
            wave = torchaudio.functional.resample(wave, sr, _SR)
        wave = wave.mean(0, keepdim=True)          # mono
        dur = wave.size(1) / float(_SR)

        display = [w for w in re.split(r"\s+", text.strip()) if w]
        tokens = [_norm_token(w) for w in display]
        keep = [i for i, t in enumerate(tokens) if t]     # alignable words only
        if not keep:
            return []

        with torch.inference_mode():
            emission, _ = self._model(wave.to(self.device))
            spans = self._aligner(emission[0], self._tokenizer([tokens[i] for i in keep]))

        sec = wave.size(1) / emission.size(1) / float(_SR)   # seconds per frame
        timed = {}
        for idx, sp in zip(keep, spans):
            if sp:
                timed[idx] = (round(sp[0].start * sec, 3), round(sp[-1].end * sec, 3))

        out = []
        for i, w in enumerate(display):
            s, e = timed.get(i, (None, None))
            out.append({"word": w, "start": s, "end": e})
        return _fill_gaps(out, dur)


_ALIGNER: Aligner | None = None


def get_aligner(cfg: dict | None = None) -> Aligner | None:
    """The shared aligner, or None if alignment is unavailable here."""
    global _ALIGNER
    if not capability(cfg)["ok"]:
        return None
    if _ALIGNER is None:
        device, _ = _probe_device()
        _ALIGNER = Aligner(device)
    return _ALIGNER


def align_words(wav: Path, text: str, lang: str, cfg: dict | None = None,
                dur: float | None = None, log=lambda *a: None) -> list[dict]:
    """Word timings for one clip. Never raises: on any failure it returns a
    proportional estimate so captions still render word-by-word.

    `dur` (clip length in seconds) is only used by the estimate; pass it when you
    already know it to skip a probe.
    """
    aligner = get_aligner(cfg)
    if aligner is not None:
        try:
            got = aligner.words(Path(wav), text, lang)
            if got:
                return got
            log("  alignment returned nothing; using estimated timing")
        except Exception as e:                       # noqa: BLE001
            log(f"  alignment failed ({e}); using estimated timing")
    if dur is None:
        try:
            from . import render
            dur = render.duration_of(Path(wav))
        except Exception:
            dur = max(1.0, len(str(text).split()) * 0.36)
    return _cap.heuristic_words(text, 0.0, dur)
