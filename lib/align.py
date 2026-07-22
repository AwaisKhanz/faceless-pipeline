"""Word-level forced alignment: *when* is each word spoken.

Captions need to know the start of every word to light it up in time. We already
have each scene's audio and its exact text, so this is forced alignment (fit
known words to the audio), which is far more accurate than transcribing blind.

Adaptive, like lib/vision.py:
  - uses the GPU only if a real kernel launches on it (a Blackwell card with the
    wrong torch says "available" and then can't run — we test for real);
  - loads a per-language alignment model once and reuses it;
  - ALWAYS degrades to a proportional estimate (lib/captions.heuristic_words) so
    a machine without the model still gets word-by-word captions, just looser.

The real engine is WhisperX (wav2vec2 forced alignment, ~50 ms, the precision
CapCut/Submagic use), which reuses the torch already installed for the voice
engine. Nothing here is imported at module load, so importing this file is free
and safe even where torch/whisperx are absent.
"""
from __future__ import annotations

from pathlib import Path

from . import captions as _cap

_OFF = ("off", "false", "no", "0", "none")
_SR = 16000                     # whisperx.load_audio resamples to 16 kHz


def _cfg_get(cfg: dict | None, key: str, default=None):
    v = (cfg or {}).get(key)
    return default if v is None or v == "" else v


def capability(cfg: dict | None = None) -> dict:
    """What alignment can do here, without loading anything.

    {ok, engine, device, reason}. ok False means the caller uses estimated
    timing; engine is still 'heuristic' in that case so the UI can say so.
    """
    cfg = cfg or {}
    if str(_cfg_get(cfg, "align", "auto")).lower() in _OFF:
        return {"ok": False, "engine": "heuristic", "device": "-",
                "reason": "turned off in config (align: off) — estimated timing"}
    try:
        import torch  # noqa: F401
    except Exception:
        return {"ok": False, "engine": "heuristic", "device": "cpu",
                "reason": "torch not installed — estimated word timing"}
    try:
        import whisperx  # noqa: F401
    except Exception:
        return {"ok": False, "engine": "heuristic", "device": "cpu",
                "reason": "whisperx not installed — estimated word timing "
                          "(pip install whisperx)"}
    device, vram = _probe_device()
    return {"ok": True, "engine": "whisperx", "device": device, "vram_gb": vram,
            "model": "wav2vec2 (per language)", "reason": "ready"}


_HW: tuple | None = None


def _device_runs(torch, dev: str) -> bool:
    """Does a real kernel actually launch? Same trap as the voice/vision code:
    a Blackwell GPU on the wrong torch reports available, then fails the first
    op. Test for real, or we claim a GPU we can't use."""
    try:
        x = torch.randn(16, 16, device=dev)
        _ = (x @ x).sum().item()
        return True
    except Exception:
        return False


def _probe_device() -> tuple:
    """cuda if it genuinely computes, else cpu. WhisperX has no MPS path, so a
    Mac aligns on CPU (fine for a few minutes of narration). Cached."""
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


def _fill_gaps(words: list[dict], dur: float) -> list[dict]:
    """Backfill any word the aligner left without a timestamp.

    wav2vec2 occasionally can't place a token (odd punctuation, a number). Rather
    than drop it — which would desync the highlight — spread the unplaced words
    evenly between their nearest timed neighbours.
    """
    if not words:
        return words
    n = len(words)
    # Ensure the ends are anchored.
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
        span = max(0.0, right - left)
        step = span / (j - i + 1)
        for k in range(i, j):
            words[k]["start"] = round(left + step * (k - i), 3)
            words[k]["end"] = round(left + step * (k - i + 1), 3)
        i = j
    return words


class Aligner:
    """Holds the per-language alignment models for the life of the process."""

    def __init__(self, device: str):
        self.device = device
        self._models: dict[str, tuple] = {}

    def _model(self, lang: str):
        import whisperx
        if lang not in self._models:
            self._models[lang] = whisperx.load_align_model(
                language_code=lang, device=self.device)
        return self._models[lang]

    def words(self, wav: Path, text: str, lang: str) -> list[dict]:
        """Per-word [{word,start,end}] for `text` as spoken in `wav`. Times are
        relative to the start of the clip."""
        import whisperx
        model_a, meta = self._model(lang)
        audio = whisperx.load_audio(str(wav))
        dur = len(audio) / float(_SR)
        segs = [{"start": 0.0, "end": dur, "text": text.strip()}]
        res = whisperx.align(segs, model_a, meta, audio, self.device,
                             return_char_alignments=False)
        out: list[dict] = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                out.append({"word": w.get("word", ""),
                            "start": w.get("start"), "end": w.get("end")})
        out = [w for w in out if w["word"]]
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
    already know it to avoid another probe.
    """
    aligner = get_aligner(cfg)
    if aligner is not None:
        try:
            got = aligner.words(Path(wav), text, lang)
            if got:
                return got
            log("  alignment returned nothing; using estimated timing")
        except Exception as e:
            log(f"  alignment failed ({e}); using estimated timing")
    # Fallback that always works.
    if dur is None:
        try:
            from . import render
            dur = render.duration_of(Path(wav))
        except Exception:
            dur = max(1.0, len(str(text).split()) * 0.36)
    return _cap.heuristic_words(text, 0.0, dur)
