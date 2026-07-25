"""Higgs Audio (Boson AI) voice engine — the optional, higher-quality backend.

Apache-2.0, multilingual (50+ languages incl. German), zero-shot voice cloning.
It is heavier than Chatterbox (a multi-GB model, its own audio tokenizer) so it
is OFF by default and only used when config `voice_engine` is "higgs" AND the
`boson_multimodal` package is installed. Everything degrades gracefully: if it is
selected but not installed, the caller falls back to Chatterbox.

Cloning difference to be aware of: Higgs clones from the reference clip's audio
PLUS a transcript of what is said in it (Chatterbox needs only the audio). We
look the transcript up from voices.json ("reference_text") or a sibling .txt
next to the clip; with neither, Higgs still speaks — just in a generic voice, and
we say so once.

Machine-aware: the device (CUDA / Apple MPS / CPU), the dtype (bfloat16 on a GPU,
float32 on CPU) and whether to use the fast static-KV-cache are all chosen from
what the running machine actually has, with a FACELESS_DEVICE / config override.

This module mirrors the small surface the pipeline uses from chatterbox_engine
(installed / available / synth / expected_paths / describe), so tts.py can treat
the two engines interchangeably.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import voice_common as _vc
from . import voices as V
from .chatterbox_engine import speech_text   # one shared text cleaner

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "voices_refs"
CACHE = ROOT / "cache" / "voice"

DEFAULT_MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"
DEFAULT_SCENE = "Audio is recorded from a quiet room."
DEFAULT_MAX_NEW_TOKENS = 2048
STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]

# One loaded engine, kept alive across scenes (load is most of the cost).
_ENGINE = {"obj": None, "key": None, "device": None}


class HiggsError(RuntimeError):
    """Higgs could not produce audio. The caller keeps whatever it can."""


# ------------------------------------------------------------- availability

def installed() -> bool:
    """Is the Higgs package importable? (We never import the heavy bits here.)"""
    import importlib.util
    return importlib.util.find_spec("boson_multimodal") is not None


def install_hint() -> str:
    return ("Higgs Audio is not installed. On the machine with the GPU:\n"
            "  pip install git+https://github.com/boson-ai/higgs-audio.git\n"
            "then set \"voice_engine\": \"higgs\" in config.json.")


def available(cfg: dict | None = None) -> bool:
    return installed()


# ------------------------------------------------------------------ device

def best_device(cfg: dict | None = None) -> str:
    """Pick the device from what the machine has: an explicit override first,
    then CUDA, then Apple MPS, then CPU."""
    cfg = cfg or {}
    forced = os.environ.get("FACELESS_DEVICE") or str(cfg.get("higgs_device") or "").strip()
    if forced and forced.lower() != "auto":
        return forced.lower()
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _vram_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


def device_info(cfg: dict | None = None) -> dict:
    dev = best_device(cfg)
    d = {"device": dev, "name": None, "vram_gb": None}
    try:
        import torch
        if dev == "cuda":
            d["name"] = torch.cuda.get_device_name(0)
            d["vram_gb"] = round(_vram_gb() or 0, 1)
    except Exception:
        pass
    return d


# -------------------------------------------------------------- references

def _transcript_for(lang: str, reference: str) -> str:
    """The words spoken in the reference clip, for cloning. Looked up from
    voices.json ("reference_text") or a sibling .txt next to the clip. Empty
    string means 'no transcript' → the caller uses a generic voice."""
    txt = (V.pref_for(lang).get("reference_text") or "").strip() \
        if hasattr(V, "pref_for") else ""
    if txt:
        return txt
    try:
        src = V.resolve(reference)                 # the raw clip on disk
        side = src.with_suffix(".txt")
        if side.exists():
            return side.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


# ------------------------------------------------------------------- model

def _engine_for(cfg: dict):
    """Load (once) and return the Higgs serve engine, sized to this machine."""
    model = str(cfg.get("higgs_model") or DEFAULT_MODEL)
    tok = str(cfg.get("higgs_tokenizer") or DEFAULT_TOKENIZER)
    dev = best_device(cfg)
    key = (model, tok, dev)
    if _ENGINE["obj"] is not None and _ENGINE["key"] == key:
        return _ENGINE["obj"]

    if not installed():
        raise HiggsError(install_hint())
    try:
        import torch
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    except Exception as e:                          # pragma: no cover - env-specific
        raise HiggsError(f"Higgs is installed but could not be imported: {e}")

    # bfloat16 on a GPU (half the memory, native speed); float32 on CPU/MPS where
    # bf16 is unreliable. Passed only if the constructor accepts it.
    dtype = None
    try:
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    except Exception:
        dtype = None

    kwargs = {"device": dev}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    try:
        engine = HiggsAudioServeEngine(model, tok, **kwargs)
    except TypeError:
        # Older signature without torch_dtype — fall back to the documented call.
        engine = HiggsAudioServeEngine(model, tok, device=dev)

    _ENGINE.update(obj=engine, key=key, device=dev)
    return engine


def _messages(scene_text: str, ref_wav: Path | None, transcript: str, scene: str):
    """Build the ChatML the serve engine expects, cloning when we have both the
    reference audio and its transcript (mirrors Higgs' own example)."""
    from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
    system = (f"Generate audio following instruction.\n\n"
              f"<|scene_desc_start|>\n{scene}\n<|scene_desc_end|>")
    msgs = [Message(role="system", content=system)]
    if ref_wav is not None and transcript:
        msgs.append(Message(role="user", content=transcript))
        msgs.append(Message(role="assistant", content=AudioContent(audio_url=str(ref_wav))))
    msgs.append(Message(role="user", content=scene_text))
    return ChatMLSample(messages=msgs)


def _norm(text: str) -> str:
    """Light, speech-friendly normalisation: strip markdown symbols (shared with
    Chatterbox), expand degree units, and guarantee a terminal punctuation so the
    model has a clean place to stop (the #1 cause of runaway on short lines)."""
    t = speech_text(text).replace("°C", " degrees Celsius").replace("°F", " degrees Fahrenheit")
    t = " ".join(t.split())
    if t and t[-1] not in ".!?,;:\"'":
        t += "."
    return t


# --------------------------------------------------------------------- api

def _key(text: str, ref: str, lang: str, model: str, temperature: float) -> str:
    blob = f"higgs|{model}|{ref}|{lang}|{temperature}|{text}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def expected_paths(scenes, lang: str, reference: str, cache: Path = CACHE,
                   opts: dict | None = None, cfg: dict | None = None) -> list[Path]:
    """Where each scene's Higgs audio WOULD be cached (a 'hg_' prefix keeps it
    separate from Chatterbox's clips, so switching engines never mixes them)."""
    o = opts or {}
    cfg = cfg or {}
    model = str(cfg.get("higgs_model") or DEFAULT_MODEL)
    temp = float(o.get("temperature", 0.7))
    return [cache / f"hg_{lang}_{s.n:03d}_"
            f"{_key(_norm(s.narration), reference, lang, model, temp)}.wav"
            for s in scenes]


def describe(lang: str, cfg: dict | None = None) -> str:
    info = device_info(cfg)
    model = str((cfg or {}).get("higgs_model") or DEFAULT_MODEL).split("/")[-1]
    where = info["device"] + (f" · {info['name']}" if info.get("name") else "")
    return f"Higgs Audio · {model} · {where}"


def synth(scenes, lang: str, ref_wav, cache: Path = CACHE, opts: dict | None = None,
          log=print, cfg: dict | None = None, reference: str = "") -> list[Path]:
    """One audio file per scene via Higgs, cached like the other engines and
    guarded by the same retry-on-artifact policy as Chatterbox."""
    o = {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "retries": 2, "best_of": 1, **(opts or {})}
    cfg = cfg or {}
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    ref_wav = Path(ref_wav) if ref_wav else None

    engine = _engine_for(cfg)
    scene_desc = str(cfg.get("higgs_scene") or DEFAULT_SCENE)
    max_new = int(cfg.get("higgs_max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    transcript = _transcript_for(lang, reference) if ref_wav else ""
    if ref_wav and not transcript:
        log("  ⚠ Higgs: no transcript for the reference clip — using a generic "
            "voice. Add \"reference_text\" in voices.json (or a .txt next to the "
            "clip) to clone your voice.")

    model_name = str(cfg.get("higgs_model") or DEFAULT_MODEL)
    out, made = [], 0
    for s in scenes:
        text = _norm(s.narration)
        p = cache / f"hg_{lang}_{s.n:03d}_{_key(text, reference, lang, model_name, float(o['temperature']))}.wav"
        if not p.exists() or p.stat().st_size < 1024:
            _synth_one(engine, text, ref_wav, transcript, scene_desc, max_new, o, p, log)
            made += 1
            log(f"S{s.n:>3} voiced  ({text[:52]}...)")
        out.append(p)
    log(f"Higgs: {made} generated, {len(scenes) - made} from cache "
        f"({device_info(cfg)['device']}).")
    return out


def _synth_one(engine, text, ref_wav, transcript, scene, max_new, o, out, log):
    import numpy as np
    sample = _messages(text, ref_wav, transcript, scene)

    def _one(i: int):
        temp = float(o["temperature"])
        if i >= max(1, int(o.get("best_of", 1))):   # steadier on a retry
            temp = max(0.3, temp - 0.1 * (i - int(o.get("best_of", 1)) + 1))
        seed = _vc.seed_for(text) + i
        # Extra knobs (seed, RAS anti-repetition) are passed defensively: some
        # serve-engine builds accept them, others don't — never let that stop us.
        for extra in ({"seed": seed, "ras_win_len": 7, "ras_win_max_num_repeat": 2},
                      {"seed": seed}, {}):
            try:
                resp = engine.generate(
                    chat_ml_sample=sample, max_new_tokens=max_new,
                    temperature=temp, top_p=float(o["top_p"]), top_k=int(o["top_k"]),
                    stop_strings=STOP_STRINGS, **extra)
                break
            except TypeError:
                continue
        else:                                        # pragma: no cover
            resp = engine.generate(chat_ml_sample=sample, max_new_tokens=max_new,
                                   temperature=temp)
        audio = getattr(resp, "audio", None)
        sr = int(getattr(resp, "sampling_rate", 24000) or 24000)
        if audio is None:
            return None, sr, True
        samples = np.asarray(audio, dtype="float32").squeeze()
        return samples, sr, False                    # RAS handles repetition inside

    samples, sr, ok, taken = _vc.best_take(
        _one, text, best_of=int(o.get("best_of", 1)), retries=int(o.get("retries", 2)), log=log)
    if samples is None:
        raise HiggsError(f"Higgs returned no audio for: {text[:60]}...")
    _save_wav(samples, sr, out)


def _save_wav(samples, sr: int, out: Path) -> Path:
    """Write a float array to a 16-bit mono wav (no torchaudio dependency)."""
    import wave

    import numpy as np
    data = np.clip(np.asarray(samples, dtype="float32").squeeze(), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return out
