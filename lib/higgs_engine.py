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
DEFAULT_ASR_MODEL = "openai/whisper-small"   # for auto-transcribing new clips
STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]

# One loaded engine (and one ASR), kept alive across scenes (load is the cost).
_ENGINE = {"obj": None, "key": None, "device": None}
_ASR = {"obj": None, "key": None}


class HiggsError(RuntimeError):
    """Higgs could not produce audio. The caller keeps whatever it can."""


# ------------------------------------------------------------- availability

_INSTALLED = None


def installed() -> bool:
    """Is Higgs actually USABLE? We check the serve engine module specifically,
    not just the top package — a `pip install git+…` wheel builds
    `boson_multimodal` but can drop the `serve` subpackage, and claiming 'ready'
    then errors a whole voice job instead of falling back to Chatterbox."""
    global _INSTALLED
    if _INSTALLED is None:
        import importlib.util
        try:
            _INSTALLED = importlib.util.find_spec(
                "boson_multimodal.serve.serve_engine") is not None
        except Exception:
            _INSTALLED = False
    return _INSTALLED


def install_hint() -> str:
    return ("Higgs Audio isn't fully installed (the serve engine is missing — a\n"
            "'pip install git+…' wheel can drop it). Install it EDITABLE from a\n"
            "clone so every sub-package is included, into THIS project's venv:\n"
            "  git clone https://github.com/boson-ai/higgs-audio.git\n"
            "  .venv/bin/pip install -e higgs-audio          # macOS/Linux\n"
            "  .venv\\Scripts\\pip install -e higgs-audio      # Windows\n"
            "then set \"voice_engine\": \"higgs\" in config.json and reload.")


def available(cfg: dict | None = None) -> bool:
    return usable()


# Once Higgs fails to load/run on THIS machine (e.g. an MPS build that can't run
# the model), it's marked unusable for the rest of the session so the WHOLE
# pipeline — synth AND the status/render lookups — agrees on Chatterbox. Without
# this, voice would fall back to Chatterbox per call while the dashboard still
# looked for Higgs' files, and report 'not voiced' even though it was.
_UNUSABLE = False
_UNUSABLE_REASON = ""


def usable() -> bool:
    return installed() and not _UNUSABLE


def unusable_reason() -> str:
    return _UNUSABLE_REASON


def mark_unusable(reason: str = "") -> None:
    global _UNUSABLE, _UNUSABLE_REASON
    _UNUSABLE = True
    _UNUSABLE_REASON = reason or "failed to run on this machine"


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

def _auto_transcribe(clip_path: Path, lang: str, cfg: dict, log) -> str:
    """Best-effort ASR so cloning is automatic when a new clip is added: Whisper
    via `transformers` (already a Higgs dependency), run ONCE per clip. Returns
    the transcript, or "" if transcription isn't available or fails — the caller
    then falls back to a generic voice, never an error."""
    try:
        from transformers import pipeline
    except Exception:
        return ""
    model = str(cfg.get("higgs_asr_model") or DEFAULT_ASR_MODEL)
    dev = best_device(cfg)
    key = (model, dev)
    if _ASR["obj"] is None or _ASR["key"] != key:
        try:
            log(f"  Higgs: transcribing the reference clip once with {model} …")
            _ASR["obj"] = pipeline("automatic-speech-recognition", model=model,
                                   device=(0 if dev == "cuda" else -1))
            _ASR["key"] = key
        except Exception as e:
            log(f"  (auto-transcript unavailable: {e})")
            return ""
    try:
        gen = {"language": lang} if lang else {}
        out = _ASR["obj"](str(clip_path), generate_kwargs=gen, chunk_length_s=30)
        return (out.get("text") or "").strip()
    except Exception as e:
        log(f"  (auto-transcript failed: {e})")
        return ""


def sibling_transcript(reference: str) -> str:
    """The transcript already saved next to a clip (the .txt), if any. Never runs
    ASR — for display in the Voices panel only, so the box shows the transcript
    that's actually in use (auto-generated or hand-written)."""
    try:
        side = V.resolve(reference).with_suffix(".txt")
        if side.exists():
            return side.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _transcript_for(lang: str, reference: str, cfg: dict | None = None, log=print) -> str:
    """The words spoken in the reference clip, needed to clone the voice. In order
    of preference: voices.json "reference_text", a sibling .txt next to the clip,
    then a one-time auto-transcription (cached as that .txt). "" means no
    transcript → the caller uses a generic voice."""
    txt = (V.pref_for(lang).get("reference_text") or "").strip()
    if txt:
        return txt
    try:
        src = V.resolve(reference)                 # the raw clip on disk
    except Exception:
        return ""
    side = src.with_suffix(".txt")
    if side.exists():
        cached = side.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    got = _auto_transcribe(src, lang, cfg or {}, log)
    if got:
        try:
            side.write_text(got, encoding="utf-8")  # cache + let the user edit it
            log(f"  Higgs: saved the transcript to {side.name} "
                f"(edit it if a word is off).")
        except Exception:
            pass
    return got


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
    transcript = _transcript_for(lang, reference, cfg, log) if ref_wav else ""
    if ref_wav and not transcript:
        log("  ⚠ Higgs: no transcript for the reference clip and auto-transcribe "
            "wasn't available — using a generic voice. Add \"reference_text\" in "
            "the Voices panel (or a .txt next to the clip) to clone your voice.")

    model_name = str(cfg.get("higgs_model") or DEFAULT_MODEL)
    out, made = [], 0
    for s in scenes:
        text = _norm(s.narration)
        p = cache / f"hg_{lang}_{s.n:03d}_{_key(text, reference, lang, model_name, float(o['temperature']))}.wav"
        if not p.exists() or p.stat().st_size < 1024:
            _synth_one(engine, text, ref_wav, transcript, scene_desc, max_new, o, p, log)
            made += 1
            log(f"S{s.n:>3} voiced  ({text[:52]}...)")
        else:
            log(f"S{s.n:>3} cached  ({text[:52]}...)")
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
    # A device that can't actually run the model (e.g. some MPS builds) returns
    # near-silence for every take. Rather than write empty clips the render then
    # can't find, raise so the caller falls back to Chatterbox for the session.
    import numpy as np
    if float(np.sqrt(np.mean(np.square(np.asarray(samples, dtype="float32"))))) < 0.003:
        raise HiggsError("produced silent audio — this device likely can't run "
                         "the Higgs model (Higgs needs an NVIDIA/CUDA GPU).")
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
