"""Chatterbox voice engine — local cloning, MIT licensed, no per-character cost.

Nothing here was testable on the machine it was written on (no PyTorch, no model
download), so every failure path is deliberately loud and specific rather than
clever. If something is wrong you should get a sentence telling you what to do,
not a stack trace.

REFERENCE AUDIO: whatever you point this at becomes your channel's voice, so it
has to be something you hold rights to — your own recording, a CC0 clip from
Mozilla Common Voice, or a public-domain LibriVox reading.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "voices_refs"          # drop reference clips here
CACHE = ROOT / "cache" / "voice"

# Calm documentary narration: low exaggeration keeps it from performing at the
# listener, which is wrong for this audience.
DEFAULTS = {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.7}

# Languages the multilingual model speaks.
SUPPORTED = ("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
             "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh")

_MODEL = {"obj": None, "kind": None, "device": None}


class ChatterboxError(RuntimeError):
    pass


def lang_id(lang: str) -> str:
    """Model language id for a pipeline language code."""
    code = lang.lower().split("-")[0]
    if code not in SUPPORTED:
        raise ChatterboxError(
            f"Chatterbox cannot speak '{lang}'. It supports: {', '.join(SUPPORTED)}")
    return code


# --------------------------------------------------------------- environment

def installed() -> bool:
    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError:
        return False


def install_hint() -> str:
    return ("Chatterbox is not installed. Run:\n"
            "    bash setup.sh\n"
            "It pulls in PyTorch and downloads ~3 GB of model files on first use.\n"
            "(Plain `pip install` will fail — Homebrew's Python blocks it.)")


def best_device() -> str:
    """Apple Silicon GPU if we can, otherwise CPU."""
    if os.environ.get("FACELESS_DEVICE"):
        return os.environ["FACELESS_DEVICE"]
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _enable_mps_fallback() -> None:
    """Let unimplemented Metal ops silently run on the CPU instead of crashing.

    Chatterbox uses operations PyTorch has not implemented for Metal. Without
    this the model refuses to load with 'not currently implemented for the MPS
    device'. With it, those few ops run on CPU and everything else stays on the
    GPU — far faster than forcing the whole model to CPU.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def load_model(multilingual: bool = True, device: str | None = None):
    """Load once and keep it — startup is most of the cost on the first line."""
    dev = device or best_device()
    kind = "mtl" if multilingual else "en"
    if _MODEL["obj"] is not None and _MODEL["kind"] == kind and _MODEL["device"] == dev:
        return _MODEL["obj"]

    if not installed():
        raise ChatterboxError(install_hint())

    try:
        if multilingual:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M
        else:
            from chatterbox.tts import ChatterboxTTS as M
    except ImportError as e:
        raise ChatterboxError(
            f"Chatterbox is installed but its modules could not be imported ({e}).\n"
            f"Try:  python3 -m pip install --upgrade --force-reinstall chatterbox-tts")

    if dev == "mps":
        _enable_mps_fallback()

    import traceback as _tb

    attempts: list[tuple[str, Exception, str]] = []
    for candidate in ([dev] if dev == "cpu" else [dev, "cpu"]):
        try:
            model = M.from_pretrained(device=candidate)
            _MODEL.update(obj=model, kind=kind, device=candidate)
            return model
        except Exception as e:
            # Keep the ORIGINAL traceback. Wrapping an exception without it
            # hides the line inside the library that actually failed, which is
            # the only part anyone needs.
            attempts.append((candidate, e, _tb.format_exc()))

    detail = "\n".join(
        f"  on {d}: {type(e).__name__}: {str(e).strip().splitlines()[0][:300]}"
        for d, e, _ in attempts)

    joined = " ".join(str(e) for _, e, _ in attempts).lower()
    cpu_cmd = ("set FACELESS_DEVICE=cpu && python make_video.py benchtts --lang en"
               if os.name == "nt" else
               "FACELESS_DEVICE=cpu python3 make_video.py benchtts --lang en")
    if "no kernel image" in joined or "sm_" in joined:
        hint = ("\nYour PyTorch build does not have kernels for this GPU — it is "
                "newer than\nthe build supports. Install the CUDA 12.8 build:\n"
                "    pip install --force-reinstall torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cu128")
    elif "mps" in joined or "not implemented" in joined:
        hint = ("\nThis looks like a Metal (GPU) limitation. Force the CPU with:\n"
                f"    {cpu_cmd}")
    elif "out of memory" in joined:
        hint = ("\nOut of memory. Close other apps, or force the CPU:\n"
                f"    {cpu_cmd}")
    elif "nonetype" in joined:
        hint = ("\n'NoneType is not callable' almost always means a dependency "
                "version clash\nrather than anything about your machine. The "
                "traceback below names the file.")
    else:
        hint = ""

    err = ChatterboxError(f"Could not load Chatterbox.\n{detail}{hint}")
    err.tracebacks = attempts          # surfaced by --verbose
    raise err


def device_in_use() -> str:
    return _MODEL["device"] or best_device()


def device_info() -> dict:
    """Name and memory of whatever will do the work, for `doctor`.

    On a discrete NVIDIA card the memory figure is the card's own VRAM, which is
    the number that matters: the model lives there instead of competing with the
    operating system for RAM.
    """
    d = {"device": best_device(), "name": None, "vram_gb": None, "note": ""}
    try:
        import torch
        if d["device"] == "cuda":
            d["name"] = torch.cuda.get_device_name(0)
            d["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            cap = torch.cuda.get_device_capability(0)
            d["note"] = f"sm_{cap[0]}{cap[1]}"
        elif d["device"] == "mps":
            d["name"] = "Apple GPU"
            d["note"] = "shares system memory — heavy jobs can stall the Mac"
        else:
            d["name"] = "CPU"
            d["note"] = "no GPU found — voicing will be slow"
    except Exception as e:
        d["note"] = f"could not query: {e}"
    return d


# ----------------------------------------------------------------- reference

def prepare_reference(src: Path, out: Path | None = None) -> Path:
    """Normalise a reference clip: mono, 24 kHz, silence trimmed, level evened.

    Cloning quality depends far more on a clean reference than on clever settings.
    """
    src = Path(src)
    if not src.exists():
        raise ChatterboxError(f"Reference clip not found: {src}")
    out = out or (REFS / f"{src.stem}_prepared.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Already prepared and still newer than the source? Nothing to redo.
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-45dB,"
                "areverse,silenceremove=start_periods=1:start_silence=0.1:"
                "start_threshold=-45dB,areverse,loudnorm=I=-18:TP=-2",
         "-ar", "24000", "-ac", "1", str(out)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise ChatterboxError(f"Could not prepare {src.name}:\n"
                              f"{r.stderr.strip().splitlines()[-3:]}")
    return out


def list_references() -> list[dict]:
    REFS.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(REFS.glob("*")):
        if f.suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac"):
            continue
        if f.stem.endswith("_prepared"):     # our own normalised copies
            continue
        try:
            d = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(f)],
                capture_output=True, text=True).stdout.strip())
        except Exception:
            d = 0.0
        out.append({"name": f.name, "path": str(f), "seconds": round(d, 1),
                    "short": d < 8})
    return out


# ---------------------------------------------------------------------- tts

def _key(text: str, ref: str, lang: str, opts: dict) -> str:
    blob = f"{ref}|{lang}|{sorted(opts.items())}|{text}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _free_device_memory() -> None:
    """Release cached GPU allocations between lines.

    Chatterbox leaves intermediate tensors cached on the device after each
    generation. On Apple Silicon that memory comes out of the same unified pool
    the OS uses, so after a few lines macOS begins swapping and sampling
    throughput collapses — measured here as 27 it/s on the first line falling to
    2.5 it/s by the third, with no relation to line length. Emptying the cache
    between lines keeps every line roughly as fast as the first.

    Best-effort by design: if the torch build has no cache API, slow is still
    better than crashing.
    """
    try:
        import torch
        dev = _MODEL["device"]
        if dev == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _save_wav(wav, sample_rate: int, out: Path) -> Path:
    """Write a torch tensor to a .wav file.

    Deliberately does NOT use torchaudio.save. As of torchaudio 2.11 that
    delegates to TorchCodec, an extra dependency Chatterbox does not install, so
    it fails with "TorchCodec is required for save_with_torchcodec".

    soundfile is already present (librosa depends on it, and Chatterbox depends
    on librosa). The stdlib `wave` module is the last resort so this can never be
    the thing that fails — writing a WAV is not worth a dependency.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    # torch tensor -> 1-D float array in [-1, 1]
    data = wav.detach().cpu().squeeze().numpy() if hasattr(wav, "detach") else wav
    if getattr(data, "ndim", 1) > 1:
        data = data.reshape(-1)

    try:
        import soundfile as sf
        sf.write(str(out), data, int(sample_rate), subtype="PCM_16")
        return out
    except Exception as sf_err:
        try:
            import wave
            import numpy as np
            clipped = np.clip(data, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype("<i2")
            with wave.open(str(out), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(sample_rate))
                w.writeframes(pcm.tobytes())
            return out
        except Exception as wave_err:
            raise ChatterboxError(
                f"Generated the audio but could not write {out.name}.\n"
                f"  soundfile: {type(sf_err).__name__}: {sf_err}\n"
                f"  wave:      {type(wave_err).__name__}: {wave_err}")


def synth_one(text: str, ref_wav: Path, lang: str, out: Path,
              opts: dict | None = None) -> Path:
    if not installed():
        raise ChatterboxError(install_hint())
    o = {**DEFAULTS, **(opts or {})}
    multilingual = lang.lower() != "en"
    model = load_model(multilingual=multilingual)

    kw = dict(audio_prompt_path=str(ref_wav),
              exaggeration=o["exaggeration"], cfg_weight=o["cfg_weight"],
              temperature=o["temperature"])
    if multilingual:
        kw["language_id"] = lang_id(lang)

    # Inference needs no autograd graph, and building one holds every
    # intermediate activation alive — the single largest avoidable allocation
    # per line. Harmless if Chatterbox already does this internally.
    import torch
    try:
        with torch.no_grad():
            wav = model.generate(text, **kw)
    except TypeError as e:
        # Only retry when this build rejected one of our tuning knobs. A
        # TypeError from *inside* generation is a real fault and must surface —
        # silently retrying it would change the voice without saying so.
        if "unexpected keyword" not in str(e):
            raise
        kw = {"audio_prompt_path": str(ref_wav)}
        if multilingual:
            kw["language_id"] = lang_id(lang)
        with torch.no_grad():
            wav = model.generate(text, **kw)

    # Some builds hand back (audio, sample_rate) instead of a bare tensor.
    sr = getattr(model, "sr", 24000)
    if isinstance(wav, (tuple, list)) and len(wav) == 2:
        wav, sr = wav

    if wav is None:
        raise ChatterboxError(
            f"Chatterbox returned no audio for: {text[:60]}...")
    try:
        return _save_wav(wav, sr, out)
    finally:
        # After the tensor is on disk nothing needs it on the GPU any more.
        del wav
        _free_device_memory()


def prepared_name(reference: str) -> str:
    """What prepare_reference() will call its output, without running ffmpeg."""
    return f"{Path(reference).stem}_prepared.wav"


def expected_paths(scenes, lang: str, reference: str, cache: Path = CACHE,
                   opts: dict | None = None) -> list[Path]:
    """Where each scene's audio WOULD be cached, generating nothing.

    Must stay in step with synth() below — same key, same filename. Used by the
    dashboard to report how much of a language is already voiced without
    loading a 3 GB model to find out.
    """
    o = {**DEFAULTS, **(opts or {})}
    ref_name = prepared_name(reference)
    return [cache / f"cb_{lang}_{s.n:03d}_{_key(s.narration, ref_name, lang, o)}.wav"
            for s in scenes]


def synth(scenes, lang: str, ref_wav: Path, cache: Path = CACHE,
          opts: dict | None = None, log=print) -> list[Path]:
    """One audio file per scene, cached exactly like the other engines."""
    ref_wav = Path(ref_wav)
    if not ref_wav.exists():
        raise ChatterboxError(f"Reference clip missing: {ref_wav}")
    o = {**DEFAULTS, **(opts or {})}
    cache.mkdir(parents=True, exist_ok=True)
    out, made = [], 0
    for s in scenes:
        k = _key(s.narration, ref_wav.name, lang, o)
        p = cache / f"cb_{lang}_{s.n:03d}_{k}.wav"
        if not p.exists() or p.stat().st_size < 1024:
            synth_one(s.narration, ref_wav, lang, p, o)
            made += 1
            log(f"S{s.n:>3} voiced  ({s.narration[:52]}...)")
        out.append(p)
    log(f"Chatterbox: {made} generated, {len(scenes) - made} from cache "
        f"({device_in_use()}).")
    return out


# ----------------------------------------------------------------- benchmark

def benchmark(lines: list[str], ref_wav: Path, lang: str = "en",
              log=print) -> dict:
    """Time a handful of real lines and extrapolate honestly.

    Worth doing before committing: if this runs at 30s a line, a 115-scene video
    in three languages is the better part of three hours and the whole approach
    is a non-starter regardless of how good it sounds.
    """
    if not installed():
        raise ChatterboxError(install_hint())
    ref_wav = Path(ref_wav)
    if not ref_wav.exists():
        raise ChatterboxError(f"Reference clip missing: {ref_wav}")

    log("Loading the model (first run also downloads it — be patient)…")
    t0 = time.time()
    load_model(multilingual=(lang.lower() != "en"))
    load_s = time.time() - t0
    log(f"  model ready in {load_s:.0f}s on {device_in_use()}")

    tmp = ROOT / "work" / "_bench"
    tmp.mkdir(parents=True, exist_ok=True)
    times, chars = [], []
    for i, line in enumerate(lines, start=1):
        t = time.time()
        synth_one(line, ref_wav, lang, tmp / f"b{i}.wav")
        dt = time.time() - t
        times.append(dt)
        chars.append(len(line))
        log(f"  line {i}: {dt:5.1f}s for {len(line)} chars")

    # Report seconds-per-character, not seconds-per-line. Line lengths vary by
    # 2x in a real script, so a per-line mean tells you very little.
    rates = [t / max(1, c) for t, c in zip(times, chars)]

    # The FIRST line is always slower — the GPU compiles kernels and warms its
    # caches on the first real call. That cost is paid once per session, not per
    # line, so including it would understate a fast machine badly. Measure the
    # steady state and report the warm-up separately.
    warmup_s = times[0] if len(times) > 1 else 0.0
    steady = rates[1:] if len(rates) > 2 else rates

    ordered = sorted(steady)
    mid = len(ordered) // 2
    median_rate = (ordered[mid] if len(ordered) % 2
                   else (ordered[mid - 1] + ordered[mid]) / 2)

    # Degradation means later lines are slower than earlier ones — the signature
    # of the machine running out of memory as it goes. A plain max/min spread
    # cannot tell that apart from a single slow warm-up line, which is the
    # opposite situation and perfectly healthy. So compare halves, in order.
    half = max(1, len(steady) // 2)
    first_half = sum(steady[:half]) / half
    second_half = sum(steady[half:]) / max(1, len(steady) - half)
    drift = second_half / first_half if first_half > 0 else 1.0

    avg_chars = sum(chars[1:]) / len(chars[1:]) if len(chars) > 1 else chars[0]
    return {"device": device_in_use(), "load_s": load_s,
            "per_line_s": median_rate * avg_chars, "per_char_s": median_rate,
            "times": times, "chars": chars, "rates": rates,
            "warmup_s": warmup_s, "drift": drift, "degrading": drift >= 2.0}
