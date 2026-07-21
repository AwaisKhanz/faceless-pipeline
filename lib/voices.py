"""Voice configuration — the single source of truth for how each language is read.

One engine: Chatterbox. It clones a reference clip, runs locally on your Mac, is
MIT licensed and costs nothing per character.

Choices live in voices.json at the project root, so they survive updates and are
shared by the studio and the command line.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFS = ROOT / "voices.json"
PREVIEWS = ROOT / "cache" / "previews"

# Languages the multilingual model speaks.
LANGS = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

# Calm documentary narration for a 60+ audience: low expression keeps it from
# performing at the listener, which is wrong for this material.
DEFAULT_EXAGGERATION = 0.4
DEFAULT_CFG = 0.5

FALLBACK_LINE = {
    "en": "You're wide awake, and it's still dark. You know, without even "
          "looking, that the clock is going to say something close to three.",
    "de": "Sie sind hellwach, und es ist noch dunkel. Sie wissen, ohne auch nur "
          "hinzusehen, dass die Uhr etwas nahe bei drei zeigen wird.",
    "es": "Está completamente despierto, y todavía es de noche. Sabe, sin "
          "siquiera mirar, que el reloj marcará algo cercano a las tres.",
}


# --------------------------------------------------------------- preferences

def _read() -> dict:
    if PREFS.exists():
        try:
            return json.loads(PREFS.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def pref_for(lang: str) -> dict:
    """Settings for a language, with defaults filled in."""
    p = _read().get(lang, {})
    return {
        "reference": p.get("reference", ""),
        "exaggeration": float(p.get("exaggeration", DEFAULT_EXAGGERATION)),
        "cfg_weight": float(p.get("cfg_weight", DEFAULT_CFG)),
    }


def save_pref(lang: str, **kw) -> dict:
    """Merge changes for one language. Unspecified fields keep their value."""
    prefs = _read()
    cur = pref_for(lang)
    cur.update({k: v for k, v in kw.items()
                if v is not None and k in cur})
    prefs[lang] = cur
    PREFS.write_text(json.dumps(prefs, indent=2) + "\n", encoding="utf-8")
    return cur


def all_prefs() -> dict:
    return {k: pref_for(k) for k in _read()}


def supported(lang: str) -> bool:
    return lang.lower().split("-")[0] in LANGS


# ----------------------------------------------------------------- catalogue

def references() -> list[dict]:
    from . import chatterbox_engine as CB
    return CB.list_references()


def status(lang: str) -> dict:
    """Everything the panel needs to know about a language's readiness."""
    from . import chatterbox_engine as CB
    p = pref_for(lang)
    ref = CB.REFS / p["reference"] if p["reference"] else None
    return {
        "installed": CB.installed(),
        "supported": supported(lang),
        "reference": p["reference"],
        "reference_ok": bool(ref and ref.exists()),
        "device": CB.best_device(),
    }


# ------------------------------------------------------------------ previews

def sample_line(lang: str, scenes=None) -> str:
    """A real line from the script — auditioning on 'hello world' tells you
    nothing about how a voice handles your actual writing."""
    if scenes:
        mid = [s for s in scenes if 12 <= len(s.narration.split()) <= 34]
        if mid:
            return mid[len(mid) // 2].narration
        return scenes[min(4, len(scenes) - 1)].narration
    return FALLBACK_LINE.get(lang, FALLBACK_LINE["en"])


def preview(text: str, lang: str, reference: str,
            exaggeration: float = DEFAULT_EXAGGERATION,
            cfg_weight: float = DEFAULT_CFG) -> Path:
    """Render (or reuse) a sample. Returns the audio path."""
    from . import chatterbox_engine as CB
    if not reference:
        raise RuntimeError("Pick a reference clip first.")
    ref = CB.REFS / reference
    if not ref.exists():
        raise RuntimeError(f"Reference clip not found: {reference}")

    PREVIEWS.mkdir(parents=True, exist_ok=True)
    sig = f"{reference}|{lang}|{exaggeration}|{cfg_weight}|{text}"
    out = PREVIEWS / f"{_hash(sig)}.wav"
    if out.exists() and out.stat().st_size > 1024:
        return out
    return CB.synth_one(text, CB.prepare_reference(ref), lang, out,
                        {"exaggeration": exaggeration, "cfg_weight": cfg_weight})


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def clear_previews() -> int:
    if not PREVIEWS.exists():
        return 0
    n = 0
    for f in PREVIEWS.iterdir():
        if f.is_file():
            f.unlink()
            n += 1
    return n
