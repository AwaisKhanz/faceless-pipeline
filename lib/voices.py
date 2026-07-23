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
            return json.loads(PREFS.read_text(encoding="utf-8"))
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
#
# Reference clips live in per-language folders:
#
#     voices_refs/
#       en/  warm-documentary-male.mp3
#       de/  ruhige-erzaehlerin.mp3
#       es/  narrador-calido.mp3
#
# Files left loose in voices_refs/ still work — they show up as "unsorted" and
# are offered for every language, so nothing breaks if you just drop a file in.
# organise() tidies them away when you ask it to.

REFS = ROOT / "voices_refs"
PREPARED = ROOT / "cache" / "refs"        # normalised copies, not user files
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")

# Words that betray a language when a file is sitting loose in voices_refs/.
# Only used to suggest a home during organise() — never to guess silently.
_LANG_HINTS = {
    "en": ("english", "eng", "_en", "-en"),
    "de": ("german", "deutsch", "ger", "_de", "-de"),
    "es": ("spanish", "espanol", "español", "spa", "_es", "-es"),
    "fr": ("french", "francais", "français", "_fr", "-fr"),
    "it": ("italian", "italiano", "_it", "-it"),
    "pt": ("portuguese", "portugues", "português", "_pt", "-pt"),
    "nl": ("dutch", "nederlands", "_nl", "-nl"),
    "pl": ("polish", "polski", "_pl", "-pl"),
    "ru": ("russian", "_ru", "-ru"),
    "tr": ("turkish", "turkce", "türkçe", "_tr", "-tr"),
    "ja": ("japanese", "_ja", "-ja"),
    "zh": ("chinese", "mandarin", "_zh", "-zh"),
}


def label_for(name: str) -> str:
    """A readable name for a file. 'warm-documentary-male.mp3' -> 'Warm documentary male'.

    The filename stays the identity — this is only what you read in the panel.
    Renaming a file to something descriptive is the whole point of the folder
    layout, and this makes that effort visible.
    """
    stem = Path(name).stem
    words = stem.replace("_", " ").replace("-", " ").split()
    if not words:
        return stem
    out = " ".join(words)
    return out[0].upper() + out[1:]


def guess_lang(name: str) -> str:
    """Which language a loose file probably belongs to, or '' if unclear."""
    low = Path(name).stem.lower()
    for code, hints in _LANG_HINTS.items():
        if any(h in low for h in hints):
            return code
    return ""


def ensure_folders(langs=("en", "de", "es", "fr", "it", "pt")) -> None:
    """Make the language folders exist so there is somewhere obvious to drop files."""
    for code in langs:
        (REFS / code).mkdir(parents=True, exist_ok=True)


def _duration(f: Path) -> float:
    import subprocess
    try:
        return float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(f)],
            capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        return 0.0


def _describe(f: Path, lang: str) -> dict:
    rel = f.relative_to(REFS).as_posix()
    d = _duration(f)
    return {
        "rel": rel,                    # the identity: "en/warm-male.mp3"
        "name": f.name,
        "label": label_for(f.name),
        "lang": lang,                  # "" means loose in voices_refs/
        "lang_name": LANGS.get(lang, "") if lang else "",
        "seconds": round(d, 1),
        "short": 0 < d < 8,            # under 8s clones poorly
        "suggest": guess_lang(f.name) if not lang else "",
    }


def references(lang: str | None = None) -> list[dict]:
    """Reference clips, optionally narrowed to one language.

    With a language: that folder's clips first, then any loose files (which
    could belong to anything, so they stay on offer).
    Without: everything, so Settings can show the whole library.
    """
    REFS.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []

    def scan(folder: Path, code: str) -> list[dict]:
        if not folder.is_dir():
            return []
        return sorted(
            (_describe(f, code) for f in folder.iterdir()
             if f.is_file() and f.suffix.lower() in AUDIO
             and not f.name.startswith(".")
             # Normalised copies are generated, not chosen. Older versions
             # wrote them alongside the originals; they live in cache/ now,
             # but any left over must not appear as pickable voices.
             and not f.stem.endswith("_prepared")),
            key=lambda d: d["label"].lower())

    if lang:
        out += scan(REFS / lang, lang)
    else:
        for d in sorted(REFS.iterdir()):
            if d.is_dir() and d.name in LANGS:
                out += scan(d, d.name)

    out += scan(REFS, "")              # loose files, always included
    return out


def resolve(ref: str) -> Path:
    """The file a saved preference points at.

    Accepts "en/warm-male.mp3" and a bare "warm-male.mp3". The bare form is
    what older versions saved, and is still what you get if you drop a file
    straight into voices_refs/ — so both have to keep working.
    """
    if not ref:
        raise FileNotFoundError("No reference clip chosen.")
    direct = REFS / ref
    if direct.is_file():
        return direct
    name = Path(ref).name
    for cand in [REFS / name] + [REFS / c / name for c in LANGS]:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Reference clip not found: {ref}")


def organise() -> list[str]:
    """Move loose clips into language folders where the name makes it obvious.

    Only moves files it can place confidently, and never overwrites. Anything
    ambiguous is left exactly where it is rather than guessed at — a clip in
    the wrong folder is worse than one that is merely untidy.
    """
    ensure_folders()
    moved = []
    for f in sorted(REFS.iterdir()):
        if not (f.is_file() and f.suffix.lower() in AUDIO):
            continue
        if f.stem.endswith("_prepared"):
            continue
        code = guess_lang(f.name)
        if not code:
            continue
        dest = REFS / code / f.name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.rename(dest)
        moved.append(f"{f.name} -> {code}/")
    return moved


def status(lang: str) -> dict:
    """Everything the panel needs to know about a language's readiness."""
    from . import chatterbox_engine as CB
    p = pref_for(lang)
    ok = False
    if p["reference"]:
        try:
            resolve(p["reference"])
            ok = True
        except FileNotFoundError:
            ok = False
    return {
        "installed": CB.installed(),
        "supported": supported(lang),
        "reference": p["reference"],
        "reference_label": label_for(p["reference"]) if p["reference"] else "",
        "reference_ok": ok,
        "device": CB.best_device(),
        "count": len(references(lang)),
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
    try:
        ref = resolve(reference)
    except FileNotFoundError as e:
        raise RuntimeError(str(e))

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
