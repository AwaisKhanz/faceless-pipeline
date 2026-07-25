"""Narration — the front the pipeline calls, and the engine ROUTER.

`synth()` is the only function the pipeline uses, so nothing downstream knows or
cares which backend produced the audio. Two backends plug in behind the same
interface:

  * chatterbox (default) — MIT, ultra-stable, on any GPU/CPU.
  * higgs         — Apache-2.0, higher quality, heavier; used only when
                    config "voice_engine" is "higgs" AND it's installed.

Selection is per run from config; if Higgs is asked for but not installed, we
fall back to Chatterbox so a render never dead-ends. One file per scene, cached
by a hash of the exact text and settings — and the two engines use different
cache prefixes, so switching never mixes their clips.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import chatterbox_engine as CB
from . import voices as V

ROOT = Path(__file__).resolve().parent.parent


def _config() -> dict:
    """Read config.json directly (no import cycle with pipeline)."""
    try:
        f = ROOT / "config.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except Exception:
        return {}


# The three engines and the config `voice_engine` spellings that select each.
# One table so every place that maps a config value to an engine agrees.
_ENGINE_ALIASES = {
    "chatterbox": ("chatterbox", "cb", "default"),
    "higgs": ("higgs", "higgs-audio"),
    "chirp": ("chirp", "chirp3", "google", "google-tts", "gtts", "vertex-tts"),
}


def selected_engine(cfg: dict) -> str:
    """The engine the config ASKS for (canonical name), before checking whether it
    can actually run here. Unknown / unset → chatterbox."""
    raw = str((cfg or {}).get("voice_engine", "chatterbox")).strip().lower()
    for canon, aliases in _ENGINE_ALIASES.items():
        if raw in aliases:
            return canon
    return "chatterbox"


def active_engine(cfg: dict) -> str:
    """The engine that will ACTUALLY narrate here: the selected one if it's usable,
    otherwise chatterbox. usable() latches off after a runtime failure, so synth
    and the status/render lookups always agree on the same engine."""
    sel = selected_engine(cfg)
    if sel == "higgs":
        from . import higgs_engine as HG
        return "higgs" if HG.usable() else "chatterbox"
    if sel == "chirp":
        from . import gtts_engine as GT
        return "chirp" if GT.usable(cfg) else "chatterbox"
    return "chatterbox"


def engine_status(cfg: dict) -> dict:
    """One snapshot of the voice engines for the UI: which is selected, which is
    active, and each engine's readiness/reason. The single source of truth so the
    Voices panel, Status page and router never disagree."""
    from . import chatterbox_engine as CB
    higgs_installed = higgs_usable = False
    higgs_reason = ""
    google_ready = False
    google_reason = ""
    try:
        from . import higgs_engine as HG
        higgs_installed, higgs_usable, higgs_reason = HG.installed(), HG.usable(), HG.unusable_reason()
    except Exception:
        pass
    try:
        from . import gtts_engine as GT
        google_ready, google_reason = GT.usable(cfg), GT.unusable_reason()
    except Exception:
        pass
    return {
        "selected": selected_engine(cfg),
        "active": active_engine(cfg),
        "chatterbox_installed": CB.installed(),
        "higgs_installed": higgs_installed,
        "higgs_usable": higgs_usable,
        "higgs_reason": higgs_reason,
        "google_ready": google_ready,
        "google_reason": google_reason,
    }


def _use_higgs(cfg: dict) -> bool:
    return active_engine(cfg) == "higgs"


def _use_gtts(cfg: dict) -> bool:
    return active_engine(cfg) == "chirp"


def _gtts_opts(cfg: dict) -> dict:
    return {"speaking_rate": float(cfg.get("google_tts_rate", 1.0) or 1.0)}


def _google_voice(lang: str, voice: str | None, cfg: dict) -> str:
    """The Google catalogue voice for a language: an explicit override, else the
    saved google_voice, else Google's default for the locale."""
    if voice:
        return voice
    gv = V.pref_for(lang).get("google_voice") or ""
    if gv:
        return gv
    from . import gtts_engine as GT
    return GT.default_voice(lang, cfg)


def _cb_opts(lang: str) -> dict:
    """The Chatterbox knobs for a language, used by both synth and voice_paths so
    their cache keys can't drift. (expected_paths ignores retries/best_of — they
    aren't part of the voice's sound — so passing the full set is safe.)"""
    p = V.pref_for(lang)
    return {"exaggeration": p["exaggeration"], "cfg_weight": p["cfg_weight"],
            "temperature": p["temperature"], "retries": p["retries"],
            "best_of": p["best_of"]}


def _fallback_notice(log, engine: str, reason: str, tip: str) -> None:
    """The one loud, consistent 'that engine couldn't run — using Chatterbox for
    the rest of the session' block, shared by every engine's fallback path."""
    for line in ("",
                 "  ⚠ VOICE ENGINE FELL BACK TO CHATTERBOX",
                 f"      {engine} couldn't run here: {reason}",
                 "      Switched to Chatterbox for the rest of this session so this "
                 "render still completes.",
                 f"      {tip}",
                 ""):
        try:
            log(line)
        except Exception:
            pass


def _raw_ref(lang: str, voice: str | None) -> str:
    """A stable reference name shared by synth and voice_paths (so their cache
    keys match). Resolved to the voices_refs-relative path when possible."""
    name = voice or V.pref_for(lang)["reference"]
    if not name:
        return ""
    try:
        return V.resolve(name).relative_to(V.REFS).as_posix()
    except (FileNotFoundError, ValueError):
        return name


def _higgs_opts(lang: str, cfg: dict) -> dict:
    p = V.pref_for(lang)
    return {"temperature": p["temperature"], "retries": p["retries"],
            "best_of": p["best_of"],
            "top_p": float(cfg.get("higgs_top_p", 0.95)),
            "top_k": int(cfg.get("higgs_top_k", 50))}


def describe(lang: str) -> str:
    """One line describing how a language will be read — used in logs and doctor."""
    p = V.pref_for(lang)
    cfg = _config()
    # Google Chirp first — it needs no reference clip (Google supplies the voice),
    # so it must not trip the "NO REFERENCE SET" guard below. No network here:
    # just the stored voice name.
    if _use_gtts(cfg):
        gv = p.get("google_voice") or "(pick a voice in Voices)"
        return f"Google Chirp 3 HD · cloud · {gv}"
    if not p["reference"]:
        return "NO REFERENCE SET"
    if _use_higgs(cfg):
        from . import higgs_engine as HG
        return f"{HG.describe(lang, cfg)} · {p['reference']}"
    return (f"Chatterbox · {p['reference']} · "
            f"expression {p['exaggeration']:.2f} · guidance {p['cfg_weight']:.2f}")


def reference_for(lang: str, override: str | None = None) -> Path:
    """The prepared reference clip for a language, or a clear error saying why not."""
    name = override or V.pref_for(lang)["reference"]
    if not name:
        raise SystemExit(
            f"No reference clip chosen for '{lang}'.\n"
            f"Pick one in the studio's Voices panel, or drop a clip into "
            f"voices_refs/{lang}/ and choose it there.")
    try:
        ref = V.resolve(name)
    except FileNotFoundError:
        raise SystemExit(
            f"The clip chosen for '{lang}' is missing: {name}\n"
            f"It may have been moved or renamed. Choose another in the "
            f"Voices panel.")
    return CB.prepare_reference(ref)


def synth(scenes, lang: str, cache: Path, voice: str | None = None,
          rate: str | None = None, pitch: str | None = None,
          log=print) -> list[Path]:
    """Generate (or reuse) one audio file per scene. Returns paths in scene order.

    `voice` names a reference clip when given, overriding the saved choice.
    `rate` and `pitch` are accepted and ignored — Chatterbox has no equivalent
    knobs, and dropping them from the signature would break existing callers.
    """
    cfg = _config()

    # Google Chirp 3 HD — cloud, no reference clip, its own language coverage, so
    # it's handled before the Chatterbox-supported guard. On any failure it marks
    # itself unusable (so status/render agree) and falls through to Chatterbox.
    if _use_gtts(cfg):
        from . import gtts_engine as GT
        gv = _google_voice(lang, voice, cfg)
        try:
            log(f"  Voice engine: {describe(lang)}")
            log(f"  Google voice: {gv or '(none — pick one in Voices)'}")
        except Exception:
            pass
        try:
            return GT.synth(scenes, lang, None, cache, _gtts_opts(cfg),
                            log=log, cfg=cfg, reference=gv)
        except SystemExit:
            raise
        except Exception as e:
            GT.mark_unusable(str(e))
            _fallback_notice(log, "Google Chirp", str(e),
                             "Check the Text-to-Speech API is enabled on your "
                             "Vertex project and a voice is chosen.")

    if not V.supported(lang):
        raise SystemExit(
            f"Chatterbox cannot speak '{lang}'. It supports: "
            f"{', '.join(sorted(V.LANGS))}")

    # Say exactly what's about to narrate, so the Activity log is self-explanatory:
    # engine · model · device · reference voice.
    ref = _raw_ref(lang, voice)
    try:
        log(f"  Voice engine: {describe(lang)}")
        log(f"  Reference clip: {ref or '(none — pick one in Voices)'}")
    except Exception:
        pass
    if _use_higgs(cfg):
        from . import higgs_engine as HG
        try:
            # Same prepared reference clip; Higgs also needs its transcript, which
            # it looks up from the raw reference name we pass through.
            return HG.synth(scenes, lang, reference_for(lang, voice), cache,
                            _higgs_opts(lang, cfg), log=log, cfg=cfg,
                            reference=_raw_ref(lang, voice))
        except SystemExit:
            raise                                   # a real "pick a voice" message
        except Exception as e:
            # Higgs was selected but couldn't run here — mark it unusable for the
            # session (so status/render lookups also switch to Chatterbox and don't
            # report 'not voiced'), then fall through to Chatterbox below.
            HG.mark_unusable(str(e))
            _fallback_notice(log, "Higgs Audio", str(e),
                             "Higgs needs an NVIDIA (CUDA) GPU — on a Mac (MPS) it "
                             "always falls back. It'll run on your RTX box.")

    return CB.synth(scenes, lang, reference_for(lang, voice), cache,
                    _cb_opts(lang), log=log)


def voice_paths(scenes, lang: str, cache: Path, voice: str | None = None) -> list[Path]:
    """Where this language's narration is (or would be) cached. Generates nothing.

    Returns an empty list when no reference clip has been chosen — that is not
    an error here, it just means nothing can have been voiced yet.
    """
    cfg = _config()
    # Google Chirp: the "voice" is a Google catalogue name, not a file, so this
    # is checked first (the reference-clip guard below would otherwise return []).
    if _use_gtts(cfg):
        from . import gtts_engine as GT
        gv = voice or V.pref_for(lang).get("google_voice") or ""
        if not gv:
            return []
        return GT.expected_paths(scenes, lang, gv, cache, _gtts_opts(cfg), cfg=cfg)

    if not (voice or V.pref_for(lang)["reference"]) or not V.supported(lang):
        return []
    name = _raw_ref(lang, voice)
    if not name:
        return []
    if _use_higgs(cfg):
        from . import higgs_engine as HG
        return HG.expected_paths(scenes, lang, name, cache,
                                 _higgs_opts(lang, cfg), cfg=cfg)
    return CB.expected_paths(scenes, lang, name, cache, _cb_opts(lang))


def list_voices(lang: str | None = None) -> None:
    """Print the reference clips available to clone from."""
    refs = V.references()
    if not refs:
        print("No reference clips yet. Put one in voices_refs/ — 30 seconds of "
              "clean speech works best.")
        return
    for r in refs:
        note = "  ← under 8s, clones poorly" if r["short"] else ""
        print(f"{r['name']:<34} {r['seconds']:>6}s{note}")
