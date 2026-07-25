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


def _use_higgs(cfg: dict) -> bool:
    """True only when Higgs is both selected and actually installed."""
    if str(cfg.get("voice_engine", "chatterbox")).strip().lower() not in ("higgs", "higgs-audio"):
        return False
    from . import higgs_engine as HG
    return HG.installed()


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
    if not p["reference"]:
        return "NO REFERENCE SET"
    cfg = _config()
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
    if not V.supported(lang):
        raise SystemExit(
            f"Chatterbox cannot speak '{lang}'. It supports: "
            f"{', '.join(sorted(V.LANGS))}")

    cfg = _config()
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
            # Higgs was selected but couldn't run — don't dead-end the render;
            # fall through to Chatterbox and say why.
            log(f"  ⚠ Higgs couldn't run ({e}). Falling back to Chatterbox for "
                f"this language. {HG.install_hint()}")

    p = V.pref_for(lang)
    return CB.synth(scenes, lang, reference_for(lang, voice), cache,
                    {"exaggeration": p["exaggeration"],
                     "cfg_weight": p["cfg_weight"],
                     "temperature": p["temperature"],
                     "retries": p["retries"],
                     "best_of": p["best_of"]}, log=log)


def voice_paths(scenes, lang: str, cache: Path, voice: str | None = None) -> list[Path]:
    """Where this language's narration is (or would be) cached. Generates nothing.

    Returns an empty list when no reference clip has been chosen — that is not
    an error here, it just means nothing can have been voiced yet.
    """
    if not (voice or V.pref_for(lang)["reference"]) or not V.supported(lang):
        return []
    name = _raw_ref(lang, voice)
    if not name:
        return []
    cfg = _config()
    if _use_higgs(cfg):
        from . import higgs_engine as HG
        return HG.expected_paths(scenes, lang, name, cache,
                                 _higgs_opts(lang, cfg), cfg=cfg)
    p = V.pref_for(lang)
    return CB.expected_paths(scenes, lang, name, cache,
                             {"exaggeration": p["exaggeration"],
                              "cfg_weight": p["cfg_weight"]})


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
