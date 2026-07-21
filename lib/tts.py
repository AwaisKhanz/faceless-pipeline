"""Narration.

A thin front for the Chatterbox engine. `synth()` is the only function the
pipeline calls, so nothing downstream knows or cares how audio is produced.

One file per scene, cached by a hash of the exact text and settings — change a
word in scene 47 and only scene 47 is regenerated.
"""
from __future__ import annotations

from pathlib import Path

from . import chatterbox_engine as CB
from . import voices as V


def describe(lang: str) -> str:
    """One line describing how a language will be read — used in logs and doctor."""
    p = V.pref_for(lang)
    if not p["reference"]:
        return "NO REFERENCE SET"
    return (f"Chatterbox · {p['reference']} · "
            f"expression {p['exaggeration']:.2f} · guidance {p['cfg_weight']:.2f}")


def reference_for(lang: str, override: str | None = None) -> Path:
    """The prepared reference clip for a language, or a clear error saying why not."""
    name = override or V.pref_for(lang)["reference"]
    if not name:
        raise SystemExit(
            f"No reference clip set for '{lang}'.\n"
            f"Pick one in the studio's Voices panel, or put a clip in "
            f"voices_refs/ and choose it there.")
    ref = CB.REFS / name
    if not ref.exists():
        raise SystemExit(f"Reference clip missing: {ref}")
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

    p = V.pref_for(lang)
    return CB.synth(scenes, lang, reference_for(lang, voice), cache,
                    {"exaggeration": p["exaggeration"],
                     "cfg_weight": p["cfg_weight"]}, log=log)


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
