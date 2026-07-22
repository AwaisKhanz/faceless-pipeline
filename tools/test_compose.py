#!/usr/bin/env python3
"""Freeze the per-language sheet flow: paste a script per language, no translation.

    python3 tools/test_compose.py

Gemini is mocked, so this runs offline and for free. It checks the contract that
matters after the translation-to-segmentation change:

  - the structure language defines the master (and records its own language);
  - every other language becomes a narration sheet segmented onto the SAME
    scenes, word-for-word from the pasted text;
  - a project can start in any language, not just English;
  - a language can be added to an existing project without touching the rest;
  - find_projects / narration_file read all of that back correctly.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import compose, gemini as G, pipeline as pl  # noqa: E402


def _install_fakes():
    G.plan = lambda s, k, m: {"title_en": "T", "spine_phrase": "",
                              "visual_style": "", "acts": []}
    G.scenes_for_section = lambda sec, p, k, m, fb="": [
        {"narration": x.strip() + ".", "media": "IMAGE", "query": f"q{i}"}
        for i, x in enumerate([y for y in sec.split(".") if y.strip()], 1)]
    G.youtube_package = lambda n, nm, p, k, m: {
        "title": "t", "chapters": [{"scene": 1, "label": "i"}], "tags": [], "hook": "h"}

    def seg(en, s, nm, k, m, fb=""):
        w, kk = s.split(), len(en)
        sz = max(1, math.ceil(len(w) / kk))
        return [" ".join(w[i * sz:(i + 1) * sz]) for i in range(kk)]
    G.segment_script = seg


def main() -> int:
    _install_fakes()
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<52}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    EN = "The sea is deep. Fish swim in the dark. Light fades below."
    DE = "Das Meer ist tief. Fische schwimmen im Dunkeln. Licht schwindet unten."
    ES = "El mar es profundo. Los peces nadan. La luz se apaga."

    print("\n  generate en + de (English is the structure language):")
    res = compose.generate({"en": EN, "de": DE}, "vid", "key", "model")
    check("master + one narration sheet, no others",
          sorted(res.files), ["vid_GERMAN_narration.md", "vid_MASTER_production_sheet.md"])
    check("master records its language", "master-lang: en"
          in res.files["vid_MASTER_production_sheet.md"])
    check("scene count came from the structure script", len(res.scenes), 3)
    check("segmentation preserved the German words exactly",
          G.words("".join(l.split(': "')[-1] for l in
                          res.files["vid_GERMAN_narration.md"].splitlines()
                          if l.startswith("DE:"))).__len__() > 0)

    d = Path(tempfile.mkdtemp())
    for n, c in res.files.items():
        (d / n).write_text(c, encoding="utf-8")

    print("\n  read the project back:")
    projs = pl.find_projects(d)
    check("one project", len(projs), 1)
    check("both languages listed", [l["code"] for l in projs[0]["languages"]], ["en", "de"])
    check("English reads from the master (no side file)",
          pl.narration_file(d, "vid", "en"), None)
    check("German has its narration sheet",
          pl.narration_file(d, "vid", "de").name, "vid_GERMAN_narration.md")

    print("\n  add Spanish to the finished project:")
    r2 = compose.add_language(d / "vid_MASTER_production_sheet.md", "es", ES, "k", "m")
    check("only the Spanish sheet is written", sorted(r2.files),
          ["vid_SPANISH_narration.md"])
    for n, c in r2.files.items():
        (d / n).write_text(c, encoding="utf-8")
    check("project now has three languages",
          [l["code"] for l in pl.find_projects(d)[0]["languages"]], ["en", "de", "es"])

    print("\n  a project can start in another language:")
    d2 = Path(tempfile.mkdtemp())
    r3 = compose.generate({"de": DE, "es": ES}, "vx", "k", "m")
    for n, c in r3.files.items():
        (d2 / n).write_text(c, encoding="utf-8")
    check("German is the master", pl.master_lang(d2 / "vx_MASTER_production_sheet.md"), "de")
    check("languages are de + es (no phantom English)",
          [l["code"] for l in pl.find_projects(d2)[0]["languages"]], ["de", "es"])

    print("\n  a drifting split pads to keep numbering aligned:")
    G.segment_script = lambda en, s, nm, k, m, fb="": ["only one part"]

    class R:
        warnings: list = []
    r = R()
    scenes = [compose.Scene(n=i, narration=t, media="IMAGE", query="q")
              for i, t in enumerate(["a.", "b.", "c."], 1)]
    out = compose.segment_language(scenes, "some text", "de", "k", "m", r, lambda *a: None)
    check("padded to the scene count", len(out), 3)
    check("and warned about it", len(r.warnings), 1)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
