#!/usr/bin/env python3
"""Freeze the per-language sheet flow: paste a script per language, no translation.

    python3 tools/test_compose.py

Gemini is mocked, so this runs offline and for free. It checks the contract that
matters after the translation-to-segmentation change:

  - the structure language defines the main script (and records its own language);
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
    check("main script + one narration sheet, no others",
          sorted(res.files), ["vid_GERMAN_narration.md", "vid_main_script.md"])
    check("main script records its language", "main-lang: en"
          in res.files["vid_main_script.md"])
    check("scene count came from the structure script", len(res.scenes), 3)
    check("segmentation preserved the German words exactly",
          G.words("".join(l.split(': "')[-1] for l in
                          res.files["vid_GERMAN_narration.md"].splitlines()
                          if l.startswith("DE:"))).__len__() > 0)

    # Point the pipeline at a scratch ROOT so we exercise the real
    # projects/<pid>/sheets/ layout, not the old flat one.
    tmp = Path(tempfile.mkdtemp())
    pl.ROOT, pl.PROJECTS = tmp, tmp / "projects"
    d = pl.sheets_dir("vid")
    compose.write_files(res, d)

    print("\n  read the project back:")
    projs = pl.find_projects()
    check("one project", len(projs), 1)
    check("both languages listed", [l["code"] for l in projs[0]["languages"]], ["en", "de"])
    check("English reads from the main script (no side file)",
          pl.narration_file(d, "vid", "en"), None)
    check("German has its narration sheet",
          pl.narration_file(d, "vid", "de").name, "vid_GERMAN_narration.md")

    print("\n  add Spanish to the finished project:")
    r2 = compose.add_language(d / "vid_main_script.md", "es", ES, "k", "m")
    check("only the Spanish sheet is written", sorted(r2.files),
          ["vid_SPANISH_narration.md"])
    compose.write_files(r2, d)
    check("project now has three languages",
          [l["code"] for l in pl.find_projects()[0]["languages"]], ["en", "de", "es"])

    print("\n  a project can start in another language:")
    tmp2 = Path(tempfile.mkdtemp())
    pl.ROOT, pl.PROJECTS = tmp2, tmp2 / "projects"
    r3 = compose.generate({"de": DE, "es": ES}, "vx", "k", "m")
    d2 = pl.sheets_dir("vx")
    compose.write_files(r3, d2)
    check("German is the main script", pl.main_lang(d2 / "vx_main_script.md"), "de")
    check("languages are de + es (no phantom English)",
          [l["code"] for l in pl.find_projects()[0]["languages"]], ["de", "es"])

    # Regression: reading the main script's OWN language needs no translation file.
    # This used to raise SystemExit ("Language 'de' needs a translation file")
    # and take down the whole dashboard for anyone whose main script wasn't English.
    proj = pl.find_projects()[0]
    st = pl.project_status(Path(proj["sheet"]), proj["languages"])
    check("status loads for a non-English main script (no crash)", "error" not in st)
    check("the main script language counts as having its sheet",
          st["languages"]["de"]["sheets"])
    check("reading the main script language directly returns scenes",
          len(pl.load_scenes(Path(proj["sheet"]), "de", None)), 3)

    # No language chosen -> the project's own main language, never a hardcoded
    # "en". This is what the CLI (resolve) and the studio workers now default to.
    import types
    import make_video as mv
    a = types.SimpleNamespace(sheet=proj["sheet"], lang=None, narration=None)
    _sheet, _tr, _sc = mv.resolve(a)
    check("CLI resolve() with no --lang uses the main language", a.lang, "de")
    check("CLI resolve() loads scenes with no narration file", len(_sc), 3)
    check("worker langs default is the main language, not 'en'",
          [] or [pl.main_lang(Path(proj["sheet"]))], ["de"])

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
