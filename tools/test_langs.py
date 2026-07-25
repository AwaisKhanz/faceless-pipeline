#!/usr/bin/env python3
"""Per-language narration files are discovered by the EXACT token in their name,
so an added language (esp. English on a non-English project, named …_EN_…) shows
up on the project page and its narration is found — without 'EN' matching 'FRENCH'.

    python3 tools/test_langs.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lib.pipeline as pl   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<52}{'' if ok else repr(got)}")
        bad += not ok

    pid = "Testingwithgermanscript"
    print("\n  filename token -> language code (exact, not substring):")
    check("…_EN_narration -> en", pl._narration_lang(pid, f"{pid}_EN_narration"), "en")
    check("…_ENGLISH_narration -> en", pl._narration_lang(pid, f"{pid}_ENGLISH_narration"), "en")
    check("…_GERMAN_narration -> de", pl._narration_lang(pid, f"{pid}_GERMAN_narration"), "de")
    check("'EN' does NOT match FRENCH", pl._narration_lang(pid, f"{pid}_FRENCH_narration"), "fr")
    check("main script -> None", pl._narration_lang(pid, f"{pid}_main_script"), None)
    check("unknown token -> None", pl._narration_lang(pid, f"{pid}_XX_narration"), None)

    print("\n  German project + added English is discovered on disk:")
    root = pathlib.Path(tempfile.mkdtemp())
    sd = root / pid / "sheets"
    sd.mkdir(parents=True)
    (sd / f"{pid}_main_script.md").write_text(
        "<!-- main-lang: de -->\n# t\n_1 scenes · language: de_\n\n---\n\n"
        "**S1 ⬜** · IMAGE\n- Narration: \"Hallo.\"\n- ALT / search: `x`\n",
        encoding="utf-8")
    (sd / f"{pid}_EN_narration.md").write_text(
        "# English narration\n\n---\n\n**S1** · EN: \"Hallo.\"\nEN: \"Hello.\"\n",
        encoding="utf-8")

    _orig = pl.PROJECTS
    pl.PROJECTS = root
    try:
        projs = [p for p in pl.find_projects() if p["id"] == pid]
        codes = [l["code"] for l in projs[0]["languages"]] if projs else []
        check("find_projects lists de + en", codes, ["de", "en"])
        nf = pl.narration_file(sd, pid, "en")
        check("narration_file finds the EN sheet", nf.name if nf else None,
              f"{pid}_EN_narration.md")
        check("narration_file(de) is None (de is the main script)",
              pl.narration_file(sd, pid, "de"), None)
    finally:
        pl.PROJECTS = _orig

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
