#!/usr/bin/env python3
"""Freeze the per-project folder layout and its one-time migration.

    python3 tools/test_layout.py

Gemini is mocked, so this runs offline and for free. It checks:

  - a fresh generate writes into projects/<pid>/sheets/ (not the old flat sheets/);
  - paths_for derives work/ and out/ inside the same project folder, with
    caches still shared at the top level;
  - migrate_layout folds a pre-existing FLAT project (sheets/, work/, out/) into
    projects/<pid>/… — including a pid that itself contains an underscore, which
    a naive split on "_" would mangle;
  - find_projects reads the migrated project back;
  - delete_project removes a project and prunes its now-empty folder.
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


def _point_pipeline_at(tmp: Path):
    """Redirect the module's ROOT/PROJECTS at a scratch dir for the test."""
    pl.ROOT = tmp
    pl.PROJECTS = tmp / "projects"


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

    tmp = Path(tempfile.mkdtemp())
    _point_pipeline_at(tmp)

    print("\n  a fresh generate lands in projects/<pid>/sheets/:")
    res = compose.generate({"en": EN, "de": DE}, "vid", "key", "model")
    written = compose.write_files(res, pl.sheets_dir("vid"))
    sdir = tmp / "projects" / "vid" / "sheets"
    check("sheets dir exists under projects/", sdir.is_dir())
    check("main script written there",
          (sdir / "vid_main_script.md").exists())
    check("nothing left in a flat sheets/ dir", (tmp / "sheets").exists(), False)

    print("\n  paths_for derives work/ + out/ in the same project folder:")
    main_script = sdir / "vid_main_script.md"
    p = pl.paths_for(main_script, "en")
    check("picks under projects/vid/work/",
          str(p["picks"]).endswith("projects/vid/work/picks.json"))
    check("mp4 under projects/vid/out/",
          str(p["out"]).endswith("projects/vid/out/vid_en.mp4"))
    check("base is the per-language working dir",
          str(p["base"]).endswith("projects/vid/work/en"))
    check("stock cache stays shared at the top",
          str(p["stockcache"]).endswith("cache/stock")
          and "projects" not in str(p["stockcache"]))

    print("\n  find_projects reads it back:")
    projs = pl.find_projects()
    check("one project", len(projs), 1)
    check("sheet is a full path", Path(projs[0]["sheet"]).is_absolute())
    check("languages are en + de", [l["code"] for l in projs[0]["languages"]], ["en", "de"])

    print("\n  the canonical topic round-trips through the sheet:")
    import lib.sheet as sheetlib
    sc = [compose.Scene(n=1, narration="A rocket lifts off.", media="VIDEO",
                        query="rocket launch", domain="spaceflight", topic="space"),
          compose.Scene(n=2, narration="An old loom in a museum.", media="IMAGE",
                        query="antique loom", domain="craft", topic="culture")]
    md = compose.render_main_script({"title_en": "T"}, sc, "vid", "en")
    check("renderer writes a Topic line", "- Topic: space" in md)
    rt = tmp / "rt_main_script.md"
    rt.write_text(md, encoding="utf-8")
    parsed = sheetlib.parse_main_script(rt)
    check("topic parses back on scene 1", parsed[0].topic, "space")
    check("topic parses back on scene 2", parsed[1].topic, "culture")

    # -- migration of a legacy flat project ----------------------------------
    print("\n  migrate a legacy FLAT project (pid has an underscore):")
    tmp2 = Path(tempfile.mkdtemp())
    _point_pipeline_at(tmp2)
    pid = "deep_sea"                       # the underscore is the trap
    flat_sheets = tmp2 / "sheets"
    flat_work = tmp2 / "work"
    flat_out = tmp2 / "out"
    for d in (flat_sheets, flat_work, flat_out):
        d.mkdir(parents=True)
    # Real sheets, written flat as the old layout did.
    res2 = compose.generate({"en": EN, "de": DE}, pid, "k", "m")
    for n, c in res2.files.items():
        (flat_sheets / n).write_text(c, encoding="utf-8")
    (flat_work / f"{pid}_picks.json").write_text("{}", encoding="utf-8")
    (flat_work / f"{pid}_assets.json").write_text("{}", encoding="utf-8")
    (flat_work / f"{pid}_en").mkdir()
    (flat_out / f"{pid}_en.mp4").write_bytes(b"fake")
    (flat_out / f"{pid}_approval.html").write_text("x", encoding="utf-8")

    rep = pl.migrate_layout()
    check("migration reported the project", rep["projects"], [pid])
    base = tmp2 / "projects" / pid
    check("main script moved",
          (base / "sheets" / f"{pid}_main_script.md").exists())
    check("narration moved",
          (base / "sheets" / f"{pid}_GERMAN_narration.md").exists())
    check("picks moved and de-prefixed", (base / "work" / "picks.json").exists())
    check("per-language work dir de-prefixed", (base / "work" / "en").is_dir())
    check("mp4 kept its name in out/", (base / "out" / f"{pid}_en.mp4").exists())
    check("approval de-prefixed to approval.html",
          (base / "out" / "approval.html").exists())
    check("legacy flat folders removed",
          any((tmp2 / d).exists() for d in ("sheets", "work", "out")), False)

    print("\n  the underscore pid is read back intact:")
    projs2 = pl.find_projects()
    check("one project", len(projs2), 1)
    check("id preserved with underscore", projs2[0]["id"], pid)

    print("\n  migrate_layout is idempotent (second run is a no-op):")
    rep2 = pl.migrate_layout()
    check("nothing moved the second time", rep2["moved"], 0)

    print("\n  delete_project removes it and prunes the empty folder:")
    proj = pl.find_project(pid)
    r = pl.delete_project(Path(proj["sheet"]), proj["languages"],
                          ["outputs", "visuals", "work", "sheets"])
    check("something was removed", r["count"] > 0)
    check("project folder is gone", base.exists(), False)
    check("no projects remain", len(pl.find_projects()), 0)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
