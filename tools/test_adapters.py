#!/usr/bin/env python3
"""Check what each adapter KEEPS and what it THROWS AWAY, without a network.

    python3 tools/test_adapters.py

`faceless sources` calls the real APIs and answers "is it reachable". That is a
different question from "does it filter correctly", and the second one is the
dangerous one: a licence filter that quietly passes CC BY-SA material does not
fail, it just puts a share-alike photograph in a monetised video. Nothing about
that is visible until someone complains.

So each adapter is fed a response containing material it MUST reject alongside
material it must keep, and the counts are asserted. These run offline, cost
nothing, and are the only check on the rules that matter legally.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import sources as S  # noqa: E402


def feed(payload):
    """Replace the HTTP layer with a fixed response."""
    S._json = lambda url, headers=None: payload


# ── Library of Congress ───────────────────────────────────────────────────
# LoC states rights per item, in prose, under several different keys. The
# phrase that grants use is "no known restrictions"; everything else is a no.

LOC = {"results": [
    {"id": "http://www.loc.gov/item/fsa/", "title": "Farm family",
     "rights": "No known restrictions on publication. For information see "
               "\"Copyright and Other Restrictions\".",
     "image_url": ["//tile.loc.gov/a/small.jpg#h=200&w=300",
                   "//tile.loc.gov/a/big.jpg#h=1200&w=1600"]},
    {"id": "b", "title": "Restricted",
     "rights_advisory": ["Publication may be restricted."],
     "image_url": ["//tile.loc.gov/b.jpg#h=1200&w=1600"]},
    {"id": "c", "title": "No rights note at all",
     "image_url": ["//tile.loc.gov/c.jpg#h=1200&w=1600"]},
    {"id": "d", "title": "Free but too small",
     "rights": "No known restrictions on publication.",
     "image_url": ["//tile.loc.gov/d.jpg#h=400&w=600"]},
    {"id": "e", "title": "Access restricted", "access_restricted": True,
     "rights": "no known restrictions",
     "image_url": ["//tile.loc.gov/e.jpg#h=1200&w=1600"]},
]}

# ── Europeana ─────────────────────────────────────────────────────────────
# Rights are a URI, so the test is exact rather than linguistic. The subtler
# case is edmIsShownBy pointing at a viewer PAGE instead of a file.

EU = {"success": True, "items": [
    {"rights": ["http://creativecommons.org/publicdomain/zero/1.0/"],
     "edmIsShownBy": ["http://x.eu/scan.jpg"],
     "edmPreview": ["http://api.eu/thumb.jpg"],
     "title": ["Book of Hours"], "dataProvider": ["KB"], "guid": "g1"},
    {"rights": ["http://creativecommons.org/licenses/by-sa/4.0/"],
     "edmIsShownBy": ["http://x.eu/sa.jpg"], "title": ["Share-alike"],
     "guid": "g2"},
    {"rights": ["http://creativecommons.org/licenses/by/4.0/"],
     "edmIsShownBy": ["http://x.eu/by.jpg"], "title": ["Attribution"],
     "guid": "g3"},
    {"rights": ["http://creativecommons.org/publicdomain/mark/1.0/"],
     "edmIsShownBy": ["http://x.eu/viewer/page?id=9"],
     "edmPreview": ["http://api.eu/t2.jpg"],
     "title": ["Viewer page, not a file"], "dataProvider": ["BnF"],
     "guid": "g4"},
]}


def main() -> int:
    bad = 0

    def check(label: str, got, want) -> None:
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<52}{got!r}"
              f"{'' if ok else f'  (wanted {want!r})'}")

    print("\n  Library of Congress — 5 records in, 1 usable:")
    feed(LOC)
    hits = S.loc("x", "IMAGE", 5, {})
    check("kept exactly the one free, large, unrestricted item", len(hits), 1)
    if hits:
        h = hits[0]
        check("chose the largest rendition, not the first", h.width, 1600)
        check("upgraded the protocol-relative URL",
              h.url.startswith("https://"), True)
        check("the word 'copyright' in LoC boilerplate is not a rejection",
              h.page.endswith("fsa/"), True)

    print("\n  Europeana — 4 records in, 2 usable:")
    feed(EU)
    hits = S.europeana("x", "IMAGE", 5, {"europeana_key": "k"})
    check("dropped CC BY and CC BY-SA, kept CC0 and PDM", len(hits), 2)
    if len(hits) == 2:
        check("a viewer page falls back to the preview file",
              hits[1].url, "https://api.eu/t2.jpg")
        check("licence is recorded on the hit",
              hits[0].license.startswith("CC0"), True)

    print("\n  refusals are explicit, never silent:")
    try:
        S.europeana("x", "IMAGE", 3, {})
        check("Europeana without a key raises", False, True)
    except S.SourceError as e:
        check("Europeana without a key raises", "europeana_key" in str(e), True)
    for name, fn in (("loc", S.loc), ("europeana", S.europeana)):
        try:
            fn("x", "VIDEO", 3, {"europeana_key": "k"})
            check(f"{name} refuses VIDEO", False, True)
        except S.SourceError:
            check(f"{name} refuses VIDEO rather than returning stills", True, True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
