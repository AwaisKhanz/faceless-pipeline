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


# ── Internet Archive ──────────────────────────────────────────────────────
# The strictest gate in the module, and the one whose failure is worst: a
# copyrighted film admitted here goes straight into a monetised video. The
# search response carries the licence signal; the metadata response carries the
# playable file. Two calls, so the fake has to answer both.

IA_SEARCH = {"response": {"docs": [
    # 0  explicit public-domain licence — keep
    {"identifier": "pd_film", "title": "A Public Domain Film",
     "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/",
     "collection": ["opensource_movies"]},
    # 1  no licence, but a curated-PD collection vouches for it — keep
    {"identifier": "prelinger_film", "title": "Prelinger Ephemeral",
     "collection": ["prelinger", "ephemera"]},
    # 2  no licence, ordinary upload collection — DROP (absence is not freedom)
    {"identifier": "mystery_upload", "title": "Someone's Upload",
     "collection": ["opensource_movies"]},
    # 3  explicit share-alike — DROP even though it is a CC licence
    {"identifier": "sa_film", "title": "Share Alike Film",
     "licenseurl": "https://creativecommons.org/licenses/by-sa/4.0/",
     "collection": ["prelinger"]},          # collection must NOT rescue it
    # 4  non-commercial — DROP, excludes monetisation
    {"identifier": "nc_film", "title": "Non-Commercial",
     "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"},
]}}

IA_META = {
    "pd_film": {"files": [
        {"name": "pd_film.ogv", "width": "640"},
        {"name": "pd_film_512kb.mp4", "width": "640", "height": "480"},
        {"name": "pd_film.mp4", "width": "1280", "height": "720"},
    ]},
    "prelinger_film": {"files": [
        {"name": "prelinger.mp4", "width": "512", "height": "384"},
    ]},
}


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

    print("\n  Internet Archive — 5 films in, 2 usable, and the gate is strict:")

    def ia_json(url, headers=None):
        if "advancedsearch" in url:
            return IA_SEARCH
        ident = url.rsplit("/", 1)[-1]
        return IA_META.get(ident, {"files": []})
    S._json = ia_json

    hits = S.internet_archive("x", "VIDEO", 5, {})
    idents = [h.page.rsplit("/", 1)[-1] for h in hits]
    check("kept only the PD-licensed and the Prelinger-collection films",
          sorted(idents), ["pd_film", "prelinger_film"])
    check("share-alike is dropped even inside a trusted collection",
          "sa_film" not in idents, True)
    check("non-commercial is dropped (it excludes monetisation)",
          "nc_film" not in idents, True)
    check("an unlicensed ordinary upload is dropped (absence is not freedom)",
          "mystery_upload" not in idents, True)
    if hits:
        pd = next((h for h in hits if h.page.endswith("pd_film")), None)
        check("chose the widest mp4 derivative, skipping the .ogv",
              pd.url.endswith("pd_film.mp4") and pd.width == 1280, True)
        check("licence reason names the collection when there is no licence URL",
              "prelinger" in next(h.license for h in hits
                                  if h.page.endswith("prelinger_film")), True)

    print("\n  a standard-definition archival clip is NOT rejected as too small:")
    check("512px archival video clears the video floor",
          S.MIN_VIDEO_WIDTH <= 512 < S.MIN_WIDTH, True)
    check("the same 512px would be rejected as a still",
          S.floor_for("IMAGE") > 512 and S.floor_for("VIDEO") <= 512, True)

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
    try:
        S.internet_archive("x", "IMAGE", 3, {})
        check("Internet Archive refuses IMAGE", False, True)
    except S.SourceError:
        check("Internet Archive refuses IMAGE (stills come from elsewhere)",
              True, True)

    print("\n  image_licenses='all' relaxes the gate (biography mode, opt-in):")
    S.configure({"image_licenses": "all"})
    check("policy flag on", S._accept_any, True)
    feed(LOC)
    # a=free, b=restricted, c=no-note now all pass the LICENCE gate; d is still
    # dropped for size and e for access_restricted — those are not licence gates.
    check("LoC now keeps restricted + un-noted (size/access still apply)",
          len(S.loc("x", "IMAGE", 5, {})), 3)
    feed(EU)
    check("Europeana now keeps CC BY and CC BY-SA too",
          len(S.europeana("x", "IMAGE", 5, {"europeana_key": "k"})), 4)
    S.configure({})                              # reset — strict for anything after
    check("strict restored after reset", S._accept_any, False)
    feed(LOC)
    check("LoC back to 1 usable under strict", len(S.loc("x", "IMAGE", 5, {})), 1)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
