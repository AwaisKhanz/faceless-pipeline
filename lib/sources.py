"""Where pictures come from, and which source is asked for what.

Adding archives to a stock-photo pipeline sounds like "more sources = better
results". It is not that simple, and the shape of this module follows from three
facts that are easy to get wrong:

1. THE SOURCES BARELY OVERLAP. Wikimedia has essentially nothing for "elderly
   woman making tea in a bright kitchen"; Pexels has thousands. Pexels has
   nothing for "Roman aqueduct engraving"; the archives have hundreds. So this
   is not a ranked list of better and worse sources — it is a map of who holds
   what, and asking the wrong one wastes a request and returns junk.

2. ALMOST EVERY ARCHIVE IS STILLS ONLY. Modern 16:9 motion comes from Pexels
   and Pixabay and essentially nowhere else that is free. A scene marked VIDEO
   must go to them first whatever its subject.

3. FREE TO VIEW IS NOT FREE TO USE. This pipeline feeds a monetised channel, so
   only CC0 and public-domain material is accepted. That is the strict end on
   purpose: it carries no attribution duty and no share-alike clause, which
   means there is nothing for anyone to remember or track later. Anything
   requiring credit, or forbidding commercial use, is dropped at the adapter.

WHO DECIDES WHAT
    Gemini tags each scene with a subject domain — a judgement call, which is
    what a model is for. This module owns domain -> source order, because which
    API holds which material is a FACT, testable and inspectable, and should not
    be re-derived by a model on every call. The capability table also makes
    impossible routes impossible: Smithsonian has no video, so a VIDEO scene can
    never be sent there however it was tagged.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

TIMEOUT = 25

# Wikimedia's User-Agent policy requires a descriptive agent WITH working
# contact information — a URL or an email — and blocks generic or vague ones
# without notice. "contact via project README" is not contact information, and
# earned a connection reset.
#
# The required shape is:
#     <client>/<version> (<contact>) <library>/<version>
#
# Every other source is happy with this too, so one compliant agent serves all
# of them. Override `contact` in config.json to point at your own page or
# address if you run this at any volume — that is what it is for.
VERSION = "1.0"
DEFAULT_CONTACT = "https://github.com/AwaisKhanz/faceless-pipeline"
_contact = DEFAULT_CONTACT


def configure(cfg: dict) -> None:
    """Adopt the contact address from config, if one is set."""
    global _contact
    _contact = (cfg or {}).get("contact") or DEFAULT_CONTACT


def user_agent() -> str:
    return (f"FacelessStudio/{VERSION} ({_contact}) "
            f"Python-urllib/{'.'.join(map(str, __import__('sys').version_info[:2]))}")

# Archive scans are frequently portrait, 4:3, or a 600px thumbnail of a
# postcard, and at 1080p those look like mistakes.
#
# Search responses do NOT carry dimensions — neither NASA's nor Smithsonian's
# — so this cannot be enforced at search time. It is checked after download in
# stock.fetch, where the file itself can be measured. A hit whose width IS
# known and is too small is dropped here as a cheap first pass.
MIN_WIDTH = 1100

# Video gets a much lower floor, and it is not an oversight. Archival footage
# is legitimately standard-definition — a digitised 1940s newsreel is 640×480
# and looks exactly as it should at that size; the graininess reads as "old",
# which is the whole point of using it. The stills floor exists to catch a
# postcard scan masquerading as a hero image, a failure mode motion does not
# have. Modern stock video is filtered to ≥1280 at its own adapter, so this
# lower bound only ever bites archival sources, which is correct.
MIN_VIDEO_WIDTH = 480


def floor_for(media: str) -> int:
    """The minimum acceptable width for this media type."""
    return MIN_VIDEO_WIDTH if media == "VIDEO" else MIN_WIDTH


class SourceError(RuntimeError):
    pass


@dataclass
class Hit:
    """One candidate picture, in the shape stock.fetch already expects."""
    url: str
    ext: str
    credit: str
    page: str
    src: str
    width: int = 0
    height: int = 0
    license: str = "public domain"


@dataclass
class Source:
    """One place pictures come from, and what it can actually do."""
    name: str
    search: object                    # (query, media, want, cfg) -> list[Hit]
    media: tuple = ("IMAGE",)         # what it can serve
    needs_key: str = ""               # config.json key, "" when none needed
    covers: frozenset = frozenset()   # canonical topics this source is strong on
    generalist: bool = False          # shallow on everything, never useless
    note: str = ""

    # Tie-break only, never a substitute for subject match — see route().
    # How often a hit from here turns out to be a usable 16:9 picture rather
    # than a 400px thumbnail, a portrait scan, or a link to a viewer page.
    reliability: float = 1.0

    # A subject this source should definitely be able to answer, used by
    # `faceless sources` to smoke-test it for real. Declared here rather than
    # in the command so that adding a source cannot leave it untested.
    probe: str = ""

    def can(self, media: str) -> bool:
        return media in self.media


# ─────────────────────────────────────────────────────────────── http

def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent(),
                      "Accept": "application/json, */*", **(headers or {})})
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:
            last = e
            # A READ TIMEOUT usually means a slow archive (the Library of
            # Congress photo search is genuinely slow), not a refusal — so give
            # it another try or two before giving up. A RESET/REFUSAL is a block:
            # retrying only stalls the run, so fall straight through and let the
            # circuit breaker skip the source.
            if "timed out" in str(e).lower() and attempt < 3:
                time.sleep(1.0 * attempt)
                continue
            break

    msg = str(last)
    # A reset can be an agent-based block OR the network refusing the host
    # outright, and the two look identical here. Do not assert either: point at
    # the command that can actually tell them apart.
    if "reset" in msg.lower() or "forcibly closed" in msg.lower():
        msg += (" — the connection was dropped. That is either this network "
                "refusing the host, or the API refusing this client. Run "
                "'faceless sources' to find out which.")
    raise SourceError(msg)


def _json(url: str, headers: dict | None = None) -> dict:
    raw = _get(url, headers)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SourceError("the server did not return JSON")


def _https(url: str) -> str:
    """Normalise a URL to https.

    Two shapes turn up and neither can be downloaded as given:

      http://...   NASA's asset manifests hand these back. Plenty of networks
                   and corporate proxies block plain http outright, and there
                   is no reason to fetch a public file in the clear.
      //host/...   protocol-relative, which the Library of Congress uses. That
                   is meaningful in a browser, which knows what scheme the page
                   was served over, and meaningless to urllib.

    Every host involved serves https, so both become https.
    """
    if url.startswith("//"):
        return "https:" + url
    return "https://" + url[7:] if url.startswith("http://") else url


def _ext(url: str, default: str = ".jpg") -> str:
    tail = urllib.parse.urlparse(url).path.rsplit(".", 1)
    if len(tail) == 2 and 2 <= len(tail[1]) <= 4:
        return "." + tail[1].lower()
    return default


# ─────────────────────────────────────────────────────── NASA (no key)
# images-api.nasa.gov. NASA material is generally public domain. Two caveats
# worth knowing, neither of which blocks ordinary documentary use:
#   - NASA must not appear to endorse a product. We are not advertising.
#   - a few items are third-party copyright, and NASA marks them. We keep only
#     items whose own metadata does not name an external rights holder.

def nasa(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    kind = "video" if media == "VIDEO" else "image"
    qs = urllib.parse.urlencode({"q": query, "media_type": kind})
    data = _json(f"https://images-api.nasa.gov/search?{qs}")
    out: list[Hit] = []

    for item in (data.get("collection", {}).get("items") or [])[: want * 3]:
        meta = (item.get("data") or [{}])[0]
        # NASA marks third-party material in these fields. Skip anything that
        # names an owner other than NASA rather than guessing.
        holder = " ".join(str(meta.get(k, "")) for k in
                          ("secondary_creator", "photographer", "keywords"))
        if any(w in holder.lower() for w in ("getty", "reuters", "associated press",
                                             "copyright", "©")):
            continue
        links = item.get("links") or []
        preview = next((l.get("href") for l in links if l.get("render") == "image"), "")
        href = item.get("href", "")
        if not href:
            continue
        try:
            files = _json(href)                     # the asset manifest
        except SourceError:
            continue
        if kind == "video":
            best = next((f for f in files if f.endswith(".mp4") and "~orig" in f), "")
            best = best or next((f for f in files if f.endswith(".mp4")), "")
        else:
            # ~orig can be enormous; ~large is the sane 1080p-ish rendition.
            best = next((f for f in files if "~large." in f), "")
            best = best or next((f for f in files if f.lower().endswith((".jpg", ".png"))), "")
        if not best:
            continue
        best = _https(best)
        out.append(Hit(url=best, ext=_ext(best, ".mp4" if kind == "video" else ".jpg"),
                       credit=meta.get("center", "NASA") or "NASA",
                       page=preview or href, src="nasa",
                       license="public domain (NASA)"))
        if len(out) >= want:
            break
    return out


# ──────────────────────────────────────── Smithsonian (free api.data.gov key)
# Every record returned by the Open Access endpoint is CC0. We still check the
# per-record rights string rather than trusting the collection-level promise.

def smithsonian(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    key = cfg.get("smithsonian_key") or "DEMO_KEY"
    qs = urllib.parse.urlencode(
        {"q": f"{query} AND online_media_type:Images", "rows": max(want * 4, 20),
         "api_key": key})
    data = _json(f"https://api.si.edu/openaccess/api/v1.0/search?{qs}")
    rows = (data.get("response") or {}).get("rows") or []
    out: list[Hit] = []

    for r in rows:
        content = (r.get("content") or {})
        media_blocks = ((content.get("descriptiveNonRepeating") or {})
                        .get("online_media") or {}).get("media") or []
        for m in media_blocks:
            rights = str(m.get("usage", {}).get("access", "")).upper()
            if rights != "CC0":
                continue                       # only the unencumbered slice
            url = m.get("content") or m.get("thumbnail") or ""
            if not url:
                continue
            out.append(Hit(
                url=_https(url), ext=_ext(url), src="smithsonian",
                credit=(content.get("freetext", {}).get("dataSource", [{}])[0]
                        .get("content", "Smithsonian") if content.get("freetext")
                        else "Smithsonian"),
                page=m.get("guid", "") or r.get("id", ""),
                license="CC0 (Smithsonian Open Access)"))
            break
        if len(out) >= want:
            break
    return out


# ────────────────────────────────────────────────── Openverse (no key needed)
# One API in front of many providers — Flickr, museums, government archives.
#
# license=cc0,pdm is doing the important work. Openverse can also filter by
# license_type=commercial, but "commercial" still includes BY (attribution
# required) and BY-SA (share-alike), and this project accepts neither. Asking
# for the two unencumbered licences by name is narrower and unambiguous.
#
# Openverse is also the only source so far that reports width and height at
# search time, so the size floor can be applied before anything is downloaded.

def openverse(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    if media == "VIDEO":
        raise SourceError("Openverse indexes images and audio, not video")

    qs = urllib.parse.urlencode({
        "q": query,
        "page_size": min(max(want * 3, 10), 40),
        "license": "cc0,pdm",         # CC0 and Public Domain Mark only
        "size": "large",
        "aspect_ratio": "wide",       # 16:9 timeline; portrait scans waste it
        "mature": "false",
    })
    headers = {}
    if cfg.get("openverse_token"):
        # Optional. Anonymous access works but is rate-limited more tightly.
        headers["Authorization"] = f"Bearer {cfg['openverse_token']}"

    data = _json(f"https://api.openverse.org/v1/images/?{qs}", headers)
    out: list[Hit] = []

    for r in (data.get("results") or []):
        lic = (r.get("license") or "").lower()
        if lic not in ("cc0", "pdm"):
            continue                   # belt and braces over the filter above
        url = r.get("url") or ""
        if not url:
            continue
        w = int(r.get("width") or 0)
        if w and w < MIN_WIDTH:
            continue                   # known too small: do not even download
        out.append(Hit(
            url=_https(url), ext=_ext(url), src="openverse",
            credit=r.get("creator") or r.get("source") or "Openverse",
            page=r.get("foreign_landing_url") or r.get("detail_url") or "",
            width=w, height=int(r.get("height") or 0),
            license=f"{lic.upper()} via Openverse"))
        if len(out) >= want:
            break
    return out


# ─────────────────────────────────────────── Wikimedia Commons (no key needed)
# The largest of the archives, and the one most likely to be misused.
#
# Commons is NOT uniformly free. It is full of CC BY-SA, whose share-alike
# clause can be read as reaching the whole video it appears in, and CC BY,
# which obliges visible credit. Only files whose own metadata says public
# domain or CC0 are kept — everything else is dropped, however good it looks.
#
# The licence lives in imageinfo's extmetadata, so this asks for it in the same
# request as the image URL rather than making a second round trip per file.

_PD_OK = re.compile(r"\b(cc0|public\s*domain|pd-|pdm)\b", re.I)
_NOT_OK = re.compile(r"\b(share\s*-?\s*alike|by-sa|non\s*-?\s*commercial|"
                     r"by-nc|nd\b|no\s*derivatives|fair\s*use|copyright)\b", re.I)


def wikimedia(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    if media == "VIDEO":
        # Commons does hold video, but almost all of it is webm/ogv that needs
        # transcoding, and the free-licensed slice is thin. Not worth the cost.
        raise SourceError("Wikimedia video is not supported")

    qs = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",           # File: namespace
        "gsrlimit": min(max(want * 4, 12), 50),
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": "1920",          # ask for a scaled rendition, not the original
    })
    data = _json(f"https://commons.wikimedia.org/w/api.php?{qs}")
    out: list[Hit] = []

    for page in ((data.get("query") or {}).get("pages") or []):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}

        def field(name: str) -> str:
            return str((meta.get(name) or {}).get("value", ""))

        blob = " ".join(field(k) for k in
                        ("License", "LicenseShortName", "UsageTerms",
                         "Copyrighted", "Permission"))
        if _NOT_OK.search(blob) or not _PD_OK.search(blob):
            continue                   # anything not plainly PD/CC0 is dropped

        url = info.get("thumburl") or info.get("url") or ""
        if not url:
            continue
        w = int(info.get("thumbwidth") or info.get("width") or 0)
        if w and w < MIN_WIDTH:
            continue

        # Artist is HTML in Commons metadata; the credit is informational here
        # since PD and CC0 require none, so a rough strip is enough.
        artist = re.sub(r"<[^>]+>", "", field("Artist")).strip()
        out.append(Hit(
            url=_https(url), ext=_ext(url), src="wikimedia",
            credit=artist[:80] or "Wikimedia Commons",
            page=info.get("descriptionurl", ""),
            width=w, height=int(info.get("thumbheight") or info.get("height") or 0),
            license=(field("LicenseShortName") or "public domain") + " (Commons)"))
        if len(out) >= want:
            break
    return out


# ──────────────────────────────────────── Library of Congress (no key needed)
# Historical photography, prints, posters and maps — the strongest free source
# for 19th and 20th century material, which is exactly the gap between "modern
# life" (stock) and "antiquity" (Smithsonian).
#
# LoC does NOT publish a blanket licence. Each item carries its own rights
# statement, and the phrase that matters is "No known restrictions on
# publication" — LoC's way of saying it has researched the item and found no
# surviving claim. Anything without such a statement is dropped, because absence
# of a rights note is not evidence of freedom.
#
# The generic _NOT_OK regex is deliberately NOT used here. LoC's own boilerplate
# ends "...see 'Copyright and Other Restrictions'" on items that are perfectly
# free, so matching the bare word "copyright" would reject almost everything.
# These two patterns read LoC's actual vocabulary instead.

_LOC_FREE = re.compile(r"no known restrictions|public domain|cc0|pdm", re.I)
_LOC_BLOCK = re.compile(
    r"publication\s+(?:may\s+be\s+|is\s+)?restricted|rights[^.]{0,20}restricted|"
    r"permission\s+(?:is\s+)?required|may\s+be\s+protected|"
    r"restricted\s+access|not\s+for\s+publication", re.I)

# LoC appends the rendition's real dimensions to each URL as a fragment:
#   //tile.loc.gov/storage-services/.../foo.jpg#h=1024&w=1536
# so the largest usable rendition can be chosen without downloading any of them.
_LOC_DIMS = re.compile(r"[#&]w=(\d+)")
_LOC_H = re.compile(r"[#&]h=(\d+)")


def loc(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    if media == "VIDEO":
        raise SourceError("Library of Congress video needs per-item negotiation")

    qs = urllib.parse.urlencode({
        "q": query, "fo": "json", "c": min(max(want * 4, 20), 40),
        "at": "results",          # results only: the full response is enormous
    })
    data = _json(f"https://www.loc.gov/photos/?{qs}")
    out: list[Hit] = []

    for r in (data.get("results") or [])[: want * 6]:
        if r.get("access_restricted"):
            continue

        # The rights note lives under different keys depending on the
        # collection, so gather them all and read the lot as one blob.
        blob = " ".join(
            " ".join(v) if isinstance(v, list) else str(v)
            for k, v in r.items()
            if "rights" in k.lower() and v)
        if _LOC_BLOCK.search(blob) or not _LOC_FREE.search(blob):
            continue

        urls = [u for u in (r.get("image_url") or []) if isinstance(u, str)]
        if not urls:
            continue

        def width_of(u: str) -> int:
            m = _LOC_DIMS.search(u)
            return int(m.group(1)) if m else 0

        # Widest first; unmeasured URLs sort last but are still usable, and get
        # measured after download like every other source that stays quiet.
        urls.sort(key=width_of, reverse=True)
        best = urls[0]
        w = width_of(best)
        if w and w < MIN_WIDTH:
            continue              # its own largest rendition is too small

        h = _LOC_H.search(best)
        out.append(Hit(
            url=_https(best), ext=_ext(best), src="loc",
            credit="Library of Congress",
            page=r.get("id") or r.get("url") or "",
            width=w, height=int(h.group(1)) if h else 0,
            license="no known restrictions (Library of Congress)"))
        if len(out) >= want:
            break
    return out


# ─────────────────────────────────────── Europeana (free key, `europeana_key`)
# Aggregates roughly 4,000 European museums, libraries and archives — the one
# place with real depth on European art, manuscripts and regional history, none
# of which the American collections hold.
#
# Europeana's own `reusability=open` filter is NOT strict enough for us: "open"
# includes CC BY (credit required) and CC BY-SA (share-alike, which can be read
# as reaching the whole video). So the query names the two unencumbered rights
# URIs explicitly, and each item's `rights` field is re-checked afterwards.
#
# A key is free from https://pro.europeana.eu/pages/get-api and takes a minute.
# Without one this source simply is not offered, which is what needs_key does.

_EU_FREE = re.compile(r"publicdomain/(zero|mark)", re.I)
_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def europeana(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    if media == "VIDEO":
        raise SourceError("Europeana video is provider-hosted and rarely usable")

    key = cfg.get("europeana_key") or ""
    if not key:
        raise SourceError("no europeana_key in config.json")

    # Every character is percent-encoded (quote, not quote_plus, so spaces become
    # %20). The earlier version marked space and " as "safe", which left them
    # literal in the URL and made the whole request malformed — Europeana replied
    # HTTP 400. The colons, slashes and quotes inside the RIGHTS filter are
    # encoded too; Europeana decodes them back, so the Lucene syntax still works.
    qs = urllib.parse.urlencode({
        "wskey": key, "query": query,
        "qf": 'RIGHTS:("http://creativecommons.org/publicdomain/zero/1.0/" OR '
              '"http://creativecommons.org/publicdomain/mark/1.0/")',
        "media": "true",           # only records that actually have a file
        "thumbnail": "true",
        "profile": "rich",
        "rows": min(max(want * 4, 20), 50),
    }, quote_via=urllib.parse.quote)
    qs += "&qf=TYPE%3AIMAGE"       # urlencode cannot repeat a key
    data = _json(f"https://api.europeana.eu/record/v2/search.json?{qs}")

    if not data.get("success", True):
        raise SourceError(data.get("error") or "Europeana rejected the request")

    out: list[Hit] = []
    for r in (data.get("items") or []):
        rights = " ".join(r.get("rights") or [])
        if not _EU_FREE.search(rights):
            continue               # belt and braces over the qf filter

        # edmIsShownBy is the real file at the providing institution;
        # edmPreview is a Europeana-hosted thumbnail. Prefer the file, but only
        # when it actually looks like a file — a fair number of records point at
        # a viewer page, and downloading HTML as a .jpg helps nobody.
        shown = (r.get("edmIsShownBy") or [""])[0]
        preview = (r.get("edmPreview") or [""])[0]
        path = urllib.parse.urlparse(shown).path.lower()
        url = shown if path.endswith(_IMAGE_EXT) else preview
        if not url:
            continue

        title = (r.get("title") or [""])[0]
        out.append(Hit(
            url=_https(url), ext=_ext(url), src="europeana",
            credit=(r.get("dataProvider") or ["Europeana"])[0],
            page=r.get("guid") or (r.get("edmIsShownAt") or [""])[0],
            license=f"{'CC0' if 'zero' in rights.lower() else 'Public Domain Mark'}"
                    f" via Europeana" + (f" — {title[:40]}" if title else "")))
        if len(out) >= want:
            break
    return out


# ──────────────────────────────────────── Internet Archive (no key needed)
# The only free source of archival MOTION — 1920s street scenes, wartime
# newsreels, mid-century educational films — none of which exists as stock and
# none of which any still-image archive can supply. This is what stock cannot
# do: a modern camera cannot film 1935.
#
# It is also the most dangerous source in the set, and is added last for that
# reason. Internet Archive hosts enormous quantities of material that is simply
# under copyright — TV recordings, uploaded films, bootlegs — sitting beside the
# public-domain holdings with nothing but a metadata field to tell them apart.
# "It was free to stream on archive.org" is not a licence. So the gate here is
# the strictest in the module, and errs towards dropping a usable clip rather
# than admitting a doubtful one:
#
#   an item passes ONLY IF
#       it carries an explicit CC0 / public-domain licence URL, OR
#       it belongs to a collection that is a curated public-domain dedication
#   AND its licence URL is not a restrictive Creative Commons variant.
#
# The collection allowlist is deliberately tiny and is NOT "absence of a
# restriction" — each entry is a positive, documented dedication to the public
# domain, which is a different thing from a missing rights note:
#
#   prelinger  Rick Prelinger's ephemeral-film archive, explicitly released to
#              the public domain. The canonical free-archival-footage source.
#   fedflix    US Government films via Public.Resource.Org — public domain by
#              statute (17 U.S.C. §105), which no upload can override.

_IA_FREE = re.compile(r"publicdomain/(zero|mark)|creativecommons\.org/publicdomain",
                      re.I)
_IA_BLOCK = re.compile(r"creativecommons\.org/licenses/"
                       r"(by|by-sa|by-nc|by-nd|by-nc-sa|by-nc-nd|nc|nd|sa)", re.I)
_IA_SAFE_COLLECTIONS = frozenset({"prelinger", "fedflix"})


def _as_set(value) -> set:
    """Internet Archive returns single-valued fields as a string and
    multi-valued ones as a list, interchangeably. Normalise to a set."""
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)} if value else set()


def internet_archive(query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    if media != "VIDEO":
        # IA holds stills too, but Openverse and Wikimedia already index the
        # freely-licensed slice of them with far better metadata. IA earns its
        # place here for motion, which nothing else can supply.
        raise SourceError("Internet Archive is used here for archival video only")

    qs = urllib.parse.urlencode({
        "q": f"({query}) AND mediatype:movies",
        "fl[]": "identifier",       # urlencode repeats these for the list
        "rows": min(max(want * 3, 8), 30),
        "output": "json",
    })
    # The fl[] list needs the other fields too; add them by hand because
    # urlencode cannot emit a repeated key from a plain dict.
    qs += "".join(f"&fl%5B%5D={f}" for f in
                  ("licenseurl", "collection", "title", "year"))
    data = _json(f"https://archive.org/advancedsearch.php?{qs}")

    docs = ((data.get("response") or {}).get("docs")) or []
    out: list[Hit] = []

    for doc in docs:
        lic = str(doc.get("licenseurl") or "")
        colls = _as_set(doc.get("collection"))

        # The gate. Restrictive CC is a hard no even if a collection would
        # otherwise vouch for it — the item's own licence wins.
        if _IA_BLOCK.search(lic):
            continue
        free = bool(_IA_FREE.search(lic)) or bool(colls & _IA_SAFE_COLLECTIONS)
        if not free:
            continue

        ident = doc.get("identifier")
        if not ident:
            continue

        # advancedsearch does not list files, so fetch the item manifest to
        # find a directly-playable derivative. One extra call per candidate,
        # like NASA's asset manifest.
        try:
            meta = _json(f"https://archive.org/metadata/{urllib.parse.quote(ident)}")
        except SourceError:
            continue

        mp4s = [f for f in (meta.get("files") or [])
                if str(f.get("name", "")).lower().endswith(".mp4")]
        if not mp4s:
            continue                 # only webm/ogv/mpeg: skip rather than transcode

        def width_of(f: dict) -> int:
            try:
                return int(f.get("width") or 0)
            except (TypeError, ValueError):
                return 0

        # Widest derivative is closest to the source scan. Unmeasured ones sort
        # last but stay eligible — archival files often omit dimensions, and a
        # missing width is not a reason to reject motion the way it is a still.
        mp4s.sort(key=width_of, reverse=True)
        pick = mp4s[0]
        w = width_of(pick)
        if w and w < MIN_VIDEO_WIDTH:
            continue                 # even the best derivative is a thumbnail

        name = urllib.parse.quote(str(pick["name"]))
        url = f"https://archive.org/download/{urllib.parse.quote(ident)}/{name}"
        reason = ("public domain" if _IA_FREE.search(lic)
                  else f"{'/'.join(sorted(colls & _IA_SAFE_COLLECTIONS))} collection")
        out.append(Hit(
            url=url, ext=".mp4", src="ia",
            credit=str(doc.get("title") or "Internet Archive")[:80],
            page=f"https://archive.org/details/{ident}",
            width=w, height=width_of({"width": pick.get("height")}),
            license=f"{reason} (Internet Archive)"))
        if len(out) >= want:
            break
    return out


# ─────────────────────────────────────────────── registry

# ══════════════════════════════════════════════ topics, coverage, routing
#
# An earlier version had eight hardcoded domains and a route table keyed on
# them. That does not scale: real scripts are about sport, food, medicine,
# farming, war, architecture, mythology, shipping, insects — and a closed list
# only ever helps if each entry routes DIFFERENTLY. Adding forty more values
# that all resolve to "ask stock" would be noise pretending to be capability.
#
# So routing is data-driven instead:
#
#   TOPICS   an open vocabulary. Many surface words map onto one canonical
#            topic, so "astrophysics", "cosmos", "orbit" and "nebula" all
#            resolve to `space` without the model needing to guess our spelling.
#
#   covers   each Source declares the topics it is genuinely strong on.
#            Adding a new archive later means declaring what it holds — no
#            central table to edit and no risk of forgetting a row.
#
#   route()  scores each available source against the topics found in BOTH the
#            domain tag and the query text, and returns the best few. The query
#            matters as much as the tag: "roman aqueduct at sunset" says
#            "historical" whatever the scene was labelled.

# Canonical topic -> the words that mean it. Extend freely; this is just data.
TOPICS: dict[str, frozenset] = {
    "space": frozenset("""space astronomy astronomical cosmos cosmic universe galaxy
        galaxies star stars stellar nebula planet planetary moon lunar solar sun
        orbit orbital spacecraft satellite rocket launch astronaut telescope mars
        jupiter saturn venus mercury comet asteroid meteor eclipse constellation
        blackhole supernova interstellar nasa apollo shuttle""".split()),
    # NOTE `solar` is deliberately absent. It means solar power far more often
    # than solar system in a modern script, and listing it here sent "solar
    # panels on a roof" to NASA. Real astronomy scenes always carry another
    # word — sun, orbit, planet, eclipse — so nothing is lost.

    "geology": frozenset("""geology geological tectonic volcano volcanic
        earthquake seismic mountain mountains canyon glacier ice arctic antarctic
        desert erosion mineral rock rocks crust mantle lava magma fossil
        continent island landscape terrain sediment strata""".split()),
    # `mine`, `mining`, `quarry` and `drilling` used to be here as well as in
    # tech. Once a word could belong to both, that pulled every mining scene
    # towards the archives — but a working mine is industry, photographed by
    # stock libraries. Rock formations are geology; digging them up is not.

    "weather": frozenset("""weather storm rain snow cloud clouds lightning thunder
        hurricane tornado wind fog mist frost drought flood monsoon sky
        sunrise sunset season seasonal""".split()),

    "ocean": frozenset("""ocean sea marine underwater reef coral wave waves tide
        coast coastal shore beach submarine deepsea diving fish whale dolphin
        shark plankton current estuary harbour harbor""".split()),

    "nature": frozenset("""nature natural wildlife animal animals bird birds
        insect insects butterfly bee forest tree trees plant plants flower
        flowers leaf jungle savanna meadow river lake waterfall wilderness
        ecosystem habitat migration species mammal reptile""".split()),

    "history": frozenset("""history historical ancient antiquity medieval
        renaissance victorian empire roman rome greek greece egypt egyptian
        pharaoh viking colonial revolution civilisation civilization archaeology
        archaeological artefact artifact ruin ruins monument castle temple
        manuscript century dynasty war battle soldier army treaty
        wartime prewar postwar veteran vintage archival historic bygone
        pioneer frontier settler immigrant steam telegraph sepia newsreel
        excavation""".split()),
    # The era words above matter more than they look. Scene tags are free text
    # written by a model, and it reaches for "wartime" or "vintage" far more
    # readily than "history" — so without them a genuinely historical scene
    # carried no historical signal at all and went to stock.

    # `portrait` belongs to people, not here: a painted one always says
    # painting, whereas "portrait of an older woman" is a photograph.
    "art": frozenset("""art artistic painting paint sculpture statue
        museum gallery drawing illustration engraving fresco mosaic ceramic
        pottery textile craft design architecture architectural cathedral""".split()),

    "science": frozenset("""science scientific laboratory lab experiment research
        microscope specimen molecule atom atomic chemistry chemical physics
        biology biological cell dna genetic evolution bacteria virus
        skeleton fossil dinosaur botany zoology taxonomy""".split()),

    # Split from science on purpose. A modern hospital is a stock subject; a
    # Victorian surgical kit is a museum one, and reaches the archives through
    # `history` instead.
    "medicine": frozenset("""medicine medical health healthcare hospital clinic
        surgery surgeon surgical doctor nurse patient diagnosis clinical
        treatment therapy anatomy organ neuron brain heart blood bone muscle
        pharmacy medication prescription stethoscope""".split()),

    "tech": frozenset("""technology computer computing software hardware machine
        machinery engineering engineer industrial industry factory robot robotic
        electronics circuit server data network internet code programming
        manufacturing construction crane welding assembly mining mine quarry
        drilling refinery pipeline warehouse logistics energy power turbine
        solar wind nuclear electricity grid""".split()),

    "transport": frozenset("""transport train railway locomotive car automobile
        truck lorry bus aircraft aeroplane airplane aviation flight ship boat
        sailing port bridge road highway traffic bicycle motorcycle""".split()),

    "people": frozenset("""people person man woman child children family home
        house kitchen bedroom living room work office worker student teacher
        doctor patient nurse elderly senior retirement friend friends couple
        neighbour neighbor community daily routine habit sleep exercise walking
        cooking eating meal conversation hands smile portrait lifestyle""".split()),

    "food": frozenset("""food cooking kitchen meal recipe ingredient vegetable
        fruit bread meat fish rice grain spice farm farming agriculture crop
        harvest field orchard livestock cattle dairy market restaurant chef""".split()),

    "sport": frozenset("""sport sports athlete athletic running runner swimming
        cycling football soccer basketball tennis golf gym fitness training
        stadium race marathon yoga stretching""".split()),

    "money": frozenset("""money finance financial economy economic business trade
        market bank banking currency coin investment stock commerce shop retail
        commercial office meeting corporate""".split()),

    # Europe gets its own topic for one concrete reason: without it Europeana
    # was strictly dominated — every topic it covers, Smithsonian and Openverse
    # also cover, with higher reliability, so it would never have led a single
    # scene and would have been dead weight in the registry. These are the
    # subjects where the American collections genuinely thin out and the
    # European institutions do not. Words that also read as general history
    # (medieval, renaissance, castle) stay in `history` as well, so a scene
    # only reaches here when it is specifically European.
    "europe": frozenset("""europe european byzantine gothic monastery abbey
        tapestry chateau flemish baroque rococo bavarian tuscan habsburg
        prussian ottoman papal vatican louvre versailles""".split()),

    "culture": frozenset("""culture cultural religion religious church mosque
        temple ritual ceremony festival tradition myth mythology legend folklore
        music musical instrument dance theatre theater book library reading
        writing language school education classroom""".split()),
}

# Reverse index, built once: surface word -> every topic that claims it.
#
# This is a SET per word, not a single topic, and that matters. The obvious
# version — {w: topic for topic, words in ... for w in words} — silently gives
# each word to whichever topic happens to be defined last in the file. Eighteen
# words were claimed twice under that scheme, and the resolution was invisible:
# `solar` landed in tech rather than space, so "solar system" routed to stock;
# `doctor`, `nurse` and `patient` landed in people rather than medicine.
#
# Words genuinely do mean several things at once — a fish is ocean AND food, a
# kitchen is people AND food — so the honest representation is all of them, and
# route() weighs the overlap. The cost is that a word placed carelessly now
# pulls a scene towards every topic listed, which is why tools/test_routing.py
# checks the cases that ambiguity would break.
_WORD2TOPIC: dict[str, frozenset] = {}
for _topic, _words in TOPICS.items():
    for _w in _words:
        _WORD2TOPIC[_w] = _WORD2TOPIC.get(_w, frozenset()) | {_topic}

# The closed set of canonical topics, for anything that needs to name one from
# outside — e.g. the scene generator asks the model to bucket each scene into one
# of these. Kept here because TOPICS is the single source of truth: add a topic
# and it is offered automatically, with no second list to keep in sync.
CANON_TOPICS: tuple = tuple(sorted(TOPICS))


def is_topic(name: str) -> bool:
    """Whether `name` is one of the canonical topics (used to validate a model tag)."""
    return name in TOPICS


def topics_in(*texts: str) -> set:
    """Canonical topics mentioned anywhere in the given text.

    Deliberately generous: a scene tagged `astronomy` whose query mentions a
    telescope should reach NASA on either signal alone.
    """
    found = set()
    for t in texts:
        for w in re.findall(r"[a-z]+", (t or "").lower()):
            topics = _WORD2TOPIC.get(w) or _WORD2TOPIC.get(w.rstrip("s"))
            if topics:
                found |= topics
    return found


# ─────────────────────────────────────────────────────────────── registry

REGISTRY: dict[str, Source] = {
    "nasa": Source(
        "nasa", nasa, media=("IMAGE", "VIDEO"),
        covers=frozenset({"space"}),
        reliability=1.8, probe="moon surface",
        note="space, Earth observation, aeronautics. Public domain."),

    "smithsonian": Source(
        "smithsonian", smithsonian, media=("IMAGE",),
        covers=frozenset({"history", "art", "science", "nature", "culture",
                          "transport", "geology"}),
        # api.data.gov accepts DEMO_KEY at a low rate limit, so this is usable
        # with no configuration; a free key just raises the ceiling.
        reliability=1.6, probe="butterfly specimen",
        note="objects, specimens, artefacts, artworks. All CC0."),

    "openverse": Source(
        "openverse", openverse, media=("IMAGE",),
        covers=frozenset({"art", "history", "nature", "science", "culture",
                          "geology", "ocean", "transport"}),
        # The only archive that reports dimensions at search time, so its
        # rejects cost nothing — a real advantage, not a guess.
        reliability=1.5, probe="roman aqueduct",
        note="many providers behind one API. CC0 and Public Domain Mark only."),

    "loc": Source(
        "loc", loc, media=("IMAGE",),
        covers=frozenset({"history", "art", "culture", "transport", "people"}),
        # `people` is here for one narrow reason: LoC holds enormous
        # documentary photography of ordinary life — FSA farm families, factory
        # floors, street scenes. It will never outscore stock for a MODERN
        # kitchen, because such a scene carries no history topic, so the
        # generalist bonus keeps stock ahead. It wins when a scene is about
        # people AND the past, which is precisely where it should.
        reliability=1.4, probe="dust bowl farm family",
        note="19th and 20th century photography, prints, posters, maps."),

    "wikimedia": Source(
        "wikimedia", wikimedia, media=("IMAGE",),
        covers=frozenset({"history", "art", "science", "nature", "geology",
                          "culture", "transport", "ocean", "space"}),
        reliability=1.3, probe="roman aqueduct",
        note="the largest archive. Only its public-domain and CC0 files are used."),

    "europeana": Source(
        "europeana", europeana, media=("IMAGE",),
        needs_key="europeana_key",
        covers=frozenset({"art", "history", "culture", "science", "europe"}),
        # Lowest of the archives on purpose: its media links point at 4,000
        # different institutions, so a fair share are viewer pages or small
        # derivatives. Unmatched for European art when it does land.
        reliability=1.0, probe="illuminated manuscript",
        note="4,000 European museums and libraries. CC0 and PDM only."),

    "ia": Source(
        "ia", internet_archive, media=("VIDEO",),
        # Only `history`, and only because that topic now absorbs the era words
        # (wartime, newsreel, vintage, archival) a model actually reaches for.
        # Keeping covers a strict subset of ARCHIVE_FIRST is what guarantees IA
        # can never fire on a MODERN video scene, where it holds nothing —
        # `route()` gates on overlap, and a modern subject has none here.
        covers=frozenset({"history"}),
        # Lowest reliability of any source: archival derivatives vary wildly,
        # some items are broken, and the strict licence gate rejects most hits.
        # It ranks below stock for every video scene by design — tried only
        # when the subject is period footage stock cannot possibly hold.
        reliability=1.1, probe="vintage newsreel city street",
        note="archival motion — the only free source for footage of the past."),

    # Stock sites are generalists: shallow on everything, and the only place
    # with modern life and modern motion. They are always a valid last resort,
    # which is what `generalist` means here.
    "pexels": Source("pexels", None, media=("IMAGE", "VIDEO"),
                     needs_key="pexels_key", generalist=True,
                     covers=frozenset({"people", "food", "sport", "money", "tech",
                                       "transport", "nature", "weather", "medicine"}),
                     reliability=1.5,
                     note="modern life, modern motion. The only source for either."),

    "pixabay": Source("pixabay", None, media=("IMAGE", "VIDEO"),
                      needs_key="pixabay_key", generalist=True,
                      covers=frozenset({"people", "food", "sport", "money", "tech",
                                        "transport", "nature", "weather", "medicine"}),
                      reliability=1.4,
                      note="second generalist, different catalogue to Pexels."),
}

# Topics where an archive is genuinely better than stock, so it should be asked
# FIRST. Everything else starts with stock, because for modern subjects the
# archives simply have nothing and asking them wastes a request.
ARCHIVE_FIRST = frozenset({"space", "history", "art", "science", "geology",
                           "culture", "ocean", "nature", "europe"})

# Never ask more than this many sources for one scene. Every routed source is
# now pooled together and CLIP picks the best across all of them, so a couple
# more places to look is a direct accuracy win — but each is still a request, so
# this stays bounded to keep a 115-scene run to minutes, not hours.
MAX_SOURCES = 4


def route(domain: str, media: str, available: set, query: str = "",
          topic: str = "") -> list[str]:
    """Which sources to ask for this scene, best first.

    `available` is the set usable right now, so a source whose key is missing
    is simply not offered and a half-configured install degrades to "fewer
    places to look" rather than an error.

    `topic` is an OPTIONAL canonical topic (one of CANON_TOPICS) — normally the
    label the scene generator's model assigned. It is how routing scales beyond
    the fixed word list: the model can bucket ANY subject on Earth into a known
    topic even when the exact word is not in the vocabulary. It is unioned with
    the words found in the domain/query, so it only ever ADDS a signal — a wrong
    or empty tag can never route worse than the words alone. The function stays
    pure and offline: no model is called here; the topic is passed in.

    Subject match dominates: every reliability bonus is smaller than one topic
    of overlap, so a source that holds the material always beats a source that
    is merely dependable. The bonus only decides between sources that hold the
    material equally — which, with five archives sharing `history`, is most
    scenes. Before it existed those ties were broken by source NAME, so
    Europeana led every historical scene for no better reason than the letter E.
    """
    found = topics_in(domain, query)
    if topic in TOPICS:                        # a valid model tag adds its topic
        found = found | {topic}

    scored: list[tuple] = []
    for name in available:
        src = REGISTRY.get(name)
        if src is None or not src.can(media):
            continue                       # declared incapable: never asked
        if is_down(name):
            continue                       # unreachable this run; stop asking

        overlap = len(found & src.covers)
        score = overlap * 10.0

        if src.generalist:
            if not (found & ARCHIVE_FIRST):
                # Modern subject: stock is not a fallback, it is the answer.
                score += 11.0
            elif overlap == 0:
                # A period subject stock holds nothing for. Still worth a place
                # at the bottom, because an archive returning nothing must not
                # leave the scene with nowhere left to look.
                score += 3.0
            # else: a period subject stock ALSO covers — deliberately no bonus.
            # Stock's strength is modern framing, which is a liability here: a
            # staged modern photo of a "farm family" is not a 1930s farm family.
        else:
            # A specialist leads on the subjects it specialises in. Without
            # this, a stock library covering two loose topics outscored the one
            # archive that actually holds the period material, by a margin
            # thinner than the tie-break — which is far too close for a
            # decision this clear-cut.
            score += len(found & ARCHIVE_FIRST & src.covers) * 6.0

        # Motion overrides subject entirely. No free archive holds modern 16:9
        # video, so a VIDEO scene starts with stock whatever it is about.
        if media == "VIDEO" and src.generalist:
            score += 20.0

        # Openverse aggregates ~800M CC images — including a great deal of modern
        # Flickr life — behind one API, so for an IMAGE scene it is worth a place
        # as extra breadth even when the subject matches no archive topic. A small
        # base keeps it below the two stock generalists (which lead modern scenes)
        # while still adding a third pool of candidates for CLIP to choose from.
        if media == "IMAGE" and name == "openverse" and score < 2.0:
            score = 2.0

        # Earning a place and being ordered within it are separate questions.
        # An archive with no topic overlap holds nothing for this scene and is
        # not asked at all — the bonus must not sneak it back in, which is why
        # it is added only after the gate.
        if score > 0:
            scored.append((score + src.reliability, name))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [n for _, n in scored[:MAX_SOURCES]]


def explain(domain: str, media: str, available: set, query: str = "",
            topic: str = "") -> str:
    """Human-readable reason for a route, for `faceless sources` and debugging."""
    found = topics_in(domain, query)
    if topic in TOPICS:
        found = found | {topic}
    r = route(domain, media, available, query, topic)
    topic_s = ", ".join(sorted(found)) or "no recognised topic"
    return f"{topic_s} -> {r}"


# ─────────────────────────────────────────────────────── circuit breaker
#
# A source the network cannot reach fails identically for every scene. Across
# 115 scenes that is 115 connection attempts, each waiting out its own timeout,
# for a source that was never going to answer. Once one has failed repeatedly
# it is marked down for the rest of the run and skipped, so the ladder moves
# straight to the next source instead of stalling.
#
# Deliberately per-process, not persisted: a network comes back, an outage
# ends, and nothing should need clearing by hand to notice that.

_FAILS: dict[str, int] = {}
FAIL_LIMIT = 3


def note_failure(name: str) -> None:
    _FAILS[name] = _FAILS.get(name, 0) + 1


def note_success(name: str) -> None:
    _FAILS.pop(name, None)


def is_down(name: str) -> bool:
    return _FAILS.get(name, 0) >= FAIL_LIMIT


def down_sources() -> list[str]:
    return sorted(n for n in _FAILS if is_down(n))


def reset_failures() -> None:
    _FAILS.clear()


def diagnose(name: str, cfg: dict | None = None) -> list[tuple]:
    """Work out WHY a source is failing, rather than guessing.

    A connection reset has several possible causes that look identical from the
    outside, and I have already guessed wrong once here. This tries the same
    host several ways and reports which combinations work, so the cause can be
    read off the results instead of theorised:

        homepage fails too      -> the network cannot reach this host at all
                                   (block, DNS, proxy, VPN, country filter)
        homepage ok, API resets -> the host is fine and the API is refusing us
        browser agent works     -> our User-Agent is the problem
        everything resets       -> TLS, or something between here and there
    """
    import ssl
    cfg = cfg or {}
    configure(cfg)

    PROBES = {
        "nasa": ("https://images-api.nasa.gov/",
                 "https://images-api.nasa.gov/search?q=moon&media_type=image"),
        "smithsonian": ("https://api.si.edu/",
                        "https://api.si.edu/openaccess/api/v1.0/search"
                        "?q=butterfly&rows=1&api_key=DEMO_KEY"),
        "openverse": ("https://api.openverse.org/",
                      "https://api.openverse.org/v1/images/?q=test&page_size=1"),
        "wikimedia": ("https://commons.wikimedia.org/",
                      "https://commons.wikimedia.org/w/api.php"
                      "?action=query&format=json&titles=File:Example.jpg&prop=imageinfo"),
        "loc": ("https://www.loc.gov/",
                "https://www.loc.gov/photos/?q=bridge&fo=json&c=1&at=results"),
        # Deliberately keyless: a 401 still proves the host is reachable, which
        # is the only thing this function is trying to establish.
        "europeana": ("https://www.europeana.eu/",
                      "https://api.europeana.eu/record/v2/search.json"
                      "?wskey=probe&query=test&rows=1"),
        "ia": ("https://archive.org/",
               "https://archive.org/advancedsearch.php"
               "?q=test&rows=1&output=json"),
    }
    if name not in PROBES:
        return [("unknown source", False, name)]

    home, api = PROBES[name]
    BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

    def attempt(label, url, ua, timeout=12):
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return (label, True, f"HTTP {r.status}, {len(r.read(2048))} bytes read")
        except urllib.error.HTTPError as e:
            # An HTTP error still proves we reached the server and it answered.
            return (label, True, f"HTTP {e.code} (reached the server)")
        except Exception as e:
            return (label, False, f"{type(e).__name__}: {str(e)[:70]}")

    return [
        attempt("homepage, our agent", home, user_agent()),
        attempt("homepage, browser agent", home, BROWSER),
        attempt("api, our agent", api, user_agent()),
        attempt("api, browser agent", api, BROWSER),
    ]


def usable(cfg: dict) -> set:
    """Sources that are configured and callable right now.

    A source needing a key it does not have is simply not offered, so a
    half-configured install degrades to "fewer places to look" rather than an
    error mid-run. Several of these need no key at all, which is why the
    pipeline still finds pictures with an empty config.json.
    """
    ok = set()
    for name, src in REGISTRY.items():
        if src.needs_key:
            if cfg.get(src.needs_key):
                ok.add(name)
        else:
            ok.add(name)
    return ok


def search(name: str, query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    """Ask one source. Raises SourceError; the caller moves on to the next."""
    configure(cfg)
    src = REGISTRY.get(name)
    if src is None or src.search is None:
        raise SourceError(f"no such source: {name}")
    try:
        hits = src.search(query, media, want, cfg) or []
    except SourceError:
        note_failure(name)
        raise
    note_success(name)
    # Second pass for sources that DO report dimensions. The ones that do not
    # are caught after download, in stock.fetch. The floor is media-aware:
    # applying the 1100px stills floor here would have discarded every archival
    # video clip, which is exactly the material this source exists to provide.
    floor = floor_for(media)
    return [h for h in hits if not h.width or h.width >= floor]
