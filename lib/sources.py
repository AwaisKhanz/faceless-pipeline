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

    def can(self, media: str) -> bool:
        return media in self.media


# ─────────────────────────────────────────────────────────────── http

def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent(),
                      "Accept": "application/json, */*", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except Exception as e:
        msg = str(e)
        # A reset rather than an HTTP error is the signature of an agent-based
        # block. Say so, because "connection reset by peer" alone sends people
        # looking at their network.
        if "reset" in msg.lower() or "forcibly closed" in msg.lower():
            msg += (f" — the server dropped the connection, which usually means "
                    f"it rejected our User-Agent ({user_agent()}). "
                    f"Set \"contact\" in config.json to a URL or email you own.")
        raise SourceError(msg)


def _json(url: str, headers: dict | None = None) -> dict:
    raw = _get(url, headers)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SourceError("the server did not return JSON")


def _https(url: str) -> str:
    """Upgrade plain http to https.

    NASA's asset manifests hand back http:// URLs. Plenty of networks and
    corporate proxies block that outright, and there is no reason to download
    a public file in the clear. The hosts all serve https.
    """
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

    "geology": frozenset("""geology geological tectonic volcano volcanic
        earthquake seismic mountain mountains canyon glacier ice arctic antarctic
        desert erosion mineral rock rocks crust mantle lava magma fossil
        continent island landscape terrain sediment strata quarry mine
        mining excavation drilling""".split()),

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
        manuscript century dynasty war battle soldier army treaty""".split()),

    "art": frozenset("""art artistic painting paint portrait sculpture statue
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

    "culture": frozenset("""culture cultural religion religious church mosque
        temple ritual ceremony festival tradition myth mythology legend folklore
        music musical instrument dance theatre theater book library reading
        writing language school education classroom""".split()),
}

# Reverse index, built once: surface word -> canonical topic.
_WORD2TOPIC: dict[str, str] = {
    w: topic for topic, words in TOPICS.items() for w in words
}


def topics_in(*texts: str) -> set:
    """Canonical topics mentioned anywhere in the given text.

    Deliberately generous: a scene tagged `astronomy` whose query mentions a
    telescope should reach NASA on either signal alone.
    """
    found = set()
    for t in texts:
        for w in re.findall(r"[a-z]+", (t or "").lower()):
            topic = _WORD2TOPIC.get(w) or _WORD2TOPIC.get(w.rstrip("s"))
            if topic:
                found.add(topic)
    return found


# ─────────────────────────────────────────────────────────────── registry

REGISTRY: dict[str, Source] = {
    "nasa": Source(
        "nasa", nasa, media=("IMAGE", "VIDEO"),
        covers=frozenset({"space"}),
        note="space, Earth observation, aeronautics. Public domain."),

    "smithsonian": Source(
        "smithsonian", smithsonian, media=("IMAGE",),
        covers=frozenset({"history", "art", "science", "nature", "culture",
                          "transport", "geology"}),
        # api.data.gov accepts DEMO_KEY at a low rate limit, so this is usable
        # with no configuration; a free key just raises the ceiling.
        note="objects, specimens, artefacts, artworks. All CC0."),

    "openverse": Source(
        "openverse", openverse, media=("IMAGE",),
        covers=frozenset({"art", "history", "nature", "science", "culture",
                          "geology", "ocean", "transport"}),
        note="many providers behind one API. CC0 and Public Domain Mark only."),

    "wikimedia": Source(
        "wikimedia", wikimedia, media=("IMAGE",),
        covers=frozenset({"history", "art", "science", "nature", "geology",
                          "culture", "transport", "ocean", "space"}),
        note="the largest archive. Only its public-domain and CC0 files are used."),

    # Stock sites are generalists: shallow on everything, and the only place
    # with modern life and modern motion. They are always a valid last resort,
    # which is what `generalist` means here.
    "pexels": Source("pexels", None, media=("IMAGE", "VIDEO"),
                     needs_key="pexels_key", generalist=True,
                     covers=frozenset({"people", "food", "sport", "money", "tech",
                                       "transport", "nature", "weather", "medicine"}),
                     note="modern life, modern motion. The only source for either."),

    "pixabay": Source("pixabay", None, media=("IMAGE", "VIDEO"),
                      needs_key="pixabay_key", generalist=True,
                      covers=frozenset({"people", "food", "sport", "money", "tech",
                                        "transport", "nature", "weather", "medicine"}),
                      note="second generalist, different catalogue to Pexels."),
}

# Topics where an archive is genuinely better than stock, so it should be asked
# FIRST. Everything else starts with stock, because for modern subjects the
# archives simply have nothing and asking them wastes a request.
ARCHIVE_FIRST = frozenset({"space", "history", "art", "science", "geology",
                           "culture", "ocean", "nature"})

# Never ask more than this many sources for one scene. Three ladder rungs times
# eight sources would be 24 requests per scene; at 115 scenes that is a sourcing
# run measured in hours.
MAX_SOURCES = 3


def route(domain: str, media: str, available: set, query: str = "") -> list[str]:
    """Which sources to ask for this scene, best first.

    `available` is the set usable right now, so a source whose key is missing
    is simply not offered and a half-configured install degrades to "fewer
    places to look" rather than an error.
    """
    found = topics_in(domain, query)

    scored: list[tuple] = []
    for name in available:
        src = REGISTRY.get(name)
        if src is None or not src.can(media):
            continue                       # declared incapable: never asked

        overlap = len(found & src.covers)
        score = overlap * 10.0

        # A generalist is never a strong match but is never useless either,
        # which is exactly the tie-break behaviour wanted at the bottom.
        if src.generalist:
            score += 3.0
            # Modern subjects: stock is not a fallback, it is the right answer.
            if not (found & ARCHIVE_FIRST):
                score += 8.0

        # Motion overrides subject entirely. No free archive holds modern 16:9
        # video, so a VIDEO scene starts with stock whatever it is about.
        if media == "VIDEO" and src.generalist:
            score += 20.0

        if score > 0:
            scored.append((score, name))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [n for _, n in scored[:MAX_SOURCES]]


def explain(domain: str, media: str, available: set, query: str = "") -> str:
    """Human-readable reason for a route, for `faceless sources` and debugging."""
    found = topics_in(domain, query)
    r = route(domain, media, available, query)
    topic_s = ", ".join(sorted(found)) or "no recognised topic"
    return f"{topic_s} -> {r}"


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
    hits = src.search(query, media, want, cfg) or []
    # Second pass for sources that DO report dimensions. The ones that do not
    # are caught after download, in stock.fetch.
    return [h for h in hits if not h.width or h.width >= MIN_WIDTH]
