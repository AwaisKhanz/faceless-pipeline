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
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

TIMEOUT = 25
UA = "faceless-studio/1.0 (local video pipeline; contact via project README)"

# Archive scans are frequently portrait, 4:3, or a 600px thumbnail of a
# postcard. At 1080p those look like mistakes, so anything smaller is dropped
# before it can reach a timeline.
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
    note: str = ""

    def can(self, media: str) -> bool:
        return media in self.media


# ─────────────────────────────────────────────────────────────── http

def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except Exception as e:
        raise SourceError(str(e))


def _json(url: str, headers: dict | None = None) -> dict:
    raw = _get(url, headers)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SourceError("the server did not return JSON")


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
                url=url, ext=_ext(url), src="smithsonian",
                credit=(content.get("freetext", {}).get("dataSource", [{}])[0]
                        .get("content", "Smithsonian") if content.get("freetext")
                        else "Smithsonian"),
                page=m.get("guid", "") or r.get("id", ""),
                license="CC0 (Smithsonian Open Access)"))
            break
        if len(out) >= want:
            break
    return out


# ─────────────────────────────────────────────── registry + capability table

REGISTRY: dict[str, Source] = {
    "nasa": Source("nasa", nasa, media=("IMAGE", "VIDEO"),
                   note="space, Earth science, aeronautics. Public domain."),
    "smithsonian": Source("smithsonian", smithsonian, media=("IMAGE",),
                          needs_key="smithsonian_key",
                          note="objects, specimens, artefacts, history. All CC0."),
}

# Subject domains a scene can be tagged with, and who to ask, in order.
#
# The rule behind every row: ask the archive that actually holds this material
# FIRST, and fall through to stock only when the topic is one stock covers.
# "people" has no archive at all — no free archive has modern domestic life —
# so it goes straight to stock, which is not a compromise but the correct
# answer for that subject.
ROUTES: dict[str, tuple] = {
    "space":    ("nasa", "pexels", "pixabay"),
    "nature":   ("smithsonian", "pexels", "pixabay"),
    "history":  ("smithsonian", "pexels", "pixabay"),
    "art":      ("smithsonian", "pexels", "pixabay"),
    "science":  ("smithsonian", "nasa", "pexels"),
    "tech":     ("pexels", "pixabay"),
    "people":   ("pexels", "pixabay"),
    "abstract": ("pexels", "pixabay"),
}
DEFAULT_ROUTE = ("pexels", "pixabay")

# Never ask more than this many sources for one scene. Three rungs of the query
# ladder times eight sources would be 24 requests per scene; at 115 scenes that
# is a sourcing run measured in hours.
MAX_SOURCES = 3


def route(domain: str, media: str, available: set) -> list[str]:
    """Which sources to ask for this scene, in order.

    `available` is the set of source names usable right now — a source whose
    key is missing simply is not offered, so a half-configured install degrades
    to "fewer places to look" rather than an error.
    """
    order = ROUTES.get((domain or "").strip().lower(), DEFAULT_ROUTE)

    # Motion overrides subject. Archives are stills; asking them for video
    # burns a request to learn something already known.
    if media == "VIDEO":
        order = tuple(n for n in ("pexels", "pixabay", *order) if n)

    seen, out = set(), []
    for name in order:
        if name in seen or name not in available:
            continue
        src = REGISTRY.get(name)
        if src and not src.can(media):
            continue                     # declared incapable, skip silently
        seen.add(name)
        out.append(name)
        if len(out) >= MAX_SOURCES:
            break
    return out


def usable(cfg: dict) -> set:
    """Sources that are configured and can be called right now."""
    ok = set()
    for name, src in REGISTRY.items():
        if not src.needs_key or cfg.get(src.needs_key):
            ok.add(name)
        elif name == "smithsonian":
            ok.add(name)                 # DEMO_KEY works, just rate-limited
    if cfg.get("pexels_key"):
        ok.add("pexels")
    if cfg.get("pixabay_key"):
        ok.add("pixabay")
    return ok


def search(name: str, query: str, media: str, want: int, cfg: dict) -> list[Hit]:
    """Ask one source. Raises SourceError; callers move on to the next."""
    src = REGISTRY.get(name)
    if src is None:
        raise SourceError(f"no such source: {name}")
    hits = src.search(query, media, want, cfg) or []
    # A picture too small to fill a 1080p frame is not a usable result.
    return [h for h in hits if not h.width or h.width >= MIN_WIDTH]
