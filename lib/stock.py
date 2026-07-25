"""Stock sourcing from Pexels, falling back to Pixabay. Cached and resumable.

Both APIs are free. Keys:
  Pexels  -> https://www.pexels.com/api/    (200 requests/hour)
  Pixabay -> https://pixabay.com/api/docs/  (100 requests/minute)

Everything is cached by (query, media, index) so re-running costs no requests and
picking an alternate take is instant.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import subprocess
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import sources as _SRC
from . import vision

UA = {"User-Agent": "faceless-pipeline/1.0"}
TIMEOUT = 30

# Below this relevance a candidate is judged not really about the scene, and
# sourcing searches harder before settling. Softmax probability over the scene
# concept vs a list of junk concepts, so 0.45 means "more likely the subject
# than any kind of junk". Tunable in config.json as clip_min.
DEFAULT_CLIP_MIN = 0.45

# How much a looser fallback query must beat the scene's own query by, per step
# down the ladder, before it is allowed to replace it. CLIP scores a short loose
# phrase higher than a long specific one, so without this the off-scene fallback
# wins on nearly every scene. At 0.12 a fallback has to be clearly, not
# marginally, better — otherwise the shot the scene actually asked for is kept.
_RUNG_PENALTY = 0.12

# Last-resort queries when a scene's own ladder finds nothing at all. Free stock
# always has neutral, atmospheric backgrounds, so these guarantee a scene can be
# filled rather than left empty — an empty scene breaks the entire render. What
# lands here is flagged as a placeholder so it stands out for a manual swap; it
# is a safety net, not a first choice. Ordered calm → generic.
_SAFETY_QUERIES = ["dark abstract background", "soft light background",
                   "blurred bokeh lights", "calm abstract background"]

# A small pool of raw image bytes, so walking the query ladder and stepping over
# duplicates does not re-download the same candidate. Bounded so memory stays
# flat over a 115-scene run.
_BYTES: "OrderedDict[str, bytes]" = OrderedDict()
_BYTES_CAP = 64
_BYTES_LOCK = threading.Lock()      # the cache is read/written from scoring threads

# Scoring downloads run in parallel and fail fast — a thumbnail that is slow to
# answer is skipped rather than waited on, because it is one of many candidates.
_SCORE_TIMEOUT = 8                  # seconds per scoring thumbnail (vs 30 for a winner)
_SCORE_WORKERS_IMAGE = 16           # concurrent thumbnail downloads for image scenes
_SCORE_WORKERS_VIDEO = 8            # ffmpeg is cheap and the machine sits idle, so
                                    # score more clips at once — video is the slowest
                                    # scene type and it is all network waiting.
_SCORE_LOCK = threading.Lock()      # one GPU scoring pass at a time when scenes run
                                    # in parallel (source_workers); downloads still
                                    # overlap — only the model forward is serialized.


def _fetch_bytes(url: str, timeout: int | None = None, retries: int = 3) -> bytes:
    with _BYTES_LOCK:
        if url in _BYTES:
            _BYTES.move_to_end(url)
            return _BYTES[url]
    b = _get(url, timeout=timeout, retries=retries)   # network OUTSIDE the lock
    with _BYTES_LOCK:
        _BYTES[url] = b
        while len(_BYTES) > _BYTES_CAP:
            _BYTES.popitem(last=False)
    return b


class StockError(RuntimeError):
    pass


def _get(url: str, headers: dict | None = None,
         timeout: int | None = None, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    to = timeout or TIMEOUT
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=to) as r:
                return r.read()
        except Exception as e:
            if attempt == retries:
                raise StockError(f"{type(e).__name__}: {e}")
            time.sleep(1.5 * attempt)
    raise StockError("unreachable")


def _slug(q: str, media: str, idx: int, sources: list[str] | None = None) -> str:
    # The source order is part of the identity: the same query routed to NASA
    # and to Pexels are different searches, and sharing a cache entry would
    # hand back whichever ran first.
    tag = ",".join(sources or [])
    h = hashlib.sha1(f"{q}|{media}|{idx}|{tag}".encode("utf-8")).hexdigest()[:14]
    safe = "".join(c if c.isalnum() else "-" for c in q.lower())[:44].strip("-")
    return f"{safe}_{media}_{idx}_{h}"


# ------------------------------------------------------------------- providers

def _pexels(query: str, media: str, key: str, want: int) -> list[dict]:
    base = "https://api.pexels.com/videos/search" if media == "VIDEO" \
        else "https://api.pexels.com/v1/search"
    qs = urllib.parse.urlencode(
        {"query": query, "per_page": max(want, 5),
         "orientation": "landscape", "size": "large"})
    data = json.loads(_get(f"{base}?{qs}", {"Authorization": key}))

    out = []
    if media == "VIDEO":
        for v in data.get("videos", []):
            files = [f for f in v.get("video_files", [])
                     if f.get("width") and f["width"] >= 1280]
            if not files:
                continue
            best = min(files, key=lambda f: abs(f["width"] - 1920))
            out.append({"url": best["link"], "ext": ".mp4",
                        "credit": v.get("user", {}).get("name", ""),
                        "page": v.get("url", ""), "src": "pexels",
                        "thumb": v.get("image", ""),      # poster frame, for scoring
                        "width": int(best.get("width") or 0),
                        "height": int(best.get("height") or 0)})
    else:
        for p in data.get("photos", []):
            out.append({"url": p["src"]["large2x"], "ext": ".jpg",
                        "credit": p.get("photographer", ""),
                        "page": p.get("url", ""), "src": "pexels",
                        "thumb": p.get("src", {}).get("medium", ""),
                        "width": int(p.get("width") or 0),
                        "height": int(p.get("height") or 0)})
    return out


def _pixabay(query: str, media: str, key: str, want: int) -> list[dict]:
    base = "https://pixabay.com/api/videos/" if media == "VIDEO" \
        else "https://pixabay.com/api/"
    params = {"key": key, "q": query, "per_page": max(want, 5), "safesearch": "true"}
    if media != "VIDEO":
        params.update({"image_type": "photo", "orientation": "horizontal",
                       "min_width": "1600"})
    data = json.loads(_get(f"{base}?{urllib.parse.urlencode(params)}"))

    out = []
    for h in data.get("hits", []):
        if media == "VIDEO":
            vids = h.get("videos", {})
            pick = vids.get("large") or vids.get("medium") or vids.get("small")
            if not pick or not pick.get("url"):
                continue
            out.append({"url": pick["url"], "ext": ".mp4",
                        "credit": h.get("user", ""),
                        "page": h.get("pageURL", ""), "src": "pixabay",
                        "width": int(pick.get("width") or 0),
                        "height": int(pick.get("height") or 0)})
        else:
            url = h.get("largeImageURL") or h.get("webformatURL")
            if not url:
                continue
            out.append({"url": url, "ext": ".jpg", "credit": h.get("user", ""),
                        "page": h.get("pageURL", ""), "src": "pixabay",
                        "thumb": h.get("webformatURL", ""),   # 640px, for scoring
                        "width": int(h.get("imageWidth") or 0),
                        "height": int(h.get("imageHeight") or 0)})
    return out


# ---------------------------------------------------------------- scoring

# The frame is 16:9. A candidate that fills it at 1080p or better is ideal;
# portrait scans and small images are what make a video look cheap.
_IDEAL_AR = 16 / 9

# How many candidates to pull per source so there is a real choice to rank.
# It costs one request whatever the number — only the winner is downloaded — so
# a wider net is close to free and turns "take the first hit" into "take the
# best of several".
POOL = 8


def _score_pool(cfg: dict | None) -> int:
    """How many pooled candidates to actually CLIP-score.

    Scoring means fetching each candidate's thumbnail, so this trades accuracy
    for time and is sized to the machine: a big net on a real GPU, a modest one
    on a laptop CPU. With scoring off it doesn't matter — the caller keeps the
    technical order — so POOL is fine.
    """
    try:
        cap = vision.capability(cfg or {})
        if not cap.get("ok"):
            return POOL
        return {"cuda": 30, "mps": 18}.get(cap.get("device"), 12)
    except Exception:
        return POOL


def _fair_pool(results: list[dict], n: int) -> list[dict]:
    """Pick up to `n` candidates to CLIP-score, giving EVERY source a fair place.

    Ranking by technical fit (16:9, resolution) before scoring quietly buried the
    sources that report no dimensions — NASA and Smithsonian score 0 on `_score`,
    so they sorted to the bottom, fell outside the scored window, and could never
    win however relevant their picture was. That is why an all-space script used
    zero NASA. This round-robins across sources instead: the first candidate from
    each source, then the second from each, and so on. Every routed source is
    always looked at, and relevance then picks the best on merit — not on who
    happened to report a width.
    """
    by_src: "OrderedDict[str, list]" = OrderedDict()
    for h in results:
        by_src.setdefault(h.get("src", ""), []).append(h)
    lists = list(by_src.values())
    out: list[dict] = []
    i = 0
    while len(out) < n and any(i < len(lst) for lst in lists):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
                if len(out) >= n:
                    break
        i += 1
    return out


def _score(hit: dict) -> float:
    """Rank a candidate by how well it fits a 16:9 1080p frame.

    Uses only the dimensions the search API already returned — nothing is
    downloaded to score. A source that reports no dimensions (NASA, Smithsonian)
    scores 0: neutral, so it keeps the order routing gave it and is judged for
    size after download, exactly as before. Sorting is stable, so equal scores
    never disturb that routed order.
    """
    w = int(hit.get("width") or 0)
    h = int(hit.get("height") or 0)
    if not w or not h:
        return 0.0
    ar = w / h
    ar_score = max(0.0, 1.0 - abs(ar - _IDEAL_AR) / _IDEAL_AR)   # 1.0 at 16:9
    if ar < 1.0:
        ar_score -= 0.6                 # portrait wastes most of a 16:9 frame
    res_score = min(w, 2560) / 1920.0   # rewards 1080p+, flattens past ~1440p
    return round(3.0 * ar_score + res_score, 4)


# ---------------------------------------------------------------------- fetch

def _pixel_width(f: Path) -> int:
    """Width of an image or video file, or 0 if it cannot be determined.

    Uses ffprobe, already a hard dependency of this project, so this adds
    nothing to install. Failure returns 0 and the caller keeps the file — a
    probe that cannot run is not evidence the picture is bad.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width", "-of", "csv=p=0", str(f)],
            capture_output=True, text=True, timeout=20)
        return int((r.stdout.strip().split(",") or ["0"])[0] or 0)
    except Exception:
        return 0


def fetch(query: str, media: str, cache: Path, pexels_key: str | None,
          pixabay_key: str | None, index: int = 0,
          sources: list[str] | None = None, cfg: dict | None = None) -> dict:
    """Return {path, credit, page, src} for the `index`-th match of `query`.

    index=0 is the top match; bump it to pull an alternate take when a pick is
    rejected on the approval sheet.
    """
    cache.mkdir(parents=True, exist_ok=True)
    slug = _slug(query, media, index, sources)
    meta_p = cache / f"{slug}.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if Path(meta["path"]).exists():
            _rescore_if_stale(meta, meta_p, query, media, cfg)
            return meta

    results: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()

    # `sources` is the routed order for this scene. EVERY routed source is
    # queried and the results pooled together — not just the first that answers —
    # so CLIP chooses the best picture across all of them instead of the best of
    # one site. Each source is a single request and only the winner is
    # downloaded, so a wide net stays cheap.
    order = sources or ["pexels", "pixabay"]
    want = index + POOL          # depth per source; grows when a swap bumps index
    # Query every routed source AT ONCE. Each is a blocking network call, so
    # doing them one after another made an all_sources scene wait out the SUM of
    # every source's latency — with eight archives, and slow ones like the
    # Library of Congress, that is many seconds per scene. In parallel the scene
    # waits only for the slowest source, not the total. Results are merged back
    # in the routed order, so dedup and telemetry stay deterministic; only the
    # winner is downloaded, later and on the main thread.
    def _query_source(name: str):
        try:
            if name == "pexels" and pexels_key:
                got = _pexels(query, media, pexels_key, want)
            elif name == "pixabay" and pixabay_key:
                got = _pixabay(query, media, pixabay_key, want)
            elif name in _SRC.REGISTRY:
                got = [
                    {"url": h.url, "ext": h.ext, "credit": h.credit,
                     "page": h.page, "src": h.src, "license": h.license,
                     "thumb": getattr(h, "thumb", "") or "",
                     "width": h.width, "height": h.height}
                    for h in _SRC.search(name, query, media, want, cfg or {})]
            else:
                got = []
            return name, got, None
        except Exception as e:
            return name, [], f"{name}: {e}"

    if len(order) > 1:
        with ThreadPoolExecutor(max_workers=min(len(order), 8)) as ex:
            per_source = list(ex.map(_query_source, order))    # keeps input order
    else:
        per_source = [_query_source(n) for n in order]

    for name, got, err in per_source:        # merge in the routed order
        if err:
            errors.append(err)
        for h in got:
            u = h.get("url")
            if u and u not in seen:           # dedupe: the same file appears on
                seen.add(u)                   # more than one aggregator
                results.append(h)

    if len(results) <= index:
        raise StockError(
            f"No {media.lower()} result #{index + 1} for '{query}'. "
            + ("; ".join(errors) if errors else "Try a simpler, more literal query.")
        )

    # Choose which candidates to CLIP-score, giving every routed source a fair
    # place (see _fair_pool) instead of pre-sorting by technical fit — that used
    # to bury NASA/Smithsonian, which report no dimensions, before they were ever
    # looked at. The pool that gets scored is sized to the machine, so a GPU
    # compares many more candidates than a laptop.
    pool_n = _score_pool(cfg)
    scored_pool = _fair_pool(results, pool_n)
    rel = _relevance(scored_pool, query, media, cfg)
    ranked_all: list = []                             # every scored (src, rel), best first
    if rel:
        for h in scored_pool:
            h["rel"] = rel.get(h["url"])

        def _combined(h):
            r = h.get("rel")
            tech = min(_score(h), 4.5) / 4.5          # 0..1
            if r is None:
                return -1.0 + tech * 0.001            # unverified: below all scored
            return r + tech * 0.12                    # relevance leads, quality breaks ties

        # Relevance decides the winner across ALL sources; technical fit only
        # breaks ties between equally-relevant pictures. Anything not scored
        # keeps a sensible technical order behind the scored head.
        ranked = sorted(scored_pool, key=_combined, reverse=True)
        ranked_all = [(h["src"], round(h["rel"], 2)) for h in ranked
                      if h.get("rel") is not None]
        scored_urls = {h["url"] for h in scored_pool}
        rest = sorted((h for h in results if h["url"] not in scored_urls),
                      key=_score, reverse=True)
        results = ranked + rest
    else:
        # No relevance signal (scoring off, or nothing scorable): rank on
        # technical fit alone, so index 0 is still the best-framed candidate.
        results.sort(key=_score, reverse=True)

    hit = results[index]
    dest = cache / f"{slug}{hit['ext']}"
    dest.write_bytes(_fetch_bytes(hit["url"]))         # reuses the scored bytes

    # Measure what actually arrived. Archives happily return a 400px scan of a
    # postcard, which looks like a mistake at 1080p — and no search API reports
    # dimensions, so this is the first honest opportunity to check.
    w = _pixel_width(dest)
    floor = _SRC.floor_for(media)
    if w and w < floor:
        dest.unlink(missing_ok=True)
        raise StockError(
            f"{hit['src']} returned {w}px for '{query}' — below the "
            f"{floor}px floor for {media.lower()}")
    meta = {"path": str(dest), "credit": hit["credit"], "page": hit["page"],
            "src": hit["src"], "license": hit.get("license", ""),
            "query": query, "media": media, "index": index,
            "score": hit.get("rel"),            # relevance, or None if not scored
            "score_v": vision.SCORE_VERSION if rel else None}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Transient telemetry for the live log — how this pick was reached: which
    # sources were searched, how many candidates were pooled and scored, and the
    # top few by relevance. Attached AFTER the cache write and stripped by
    # fetch_all before assets.json, so it never persists.
    counts: dict[str, int] = {}
    for h in results:
        counts[h["src"]] = counts.get(h["src"], 0) + 1
    meta["_detail"] = {"sources": list(order), "counts": counts,
                       "pooled": len(results),
                       "scored": len(scored_pool) if rel else 0,
                       "ranked": ranked_all}
    return meta


def _rescore_if_stale(meta: dict, meta_p: Path, query: str, media: str,
                      cfg: dict | None) -> None:
    """Refresh a cached pick's match score in place when the scorer has moved on.

    Re-scores the file already on disk — a frame for a video, the image itself
    otherwise — so a calibration change is picked up on the next source without
    re-downloading anything or clearing the cache. Silent on any failure; a
    missing score just means the old ranking stands until a real re-source.
    """
    scorer = vision.get_scorer(cfg or {})
    if scorer is None or meta.get("score_v") == vision.SCORE_VERSION:
        return
    path = meta.get("path", "")
    try:
        if (meta.get("media") or media) == "VIDEO":
            raw = _video_frame(path)            # ffmpeg reads the local file too
        else:
            raw = Path(path).read_bytes()
        if raw:
            r = scorer.relevance(query, [(path, raw)])
            meta["score"] = r.get(path)
        meta["score_v"] = vision.SCORE_VERSION
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def _relevance(pool: list[dict], query: str, media: str,
               cfg: dict | None) -> dict[str, float]:
    """CLIP relevance for the candidates that have something cheap to look at.

    Downloads each candidate's thumbnail (a small poster or web-size image, not
    the full asset) and scores it. Returns url -> relevance. Empty when scoring
    is off or nothing was scorable, in which case the caller keeps the technical
    order. Never raises.
    """
    scorer = vision.get_scorer(cfg or {})
    if scorer is None:
        return {}

    # Fetch every candidate's thumbnail AT ONCE. This used to be a sequential
    # loop, and a single slow or dead thumbnail — retried three times at the full
    # 30s timeout — could stall a whole scene for minutes. Now the downloads run
    # in parallel AND fail fast: a scoring thumbnail is one of ~30 candidates, so
    # if it doesn't answer in a few seconds it is skipped, not waited on. Only the
    # eventual winner is downloaded properly (with full retries) later.
    def _grab(h: dict):
        try:
            if media == "IMAGE":
                # A small web-size copy is plenty for CLIP (it works at 224px).
                raw = _fetch_bytes(h.get("thumb") or h["url"],
                                   timeout=_SCORE_TIMEOUT, retries=1)
            elif h.get("thumb"):
                raw = _fetch_bytes(h["thumb"], timeout=_SCORE_TIMEOUT, retries=1)
            else:
                raw = _video_frame(h["url"])           # else pull one frame
            return (h["url"], raw) if raw else None
        except Exception:
            return None                                # unscorable: keep tech order

    workers = _SCORE_WORKERS_VIDEO if media == "VIDEO" else _SCORE_WORKERS_IMAGE
    n = min(len(pool), workers)
    if n > 1:
        with ThreadPoolExecutor(max_workers=n) as ex:
            fetched = list(ex.map(_grab, pool))
    else:
        fetched = [_grab(h) for h in pool]
    items = [x for x in fetched if x]
    if not items:
        return {}
    with _SCORE_LOCK:                  # only one scene on the GPU at a time
        return scorer.relevance(query, items)


def _video_frame(url: str) -> bytes:
    """One representative frame from a video URL, as JPEG bytes, for scoring.

    ffmpeg streams over http and stops after the first frame it needs, so this
    reads only the opening of the clip — it does NOT download the whole file to
    look at it. A second in avoids a black or fade-in opening frame. Returns b""
    on any failure, and the candidate simply goes unscored.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-ss", "1", "-i", url,
             "-frames:v", "1", "-vf", "scale=384:-1",
             "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True, timeout=15)     # a frame comes fast or not at all
        return r.stdout if r.returncode == 0 and r.stdout else b""
    except Exception:
        return b""


def _detail_line(detail: dict | None, full: bool = False) -> str:
    """The dim second line under a scene: which sources were searched, how deep
    the pool went, and the candidates by relevance. Shows the top 6 by default;
    with `full` it lists EVERY scored candidate. Empty when there is nothing to
    say (a cache hit carries no fresh telemetry)."""
    if not detail:
        return ""
    # Show every source that was ASKED with how many it actually returned, so a
    # source that was queried but came back empty (e.g. Wikimedia blocked on this
    # network, or a source the topic did not really suit) reads as "wikimedia 0"
    # instead of silently looking like it contributed. Nothing is hidden: if a
    # name shows 0 it found nothing this scene.
    order = detail.get("sources") or []
    counts = detail.get("counts") or {}
    srcs = "·".join(f"{name} {counts.get(name, 0)}" for name in order) or "stock"
    parts = [f"searched {srcs}", f"pooled {detail.get('pooled', 0)}"]
    if detail.get("scored"):
        parts.append(f"scored {detail['scored']}")
    ranked = detail.get("ranked") or []
    shown = ranked if full else ranked[:6]
    if shown:
        cand = " · ".join(f"{s} {int(r * 100)}%" for s, r in shown)
        if not full and len(ranked) > 6:
            cand += f" (+{len(ranked) - 6} more)"
        parts.append(("all: " if full else "top: ") + cand)
    return "       " + " · ".join(parts)


def _emit_scene_header(log, s) -> None:
    """Open a scene's step-by-step block (source_log: 'full'): a blank line, then
    what this scene is and what it wants on screen."""
    topic = getattr(s, "topic", "") or "—"
    text = (getattr(s, "text", "") or s.query or "").strip()
    log("")
    log(f"── Scene {s.n} · {s.media.lower()} · {topic}")
    if text:
        log(f"   narration: \"{text[:72]}\"")


def _emit_rung(log, query, detail, label, picked: bool, scorer_on: bool) -> None:
    """One rung of the ladder, step by step: the query, then EVERY source it hit
    with how many results each returned, then how many were pooled/scored and the
    best match. Nothing condensed — this is the 'show me everything' view."""
    log(f"   {label}: \"{query[:60]}\"")
    if not detail:
        log("      → no source returned anything for this query")
        return
    order = detail.get("sources") or []
    counts = detail.get("counts") or {}
    per = "  ".join(f"{name} {counts.get(name, 0)}" for name in order) or "stock"
    log(f"      sources: {per}")
    pooled = detail.get("pooled", 0)
    scored = detail.get("scored", 0)
    ranked = detail.get("ranked") or []
    tail = ""
    if scorer_on and ranked and ranked[0][1] is not None:
        bs, br = ranked[0]
        tail = f" · best {bs} {int(br * 100)}%"
    dupe = "" if picked else " · (all already used elsewhere)"
    log(f"      → {pooled} pooled · {scored} scored{tail}{dupe}")


def _generate_one(gen, s, cache: Path, cfg: dict, used: set,
                  log=lambda *_: None) -> dict | None:
    """Generate ONE image for a scene and return an asset dict, or None on any
    failure so the caller falls back to search. The file is named by a hash of
    the prompt, so an identical prompt reuses the picture already made — a
    re-source never pays for the same generation twice.

    Retries once: the Gemini image model occasionally replies with TEXT instead
    of a picture, and a second attempt usually returns the image. A genuine block
    (safety, or a named real person) fails both times and is reported, so the
    caller can search instead — the scene is never left empty."""
    subject = s.query or getattr(s, "text", "") or ""
    prompt = gen.prompt_for(subject, cfg)
    if not prompt.strip():
        return None
    slug = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    dest = cache / f"gen_{slug}.png"
    for attempt in (1, 2):
        try:
            gen.image(prompt, cfg, dest)
            break
        except Exception as e:
            if attempt == 2:
                log(f"  ⚠ S{s.n:>3} could not generate ({str(e)[:64]}) — "
                    f"searching instead")
                return None
    path = str(dest)
    if path in used:
        return None                        # already on screen elsewhere
    return {"path": path, "src": "imagen", "query": s.query or prompt[:60],
            "media": "IMAGE", "credit": "AI-generated (Imagen)", "page": "",
            "license": "AI-generated", "score": None, "generated": True}


def fetch_all(scenes, cache: Path, pexels_key, pixabay_key,
              picks: dict[int, int] | None = None, log=print,
              cfg: dict | None = None, already: dict | None = None,
              on_progress=None, should_cancel=None) -> dict[int, dict]:
    """Fetch a visual for every scene. Failures are reported, not fatal.

    Two things happen here beyond a plain search.

    THE LADDER. Free stock does not have every shot a script asks for. Each
    scene carries progressively looser queries, and we walk down until one
    returns. A slightly generic but on-topic clip beats an empty scene, and it
    beats the junk a single over-specific query returns.

    NO REPEATS. The same clip appearing twice is the clearest sign of a cheap
    video, and it happens easily: two scenes about the ocean will happily
    return the identical stock file. Assets already used in this video are
    skipped, taking the next match down instead.
    """
    picks = picks or {}
    cfg = cfg or {}
    have = _SRC.usable({**cfg, "pexels_key": pexels_key, "pixabay_key": pixabay_key})
    out, failed, weak, placeholder = {}, [], [], []
    # Assets already assigned on a previous run count as used too, or a
    # re-source of three scenes would happily pick something on screen
    # elsewhere in the same video.
    used: set[str] = {a.get("path") for a in (already or {}).values() if a.get("path")}

    # Bring the relevance scorer up once (it logs on/off and the chosen tier a
    # single time), and read the "good enough" bar. When scoring is off both are
    # inert and the ladder behaves exactly as it always did.
    scorer_on = vision.get_scorer(cfg, log) is not None
    clip_min = float(cfg.get("clip_min") or DEFAULT_CLIP_MIN)
    # "full" lists EVERY scored candidate per scene; anything else keeps the
    # clean top-6 view. Set "source_log": "full" in config.json for the firehose.
    full_log = str(cfg.get("source_log", "")).strip().lower() in ("full", "all", "verbose")
    # Ask EVERY capable source per scene (then CLIP picks), rather than the top
    # few by subject. Catches named people / off-topic subjects the topic router
    # would miss. More requests per scene, so it is opt-in.
    all_sources = str(cfg.get("search_all_sources", "")).strip().lower() \
        in ("1", "true", "yes", "on", "all")
    # Biography mode: for a scene that shows a PERSON, stock has no real named
    # people, yet a crisp generic stock photo can out-score the actual (often
    # lower-res) archive shot. So skip stock on people scenes and let the archives
    # — which DO hold the person — win. Stock still rescues an empty scene below.
    name_people = str(cfg.get("name_real_people", "")).strip().lower() \
        in ("1", "true", "yes", "on")

    # IMAGE GENERATION (config `generate`): off = search only (default); all =
    # generate every non-person scene, no search; mixed = search, but replace any
    # scene whose best match is below `generate_min` with one generated image.
    # Real-people scenes never generate — Imagen will not render a named person —
    # so they always search. A per-run cap and one-image-per-scene keep cost in
    # hand. The module is imported lazily so a search-only install never needs it.
    gen_mode = str(cfg.get("generate", "")).strip().lower()
    gen_mode = {"on": "all", "generate": "all", "true": "all"}.get(gen_mode, gen_mode)
    if gen_mode not in ("mixed", "all"):
        gen_mode = "off"
    gen_min = float(cfg.get("generate_min") or 0.60)
    gen_cap = int(cfg.get("generate_max") or 40)
    # Scenes the sheet marks `exact` (a specific named thing a search can't be
    # trusted to get) are GENERATED regardless of any search score — but only when
    # generation is on. Toggle with generate_exact (default on).
    gen_exact = str(cfg.get("generate_exact", "auto")).strip().lower() \
        not in ("off", "false", "no", "0", "none")
    _gen = None
    if gen_mode != "off":
        try:
            from . import imagen as _gen
            if not _gen.available(cfg):
                _gen = None
        except Exception:
            _gen = None
    if gen_mode != "off" and _gen is None:
        log("  ⚠ generation is on but Vertex is not configured — searching only.")
        gen_mode = "off"
    # Tell the user when scenes that WANT an exact (generated) visual won't get one
    # because generation is off — those fall back to a best-effort search.
    n_exact = sum(1 for s in scenes if getattr(s, "exact", False))
    if n_exact and (gen_mode == "off" or not gen_exact):
        why = "generation is off" if gen_mode == "off" else "generate_exact is off"
        log(f"  ⓘ {n_exact} scene(s) are marked for an exact AI-generated visual, "
            f"but {why} — searching for them instead. Turn generation on for "
            f"exact visuals, or generate them by hand in Review.")
    gen_count = [0]                     # images generated so far (mutable closure)
    # PARALLEL SCENES. The machine sits idle during sourcing — it is all network
    # waiting — so running several scenes at once is close to a linear speed-up.
    # Opt-in via `source_workers` (1 = the old sequential behaviour, unchanged).
    # Shared state is guarded: `used` (the anti-repeat set) via _claim, the GPU
    # scorer via _SCORE_LOCK, and each scene's log lines are buffered and flushed
    # together so a scene's block stays contiguous.
    workers = max(1, int(cfg.get("source_workers") or 1))
    _state_lock = threading.Lock()
    _log_lock = threading.Lock()
    _progress = [0]

    def _claim(path: str, mine: set) -> bool:
        """Reserve a candidate path so no two scenes ever use the same file — the
        anti-repeat guarantee, kept correct when scenes run in parallel. Returns
        False when another scene already holds it (the caller tries the next)."""
        if not path:
            return False
        with _state_lock:
            if path in used:
                return False
            used.add(path)
            mine.add(path)
            return True

    def _source_scene(i, s) -> None:
        # Stop was pressed: don't start this scene. Scenes already in flight
        # finish (a few seconds), but the queue empties fast instead of grinding
        # through every remaining scene.
        if should_cancel and should_cancel():
            return
        lines: list = []
        emit = lines.append             # buffer this scene's log; flush at the end
        mine: set = set()               # paths this scene reserved
        chosen = ""                     # the one it keeps (the rest are released)
        try:
            base = picks.get(s.n, 0)
            ladder = [q for q in [s.query, *getattr(s, "fallbacks", [])] if q]
            route = _SRC.route(getattr(s, "domain", ""), s.media, have,
                               query=" ".join(ladder), topic=getattr(s, "topic", ""),
                               all_sources=all_sources)
            # Biography mode + a people scene: drop stock so the real person (from
            # the archives) wins over a generic stock look-alike.
            real_person = (name_people and getattr(s, "topic", "") == "people"
                           and s.media == "IMAGE")
            if real_person:
                archives_only = [r for r in route if r not in ("pexels", "pixabay")]
                if archives_only:
                    route = archives_only

            with _state_lock:
                can_gen = (_gen is not None and s.media == "IMAGE"
                           and not real_person and gen_count[0] < gen_cap)

            # An `exact` scene needs a specific visual a search can't be trusted to
            # get, so it's generated up front (like all-mode) rather than searched.
            want_exact = gen_exact and bool(getattr(s, "exact", False))

            # ALL mode, OR an exact scene in any generation mode: generate instead
            # of searching. A generation failure falls through to a normal search,
            # so a scene is never left empty by it.
            if can_gen and (gen_mode == "all" or want_exact):
                a = _generate_one(_gen, s, cache, cfg, used, emit)
                if a and _claim(a["path"], mine):
                    with _state_lock:
                        out[s.n] = a
                        gen_count[0] += 1
                    chosen = a["path"]
                    tag = " (exact)" if want_exact and gen_mode != "all" else ""
                    emit(f"✦ S{s.n:>3} image · generated{tag} · \"{a['query'][:46]}\"")
                    return

            got = None
            got_rel = -1.0
            best_below = None
            best_below_rel = -1.0
            best_below_eff = -1.0
            notes: list = []
            rungs: list = []
            rung_log: list = []

            for rung, query in enumerate(ladder):
                pick = None
                rdetail = None
                for bump in range(4):
                    try:
                        hit = fetch(query, s.media, cache, pexels_key, pixabay_key,
                                    base + bump, sources=route, cfg=cfg)
                    except StockError as e:
                        if bump == 0:
                            notes.append(f"{query[:34]!r}: {e}")
                        break                   # this query is exhausted
                    if rdetail is None:
                        rdetail = hit.get("_detail")
                    if not _claim(hit["path"], mine):
                        continue                # already on screen elsewhere
                    pick = hit
                    break
                rung_log.append((query, rdetail, pick is not None))
                if pick is None:
                    continue

                rel = pick.get("score")
                rungs.append((query, rel))
                if not scorer_on or rel is None:
                    got, got_rel = pick, None
                    break

                # Stay on the scene's own shot: the first rung to clear the bar
                # wins outright; otherwise a looser rung must beat an earlier one
                # by more than a per-step handicap. (See the long note in history.)
                if rel >= clip_min:
                    got, got_rel = pick, rel
                    break
                eff = rel - rung * _RUNG_PENALTY
                if eff > best_below_eff:
                    best_below, best_below_rel, best_below_eff = pick, rel, eff

            if got is None and best_below is not None:
                got, got_rel = best_below, best_below_rel

            for nm, reason in _SRC.drain_newly_down():
                if "429" in reason or "rate-limit" in reason.lower():
                    emit(f"  ⚠ {nm} disabled for the rest of this run after "
                         f"{_SRC.FAIL_LIMIT} rate-limit responses — it is reachable, "
                         f"you just asked too often. Turn search_all_sources off, or "
                         f"add {nm}'s API token, then re-source.")
                else:
                    emit(f"  ⚠ {nm} disabled for the rest of this run after "
                         f"{_SRC.FAIL_LIMIT} failed requests — unreachable on this "
                         f"network (run 'faceless sources' to check).")

            if full_log:
                _emit_scene_header(emit, s)
                for idx, (q, det, picked) in enumerate(rung_log):
                    label = "search" if idx == 0 else f"fallback {idx}"
                    _emit_rung(emit, q, det, label, picked, scorer_on)

            # MIXED mode: replace an empty or below-bar match with one generated
            # image. (Scoring off → cannot judge "below 60%", so only rescue empty.)
            if gen_mode == "mixed" and can_gen:
                below = got is None or (got_rel is not None and got_rel < gen_min)
                if got is not None and got_rel is None:
                    below = False
                if below:
                    a = _generate_one(_gen, s, cache, cfg, used, emit)
                    if a and _claim(a["path"], mine):
                        with _state_lock:
                            out[s.n] = a
                            gen_count[0] += 1
                        chosen = a["path"]
                        why = ("nothing found" if got is None
                               else f"best match {int(got_rel * 100)}% < "
                                    f"{int(gen_min * 100)}%")
                        emit(f"✦ S{s.n:>3} image · generated ({why}) · "
                             f"\"{a['query'][:46]}\"")
                        return

            if got is None:
                # Nothing real: fall back to a neutral background stock always has.
                for gq in _SAFETY_QUERIES:
                    for bump in range(3):
                        try:
                            hit = fetch(gq, s.media, cache, pexels_key, pixabay_key,
                                        bump, sources=None, cfg=cfg)
                        except StockError:
                            break
                        if not _claim(hit["path"], mine):
                            continue
                        got = hit
                        break
                    if got is not None:
                        break

                if got is not None:
                    got = dict(got)
                    got.pop("_detail", None)
                    got["placeholder"] = True
                    got["score"] = None
                    with _state_lock:
                        out[s.n] = got
                        placeholder.append(s.n)
                    chosen = got["path"]
                    emit(f"⚑ S{s.n:>3} {s.media.lower():<5} · placeholder · no real "
                         f"match for \"{(s.query or '')[:40]}\"")
                    return

                with _state_lock:
                    failed.append((s.n, "; ".join(notes) or "no match"))
                emit(f"✗ S{s.n:>3} {s.media.lower():<5} · FAILED · "
                     f"{notes[0] if notes else 'no match found'}")
                return

            weak_pick = scorer_on and got_rel is not None and got_rel < clip_min
            detail = got.pop("_detail", None)
            with _state_lock:
                if weak_pick:
                    weak.append(s.n)
                out[s.n] = got
            chosen = got["path"]
            sym = "~" if weak_pick else "✓"
            pct = f"{got_rel * 100:.0f}%" if (scorer_on and got_rel is not None) else "  —"
            topic = f" · {s.topic}" if getattr(s, "topic", "") else ""
            note = "  (weak — below the match bar; worth a manual swap)" if weak_pick else ""
            emit(f"{sym} S{s.n:>3} {s.media.lower():<5} · {got['src']:<11} "
                 f"{pct:>4} · \"{got['query'][:46]}\"{topic}{note}")
            if not full_log and len(rungs) > 1:
                steps = []
                for q, r in rungs:
                    mark = " ✓" if q == got["query"] else ""
                    rr = f"{int(r * 100)}%" if (scorer_on and r is not None) else "—"
                    steps.append(f"\"{q[:30]}\" {rr}{mark}")
                emit("       ladder: " + " → ".join(steps))
            d2 = _detail_line(detail, full_log)
            if d2:
                emit(d2)
        finally:
            # Give back any candidates this scene reserved but did not keep, so
            # another scene can use them (only the winner stays claimed).
            extra = mine - ({chosen} if chosen else set())
            if extra:
                with _state_lock:
                    used.difference_update(extra)
            with _log_lock:              # flush this scene's block in one piece
                for m in lines:
                    log(m)
            if on_progress:
                with _state_lock:
                    _progress[0] += 1
                    done = _progress[0]
                on_progress(done, len(scenes), f"S{s.n} {s.media.lower()}")

    if workers > 1 and len(scenes) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(_source_scene, i, s) for i, s in enumerate(scenes)]:
                fut.result()                 # surface any scene's exception
    else:
        for i, s in enumerate(scenes):
            _source_scene(i, s)

    down = _SRC.down_sources()
    if down:
        log(f"\nUnreachable this run, skipped after {_SRC.FAIL_LIMIT} failures: "
            f"{', '.join(down)}")
        log("Run 'faceless sources' to see whether that is your network or theirs.")

    if failed:
        log(f"\n{len(failed)} scene(s) had no usable match: "
            f"{[n for n, _ in failed]}")
        log("Edit those 'ALT / search' lines in the main script and re-run 'stock'.")

    if weak:
        log(f"\n{len(weak)} scene(s) matched only weakly: {weak}")
        log("Nothing free fit them well. Review & swap those, or reword their "
            "'ALT / search' line for a shot that exists.")

    if placeholder:
        log(f"\n{len(placeholder)} scene(s) had NO match and got a neutral "
            f"placeholder: {placeholder}")
        log("The video will build, but these carry a generic background. Reword "
            "their 'ALT / search' line for a shot that exists, then re-source.")
    return out


def credits_block(assets: dict[int, dict]) -> str:
    """Attribution text for the video description. Neither site requires it,
    but both ask for it, and it costs nothing."""
    seen = {}
    for a in assets.values():
        if a.get("credit"):
            seen.setdefault((a["credit"], a["src"]), 0)
            seen[(a["credit"], a["src"])] += 1
    if not seen:
        return ""
    names = sorted({f"{c}" for (c, _), _ in seen.items()})
    return ("Stock footage and photography via Pexels and Pixabay. "
            "Thanks to: " + ", ".join(names) + ".")
