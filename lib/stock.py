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
from collections import OrderedDict
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


def _fetch_bytes(url: str) -> bytes:
    if url in _BYTES:
        _BYTES.move_to_end(url)
        return _BYTES[url]
    b = _get(url)
    _BYTES[url] = b
    while len(_BYTES) > _BYTES_CAP:
        _BYTES.popitem(last=False)
    return b


class StockError(RuntimeError):
    pass


def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:
            if attempt == 3:
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
    for name in order:           # route() already caps how many sources this is
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
        except Exception as e:
            errors.append(f"{name}: {e}")
            got = []
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
    items: list[tuple[str, bytes]] = []
    for h in pool:
        try:
            if media == "IMAGE":
                # A small web-size copy is plenty for CLIP (it works at 224px).
                raw = _fetch_bytes(h.get("thumb") or h["url"])
            elif h.get("thumb"):
                raw = _fetch_bytes(h["thumb"])         # a poster frame, if given
            else:
                raw = _video_frame(h["url"])           # else pull one frame
            if raw:
                items.append((h["url"], raw))
        except StockError:
            continue
    return scorer.relevance(query, items) if items else {}


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
            capture_output=True, timeout=45)
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


def fetch_all(scenes, cache: Path, pexels_key, pixabay_key,
              picks: dict[int, int] | None = None, log=print,
              cfg: dict | None = None, already: dict | None = None,
              on_progress=None) -> dict[int, dict]:
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

    for i, s in enumerate(scenes):
        if on_progress:
            on_progress(i + 1, len(scenes), f"S{s.n} {s.media.lower()}")
        base = picks.get(s.n, 0)
        ladder = [q for q in [s.query, *getattr(s, "fallbacks", [])] if q]
        # The query text is as strong a signal as the tag: "roman aqueduct"
        # says historical whatever the scene was labelled. `topic` is the model's
        # canonical bucket, which routes any subject even when its words are not
        # in the vocabulary.
        route = _SRC.route(getattr(s, "domain", ""), s.media, have,
                           query=" ".join(ladder), topic=getattr(s, "topic", ""),
                           all_sources=all_sources)
        # Biography mode + a people scene: drop stock so the real person (from the
        # archives) wins over a generic stock look-alike. Only if archives remain.
        if name_people and getattr(s, "topic", "") == "people" and s.media == "IMAGE":
            archives_only = [r for r in route if r not in ("pexels", "pixabay")]
            if archives_only:
                route = archives_only
        got = None
        got_rel = -1.0
        best_below = None                       # best weak pick if nothing clears
        best_below_rel = -1.0
        best_below_eff = -1.0
        notes: list[str] = []
        rungs: list = []            # (query, rel) for each rung tried — the ladder

        for rung, query in enumerate(ladder):
            # Walk a few matches deep so a duplicate can be stepped over
            # without giving up on this query.
            pick = None
            for bump in range(4):
                try:
                    hit = fetch(query, s.media, cache, pexels_key, pixabay_key,
                                base + bump, sources=route, cfg=cfg)
                except StockError as e:
                    if bump == 0:
                        notes.append(f"{query[:34]!r}: {e}")
                    break                       # this query is exhausted
                if hit["path"] in used:
                    continue                    # already on screen elsewhere
                pick = hit
                break
            if pick is None:
                continue                        # nothing usable from this query

            rel = pick.get("score")
            rungs.append((query, rel))          # record this rung for the ladder line
            if not scorer_on or rel is None:
                # No relevance signal (scoring off, or an unscorable video):
                # first usable match wins, exactly as before.
                got, got_rel = pick, None
                break

            # STAY ON THE SCENE'S OWN SHOT. The ladder runs specific → loose, and
            # CLIP quietly scores a loose query higher than a specific one (a short
            # generic phrase matches any photo better than a long exact one). Taking
            # the global best across rungs therefore let the loose, off-scene
            # fallback beat the on-scene primary almost every time. Two rules stop
            # that: (1) the FIRST rung that clears the bar wins outright — since the
            # primary is tried first, a good primary ends it before any fallback is
            # even seen; (2) if nothing clears the bar, a looser rung must beat the
            # earlier one by more than a per-step handicap to replace it, so the
            # video only drifts off-scene when the fallback is clearly better.
            if rel >= clip_min:
                got, got_rel = pick, rel
                break                           # on-scene and good enough — take it
            eff = rel - rung * _RUNG_PENALTY
            if eff > best_below_eff:
                best_below, best_below_rel, best_below_eff = pick, rel, eff

        if got is None and best_below is not None:
            # No rung cleared the bar: ship the most on-scene of the weak matches
            # (the handicap already favoured the earlier, more specific queries).
            got, got_rel = best_below, best_below_rel

        # Announce any source the circuit breaker just disabled, the instant it
        # happens. Otherwise a source (e.g. Wikimedia blocked on this network)
        # simply vanishes from later scenes' "searched" line with no explanation.
        for nm in _SRC.drain_newly_down():
            log(f"  ⚠ {nm} disabled for the rest of this run after "
                f"{_SRC.FAIL_LIMIT} failed requests — unreachable on this network "
                f"(run 'faceless sources' to check).")

        if got is None:
            # The scene's own ladder found nothing real. An empty scene breaks
            # the whole render, so fall back to a neutral background that free
            # stock always has — always via the general stock providers, never a
            # specialised source that would have no such thing. Flag it as a
            # placeholder so it is obvious this one still needs a real picture.
            for gq in _SAFETY_QUERIES:
                for bump in range(3):
                    try:
                        hit = fetch(gq, s.media, cache, pexels_key, pixabay_key,
                                    bump, sources=None, cfg=cfg)
                    except StockError:
                        break                       # this generic query is dry too
                    if hit["path"] in used:
                        continue                    # already on screen; try next
                    got = hit
                    break
                if got is not None:
                    break

            if got is not None:
                got = dict(got)
                got.pop("_detail", None)
                got["placeholder"] = True           # a fill, not a real match
                got["score"] = None
                used.add(got["path"])
                out[s.n] = got
                placeholder.append(s.n)
                log(f"⚑ S{s.n:>3} {s.media.lower():<5} · placeholder · no real "
                    f"match for \"{(s.query or '')[:40]}\"")
                continue

            failed.append((s.n, "; ".join(notes) or "no match"))
            log(f"✗ S{s.n:>3} {s.media.lower():<5} · FAILED · "
                f"{notes[0] if notes else 'no match found'}")
            continue

        # Searched the whole ladder and nothing really matched: use the best we
        # found, but flag it so the scene can be fixed by hand rather than left
        # empty (an empty scene breaks the video).
        weak_pick = scorer_on and got_rel is not None and got_rel < clip_min
        if weak_pick:
            weak.append(s.n)

        used.add(got["path"])
        detail = got.pop("_detail", None)          # strip telemetry before storing
        out[s.n] = got
        # Result line: ~ for a weak match, ✓ for a good one. Then, when the scene
        # dropped to a looser query, a ladder line showing each rung it tried; and
        # a dim detail line showing the sources, pool depth and the candidates.
        sym = "~" if weak_pick else "✓"
        pct = f"{got_rel * 100:.0f}%" if (scorer_on and got_rel is not None) else "  —"
        topic = f" · {s.topic}" if getattr(s, "topic", "") else ""
        log(f"{sym} S{s.n:>3} {s.media.lower():<5} · {got['src']:<11} "
            f"{pct:>4} · \"{got['query'][:46]}\"{topic}")
        if len(rungs) > 1:                         # a genuine fallback happened
            steps = []
            for q, r in rungs:
                mark = " ✓" if q == got["query"] else ""
                rr = f"{int(r * 100)}%" if (scorer_on and r is not None) else "—"
                steps.append(f"\"{q[:30]}\" {rr}{mark}")
            log("       ladder: " + " → ".join(steps))
        d2 = _detail_line(detail, full_log)
        if d2:
            log(d2)

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
