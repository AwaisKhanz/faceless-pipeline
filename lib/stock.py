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

    # `sources` is the routed order for this scene. Without it we behave
    # exactly as before, so every existing caller is unaffected.
    order = sources or ["pexels", "pixabay"]
    want = index + POOL          # a pool to rank, not just enough to index
    for name in order:
        if len(results) > index:
            break
        try:
            if name == "pexels" and pexels_key:
                results += _pexels(query, media, pexels_key, want)
            elif name == "pixabay" and pixabay_key:
                results += _pixabay(query, media, pixabay_key, want)
            elif name in _SRC.REGISTRY:
                results += [
                    {"url": h.url, "ext": h.ext, "credit": h.credit,
                     "page": h.page, "src": h.src, "license": h.license,
                     "thumb": "", "width": h.width, "height": h.height}
                    for h in _SRC.search(name, query, media, want, cfg or {})]
        except Exception as e:
            errors.append(f"{name}: {e}")

    if len(results) <= index:
        raise StockError(
            f"No {media.lower()} result #{index + 1} for '{query}'. "
            + ("; ".join(errors) if errors else "Try a simpler, more literal query.")
        )

    # Rank the pool so index 0 is the best-fitting candidate, not the first one
    # the API happened to return. Stable sort keeps the routed order for ties,
    # so a source that reports no dimensions is never pushed below one that does
    # purely for being un-measured. Bumping the index (a swap on the review
    # sheet) then walks down to the next-best.
    results.sort(key=_score, reverse=True)

    # Then, if this machine can run it, re-rank by what the picture is actually
    # OF. Relevance (0..1, whether the image matches the scene) dominates; the
    # technical score is a small tiebreak between equally-relevant candidates.
    # rel stays None when scoring is unavailable or a candidate could not be
    # scored, and those fall to the bottom in technical order — never above a
    # candidate we actually verified.
    rel = _relevance(results[:POOL], query, media, cfg)
    if rel:
        for h in results[:POOL]:
            h["rel"] = rel.get(h["url"])

        def _combined(h):
            r = h.get("rel")
            tech = min(_score(h), 4.5) / 4.5          # 0..1
            if r is None:
                return -1.0 + tech * 0.001            # unverified: below all scored
            return r + tech * 0.12                    # relevance leads, quality breaks ties

        head = sorted(results[:POOL], key=_combined, reverse=True)
        results[:len(head)] = head

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

    for i, s in enumerate(scenes):
        if on_progress:
            on_progress(i + 1, len(scenes), f"S{s.n} {s.media.lower()}")
        base = picks.get(s.n, 0)
        ladder = [q for q in [s.query, *getattr(s, "fallbacks", [])] if q]
        # The query text is as strong a signal as the tag: "roman aqueduct"
        # says historical whatever the scene was labelled.
        route = _SRC.route(getattr(s, "domain", ""), s.media, have,
                           query=" ".join(ladder))
        got = None
        got_rel = -1.0
        notes: list[str] = []

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
            if not scorer_on or rel is None:
                # No relevance signal (scoring off, or an unscorable video):
                # first usable match wins, exactly as before.
                got, got_rel = pick, None
                if rung:
                    notes.append(f"fell back to {query[:38]!r}")
                break

            # Scoring is on: keep the most relevant candidate across the rungs,
            # and only stop searching once one clears the bar.
            if rel > got_rel:
                got, got_rel = pick, rel
                if rung:
                    notes.append(f"fell back to {query[:34]!r} (match {rel:.2f})")
            if got_rel >= clip_min:
                break                           # good enough — stop here

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
                got["placeholder"] = True           # a fill, not a real match
                got["score"] = None
                used.add(got["path"])
                out[s.n] = got
                placeholder.append(s.n)
                log(f"  S{s.n:>3} PLACEHOLDER  neutral fill — no real match for "
                    f"{(s.query or '')[:34]!r}")
                continue

            failed.append((s.n, "; ".join(notes) or "no match"))
            log(f"  S{s.n:>3} FAILED  {notes[0] if notes else 'no match'}")
            continue

        # Searched the whole ladder and nothing really matched: use the best we
        # found, but flag it so the scene can be fixed by hand rather than left
        # empty (an empty scene breaks the video).
        if scorer_on and got_rel is not None and got_rel < clip_min:
            notes.append(f"weak visual match ({got_rel:.2f})")
            weak.append(s.n)

        used.add(got["path"])
        out[s.n] = got
        tail = f"  ({notes[-1]})" if notes else ""
        rtag = f" [{got_rel:.2f}]" if (scorer_on and got_rel is not None) else ""
        log(f"  S{s.n:>3} {s.media:<5} {got['src']:<11} {got['query'][:40]}{rtag}{tail}")

    down = _SRC.down_sources()
    if down:
        log(f"\nUnreachable this run, skipped after {_SRC.FAIL_LIMIT} failures: "
            f"{', '.join(down)}")
        log("Run 'faceless sources' to see whether that is your network or theirs.")

    if failed:
        log(f"\n{len(failed)} scene(s) had no usable match: "
            f"{[n for n, _ in failed]}")
        log("Edit those 'ALT / search' lines in the master sheet and re-run 'stock'.")

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
