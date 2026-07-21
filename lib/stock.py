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
from pathlib import Path

from . import sources as _SRC

UA = {"User-Agent": "faceless-pipeline/1.0"}
TIMEOUT = 30


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
                        "page": v.get("url", ""), "src": "pexels"})
    else:
        for p in data.get("photos", []):
            out.append({"url": p["src"]["large2x"], "ext": ".jpg",
                        "credit": p.get("photographer", ""),
                        "page": p.get("url", ""), "src": "pexels"})
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
                        "page": h.get("pageURL", ""), "src": "pixabay"})
        else:
            url = h.get("largeImageURL") or h.get("webformatURL")
            if not url:
                continue
            out.append({"url": url, "ext": ".jpg", "credit": h.get("user", ""),
                        "page": h.get("pageURL", ""), "src": "pixabay"})
    return out


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
            return meta

    results: list[dict] = []
    errors: list[str] = []

    # `sources` is the routed order for this scene. Without it we behave
    # exactly as before, so every existing caller is unaffected.
    order = sources or ["pexels", "pixabay"]
    for name in order:
        if len(results) > index:
            break
        try:
            if name == "pexels" and pexels_key:
                results += _pexels(query, media, pexels_key, index + 3)
            elif name == "pixabay" and pixabay_key:
                results += _pixabay(query, media, pixabay_key, index + 3)
            elif name in _SRC.REGISTRY:
                results += [
                    {"url": h.url, "ext": h.ext, "credit": h.credit,
                     "page": h.page, "src": h.src, "license": h.license}
                    for h in _SRC.search(name, query, media, index + 3, cfg or {})]
        except Exception as e:
            errors.append(f"{name}: {e}")

    if len(results) <= index:
        raise StockError(
            f"No {media.lower()} result #{index + 1} for '{query}'. "
            + ("; ".join(errors) if errors else "Try a simpler, more literal query.")
        )

    hit = results[index]
    dest = cache / f"{slug}{hit['ext']}"
    dest.write_bytes(_get(hit["url"]))

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
            "query": query, "media": media, "index": index}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


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
    out, failed = {}, []
    # Assets already assigned on a previous run count as used too, or a
    # re-source of three scenes would happily pick something on screen
    # elsewhere in the same video.
    used: set[str] = {a.get("path") for a in (already or {}).values() if a.get("path")}

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
        notes: list[str] = []

        for rung, query in enumerate(ladder):
            # Walk a few matches deep so a duplicate can be stepped over
            # without giving up on this query.
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
                got = hit
                if rung:
                    notes.append(f"fell back to {query[:38]!r}")
                break
            if got:
                break

        if got is None:
            failed.append((s.n, "; ".join(notes) or "no match"))
            log(f"  S{s.n:>3} FAILED  {notes[0] if notes else 'no match'}")
            continue

        used.add(got["path"])
        out[s.n] = got
        # The last note is the useful one ("fell back to ..."); earlier
        # entries are just the queries that missed on the way down.
        tail = f"  ({notes[-1]})" if notes else ""
        log(f"  S{s.n:>3} {s.media:<5} {got['src']:<11} {got['query'][:40]}{tail}")

    down = _SRC.down_sources()
    if down:
        log(f"\nUnreachable this run, skipped after {_SRC.FAIL_LIMIT} failures: "
            f"{', '.join(down)}")
        log("Run 'faceless sources' to see whether that is your network or theirs.")

    if failed:
        log(f"\n{len(failed)} scene(s) had no usable match: "
            f"{[n for n, _ in failed]}")
        log("Edit those 'ALT / search' lines in the master sheet and re-run 'stock'.")
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
