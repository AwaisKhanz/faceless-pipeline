"""Image generation via Vertex AI — Gemini image models ("Nano Banana").

Google retired the standalone Imagen `:predict` models in 2026 and folded image
generation into the Gemini models, called through the ordinary `:generateContent`
endpoint (the same one the LLM uses) with an IMAGE response modality. The picture
comes back as a base64 part in the response. The module name stays `imagen` for
familiarity; the model underneath is now e.g. gemini-2.5-flash-image.

Deliberate limits, because generation costs money and the point is control:
  - ONE image per call, never a batch.
  - 16:9, sized for the timeline.
  - Cached by prompt: an identical prompt reuses the file it already made, so a
    re-source never pays twice.
  - NEVER used for a real named person. A biography needs the real face, so the
    caller keeps person scenes on the photo archives and only generates concept /
    b-roll beats.

The pipeline decides WHEN to call this (config `generate`: off / mixed / all);
this module only knows HOW.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import llm

# These are Gemini models on the same generateContent endpoint as the LLM, so
# they run on the "global" location the LLM already uses. Override in config.
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-2.5-flash-image"       # "Nano Banana", ~$0.039/image
DEFAULT_ASPECT = "16:9"
# Strong photoreal styling so a generated scene reads as a REAL photograph, not
# an AI illustration/render. Kept explicit (and with the "not a…" negatives that
# these models respond to) because the default otherwise drifts arty.
PHOTO_STYLE = ("a real photograph, photorealistic, shot on a full-frame DSLR, "
               "50mm lens, natural lighting, realistic skin and textures with "
               "natural imperfections, documentary photography, high detail, "
               "shallow depth of field")
PHOTO_NEG = ("Not an illustration, not a drawing, not a painting, not a cartoon, "
             "not anime, not a 3D render, not CGI, not digital art, no "
             "over-smoothing, no plastic look, no text or watermark.")
DEFAULT_STYLE = PHOTO_STYLE               # back-compat alias
# When the scene ITSELF asks for a non-photographic look, respect it — don't
# force realism onto a deliberately illustrated/animated/rendered request.
_NONPHOTO = re.compile(
    r"\b(illustrat\w*|drawing|drawn|sketch\w*|cartoon|anime|manga|"
    r"paint\w*|watercolou?r|oil painting|vector|flat design|3d ?render|"
    r"\brender\b|cgi|clay|claymation|pixar|comic|caricature|logo|icon|"
    r"diagram|chart|infographic|blueprint|low ?poly|pixel art|graffiti|"
    r"mural|concept art|storybook|cel[- ]?shaded|stylised|stylized)\b", re.I)
TIMEOUT = 120                              # a generation is slower than a search


class GenError(RuntimeError):
    """Generation could not produce an image. The caller falls back to search."""


def available(cfg: dict | None) -> bool:
    """Can we generate at all? True only when Vertex is genuinely ready — a
    project, the service-account file if one is named, and google-auth installed.
    So the UI never offers generation that would then fail, and sourcing degrades
    to 'search only' when it is not set up."""
    return llm.vertex_ready(cfg)


def prompt_for(query: str, cfg: dict | None = None) -> str:
    """Turn a scene's search phrase into a generation prompt.

    Default is a strong PHOTOREAL look (real photograph, not an illustration) so
    generated scenes match the stock footage around them. Two escapes:
      • config `generate_style` — if you set one, it's used verbatim (you're in
        charge of the look);
      • the scene subject itself asking for a non-photo style (e.g. 'watercolour
        illustration of…') — then realism is NOT forced, and the request stands.
    """
    cfg = cfg or {}
    subject = (query or "").strip().rstrip(".")
    override = (cfg.get("generate_style") or "").strip()
    if override:                                   # user took control of the look
        return f"{subject}. {override}." if subject else override
    if _NONPHOTO.search(subject):                  # scene deliberately non-photo
        return f"{subject}. High quality, detailed." if subject else "A detailed image."
    if not subject:
        return f"{PHOTO_STYLE}. {PHOTO_NEG}"
    return f"A real photograph of {subject}. {PHOTO_STYLE}. {PHOTO_NEG}"


def _endpoint(project: str, location: str, model: str) -> str:
    host = ("aiplatform.googleapis.com" if location == "global"
            else f"{location}-aiplatform.googleapis.com")
    return (f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent")


def _generate(url: str, body: dict, token: str) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _first_image(out: dict) -> str:
    """The base64 image bytes from the first image part of a generateContent
    response, or '' if the model returned only text (or nothing)."""
    for cand in (out.get("candidates") or []):
        for part in ((cand.get("content") or {}).get("parts") or []):
            blob = part.get("inlineData") or part.get("inline_data") or {}
            if blob.get("data"):
                return blob["data"]
    return ""


def _reason(out: dict) -> str:
    """Why no image came back — a safety block reads far better than 'empty'."""
    fb = out.get("promptFeedback") or {}
    if fb.get("blockReason"):
        return f"blocked: {fb['blockReason']}"
    cands = out.get("candidates") or []
    if cands and cands[0].get("finishReason") not in (None, "STOP"):
        return f"stopped: {cands[0]['finishReason']}"
    return "no image in the response (the model may have replied with text only)"


# ─────────────────────────────────────────────────────── rate limiting
#
# The image model has a per-minute quota. Sourcing runs many scenes in parallel,
# so without pacing a burst of generations all fire at once, every one comes back
# 429, and the scene silently falls back to a stock photo instead of the picture
# it needed. Three cooperating guards keep us a good citizen of the quota:
#
#   generate_workers      how many generations may run at once (default 1)
#   generate_min_interval the FLOOR on the gap between calls (default 4s)
#   generate_retries      how many times to wait out a 429/5xx before giving up
#
# The gap is ADAPTIVE. Every 429 widens it (so the next scene waits instead of
# charging into the same limit — the mistake that makes a client hammer the API);
# a clean run of successes eases it back toward the floor. This is ordinary
# cooperative backoff: a 429 is a normal "slow down" signal, and heeding it for
# ALL following calls — not just the one that got it — is exactly what the server
# wants. The gap is shared across every worker via one process-wide throttle.
_CAP = 90.0                            # never pace slower than this (seconds)
_GROW = 2.0                            # multiply the gap by this on a 429
_SHRINK = 0.8                          # multiply the gap by this after a clean run
_EASE_AFTER = 3                        # successes needed before easing the gap down


class _Throttle:
    """Process-wide adaptive pacing shared by every generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.gap = 0.0                 # current spacing between calls (seconds)
        self._next = 0.0               # earliest monotonic time a call may start
        self._streak = 0               # consecutive successes since the last 429

    def reset(self) -> None:           # for tests
        with self._lock:
            self.gap, self._next, self._streak = 0.0, 0.0, 0

    def pace(self, floor: float) -> None:
        """Sleep until this call is allowed to start, honouring the current
        (possibly widened) gap, never going below `floor`."""
        with self._lock:
            if self.gap < floor:
                self.gap = floor
            now = time.monotonic()
            start = self._next if self._next > now else now
            self._next = start + self.gap
            delay = start - now
        if delay > 0:
            time.sleep(delay)

    def hit_limit(self, retry_after: float) -> None:
        """A 429/5xx just happened: widen the gap and hold every worker off until
        the cooldown the server asked for has passed."""
        with self._lock:
            self._streak = 0
            self.gap = min(_CAP, max(self.gap * _GROW, retry_after, 1.0))
            self._next = max(self._next, time.monotonic() + max(retry_after, 0.0))

    def ok(self, floor: float) -> None:
        """A success: after a clean streak, relax the gap back toward the floor."""
        with self._lock:
            self._streak += 1
            if self._streak >= _EASE_AFTER and self.gap > floor:
                self.gap = max(floor, self.gap * _SHRINK)
                self._streak = 0


_THROTTLE = _Throttle()
_SEM_LOCK = threading.Lock()
_SEM = None
_SEM_N = 0


def _semaphore(cfg: dict) -> threading.Semaphore:
    global _SEM, _SEM_N
    n = max(1, int(cfg.get("generate_workers") or 1))
    with _SEM_LOCK:
        if _SEM is None or _SEM_N != n:
            _SEM, _SEM_N = threading.Semaphore(n), n
        return _SEM


def _retry_delay(detail: str, headers, attempt: int) -> float:
    """How long to wait before retrying. Prefer the server's own instruction —
    Google puts a 'retryDelay' like '17s' in the 429 body, and HTTP may carry a
    Retry-After header — and only fall back to exponential backoff (2, 4, 8, …)
    when neither is present. Capped so one call can't stall forever."""
    try:
        for item in (json.loads(detail).get("error", {}).get("details") or []):
            rd = str(item.get("retryDelay") or "")
            if rd.endswith("s") and rd[:-1].replace(".", "", 1).isdigit():
                return min(_CAP, float(rd[:-1]) + 0.5)
    except Exception:
        pass
    try:
        ra = headers.get("Retry-After") if headers else None
        if ra is not None and str(ra).strip().isdigit():
            return min(_CAP, float(ra) + 0.5)
    except Exception:
        pass
    return min(_CAP, 2.0 ** attempt)


def image(prompt: str, cfg: dict, dest: Path, log=None) -> Path:
    """Generate ONE 16:9 image for `prompt` and write it to `dest`.

    Cached: if `dest` already holds an image, it is returned without spending a
    call. Rate limits (429) are throttled and retried with backoff rather than
    failing; only a genuine, persistent failure raises GenError so the caller can
    fall back to search. `log` (optional) receives one line per retry.
    """
    def _note(msg: str) -> None:
        if callable(log):
            try:
                log(msg)
            except Exception:
                pass
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest                        # already generated — reuse, no cost

    project = cfg.get("vertex_project")
    if not project:
        raise GenError("image generation needs \"vertex_project\" in config.json")
    location = cfg.get("generate_location") or DEFAULT_LOCATION
    model = cfg.get("generate_model") or DEFAULT_MODEL
    sa_path = cfg.get("vertex_service_account") or ""
    aspect = cfg.get("generate_aspect") or DEFAULT_ASPECT

    try:
        token = llm._vertex_token(sa_path)
    except llm.LLMError as e:
        raise GenError(str(e)) from None

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect},
        },
    }
    url = _endpoint(project, location, model)

    def _call(b: dict, tok: str) -> dict:
        return _generate(url, b, tok)

    retries = max(0, int(cfg.get("generate_retries", 5) or 0))
    floor = float(cfg.get("generate_min_interval", 4.0) or 0)
    auth_retried = False
    out = None
    # One generation at a time (per generate_workers), paced by the shared adaptive
    # throttle. A 429 is waited out AND widens the gap for the next scene, so we
    # settle onto the real quota instead of firing rejected request after rejected
    # request.
    with _semaphore(cfg):
        attempt = 0
        while out is None:
            _THROTTLE.pace(floor)
            try:
                out = _call(body, token)
                _THROTTLE.ok(floor)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:1000]
                except Exception:
                    pass
                if e.code in (401, 403):       # token expired mid-run: mint once more
                    if auth_retried:
                        raise GenError(f"image auth failed (HTTP {e.code})") from None
                    auth_retried = True
                    llm._VERTEX_CREDS.clear()
                    try:
                        token = llm._vertex_token(sa_path)
                    except Exception as e2:
                        raise GenError(f"image auth failed: {e2}") from None
                    continue
                if e.code == 400 and "imageConfig" in detail:
                    # Older model that doesn't accept imageConfig — retry without it.
                    body["generationConfig"].pop("imageConfig", None)
                    continue
                if e.code in (429, 500, 503) and attempt < retries:
                    attempt += 1
                    wait = _retry_delay(detail, getattr(e, "headers", None), attempt)
                    _THROTTLE.hit_limit(wait)   # widen the gap for THIS + later scenes
                    _note(f"· image model busy ({e.code}) — easing to "
                          f"{_THROTTLE.gap:.0f}s between images; waiting {wait:.0f}s "
                          f"(retry {attempt}/{retries})")
                    continue                    # pace() will serve the cooldown
                if e.code == 429:
                    raise GenError(
                        f"image model is rate-limiting (429) after {retries} retries — "
                        "the quota is very tight; raise generate_min_interval or lower "
                        "generate_max, or try again later."
                    ) from None
                raise GenError(f"image HTTP {e.code}: {detail}") from None
            except Exception as e:
                raise GenError(f"image request failed: {e}") from None

    b64 = _first_image(out)
    if not b64:
        raise GenError(f"image model produced nothing: {_reason(out)}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
    return dest
