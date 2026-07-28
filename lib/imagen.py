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
import random
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

# Which service actually makes the picture. The engine is chosen in Settings
# (config `generate_engine`); if the chosen one fails or isn't configured, and
# generate_failover is on, we fall through the chain to the next available one.
#   cloudflare — Cloudflare Workers AI (needs an account id + API token)
#   vertex     — Google Vertex image models, POOLED across models × regions
DEFAULT_ENGINE = "cloudflare"
ENGINE_CHAIN = ("cloudflare", "vertex")
CF_RUN_URL = "https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}"
CF_DEFAULT_MODEL = "@cf/black-forest-labs/flux-1-schnell"

# Vertex image generation is POOLED across (model, region) combinations. Each
# combo is served by its own regional backend, so rotating across several spreads
# the load (and the Gemini image models' dynamic shared quota) over many endpoints
# instead of hammering one — the same trick as the Cloudflare account pool, and it
# all draws on Google Cloud credit.
#
# NOTE: the Imagen models (imagegeneration@*, imagen-3.0-*) were DEPRECATED and
# shut down (as early as 2026-06-30) — calling them now returns HTTP 404. Google's
# replacement is the Gemini "Nano Banana" image family, called via generateContent:
#   gemini-2.5-flash-image        GA, ~13 regions + global — the reliable workhorse
#   gemini-3.1-flash-image        Nano Banana 2, higher quality (fewer regions)
#   gemini-3.1-flash-lite-image   Nano Banana 2 Lite, tuned for high-volume batches
#   gemini-3-pro-image            Nano Banana Pro, premium 4K — GLOBAL endpoint only
VERTEX_DEFAULT_MODELS = ("gemini-2.5-flash-image",)
# Regions that serve gemini-2.5-flash-image (a combo a project can't reach just
# 404s and is auto-skipped, so an over-broad list is safe).
VERTEX_DEFAULT_REGIONS = ("global", "us-central1", "us-east4", "us-west1",
                          "europe-west1", "europe-west4")
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
TIMEOUT = 60                               # a generation is slower than a search, but a
                                           # socket still hung >60s is dead — fail fast and
                                           # rotate to the next region rather than block a
                                           # worker (the old 120s cost a 2-min stall on a hang)


class GenError(RuntimeError):
    """Generation could not produce an image. The caller falls back to search."""


class Cancelled(Exception):
    """The user pressed Stop while this generation was waiting. Raised so a long
    throttle/backoff wait (or a blocked concurrency slot) aborts promptly instead
    of finishing the whole image first."""


class _RateLimited(Exception):
    """An engine said 'slow down' (HTTP 429/5xx). Carries the server's requested
    cooldown when it gave one, else None so the caller backs off exponentially.
    Handled by the shared throttle in image(); never leaks to callers."""

    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited")
        self.retry_after = retry_after


def _interruptible_sleep(seconds: float, should_cancel=None) -> None:
    """Sleep, but wake every ~200ms to see if the job was cancelled. A single
    long time.sleep() is why Stop 'did nothing for ages' — the worker was parked
    inside a 30-90s backoff and couldn't notice the flag until it returned."""
    if seconds <= 0:
        return
    if not callable(should_cancel):
        time.sleep(seconds)
        return
    end = time.monotonic() + seconds
    while True:
        if should_cancel():
            raise Cancelled()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.2, left))


def engine_ready(engine: str, cfg: dict | None) -> bool:
    """Is one engine usable with the current config? Cloudflare needs an account
    id + token, Vertex needs a project (+ google-auth)."""
    cfg = cfg or {}
    if engine == "cloudflare":
        return bool(_cloudflare_accounts(cfg))         # single pair OR a pool
    if engine == "vertex":
        return llm.vertex_ready(cfg)
    return False


def selected_engine(cfg: dict | None) -> str:
    """The engine the user picked in Settings, defaulting to Cloudflare."""
    e = str((cfg or {}).get("generate_engine") or DEFAULT_ENGINE).strip().lower()
    return e if e in ENGINE_CHAIN else DEFAULT_ENGINE


_ENGINE_LABELS = {"cloudflare": "Cloudflare", "vertex": "Vertex"}


def engine_label(engine: str | None) -> str:
    """A human-friendly name for an engine, for the 'credit' field."""
    return _ENGINE_LABELS.get(engine or "", "AI")


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("", "off", "false", "no", "0", "none")


def engine_order(cfg: dict | None) -> list[str]:
    """The engines to try, in order. Normally the chosen one first, then the rest
    of the failover chain (ready ones only). With generate_failover off, ONLY the
    chosen engine is used — so 'vertex' means vertex or nothing (the scene then
    falls back to a stock search, never to another generator)."""
    cfg = cfg or {}
    chosen = selected_engine(cfg)
    if not _truthy(cfg.get("generate_failover", True)):
        return [chosen] if engine_ready(chosen, cfg) else []
    seq = [chosen] + [e for e in ENGINE_CHAIN if e != chosen]
    return [e for e in seq if engine_ready(e, cfg)]


def available(cfg: dict | None) -> bool:
    """Can we generate at all? True when ANY engine is usable — Cloudflare (with a
    token) or Vertex (with a project). The UI only offers generation, and sourcing
    only falls back to it, when one is configured."""
    cfg = cfg or {}
    return any(engine_ready(e, cfg) for e in ENGINE_CHAIN)


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

    def pace(self, floor: float, should_cancel=None) -> None:
        """Sleep until this call is allowed to start, honouring the current
        (possibly widened) gap, never going below `floor`. The wait is
        interruptible so Stop is honoured even mid-cooldown."""
        with self._lock:
            if self.gap < floor:
                self.gap = floor
            now = time.monotonic()
            start = self._next if self._next > now else now
            self._next = start + self.gap
            delay = start - now
        _interruptible_sleep(delay, should_cancel)

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


def _server_retry_after(detail: str, headers) -> float | None:
    """The cooldown the server explicitly asked for, or None. Google puts a
    'retryDelay' like '17s' in the 429 body; HTTP may carry a Retry-After header.
    When neither is present the caller backs off exponentially instead."""
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
    return None


def _read_error(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", "replace")[:1000]
    except Exception:
        return ""


def _short(detail: str, limit: int = 140) -> str:
    """One-line gist of an API error body for a log line: prefer the JSON
    error.message / error.status if present, else a trimmed raw string."""
    try:
        d = json.loads(detail).get("error", {})
        msg = (d.get("message") or d.get("status") or "").strip()
        if msg:
            return msg[:limit]
    except Exception:
        pass
    return " ".join(str(detail).split())[:limit]


def _aspect_wh(cfg: dict) -> tuple[int, int]:
    """Pixel size for a requested aspect (default 16:9 → 1280×720). Engines that
    take width/height get a frame-shaped image instead of a square one that would
    later be cropped."""
    asp = str((cfg or {}).get("generate_aspect") or DEFAULT_ASPECT)
    try:
        a, b = (float(x) for x in asp.split(":"))
    except Exception:
        a, b = 16.0, 9.0
    long = 1280.0
    w, h = (long, long * b / a) if a >= b else (long * a / b, long)
    return int(w) // 2 * 2, int(h) // 2 * 2       # even dimensions


def _engine_floor(engine: str, cfg: dict) -> float:
    """Minimum seconds between calls for an engine."""
    return float((cfg or {}).get("generate_min_interval", 4.0) or 0)


# ───────────────────────────────────────────── engines (one raw attempt each)
#
# Each `_*_raw` makes ONE request and either writes the image to `dest` or raises:
# _RateLimited for a 429/5xx (the shared throttle waits it out), GenError for a
# real failure (image() moves on to the next engine). None of them retry or
# sleep — pacing, backoff and cancellation all live in image().

# Cloudflare accounts each get their OWN free daily Neuron allowance, so pooling
# several (yours + friends') multiplies the free images per day. We rotate through
# them and, when one hits its cap, rest it for a while and move to the next — the
# resting map is process-wide so every scene/worker shares the same knowledge.
_CF_LOCK = threading.Lock()
_CF_REST: dict[str, float] = {}                # account_id → monotonic time it may be used again


class _CFRateLimited(Exception):
    """One Cloudflare account is rate-limited / over its daily cap — rest it and
    rotate to the next account (distinct from _RateLimited, which pauses the whole
    engine)."""


def _cloudflare_accounts(cfg: dict) -> list[tuple[str, str]]:
    """Every configured (account_id, token) pair: the single cf_account_id/token
    first, then one 'account_id:token' per line of cf_accounts (a comma or space
    between the two also works). Deduped by account id, blanks skipped."""
    out, seen = [], set()

    def add(aid, tok):
        aid, tok = (aid or "").strip(), (tok or "").strip()
        if aid and tok and aid not in seen:
            seen.add(aid)
            out.append((aid, tok))

    add(cfg.get("cf_account_id"), cfg.get("cf_api_token"))
    for line in str(cfg.get("cf_accounts") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\s:,]+", line, maxsplit=1)
        if len(parts) == 2:
            add(parts[0], parts[1])
    return out


def _cf_ready(accounts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    now = time.monotonic()
    with _CF_LOCK:
        return [(a, t) for (a, t) in accounts if _CF_REST.get(a, 0.0) <= now]


def _cf_put_to_rest(account_id: str, minutes: float) -> None:
    with _CF_LOCK:
        _CF_REST[account_id] = time.monotonic() + max(0.0, minutes) * 60.0


def _cf_reset() -> None:                       # for tests
    with _CF_LOCK:
        _CF_REST.clear()


def _cf_is_cap(code: int, detail: str) -> bool:
    """True when a Cloudflare error means 'this account is out of allowance / too
    many requests' (rest it) rather than a bad prompt (fail over engines). The
    daily-cap error (code 3040/4006 'neuron limit exceeded') can arrive as 400."""
    if code in (429, 500, 502, 503, 504):
        return True
    low = detail.lower()
    return any(s in low for s in ("neuron", "exceeded", "rate limit",
                                  "too many", "3040", "4006"))


def _cf_is_flux2(model: str) -> bool:
    m = model.lower()
    return "flux-2" in m or "flux2" in m


def _cf_steps(model: str, cfg: dict) -> int:
    """Diffusion steps for a Cloudflare model, clamped to what it accepts.
    More steps = more detail (and, for flux-2-dev, more cost)."""
    m = model.lower()
    want = cfg.get("cf_steps")
    if _cf_is_flux2(model):
        if "klein" in m:
            return max(1, int(want or 8))              # distilled FLUX.2 — few steps
        return max(1, int(want or 28))                 # flux-2-dev — flagship
    if "schnell" in m:
        return max(1, min(int(want or 8), 8))          # distilled flux: max 8
    return max(1, min(int(want or 20), 20))            # SDXL: max 20


def _cf_multipart(fields: dict) -> tuple[bytes, str]:
    """Encode form fields as multipart/form-data — the format FLUX.2 requires."""
    boundary = "----faceless" + "".join(random.choices("0123456789abcdef", k=20))
    buf = []
    for k, v in fields.items():
        buf.append(f"--{boundary}\r\n"
                   f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    buf.append(f"--{boundary}--\r\n")
    return "".join(buf).encode("utf-8"), f"multipart/form-data; boundary={boundary}"


def _cf_request(model: str, prompt: str, cfg: dict) -> tuple[bytes, str]:
    """The request body + Content-Type for a model. FLUX.2 takes multipart form
    fields (prompt/steps/width/height); everything else takes JSON, with SDXL
    getting a negative prompt + guidance for the best photoreal look."""
    w, h = _aspect_wh(cfg)
    steps = _cf_steps(model, cfg)
    if _cf_is_flux2(model):                            # FLUX.2 [dev] — flagship quality
        return _cf_multipart({"prompt": prompt, "steps": steps,
                              "width": w, "height": h})
    m = model.lower()
    seed = random.randint(1, 2_000_000_000)
    if "flux" in m:                                    # flux-1-schnell — fixed square
        body = {"prompt": prompt, "steps": steps, "seed": seed}
    else:                                              # SDXL / Stable-Diffusion
        body = {"prompt": prompt, "width": w, "height": h, "num_steps": steps,
                "guidance": float(cfg.get("cf_guidance") or 7.5), "seed": seed}
        neg = str(cfg.get("cf_negative") or PHOTO_NEG).strip()
        if neg:
            body["negative_prompt"] = neg
    return json.dumps(body).encode("utf-8"), "application/json"


def _cf_extract_b64(raw: bytes) -> str:
    """Base64 image data out of a JSON Cloudflare response, across its shapes:
    {"image": …} (FLUX.2), {"result":{"image": …}} (flux-1), or …data[0].b64_json."""
    try:
        j = json.loads(raw)
    except Exception:
        return ""
    if not isinstance(j, dict):
        return ""
    if j.get("image"):
        return j["image"]
    res = j.get("result") if isinstance(j.get("result"), dict) else {}
    if res.get("image"):
        return res["image"]
    try:
        return res["data"][0]["b64_json"]
    except Exception:
        return ""


def _cf_one(account_id: str, token: str, model: str, prompt: str,
            dest: Path, cfg: dict) -> None:
    """One request to one Cloudflare account. Raises _CFRateLimited when THAT
    account is capped/limited, GenError for a genuine (prompt/model) failure."""
    url = CF_RUN_URL.format(acct=account_id, model=model)
    data, ctype = _cf_request(model, prompt, cfg)
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {token}", "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            resp_ctype = r.headers.get("Content-Type", "")
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = _read_error(e)
        if _cf_is_cap(e.code, detail):
            raise _CFRateLimited() from None
        raise GenError(f"Cloudflare HTTP {e.code}: {detail}") from None
    except Exception as e:
        raise GenError(f"Cloudflare request failed: {e}") from None
    if resp_ctype.lower().startswith("image"):         # SDXL returns raw image bytes
        dest.write_bytes(raw)
        return
    b64 = _cf_extract_b64(raw)
    if not b64:
        why = raw[:200].decode("utf-8", "replace")
        if _cf_is_cap(200, why):                       # cap sometimes reported in a 200 body
            raise _CFRateLimited()
        raise GenError(f"Cloudflare produced nothing: {why[:120]}")
    dest.write_bytes(base64.b64decode(b64))


def _cloudflare_raw(prompt: str, cfg: dict, dest: Path, log=None) -> None:
    accounts = _cloudflare_accounts(cfg)
    if not accounts:
        raise GenError("Cloudflare needs an account id + token "
                       "(cf_account_id/cf_api_token, or cf_accounts)")
    model = (cfg.get("cf_model") or CF_DEFAULT_MODEL).strip()
    rest_min = float(cfg.get("cf_rest_minutes", 15) or 15)
    usable = _cf_ready(accounts)
    if not usable:                                     # whole pool resting → fail fast
        raise GenError(f"all {len(accounts)} Cloudflare account(s) are resting "
                       "(daily cap) — will retry after their rest / the UTC reset")
    for account_id, token in usable:
        try:
            _cf_one(account_id, token, model, prompt, dest, cfg)
            return                                     # success
        except _CFRateLimited:
            _cf_put_to_rest(account_id, rest_min)
            if callable(log):
                log(f"  · Cloudflare account …{account_id[-6:]} over its limit — "
                    f"resting {rest_min:.0f}m, trying the next account")
            continue                                   # rotate to the next account
        # a real GenError (bad prompt/model) is the same for every account: re-raise
    raise GenError(f"all {len(accounts)} Cloudflare account(s) are rate-limited")


# ─────────────────────────────────────── Vertex model × region pool
#
# Vertex quotas are tracked per base_model per region, so a single model in one
# region is one small quota bucket (and the Gemini image models share a fixed
# "dynamic shared quota" you can't raise). We POOL instead: rotate across several
# models (Imagen, which has its OWN quota) and several regions, resting any combo
# that's busy and moving to the next — so the effective throughput is the sum of
# all the buckets, and it all draws on Google Cloud credit.
_VERTEX_LOCK = threading.Lock()
_VERTEX_REST: dict[tuple, float] = {}          # (model, region) → monotonic time free again


class _VertexBusy(Exception):
    """A Vertex (model, region) combo is busy / rate-limited / unavailable. Rest it
    and rotate to the next combo. rest_minutes overrides the default (e.g. a long
    rest for a 404, meaning the model isn't served in that region). reason carries
    the real HTTP status/message so it can be surfaced in the log."""

    def __init__(self, rest_minutes: float | None = None, reason: str = ""):
        super().__init__(reason or "vertex busy")
        self.rest_minutes = rest_minutes
        self.reason = reason


def _split_list(v) -> list[str]:
    return [x.strip() for x in re.split(r"[\s,]+", str(v or "")) if x.strip()]


def _vertex_combos(cfg: dict) -> list[tuple[str, str]]:
    """(model, region) pairs to rotate over — vertex_models × vertex_regions,
    defaulting to the Imagen models across a few regions."""
    models = _split_list(cfg.get("vertex_models")) or list(VERTEX_DEFAULT_MODELS)
    regions = _split_list(cfg.get("vertex_regions")) or list(VERTEX_DEFAULT_REGIONS)
    out, seen = [], set()
    for m in models:
        for r in regions:
            if (m, r) not in seen:
                seen.add((m, r))
                out.append((m, r))
    return out


def _vertex_ready(combos: list[tuple[str, str]]) -> list[tuple[str, str]]:
    now = time.monotonic()
    with _VERTEX_LOCK:
        return [c for c in combos if _VERTEX_REST.get(c, 0.0) <= now]


def _vertex_put_to_rest(combo: tuple[str, str], minutes: float) -> None:
    with _VERTEX_LOCK:
        _VERTEX_REST[combo] = time.monotonic() + max(0.0, minutes) * 60.0


def _vertex_reset() -> None:                   # for tests
    with _VERTEX_LOCK:
        _VERTEX_REST.clear()


def _vertex_predict_endpoint(project: str, region: str, model: str) -> str:
    return (f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{region}/publishers/google/models/{model}:predict")


def _vertex_generate_one(project, region, model, sa_path, prompt, cfg, dest) -> None:
    """One Gemini image request (generateContent) to one region."""
    aspect = cfg.get("generate_aspect") or DEFAULT_ASPECT
    try:
        token = llm._vertex_token(sa_path)
    except llm.LLMError as e:
        raise GenError(str(e)) from None
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                 "imageConfig": {"aspectRatio": aspect}}}
    url = _endpoint(project, region, model)
    auth_retried, out = False, None
    while out is None:
        try:
            out = _generate(url, body, token)
        except urllib.error.HTTPError as e:
            detail = _read_error(e)
            if e.code in (401, 403) and not auth_retried:
                auth_retried = True
                llm._VERTEX_CREDS.clear()
                try:
                    token = llm._vertex_token(sa_path)
                except Exception as e2:
                    raise GenError(f"Vertex auth failed: {e2}") from None
                continue
            if e.code == 400 and "imageConfig" in detail:
                body["generationConfig"].pop("imageConfig", None)
                continue
            if e.code == 404:
                raise _VertexBusy(rest_minutes=720,
                                  reason=f"HTTP 404 {_short(detail)}") from None
            if e.code in (429, 500, 503):
                raise _VertexBusy(reason=f"HTTP {e.code} {_short(detail)}") from None
            raise GenError(f"Vertex HTTP {e.code}: {detail}") from None
        except Exception as e:
            raise GenError(f"Vertex request failed: {e}") from None
    b64 = _first_image(out)
    if not b64:
        raise GenError(f"Vertex produced nothing: {_reason(out)}")
    dest.write_bytes(base64.b64decode(b64))


def _vertex_predict_one(project, region, model, sa_path, prompt, cfg, dest) -> None:
    """One Imagen request (:predict) to one region. Imagen is a dedicated image
    model with a separate quota from the Gemini image models."""
    aspect = cfg.get("generate_aspect") or DEFAULT_ASPECT
    try:
        token = llm._vertex_token(sa_path)
    except llm.LLMError as e:
        raise GenError(str(e)) from None
    body = {"instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": 1, "aspectRatio": aspect,
                           "personGeneration": "allow_adult", "safetySetting": "block_few"}}
    url = _vertex_predict_endpoint(project, region, model)
    auth_retried, out = False, None
    while out is None:
        try:
            out = _generate(url, body, token)
        except urllib.error.HTTPError as e:
            detail = _read_error(e)
            if e.code in (401, 403) and not auth_retried:
                auth_retried = True
                llm._VERTEX_CREDS.clear()
                try:
                    token = llm._vertex_token(sa_path)
                except Exception as e2:
                    raise GenError(f"Imagen auth failed: {e2}") from None
                continue
            if e.code == 404:
                raise _VertexBusy(rest_minutes=720,
                                  reason=f"HTTP 404 {_short(detail)}") from None
            if e.code in (429, 500, 503):
                raise _VertexBusy(reason=f"HTTP {e.code} {_short(detail)}") from None
            raise GenError(f"Imagen HTTP {e.code}: {detail}") from None
        except Exception as e:
            raise GenError(f"Imagen request failed: {e}") from None
    preds = out.get("predictions") or []
    b64 = preds[0].get("bytesBase64Encoded") if preds and isinstance(preds[0], dict) else ""
    if not b64:
        raise GenError("Imagen produced nothing (safety filter or empty response)")
    dest.write_bytes(base64.b64decode(b64))


def _vertex_one(project, region, model, sa_path, prompt, cfg, dest) -> None:
    if "imagen" in model.lower():
        _vertex_predict_one(project, region, model, sa_path, prompt, cfg, dest)
    else:
        _vertex_generate_one(project, region, model, sa_path, prompt, cfg, dest)


def _vertex_raw(prompt: str, cfg: dict, dest: Path, log=None) -> None:
    if not llm.vertex_ready(cfg):
        raise GenError("Vertex needs \"vertex_project\" in config")
    project = cfg["vertex_project"]
    sa_path = cfg.get("vertex_service_account") or ""
    combos = _vertex_combos(cfg)
    if not combos:
        raise GenError("no Vertex models/regions configured")
    rest_min = float(cfg.get("vertex_rest_minutes", 2) or 2)
    usable = _vertex_ready(combos)
    if not usable:                                     # whole pool resting → fail fast
        raise GenError(f"all {len(combos)} Vertex model/region combos are resting")
    for model, region in usable:
        try:
            _vertex_one(project, region, model, sa_path, prompt, cfg, dest)
            return                                     # success
        except _VertexBusy as vb:
            _vertex_put_to_rest((model, region), vb.rest_minutes or rest_min)
            if callable(log):
                mins = vb.rest_minutes or rest_min
                why = f" [{vb.reason}]" if vb.reason else ""
                log(f"  · Vertex {model}@{region} unavailable{why} — resting "
                    f"{mins:.0f}m, trying the next model/region")
            continue
        # a real GenError (bad prompt / safety) is combo-independent: re-raise
    raise GenError(f"all {len(combos)} Vertex model/region combos are busy")


_ENGINE_RAW = {"cloudflare": _cloudflare_raw,
               "vertex": _vertex_raw}


def image(prompt: str, cfg: dict, dest: Path, log=None, should_cancel=None) -> str:
    """Generate ONE image for `prompt`, write it to `dest`, and RETURN the name of
    the engine that produced it ("cloudflare" / "vertex"), so the caller can label
    the asset's source accurately (not always "imagen"). Uses the engine chosen in
    Settings and falls through the failover chain if it can't.

    Cached: if `dest` already holds an image it is returned without a call. Each
    engine is paced by the shared adaptive throttle and its 429s are waited out
    with backoff; a hard failure moves on to the next available engine, and only
    when EVERY engine has failed does GenError bubble up (so the caller can fall
    back to search). `log` receives one line per retry / engine switch.

    `should_cancel` is polled while WAITING — for a free concurrency slot and
    during every throttle/backoff sleep — and raises Cancelled the moment Stop is
    pressed, so a queued generation never keeps a job alive after the user stops.
    """
    def _note(msg: str) -> None:
        if callable(log):
            try:
                log(msg)
            except Exception:
                pass

    def _stop() -> bool:
        return bool(callable(should_cancel) and should_cancel())

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return selected_engine(cfg)        # cached — reuse, best-effort engine name
    dest.parent.mkdir(parents=True, exist_ok=True)

    order = engine_order(cfg)
    if not order:
        raise GenError("no image engine is configured")
    retries = max(0, int(cfg.get("generate_retries", 5) or 0))
    workers = max(1, int(cfg.get("generate_workers") or 1))

    # generate_workers many at once. Acquire the slot cancellably so Stop is
    # honoured even while another job holds it.
    sem = _semaphore(cfg)
    while not sem.acquire(timeout=0.25):
        if _stop():
            raise Cancelled()
    try:
        errors = []
        for eng in order:
            # The pace floor spaces the START of each call. Splitting it across the
            # workers means N workers actually fire ~N× as often (floor/N apart)
            # rather than all queuing behind one global gap — so raising
            # generate_workers genuinely speeds generation up. A rate limit still
            # widens the shared gap adaptively, so this can't run away.
            floor = _engine_floor(eng, cfg) / workers
            attempt = 0
            while True:
                if _stop():
                    raise Cancelled()
                _THROTTLE.pace(floor, should_cancel if callable(should_cancel) else None)
                try:
                    _ENGINE_RAW[eng](prompt, cfg, dest, log)
                    _THROTTLE.ok(floor)
                    if eng != order[0]:
                        _note(f"· generated with {eng} (failover)")
                    return eng
                except Cancelled:
                    raise
                except _RateLimited as rl:
                    if attempt < retries:
                        attempt += 1
                        wait = rl.retry_after or min(_CAP, 2.0 ** attempt)
                        _THROTTLE.hit_limit(wait)   # widen the gap for later scenes too
                        _note(f"· {eng} busy — easing to {_THROTTLE.gap:.0f}s "
                              f"between images; waiting {wait:.0f}s "
                              f"(retry {attempt}/{retries})")
                        continue                    # pace() serves the cooldown
                    errors.append(f"{eng}: still rate-limited after {retries} tries")
                    _note(f"· {eng} still busy — switching engine")
                    break
                except GenError as e:
                    errors.append(f"{eng}: {e}")
                    _note(f"· {eng} failed ({str(e)[:80]}) — switching engine")
                    break
        raise GenError("all image engines failed — " + " | ".join(errors))
    finally:
        sem.release()
