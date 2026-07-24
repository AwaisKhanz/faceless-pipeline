"""Image generation via Vertex AI Imagen.

A companion to the search sources: instead of finding a photo, this generates one
from the scene's own words. It reuses the Vertex service-account token minted in
lib.llm, so the same credentials that power the LLM also make pictures — nothing
new to configure beyond turning it on.

Deliberate limits, because generation costs money and the point is control:
  - ONE image per call (sampleCount=1), never a pool.
  - 16:9, sized for the timeline.
  - Cached by prompt: an identical prompt reuses the file it already made, so a
    re-source never pays twice.
  - NEVER used for a real named person. Imagen will not render a specific public
    figure, and a biography needs the real face — so the caller keeps person
    scenes on the photo archives and only generates concept / b-roll beats.

The pipeline decides WHEN to call this (config `generate`: off / mixed / all);
this module only knows HOW.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from . import llm

# Imagen is regional; `global` (used for Gemini) is not a valid Imagen endpoint,
# so generation has its own location, defaulting to a region that always has it.
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "imagen-3.0-generate-002"
DEFAULT_STYLE = ("cinematic documentary photograph, natural light, realistic, "
                 "high detail, shallow depth of field")
TIMEOUT = 120                              # a generation is slower than a search


class GenError(RuntimeError):
    """Generation could not produce an image. The caller falls back to search."""


def available(cfg: dict | None) -> bool:
    """Can we generate at all? True when a Vertex project is configured. The
    service-account file and google-auth are checked at call time, so a
    half-configured setup degrades to 'search only' rather than erroring."""
    cfg = cfg or {}
    return bool(cfg.get("vertex_project"))


def prompt_for(query: str, cfg: dict | None = None) -> str:
    """Turn a scene's search phrase into a generation prompt: the literal subject,
    then a consistent style so the whole video looks like it belongs together."""
    cfg = cfg or {}
    style = (cfg.get("generate_style") or DEFAULT_STYLE).strip()
    subject = (query or "").strip().rstrip(".")
    return f"{subject}. {style}." if subject else style


def _endpoint(project: str, location: str, model: str) -> str:
    return (f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:predict")


def _predict(url: str, body: dict, token: str) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def image(prompt: str, cfg: dict, dest: Path) -> Path:
    """Generate ONE 16:9 image for `prompt` and write it to `dest`.

    Cached: if `dest` already holds an image, it is returned without spending a
    call. Raises GenError on any failure so the caller can fall back to search.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest                        # already generated — reuse, no cost

    project = cfg.get("vertex_project")
    if not project:
        raise GenError("image generation needs \"vertex_project\" in config.json")
    location = cfg.get("generate_location") or DEFAULT_LOCATION
    model = cfg.get("generate_model") or DEFAULT_MODEL
    sa_path = cfg.get("vertex_service_account") or ""

    try:
        token = llm._vertex_token(sa_path)
    except llm.LLMError as e:
        raise GenError(str(e)) from None

    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,              # exactly one image — never a pool
            "aspectRatio": "16:9",
            "personGeneration": cfg.get("generate_person") or "allow_adult",
        },
    }
    url = _endpoint(project, location, model)

    try:
        out = _predict(url, body, token)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if e.code in (401, 403):           # token expired mid-run: mint once more
            llm._VERTEX_CREDS.clear()
            try:
                out = _predict(url, body, llm._vertex_token(sa_path))
            except Exception as e2:
                raise GenError(f"Imagen auth failed: {e2}") from None
        elif e.code == 429:
            raise GenError("Imagen is rate-limiting (429) — try again shortly.") from None
        else:
            raise GenError(f"Imagen HTTP {e.code}: {detail}") from None
    except Exception as e:
        raise GenError(f"Imagen request failed: {e}") from None

    preds = out.get("predictions") or []
    b64 = preds[0].get("bytesBase64Encoded") if preds else ""
    if not b64:
        # A safety block returns no image but a reason — surface it plainly.
        reason = (preds[0].get("raiFilteredReason") if preds else "") \
            or "no image returned (possibly a safety filter)"
        raise GenError(f"Imagen produced nothing: {reason}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
    return dest
