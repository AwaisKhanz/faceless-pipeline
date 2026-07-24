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
import urllib.error
import urllib.request
from pathlib import Path

from . import llm

# These are Gemini models on the same generateContent endpoint as the LLM, so
# they run on the "global" location the LLM already uses. Override in config.
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-2.5-flash-image"       # "Nano Banana", ~$0.039/image
DEFAULT_ASPECT = "16:9"
DEFAULT_STYLE = ("cinematic documentary photograph, natural light, realistic, "
                 "high detail, shallow depth of field")
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
    """Turn a scene's search phrase into a generation prompt: the literal subject,
    then a consistent style so the whole video looks like it belongs together."""
    cfg = cfg or {}
    style = (cfg.get("generate_style") or DEFAULT_STYLE).strip()
    subject = (query or "").strip().rstrip(".")
    return f"{subject}. {style}." if subject else style


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

    try:
        out = _call(body, token)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if e.code in (401, 403):           # token expired mid-run: mint once more
            llm._VERTEX_CREDS.clear()
            try:
                out = _call(body, llm._vertex_token(sa_path))
            except Exception as e2:
                raise GenError(f"image auth failed: {e2}") from None
        elif e.code == 400 and "imageConfig" in detail:
            # Older model that doesn't accept imageConfig — retry without it.
            body["generationConfig"].pop("imageConfig", None)
            try:
                out = _call(body, token)
            except Exception as e2:
                raise GenError(f"image HTTP 400: {e2}") from None
        elif e.code == 429:
            raise GenError("image model is rate-limiting (429) — try again shortly.") from None
        else:
            raise GenError(f"image HTTP {e.code}: {detail}") from None
    except Exception as e:
        raise GenError(f"image request failed: {e}") from None

    b64 = _first_image(out)
    if not b64:
        raise GenError(f"image model produced nothing: {_reason(out)}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
    return dest
