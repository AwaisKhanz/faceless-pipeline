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
