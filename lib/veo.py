"""Video generation via Vertex AI Veo.

The video counterpart to lib.imagen, and deliberately kept at arm's length
because it is the expensive one: a few seconds of Veo costs dollars, not cents.
So there is no automatic mode — video is only ever generated when the user asks
for a specific scene in review, one clip at a time, under a hard per-run cap.

Veo is asynchronous: submit the prompt (`:predictLongRunning`) to get an
operation, then poll (`:fetchPredictOperation`) until the clip is ready, and
finally decode the base64 MP4. The whole round trip takes a minute or two.

Reuses the same Vertex service-account token as the LLM and Imagen — nothing new
to configure beyond turning it on and having the Google Cloud credit.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import llm

DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "veo-3.0-generate-001"
DEFAULT_SECONDS = 8
POLL_EVERY = 12                   # seconds between "is it done yet?" checks
POLL_TIMEOUT = 360                # give up after six minutes of rendering


class VeoError(RuntimeError):
    """Video generation failed. The caller keeps the scene's current asset."""


DEFAULT_STYLE = "cinematic documentary footage, natural light, gentle camera motion"


def available(cfg: dict | None) -> bool:
    """Can we generate video? True when a Vertex project is configured (the same
    setup Imagen and the LLM use). Credentials are checked at call time."""
    cfg = cfg or {}
    return bool(cfg.get("vertex_project"))


def prompt_for(query: str, cfg: dict | None = None) -> str:
    """Turn a scene's phrase into a video prompt: the subject, then a consistent
    motion/style so generated clips sit alongside the stock footage."""
    cfg = cfg or {}
    style = (cfg.get("veo_style") or DEFAULT_STYLE).strip()
    subject = (query or "").strip().rstrip(".")
    return f"{subject}. {style}." if subject else style


def _post(url: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def _extract(op: dict) -> str:
    """Pull the base64 MP4 out of a finished operation, across the response
    shapes Veo has used ({videos:[{bytesBase64Encoded}]} or a sample list)."""
    resp = op.get("response") or {}
    vids = resp.get("videos") or resp.get("generatedSamples") or []
    if vids:
        v = vids[0]
        return (v.get("bytesBase64Encoded")
                or (v.get("video") or {}).get("bytesBase64Encoded") or "")
    return ""


def _reason(op: dict) -> str:
    resp = op.get("response") or {}
    r = resp.get("raiMediaFilteredReasons") or op.get("error", {}).get("message")
    return (r[0] if isinstance(r, list) and r else r) \
        or "no video returned (possibly a safety filter)"


def video(prompt: str, cfg: dict, dest: Path, on_wait=None) -> Path:
    """Generate ONE 16:9 clip for `prompt` and write the MP4 to `dest`.

    Cached: an existing `dest` is returned without spending a call. `on_wait` is
    pinged on every poll so the UI can show progress. Raises VeoError on any
    failure so the caller can keep the scene's current picture.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest                        # already generated — reuse, no cost

    project = cfg.get("vertex_project")
    if not project:
        raise VeoError("video generation needs \"vertex_project\" in config.json")
    location = cfg.get("veo_location") or DEFAULT_LOCATION
    model = cfg.get("veo_model") or DEFAULT_MODEL
    sa_path = cfg.get("vertex_service_account") or ""
    seconds = int(cfg.get("veo_seconds") or DEFAULT_SECONDS)
    base = (f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}")

    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "aspectRatio": "16:9",
            "sampleCount": 1,              # one clip, never a batch
            "durationSeconds": seconds,
            "personGeneration": cfg.get("veo_person") or "allow_adult",
            "generateAudio": False,        # the scene has its own narration
        },
    }

    def _auth():
        try:
            return llm._vertex_token(sa_path)   # cached; auto-refreshes if expired
        except llm.LLMError as e:
            raise VeoError(str(e)) from None

    try:
        op = _post(f"{base}:predictLongRunning", body, _auth())
    except urllib.error.HTTPError as e:
        raise VeoError(f"Veo submit failed (HTTP {e.code})") from None
    except Exception as e:
        raise VeoError(f"Veo submit failed: {e}") from None

    name = op.get("name")
    if not name:
        raise VeoError("Veo did not start an operation")

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_EVERY)
        if on_wait:
            on_wait()
        try:
            op = _post(f"{base}:fetchPredictOperation",
                       {"operationName": name}, _auth())
        except Exception:
            continue                       # a transient poll hiccup — keep waiting
        if op.get("done"):
            b64 = _extract(op)
            if not b64:
                raise VeoError(f"Veo produced nothing: {_reason(op)}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(b64))
            return dest

    raise VeoError("Veo timed out — still rendering after the wait limit.")
