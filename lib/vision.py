"""Does this picture actually match the scene? — a free, local relevance scorer.

Size and aspect say a candidate FILLS the frame. They say nothing about whether
it is about the right thing. This module answers that, using CLIP: an open model
that embeds an image and a piece of text into the same space, so the cosine
between them measures how well the picture matches a description.

It is completely free and completely local:
  - open weights, no API, no per-use cost, offline after a one-time download;
  - it runs on the torch install the voice engine already uses, and very likely
    needs no new package (Chatterbox/diffusers already pull in `transformers`).

IT ADAPTS TO THE MACHINE. There is one code path and three model tiers, chosen
by what the hardware can actually do:

    CUDA, >=10 GB VRAM   ViT-L/14   sharpest    (the RTX box)
    CUDA <10 GB / Apple  ViT-B/16   balanced    (a Mac, a gaming laptop)
    CPU only             ViT-B/32   lightest    (a small laptop)

and if torch or transformers is missing, or the model will not load, it reports
that plainly and the caller falls back to size/aspect scoring. Nothing breaks;
a weaker machine simply gets the older behaviour.

CALIBRATION. Raw CLIP cosines sit in a narrow band and are not comparable across
queries, so a bare number cannot answer "is this good enough". Instead each image
is scored by a softmax over [the scene concept] + [a list of junk concepts]
(text, watermark, chart, clip-art …). The result is the probability the image is
the scene concept rather than junk — a real 0..1 number that both RANKS the pool
and gives an absolute bar to decide whether to search harder.
"""
from __future__ import annotations

import hashlib
import io
import sys

# Model tiers. All three are open weights on the Hugging Face hub.
LARGE = "openai/clip-vit-large-patch14"    # ~1.7 GB
BASE16 = "openai/clip-vit-base-patch16"    # ~600 MB
BASE32 = "openai/clip-vit-base-patch32"    # ~350 MB

# Concepts an image can be "about" instead of the scene. Softmaxing the scene
# concept against these turns a bare cosine into a calibrated relevance, and
# doubles as a free junk filter: a diagram scores as a diagram, not the subject.
JUNK = [
    "a screenshot of text", "a chart or diagram", "a logo or watermark",
    "clip art", "a blank or solid colour image", "an advertisement",
]

# Text templates. Averaging a couple of phrasings is steadier than one.
TEMPLATES = ("a photo of {}", "{}")

# Relevance is a cosine similarity between the picture and the scene, mapped to
# 0..1 over the band a CLIP image-text pair actually occupies: a strong,
# on-subject match sits around 0.30, something unrelated around 0.17. The band
# is fixed so a percentage means the same thing on every scene. (An earlier
# softmax against junk concepts saturated — any real photograph scored ~100%,
# because it only measured "photo vs junk", not "matches THIS line".)
_COS_LOW = 0.17
_COS_HIGH = 0.30

# Bumped whenever the scoring maths change. A cached pick tagged with an older
# version is re-scored from the file already on disk — no re-download — so a
# calibration fix takes effect on the next source without clearing the cache.
#   1: softmax vs junk (saturated to ~100%)   2: normalised cosine
SCORE_VERSION = 2


def _cfg_get(cfg: dict, key: str, default):
    v = (cfg or {}).get(key)
    return default if v in (None, "") else v


def capability(cfg: dict | None = None) -> dict:
    """What relevance scoring can do on this machine, without loading anything.

    Returns a dict: {ok, device, vram_gb, model, reason}. `ok` False means the
    caller should fall back to size/aspect scoring; `reason` says why.
    """
    cfg = cfg or {}
    if str(_cfg_get(cfg, "clip", "auto")).lower() in ("off", "false", "no", "0"):
        return {"ok": False, "reason": "turned off in config (clip: off)",
                "device": "-", "vram_gb": None, "model": "-"}
    try:
        import torch  # noqa: F401
    except Exception:
        return {"ok": False, "reason": "torch not installed", "device": "-",
                "vram_gb": None, "model": "-"}
    try:
        import transformers  # noqa: F401
    except Exception:
        return {"ok": False, "reason": "transformers not installed "
                "(pip install transformers)", "device": "-", "vram_gb": None,
                "model": "-"}

    import torch
    device, vram = "cpu", None
    try:
        if torch.cuda.is_available():
            device = "cuda"
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device, vram = "cpu", None

    override = _cfg_get(cfg, "clip_model", "")
    model = override or _pick_model(device, vram)
    return {"ok": True, "device": device, "vram_gb": vram, "model": model,
            "reason": "ready"}


def _pick_model(device: str, vram_gb: float | None) -> str:
    """The heaviest model the machine can comfortably run."""
    if device == "cuda" and vram_gb and vram_gb >= 10:
        return LARGE
    if device in ("cuda", "mps"):
        return BASE16
    return BASE32                     # cpu / unknown: keep it light


# ───────────────────────────────────────────────────────── the scorer

class Scorer:
    """Loads one CLIP model and scores images against a scene concept.

    One instance per process (see get_scorer). Loading downloads the weights the
    first time only. Relevance is cached by (model, query, image-bytes hash) so
    walking the query ladder or re-sourcing never recomputes an image.
    """

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._proc = None
        self._cache: dict[tuple, float] = {}

    def _load(self):
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import CLIPModel, CLIPProcessor
        transformers.logging.set_verbosity_error()   # skip the load-report noise
        self._torch = torch
        # use_safetensors avoids also pulling the legacy pytorch_model.bin — the
        # repo ships both, and fetching each is a needless second ~600 MB.
        self._model = CLIPModel.from_pretrained(
            self.model_id, use_safetensors=True).to(self.device).eval()
        self._proc = CLIPProcessor.from_pretrained(self.model_id)

    def relevance(self, query: str, items: list[tuple[str, bytes]]) -> dict[str, float]:
        """Score each (key, image-bytes) for how well it matches `query`, 0..1.

        The number is the picture's cosine similarity to the scene, normalised
        over _COS_LOW.._COS_HIGH, so it spreads across candidates and means the
        same thing on every scene. A picture that looks more like junk (text, a
        chart, a watermark) than like the subject is knocked down, which keeps
        clip-art and screenshots out. Undecodable images score 0. Never raises:
        on any failure it returns 0 for everything and the caller falls back to
        size/aspect ranking.
        """
        if not items:
            return {}
        try:
            self._load()
            from PIL import Image
            torch = self._torch

            pos_texts = [t.format(query) for t in TEMPLATES]
            texts = pos_texts + JUNK
            n_pos = len(pos_texts)

            out: dict[str, float] = {}
            todo: list[tuple[str, tuple, "Image.Image"]] = []
            for key, raw in items:
                ck = (self.model_id, query, hashlib.sha1(raw).hexdigest())
                if ck in self._cache:
                    out[key] = self._cache[ck]
                    continue
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    todo.append((key, ck, img))
                except Exception:
                    out[key] = 0.0

            if todo:
                inputs = self._proc(text=texts, images=[im for _, _, im in todo],
                                    return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    m = self._model(**inputs)
                    ie = m.image_embeds / m.image_embeds.norm(dim=-1, keepdim=True)
                    te = m.text_embeds / m.text_embeds.norm(dim=-1, keepdim=True)
                    cos = (ie @ te.t()).tolist()          # images x texts, -1..1
                span = _COS_HIGH - _COS_LOW
                for (key, ck, _img), row in zip(todo, cos):
                    pos = sum(row[:n_pos]) / n_pos          # match to the subject
                    junk = max(row[n_pos:]) if len(row) > n_pos else 0.0
                    rel = max(0.0, min(1.0, (pos - _COS_LOW) / span))
                    if junk > pos:                          # more junk than subject
                        rel *= 0.35
                    rel = round(rel, 4)
                    out[key] = rel
                    self._cache[ck] = rel
            return out
        except Exception:
            # Model failed at runtime (OOM, corrupt download, …). Degrade.
            return {k: 0.0 for k, _ in items}


# ───────────────────────────────────────────────── process-wide singleton

_SCORER: Scorer | None = None
_TRIED = False


def get_scorer(cfg: dict | None = None, log=lambda *a: None) -> Scorer | None:
    """The shared scorer, or None if this machine cannot run it.

    Detection and loading are attempted once per process. The reason for any
    fallback is logged so `faceless sources` / the run log explains itself.
    """
    global _SCORER, _TRIED
    if _SCORER is not None:
        return _SCORER
    if _TRIED:
        return None
    _TRIED = True

    cap = capability(cfg)
    if not cap["ok"]:
        log(f"  visual matching off — {cap['reason']}. "
            f"Ranking by size and aspect only.")
        return None

    where = cap["device"] + (f" {cap['vram_gb']}GB" if cap["vram_gb"] else "")
    log(f"  visual matching on — {cap['model'].split('/')[-1]} on {where}. "
        f"First run downloads the model once.")
    _SCORER = Scorer(cap["model"], cap["device"])
    return _SCORER


def reset() -> None:
    """Drop the singleton — for tests that swap the scorer."""
    global _SCORER, _TRIED
    _SCORER, _TRIED = None, False
