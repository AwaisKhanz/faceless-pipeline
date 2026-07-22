"""Does this picture actually match the scene? — a free, local relevance scorer.

Size and aspect say a candidate FILLS the frame. They say nothing about whether
it is about the right thing. This module answers that, using CLIP: an open model
that embeds an image and a piece of text into the same space, so the cosine
between them measures how well the picture matches a description.

It is completely free and completely local:
  - open weights, no API, no per-use cost, offline after a one-time download;
  - it runs on the torch install the voice engine already uses, and very likely
    needs no new package (Chatterbox/diffusers already pull in `transformers`).

IT ADAPTS TO THE MACHINE. One code path, chosen by what the hardware can do:

    CUDA >=12 GB VRAM    SigLIP 2 so400m   strongest   (the RTX box)
    CUDA >=8 GB          SigLIP 2 large    strong
    CUDA <8 GB           CLIP ViT-B/16     reliable
    Apple MPS            CLIP ViT-B/16     reliable     (a Mac)
    CPU only             CLIP ViT-B/32     lightest     (a small laptop)

SigLIP 2 (2025) is a markedly better image-text matcher than CLIP, so it leads
where there is a real GPU and the VRAM for it; smaller GPUs, Apple MPS and CPU
stay on dependable CLIP. If the picked model will not load (old transformers, a
bad download, OOM) it drops to CLIP at load time, so the choice only ever goes
UP — a weak or old environment silently gets the reliable path, never nothing.
Missing torch/transformers reports plainly and the caller ranks by size/aspect.

CALIBRATION. Cosines sit in a narrow, family-specific band, so each image's
cosine to the scene is mapped to 0..1 over that band. Crucially the RANKING —
which picture wins — uses the raw cosine and is band-independent, so a stronger
model improves the actual pick regardless of how the % is calibrated. A short
list of junk concepts (text, chart, watermark, clip-art) knocks down anything
that looks more like junk than like the subject.
"""
from __future__ import annotations

import hashlib
import io
import sys

# Model tiers, all open weights on the Hugging Face hub. Two families:
#   CLIP   — the original; reliable everywhere, incl. Apple MPS and CPU.
#   SigLIP 2 (2025) — stronger image-text retrieval, so it leads on a real GPU
#            where the VRAM and (for SigLIP) a recent transformers are present.
# The machine picks the strongest it can run; anything that won't load falls back
# to CLIP, so a weak or old environment simply gets the dependable path.
SIGLIP_SO400M = "google/siglip2-so400m-patch14-384"   # strongest, big GPU
SIGLIP_L = "google/siglip2-large-patch16-256"          # strong, mid GPU
BASE_SIGLIP = "google/siglip2-base-patch16-224"        # (override only)
LARGE = "openai/clip-vit-large-patch14"    # ~1.7 GB
BASE16 = "openai/clip-vit-base-patch16"    # ~600 MB  (Apple/MPS default)
BASE32 = "openai/clip-vit-base-patch32"    # ~350 MB  (CPU default)

# Concepts an image can be "about" instead of the scene. Softmaxing the scene
# concept against these turns a bare cosine into a calibrated relevance, and
# doubles as a free junk filter: a diagram scores as a diagram, not the subject.
JUNK = [
    "a screenshot of text", "a chart or diagram", "a logo or watermark",
    "clip art", "a blank or solid colour image", "an advertisement",
]

# Text templates. Averaging several phrasings is steadier than one — it is the
# standard prompt-ensembling trick, and it nudges a true match's score up a
# little because the picture matches the *idea* across wordings, not one exact
# phrase. Kept neutral (no "cinematic", no style words) so the score measures
# subject, not aesthetic.
TEMPLATES = ("a photo of {}", "a picture of {}", "an image showing {}", "{}")

# Relevance is a cosine similarity between the picture and the scene, mapped to
# 0..1 over the band a CLIP image-text pair actually occupies: a strong,
# on-subject match sits around 0.30, something unrelated around 0.17. The band
# is fixed so a percentage means the same thing on every scene. (An earlier
# softmax against junk concepts saturated — any real photograph scored ~100%,
# because it only measured "photo vs junk", not "matches THIS line".)
_COS_LOW = 0.17
_COS_HIGH = 0.29

# Each family's cosines sit in a different band, so the map from cosine to the
# 0..1 match% is per-family. IMPORTANT: ranking (which picture wins) uses cosine
# directly and is band-independent, so a slightly-off band never hurts the pick —
# it only shifts the displayed % and how eagerly a scene escalates. SigLIP's
# numbers are a sensible starting point; easy to tune after a real run because a
# version bump re-scores from disk with no re-download.
_BANDS = {
    "clip":   (0.17, 0.29),
    "siglip": (0.02, 0.24),
}


def _family_of(model_id: str) -> str:
    return "siglip" if "siglip" in (model_id or "").lower() else "clip"


def _band_of(model_id: str) -> tuple:
    return _BANDS.get(_family_of(model_id), (_COS_LOW, _COS_HIGH))


# Bumped whenever the scoring maths change. A cached pick tagged with an older
# version is re-scored from the file already on disk — no re-download — so a
# calibration fix takes effect on the next source without clearing the cache.
#   1: softmax vs junk (saturated to ~100%)   2: normalised cosine
#   3: 4-template ensemble, gentler junk penalty, band top 0.29
#   4: model family tiers (SigLIP 2 on a real GPU), per-family band
SCORE_VERSION = 4


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

    device, vram = _probe_device()
    override = _cfg_get(cfg, "clip_model", "")
    model = override or _pick_model(device, vram)
    return {"ok": True, "device": device, "vram_gb": vram, "model": model,
            "family": _family_of(model),
            "fallback": _clip_fallback(device, vram),
            "reason": "ready"}


_HW: tuple | None = None       # (device, vram_gb), probed once per process


def _device_runs(torch, dev: str) -> bool:
    """Does a real kernel actually launch on this device?

    torch.cuda.is_available() returns True on a driver the installed torch may
    not be able to run a kernel on — a Blackwell card (RTX 50-series, sm_120)
    with a torch built for older architectures is the case that bit us: the
    flag says yes, the first matmul says "no kernel image". The voice engine
    already tests for real; relevance must too, or it claims a GPU it cannot use
    and then silently scores every picture 0. On failure the caller drops to CPU.
    """
    try:
        x = torch.randn(16, 16, device=dev)
        _ = (x @ x).sum().item()          # forces an actual launch
        return True
    except Exception:
        return False


def _probe_device() -> tuple:
    """Pick the best device that genuinely computes. Cached — hardware is fixed
    for the life of the process, and the probe is not free."""
    global _HW
    if _HW is not None:
        return _HW
    import torch
    device, vram = "cpu", None
    try:
        if torch.cuda.is_available() and _device_runs(torch, "cuda"):
            device = "cuda"
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        elif (getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available() and _device_runs(torch, "mps")):
            device = "mps"
    except Exception:
        device, vram = "cpu", None
    _HW = (device, vram)
    return _HW


def _pick_model(device: str, vram_gb: float | None) -> str:
    """The strongest model the machine can comfortably run.

    SigLIP 2 leads on a real GPU with the VRAM for it; a smaller GPU or Apple
    MPS stays on dependable CLIP (SigLIP on MPS is unproven and not worth a
    silent regression); CPU keeps the lightest CLIP. Anything that then fails to
    load (old transformers, a bad download, OOM) drops back to CLIP at load time,
    so this only ever chooses UP — it can't strand a machine.
    """
    if device == "cuda" and vram_gb:
        if vram_gb >= 12:
            return SIGLIP_SO400M       # RTX-class: the strongest we ship
        if vram_gb >= 8:
            return SIGLIP_L            # mid GPU: still SigLIP, lighter
        return BASE16                  # small GPU: reliable CLIP
    if device == "mps":
        return BASE16                  # Apple Silicon: CLIP, known-good on MPS
    return BASE32                      # cpu / unknown: keep it light


def _clip_fallback(device: str, vram_gb: float | None) -> str:
    """The CLIP model to drop to if the picked model won't load."""
    if device == "cuda" and vram_gb and vram_gb >= 8:
        return LARGE
    if device in ("cuda", "mps"):
        return BASE16
    return BASE32


# ───────────────────────────────────────────────────────── the scorer

class Scorer:
    """Loads one CLIP model and scores images against a scene concept.

    One instance per process (see get_scorer). Loading downloads the weights the
    first time only. Relevance is cached by (model, query, image-bytes hash) so
    walking the query ladder or re-sourcing never recomputes an image.
    """

    def __init__(self, model_id: str, device: str, fallback: str | None = None):
        self.model_id = model_id
        self.device = device
        self.family = _family_of(model_id)
        self.band = _band_of(model_id)
        self._fallback = fallback           # a CLIP id to drop to if this won't load
        self._model = None
        self._proc = None
        self._cache: dict[tuple, float] = {}

    def _load_one(self, model_id: str):
        """Load a specific model, family chosen by its id. Raises on failure."""
        import transformers
        transformers.logging.set_verbosity_error()
        if _family_of(model_id) == "siglip":
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(
                model_id, use_safetensors=True).to(self.device).eval()
            proc = AutoProcessor.from_pretrained(model_id)
        else:
            from transformers import CLIPModel, CLIPProcessor
            # use_safetensors avoids also pulling the legacy pytorch_model.bin —
            # the repo ships both, and fetching each is a needless ~600 MB.
            model = CLIPModel.from_pretrained(
                model_id, use_safetensors=True).to(self.device).eval()
            proc = CLIPProcessor.from_pretrained(model_id)
        return model, proc

    def _load(self):
        if self._model is not None:
            return
        import torch
        self._torch = torch
        try:
            self._model, self._proc = self._load_one(self.model_id)
        except Exception:
            # The picked model won't load here (old transformers, bad download,
            # OOM). Drop to the dependable CLIP fallback rather than losing
            # scoring entirely — this is the whole point of choosing UP only.
            if not self._fallback or self._fallback == self.model_id:
                raise
            self.model_id = self._fallback
            self.family = _family_of(self.model_id)
            self.band = _band_of(self.model_id)
            self._model, self._proc = self._load_one(self.model_id)

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
                # SigLIP was trained with fixed-length text padding; CLIP uses
                # dynamic. Getting this wrong quietly wrecks SigLIP's scores.
                pad = "max_length" if self.family == "siglip" else True
                inputs = self._proc(text=texts, images=[im for _, _, im in todo],
                                    return_tensors="pt", padding=pad).to(self.device)
                with torch.no_grad():
                    m = self._model(**inputs)
                    ie = m.image_embeds / m.image_embeds.norm(dim=-1, keepdim=True)
                    te = m.text_embeds / m.text_embeds.norm(dim=-1, keepdim=True)
                    cos = (ie @ te.t()).tolist()          # images x texts, -1..1
                low, high = self.band
                span = (high - low) or 1e-6
                for (key, ck, _img), row in zip(todo, cos):
                    pos = sum(row[:n_pos]) / n_pos          # match to the subject
                    junk = max(row[n_pos:]) if len(row) > n_pos else 0.0
                    rel = max(0.0, min(1.0, (pos - low) / span))
                    # Only knock a picture down when it looks CLEARLY more like
                    # junk (text, a chart, clip-art) than like the subject — a
                    # small margin, so a real photo that happens to score close on
                    # a junk concept is not wrongly sent to 0. The penalty is
                    # gentler now too (0.55, was 0.35): enough to sink screenshots
                    # below real photos, not enough to erase a borderline match.
                    if junk > pos + 0.02:
                        rel *= 0.55
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
    _SCORER = Scorer(cap["model"], cap["device"], fallback=cap.get("fallback"))
    return _SCORER


def reset() -> None:
    """Drop the singleton — for tests that swap the scorer."""
    global _SCORER, _TRIED
    _SCORER, _TRIED = None, False
