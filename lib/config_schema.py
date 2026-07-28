"""The config schema: one description of every setting, driving the Settings UI.

This is the single source of truth the Settings page renders from and validates
against. Each field says WHAT it is (a select, a toggle, a number, a path…), its
choices, its range, and when it should even be shown (`show_if`). The page draws
itself from this — add a setting here and a properly-typed control appears, with
no bespoke form code — and the server validates every save against it, so nothing
malformed ever reaches config.json.

Help text and defaults are NOT duplicated here: they are read from
config.example.json (the `_key` doc lines and the values), so the docs stay in
one place. This module only adds the machine-readable shape.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config.example.json"

# The open archives disable_sources can switch off (used to build its checkboxes).
SOURCE_NAMES = ["pexels", "pixabay", "openverse", "wikimedia", "loc", "nasa",
                "smithsonian", "europeana", "ia", "met", "aic", "cleveland"]


def _opt(*values) -> list[dict]:
    """Options as {value,label}. Pass scalars, or (value,label) pairs."""
    out = []
    for v in values:
        if isinstance(v, tuple):
            out.append({"value": v[0], "label": v[1]})
        else:
            out.append({"value": v, "label": str(v)})
    return out


# The visible groups, in order. `id` links fields to a section.
SECTIONS = [
    {"id": "stock", "title": "Stock image keys",
     "help": "Free API keys for Pexels and Pixabay — the fastest sources."},
    {"id": "archives", "title": "Public archives",
     "help": "Optional keys that widen the museum / public-domain sources."},
    {"id": "llm", "title": "Language model",
     "help": "Writes the sheets, image queries and descriptions. Pick a provider; "
             "only its fields show. Vertex credentials are shared with Imagen, "
             "Veo and Chirp voice."},
    {"id": "sourcing", "title": "Finding visuals",
     "help": "How scenes are matched to pictures, and how much runs at once."},
    {"id": "matching", "title": "Visual matching",
     "help": "Scores each candidate for how well it fits the scene."},
    {"id": "imagegen", "title": "AI image generation",
     "help": "Generate stills with Vertex when search falls short. Uses your "
             "Google Cloud credit."},
    {"id": "videogen", "title": "AI video (Veo)",
     "help": "Expensive, manual-only motion clips."},
    {"id": "voice", "title": "Voice / narration",
     "help": "Which engine speaks, and its options."},
    {"id": "audio", "title": "Captions & audio",
     "help": "Timing, loudness and the little polish steps."},
]

# Field definitions. `help` and `default` are filled from config.example.json.
#   type: text | secret | bool | number | select | multiselect | dict
#   show_if: {other_key: [allowed values]} — the field shows only when it matches.
_FIELDS: list[dict] = [
    # ── stock keys ──────────────────────────────────────────────────────────
    {"key": "pexels_key", "section": "stock", "type": "secret"},
    {"key": "pixabay_key", "section": "stock", "type": "secret"},

    # ── archives ────────────────────────────────────────────────────────────
    {"key": "contact", "section": "archives", "type": "text"},
    {"key": "smithsonian_key", "section": "archives", "type": "secret"},
    {"key": "europeana_key", "section": "archives", "type": "secret"},
    {"key": "openverse_client_id", "section": "archives", "type": "secret"},
    {"key": "openverse_client_secret", "section": "archives", "type": "secret"},
    {"key": "openverse_token", "section": "archives", "type": "secret"},

    # ── language model ──────────────────────────────────────────────────────
    {"key": "llm", "section": "llm", "type": "select",
     "options": _opt(("gemini", "Gemini (AI Studio key)"),
                     ("vertex", "Vertex (Google Cloud)"),
                     ("ollama", "Ollama (local)"),
                     ("grok", "Grok (xAI)"),
                     ("openrouter", "OpenRouter"))},
    {"key": "gemini_key", "section": "llm", "type": "secret"},
    {"key": "gemini_model", "section": "llm", "type": "select", "allow_custom": True,
     "options": _opt(("auto", "auto (pick the best Flash)"),
                     "gemini-2.5-flash", "gemini-2.5-pro"),
     "show_if": {"llm": ["gemini"]}},
    # Vertex credentials are shared (LLM + Imagen + Veo + Chirp), so always shown.
    {"key": "vertex_project", "section": "llm", "type": "text"},
    {"key": "vertex_location", "section": "llm", "type": "text"},
    {"key": "vertex_service_account", "section": "llm", "type": "text"},
    {"key": "vertex_model", "section": "llm", "type": "select", "allow_custom": True,
     "options": _opt("gemini-2.5-flash", "gemini-2.5-pro"),
     "show_if": {"llm": ["vertex"]}},
    {"key": "ollama_host", "section": "llm", "type": "text", "show_if": {"llm": ["ollama"]}},
    {"key": "ollama_model", "section": "llm", "type": "text", "show_if": {"llm": ["ollama"]}},
    {"key": "grok_key", "section": "llm", "type": "secret", "show_if": {"llm": ["grok"]}},
    {"key": "grok_model", "section": "llm", "type": "text", "show_if": {"llm": ["grok"]}},
    {"key": "openrouter_key", "section": "llm", "type": "secret", "show_if": {"llm": ["openrouter"]}},
    {"key": "openrouter_model", "section": "llm", "type": "text", "show_if": {"llm": ["openrouter"]}},

    # ── finding visuals ─────────────────────────────────────────────────────
    {"key": "search_all_sources", "section": "sourcing", "type": "bool"},
    {"key": "disable_sources", "section": "sourcing", "type": "multiselect",
     "options": _opt(*SOURCE_NAMES)},
    {"key": "source_workers", "section": "sourcing", "type": "number",
     "int": True, "min": 1, "max": 8, "step": 1},
    {"key": "max_concurrent_jobs", "section": "sourcing", "type": "select",
     "auto_or_int": True,
     "options": _opt("auto", 1, 2, 3, 4, 5, 6)},
    {"key": "name_real_people", "section": "sourcing", "type": "bool"},
    {"key": "image_licenses", "section": "sourcing", "type": "select",
     "options": _opt(("strict", "strict — CC0 / public domain only"),
                     ("all", "all — accept every open license"))},
    {"key": "expand_queries", "section": "sourcing", "type": "select",
     "options": _opt(("auto", "auto"), ("off", "off"))},
    {"key": "auto_split", "section": "sourcing", "type": "select",
     "options": _opt(("auto", "auto"), ("off", "off"))},
    {"key": "source_log", "section": "sourcing", "type": "select",
     "options": _opt(("", "clean (default)"), ("full", "full — every candidate"))},
    {"key": "log_detail", "section": "sourcing", "type": "select",
     "options": _opt(("normal", "normal"), ("full", "full — verbose everywhere"))},

    # ── visual matching ─────────────────────────────────────────────────────
    {"key": "clip", "section": "matching", "type": "select",
     "options": _opt(("auto", "auto"), ("off", "off"))},
    {"key": "clip_model", "section": "matching", "type": "select",
     "allow_custom": True, "allow_empty": True,
     "options": _opt(("", "auto — pick best for this machine"),
                     "openai/clip-vit-base-patch32",
                     "google/siglip2-so400m-patch14-384"),
     "show_if": {"clip": ["auto"]}},
    {"key": "clip_min", "section": "matching", "type": "number",
     "min": 0, "max": 1, "step": 0.01, "show_if": {"clip": ["auto"]}},

    # ── image generation ────────────────────────────────────────────────────
    {"key": "generate", "section": "imagegen", "type": "select",
     "options": _opt(("off", "off — search only"),
                     ("mixed", "mixed — fill weak matches"),
                     ("all", "all — generate every scene"))},
    {"key": "generate_engine", "section": "imagegen", "type": "select",
     "options": _opt(("pollinations", "Pollinations.ai — free, no key (default)"),
                     ("cloudflare", "Cloudflare Workers AI — needs account + token"),
                     ("vertex", "Vertex AI — Google Gemini image (needs project)"))},
    {"key": "generate_min", "section": "imagegen", "type": "number",
     "min": 0, "max": 1, "step": 0.01, "show_if": {"generate": ["mixed"]}},
    {"key": "generate_max", "section": "imagegen", "type": "number",
     "int": True, "min": 1, "max": 500, "step": 1, "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_model", "section": "imagegen", "type": "select", "allow_custom": True,
     "options": _opt(("gemini-3.1-flash-lite-image", "Gemini 3.1 Flash Lite image"),
                     ("gemini-2.5-flash-image", "Nano Banana (2.5 flash) — ~$0.04"),
                     ("gemini-3-pro-image", "Gemini 3 Pro image (4K, pricier)")),
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_location", "section": "imagegen", "type": "text",
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_aspect", "section": "imagegen", "type": "select",
     "options": _opt("16:9", "1:1", "9:16", "4:3", "3:4"),
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_style", "section": "imagegen", "type": "text",
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_exact", "section": "imagegen", "type": "bool",
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_workers", "section": "imagegen", "type": "number",
     "int": True, "min": 1, "max": 4, "step": 1,
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_min_interval", "section": "imagegen", "type": "number",
     "min": 0, "max": 30, "step": 0.5,
     "show_if": {"generate": ["mixed", "all"]}},
    {"key": "generate_retries", "section": "imagegen", "type": "number",
     "int": True, "min": 0, "max": 8, "step": 1,
     "show_if": {"generate": ["mixed", "all"]}},
    # Pollinations engine settings
    {"key": "pollinations_model", "section": "imagegen", "type": "select", "allow_custom": True,
     "options": _opt(("flux", "flux — photoreal (default)"), ("turbo", "turbo — faster")),
     "show_if": {"generate_engine": ["pollinations"]}},
    {"key": "pollinations_interval", "section": "imagegen", "type": "number",
     "min": 0, "max": 30, "step": 0.5, "show_if": {"generate_engine": ["pollinations"]}},
    {"key": "pollinations_token", "section": "imagegen", "type": "secret",
     "show_if": {"generate_engine": ["pollinations"]}},
    # Cloudflare Workers AI engine settings
    {"key": "cf_account_id", "section": "imagegen", "type": "text",
     "show_if": {"generate_engine": ["cloudflare"]}},
    {"key": "cf_api_token", "section": "imagegen", "type": "secret",
     "show_if": {"generate_engine": ["cloudflare"]}},
    {"key": "cf_accounts", "section": "imagegen", "type": "textarea",
     "show_if": {"generate_engine": ["cloudflare"]}},
    {"key": "cf_rest_minutes", "section": "imagegen", "type": "number",
     "int": True, "min": 1, "max": 720, "step": 1,
     "show_if": {"generate_engine": ["cloudflare"]}},
    {"key": "cf_model", "section": "imagegen", "type": "select", "allow_custom": True,
     "options": _opt(("@cf/black-forest-labs/flux-2-dev",
                      "FLUX.2 [dev] — best quality, PAID (~$0.07/image)"),
                     ("@cf/black-forest-labs/flux-1-schnell",
                      "FLUX.1 schnell — free, fast, decent"),
                     ("@cf/bytedance/stable-diffusion-xl-lightning",
                      "SDXL-Lightning — free, fast, softer"),
                     ("@cf/stabilityai/stable-diffusion-xl-base-1.0",
                      "Stable Diffusion XL base — free")),
     "show_if": {"generate_engine": ["cloudflare"]}},
    {"key": "cf_steps", "section": "imagegen", "type": "number",
     "int": True, "min": 1, "max": 50, "step": 1,
     "show_if": {"generate_engine": ["cloudflare"]}},
    {"key": "cf_guidance", "section": "imagegen", "type": "number",
     "min": 1, "max": 20, "step": 0.5,
     "show_if": {"generate_engine": ["cloudflare"]}},

    # ── video (Veo) ─────────────────────────────────────────────────────────
    {"key": "veo_max", "section": "videogen", "type": "number", "int": True, "min": 1, "max": 20, "step": 1},
    {"key": "veo_seconds", "section": "videogen", "type": "select", "int": True,
     "options": _opt(4, 6, 8)},
    {"key": "veo_model", "section": "videogen", "type": "select", "allow_custom": True,
     "options": _opt(("veo-3.1-generate-001", "veo-3.1 (GA)"),
                     ("veo-3.1-fast-generate-001", "veo-3.1 fast (cheaper)"))},
    {"key": "veo_location", "section": "videogen", "type": "text"},
    {"key": "veo_smart_prompt", "section": "videogen", "type": "select",
     "options": _opt(("auto", "auto"), ("off", "off"))},
    {"key": "veo_style", "section": "videogen", "type": "text", "show_if": {"veo_smart_prompt": ["off"]}},

    # ── voice ───────────────────────────────────────────────────────────────
    {"key": "voice_engine", "section": "voice", "type": "select",
     "options": _opt(("chatterbox", "Chatterbox — local, clones a clip (default)"),
                     ("higgs", "Higgs — local, CUDA, clones a clip"),
                     ("chirp", "Chirp 3 HD — Google Cloud, catalogue voices"))},
    {"key": "higgs_model", "section": "voice", "type": "text", "show_if": {"voice_engine": ["higgs"]}},
    {"key": "higgs_tokenizer", "section": "voice", "type": "text", "show_if": {"voice_engine": ["higgs"]}},
    {"key": "higgs_device", "section": "voice", "type": "select",
     "options": _opt(("auto", "auto"), ("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")),
     "show_if": {"voice_engine": ["higgs"]}},
    {"key": "higgs_asr_model", "section": "voice", "type": "select", "allow_custom": True,
     "options": _opt("openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"),
     "show_if": {"voice_engine": ["higgs"]}},
    {"key": "google_tts_rate", "section": "voice", "type": "number",
     "min": 0.25, "max": 2.0, "step": 0.05, "show_if": {"voice_engine": ["chirp"]}},
    {"key": "google_tts_locale", "section": "voice", "type": "dict",
     "show_if": {"voice_engine": ["chirp"]}},
    {"key": "voice_flow", "section": "voice", "type": "select",
     "options": _opt(("off", "off — one clip per scene"),
                     ("sentence", "sentence — join a sentence, split the audio"))},

    # ── captions & audio ────────────────────────────────────────────────────
    {"key": "align", "section": "audio", "type": "select",
     "options": _opt(("auto", "auto"), ("off", "off"))},
    {"key": "audio_master", "section": "audio", "type": "select",
     "options": _opt(("auto", "auto"), ("off", "off"))},
    {"key": "lufs_target", "section": "audio", "type": "number",
     "int": True, "min": -24, "max": -8, "step": 1, "show_if": {"audio_master": ["auto"]}},
    {"key": "music_duck", "section": "audio", "type": "bool"},
    {"key": "trim_silence", "section": "audio", "type": "bool"},
    {"key": "scene_gap", "section": "audio", "type": "number", "min": 0, "max": 1.5, "step": 0.05},
    {"key": "scene_flow_gap", "section": "audio", "type": "number", "min": 0, "max": 0.5, "step": 0.02},
    {"key": "scene_dissolve", "section": "audio", "type": "number", "min": 0, "max": 2.0, "step": 0.05},
    {"key": "caption_lead", "section": "audio", "type": "number", "min": 0, "max": 0.5, "step": 0.01},
]


@lru_cache(maxsize=1)
def _example() -> dict:
    try:
        return json.loads(EXAMPLE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fields() -> list[dict]:
    """Every field, enriched with help + default from config.example.json."""
    ex = _example()
    out = []
    for f in _FIELDS:
        key = f["key"]
        out.append({**f, "help": ex.get("_" + key, ""),
                    "default": ex.get(key)})
    return out


def known_keys() -> set[str]:
    return {f["key"] for f in _FIELDS}


def schema(values: dict | None = None) -> list[dict]:
    """Sections, each with its fields (type/options/help/default/show_if), and —
    when `values` is given — the current value of each field folded in."""
    values = values or {}
    by_section: dict[str, list] = {s["id"]: [] for s in SECTIONS}
    for f in _fields():
        entry = dict(f)
        if f["key"] in values:
            entry["value"] = values[f["key"]]
        elif f.get("default") is not None:
            entry["value"] = f["default"]
        by_section.setdefault(f["section"], []).append(entry)
    return [{**s, "fields": by_section.get(s["id"], [])} for s in SECTIONS]


# ----------------------------------------------------------------- validation

class _Bad(ValueError):
    pass


def _coerce(field: dict, raw):
    """Coerce a raw UI value to the correct Python type, raising _Bad on anything
    that can't be made valid. Numbers are clamped to their range (forgiving);
    selects must match an option (strict)."""
    t = field["type"]

    if t in ("text", "secret", "textarea"):
        return "" if raw is None else str(raw)

    if t == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("true", "1", "yes", "on")

    if t == "number":
        try:
            n = float(raw)
        except (TypeError, ValueError):
            raise _Bad("must be a number")
        if "min" in field:
            n = max(field["min"], n)
        if "max" in field:
            n = min(field["max"], n)
        return int(round(n)) if field.get("int") else round(n, 4)

    if t == "select":
        if field.get("auto_or_int"):
            if str(raw).strip().lower() == "auto":
                return "auto"
            try:
                return int(raw)
            except (TypeError, ValueError):
                raise _Bad("must be 'auto' or a whole number")
        val = int(raw) if field.get("int") else raw
        # allow_custom fields (e.g. model names, which change over time) accept any
        # value — the options are just suggestions, not a closed set. Empty is
        # rejected unless allow_empty (e.g. clip_model, where "" means auto-pick).
        if field.get("allow_custom"):
            if isinstance(val, str) and not val.strip():
                if field.get("allow_empty"):
                    return ""
                raise _Bad("cannot be empty")
            return val
        allowed = [o["value"] for o in field.get("options", [])]
        if allowed and val not in allowed:
            raise _Bad(f"must be one of {allowed}")
        return val

    if t == "multiselect":
        if not isinstance(raw, list):
            raise _Bad("must be a list")
        allowed = {o["value"] for o in field.get("options", [])}
        return [v for v in raw if v in allowed]

    if t == "dict":
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return {}
            try:
                got = json.loads(s)
            except json.JSONDecodeError as e:
                raise _Bad(f"invalid JSON ({e.msg})")
            if not isinstance(got, dict):
                raise _Bad("must be a JSON object")
            return got
        raise _Bad("must be an object")

    return raw


def validate_and_merge(current: dict, updates: dict) -> tuple[dict, dict]:
    """Merge validated `updates` onto `current`.

    Only KNOWN keys are ever written, so the UI can't inject arbitrary keys, and
    every other key in the file — the `_label` docs, `_section_*` headers, and
    anything unknown — is preserved untouched. Returns (merged, errors); if
    errors is non-empty the caller should not save.
    """
    fields = {f["key"]: f for f in _FIELDS}
    merged = json.loads(json.dumps(current))     # deep copy, key order preserved
    errors: dict[str, str] = {}
    for key, raw in (updates or {}).items():
        f = fields.get(key)
        if not f:
            continue                              # ignore unknown / non-editable keys
        try:
            merged[key] = _coerce(f, raw)
        except _Bad as e:
            errors[key] = str(e)
    return merged, errors
