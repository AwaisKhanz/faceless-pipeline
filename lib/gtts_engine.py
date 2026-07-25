"""Narration via Google Cloud Text-to-Speech — Chirp 3 HD voices.

The third voice engine, alongside Chatterbox and Higgs, and the odd one out: it
runs in the CLOUD, not on your GPU. That is the whole point of offering it —
Chirp 3 HD is rock-steady (no artefacts, no dropouts, no "noise instead of
speech"), studio-clean, and covers dozens of languages, at the cost of a few
cents per video and needing a network.

WHERE THE VOICE COMES FROM is the key difference. Chatterbox and Higgs CLONE the
reference clip you drop in voices_refs/. Chirp does NOT clone — Google supplies a
catalogue of ~248 named voices (e.g. en-US-Chirp3-HD-Kore), and you pick one per
language in the Voices panel. So for this engine the "voice" is a Google voice
NAME, not a file, and no reference clip is needed.

PERFECT-FIT AUTH. It reuses the exact Google Cloud credentials the pipeline
already uses for Imagen and Veo: the OAuth token minted by lib/llm._vertex_token
from `vertex_project` + `vertex_service_account` (or Application Default
Credentials). No new key, no separate setup beyond enabling the Text-to-Speech
API on the same project.

Like the Higgs engine, a runtime failure marks the engine UNUSABLE for the
session so the status/render lookups switch to Chatterbox in lockstep — the
dashboard can never say "voiced" while looking at the wrong files.
"""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import llm
from .chatterbox_engine import speech_text   # the shared text cleaner

CACHE = Path(__file__).resolve().parent.parent / "cache" / "voice"

DEFAULT_MODEL = "Chirp3-HD"                    # the voice family we offer
_HOST = "texttospeech.googleapis.com"
_SYNTH_URL = f"https://{_HOST}/v1/text:synthesize"
_VOICES_URL = f"https://{_HOST}/v1/voices"
TIMEOUT = 60

# Our 2-letter language code -> the BCP-47 locale Google keys voices by. Only a
# default: the panel lists every Chirp3-HD voice the locale actually returns, so
# a user can still pick a regional variant. Mandarin is "cmn-CN" at Google, not
# "zh", which is why the map is explicit rather than f"{lang}-{LANG}".
_LOCALE = {
    "ar": "ar-XA", "de": "de-DE", "en": "en-US", "es": "es-ES", "fi": "fi-FI",
    "fr": "fr-FR", "hi": "hi-IN", "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR",
    "nl": "nl-NL", "pl": "pl-PL", "pt": "pt-BR", "ru": "ru-RU", "sv": "sv-SE",
    "tr": "tr-TR", "zh": "cmn-CN", "el": "el-GR", "da": "da-DK", "no": "nb-NO",
}


class GTTSError(RuntimeError):
    """Chirp could not synthesize. The router falls back to Chatterbox."""


# ---------------------------------------------------------------- usability

# Same sticky-failure contract as higgs_engine: once Chirp fails on this machine
# (API not enabled, no credentials, no network), it's marked unusable for the
# session so synth AND the status/render lookups all agree on Chatterbox.
_UNUSABLE = False
_UNUSABLE_REASON = ""


def available(cfg: dict | None = None) -> bool:
    """Offline readiness: a Vertex project is set, the service-account file (if
    named) exists, and google-auth is importable. The Text-to-Speech API being
    enabled and the network being up are proven at call time, not here."""
    return llm.vertex_ready(cfg)


def usable(cfg: dict | None = None) -> bool:
    return available(cfg) and not _UNUSABLE


def unusable_reason() -> str:
    return _UNUSABLE_REASON


def mark_unusable(reason: str = "") -> None:
    global _UNUSABLE, _UNUSABLE_REASON
    _UNUSABLE = True
    _UNUSABLE_REASON = reason or "failed to run on this machine"


def install_hint() -> str:
    return ("Google Chirp 3 HD uses your existing Vertex credentials. To turn it "
            "on:\n"
            "  1. Enable the Text-to-Speech API on your Google Cloud project:\n"
            "       https://console.cloud.google.com/apis/library/texttospeech.googleapis.com\n"
            "  2. Make sure \"vertex_project\" (and \"vertex_service_account\" if "
            "you use one) are set in config.json — the same ones Imagen/Veo use.\n"
            "  3. Set \"voice_engine\": \"chirp\" and pick a voice per language in "
            "the Voices panel.")


# ------------------------------------------------------------------- auth

def _project(cfg: dict) -> str:
    return str((cfg or {}).get("vertex_project") or "").strip()


def _headers(cfg: dict) -> dict:
    """Bearer token, reused from the LLM's Vertex auth.

    The x-goog-user-project (quota/billing) header is set ONLY for user/ADC
    credentials, which need it to name a quota project. A service account already
    carries its own project, and adding the header then demands the
    serviceusage.services.use permission on it — which a minimal SA often lacks,
    causing a 403 even when the Text-to-Speech API is enabled. So we omit it when
    a service-account file is configured.
    """
    sa = str((cfg or {}).get("vertex_service_account") or "")
    token = llm._vertex_token(sa)
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if not sa:
        proj = _project(cfg)
        if proj:
            h["x-goog-user-project"] = proj
    return h


def locale_for(lang: str, cfg: dict | None = None) -> str:
    """The BCP-47 locale for a 2-letter language. A config override wins:
    "google_tts_locale": {"en": "en-GB"} lets you force a regional accent."""
    over = (cfg or {}).get("google_tts_locale") or {}
    if isinstance(over, dict) and over.get(lang):
        return str(over[lang])
    base = lang.lower().split("-")[0]
    return _LOCALE.get(base, f"{base}-{base.upper()}")


# --------------------------------------------------------------- catalogue

_VOICE_CACHE: dict = {}
_LAST_ERROR = ""


def last_voice_error() -> str:
    """Why the most recent voices() call came back empty (for the UI to show).
    Empty string means the last call succeeded."""
    return _LAST_ERROR


def voices(lang: str, cfg: dict | None = None, log=lambda *_: None) -> list[dict]:
    """Google's Chirp 3 HD voices for a language, as [{name, locale, gender}].

    This is what makes "Google supplies the voices" real: the Voices panel calls
    this and shows the list to pick from. Cached per locale for the session.
    Returns [] if the catalogue can't be fetched — but records WHY in
    last_voice_error() so the panel can tell you (usually: the Text-to-Speech API
    isn't enabled on the project) instead of a mystifying "0 available".
    """
    global _LAST_ERROR
    cfg = cfg or {}
    loc = locale_for(lang, cfg)
    if loc in _VOICE_CACHE:
        return _VOICE_CACHE[loc]
    out: list[dict] = []
    try:
        url = f"{_VOICES_URL}?languageCode={urllib.parse.quote(loc)}"
        req = urllib.request.Request(url, headers=_headers(cfg))
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
        for v in data.get("voices", []):
            name = v.get("name", "")
            if "Chirp3-HD" not in name:
                continue
            out.append({"name": name,
                        "locale": (v.get("languageCodes") or [loc])[0],
                        "gender": (v.get("ssmlGender") or "").title()})
        out.sort(key=lambda d: d["name"])
        if not out:
            _LAST_ERROR = (f"The catalogue returned no Chirp 3 HD voices for "
                           f"{loc}. Try a different locale via google_tts_locale.")
        else:
            _LAST_ERROR = ""
    except urllib.error.HTTPError as e:
        _LAST_ERROR = _http_detail(e)
        log(f"  (couldn't list Google voices for {loc}: {_LAST_ERROR})")
        return []
    except Exception as e:                              # noqa: BLE001
        _LAST_ERROR = str(e)
        log(f"  (couldn't list Google voices for {loc}: {e})")
        return []
    _VOICE_CACHE[loc] = out
    return out


def default_voice(lang: str, cfg: dict | None = None) -> str:
    """A reasonable Chirp voice for a language when none is chosen yet — the first
    from the catalogue, or "" if the catalogue can't be read."""
    got = voices(lang, cfg)
    return got[0]["name"] if got else ""


# ------------------------------------------------------------ normalisation

def _norm(text: str) -> str:
    """Same speech-friendly cleaner the other engines use: strip markdown, expand
    degree units, guarantee terminal punctuation."""
    t = speech_text(text).replace("°C", " degrees Celsius").replace("°F", " degrees Fahrenheit")
    t = " ".join(t.split())
    if t and t[-1] not in ".!?,;:\"'":
        t += "."
    return t


def describe(lang: str, cfg: dict | None = None) -> str:
    return "Google Chirp 3 HD · cloud"


# --------------------------------------------------------------------- api

def _key(text: str, voice: str, lang: str, rate: float) -> str:
    blob = f"chirp|{voice}|{lang}|{rate}|{text}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def expected_paths(scenes, lang: str, reference: str, cache: Path = CACHE,
                   opts: dict | None = None, cfg: dict | None = None) -> list[Path]:
    """Where each scene's Chirp audio WOULD be cached. The 'gc_' prefix keeps it
    apart from Chatterbox (cb_) and Higgs (hg_), so switching never mixes clips.
    `reference` here is the Google VOICE NAME, not a file."""
    o = opts or {}
    voice = reference or default_voice(lang, cfg)
    rate = float(o.get("speaking_rate", 1.0))
    return [cache / f"gc_{lang}_{s.n:03d}_"
            f"{_key(_norm(s.narration), voice, lang, rate)}.wav"
            for s in scenes]


def _synth_one(text: str, voice: str, locale: str, rate: float, out: Path,
               cfg: dict, log=print) -> None:
    """One request to text:synthesize. LINEAR16 comes back as a base64 WAV, so
    the bytes are written straight to disk. Retries a couple of times on the
    transient 429/5xx that any cloud call occasionally throws."""
    body = json.dumps({
        "input": {"text": text},
        "voice": {"languageCode": locale, "name": voice},
        "audioConfig": {"audioEncoding": "LINEAR16",
                        "sampleRateHertz": 48000,
                        "speakingRate": rate},
    }).encode("utf-8")

    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(_SYNTH_URL, data=body,
                                         headers=_headers(cfg), method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8"))
            audio = data.get("audioContent")
            if not audio:
                raise GTTSError("response had no audioContent")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(audio))
            if out.stat().st_size < 1024:
                raise GTTSError("wrote a suspiciously empty clip")
            return
        except urllib.error.HTTPError as e:
            detail = _http_detail(e)
            # 4xx that isn't rate-limiting won't get better on retry — fail fast
            # with the real reason (usually: API not enabled, or a bad voice name).
            if e.code not in (429, 500, 502, 503, 504):
                raise GTTSError(f"HTTP {e.code}: {detail}") from None
            last = f"HTTP {e.code}: {detail}"
        except Exception as e:                          # noqa: BLE001
            last = str(e)
        log(f"  Chirp retry {attempt + 1}/3 ({last})")
    raise GTTSError(last or "synthesis failed")


def _http_detail(e: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(e.read().decode("utf-8"))
        return str(payload.get("error", {}).get("message") or payload)[:300]
    except Exception:
        return e.reason if getattr(e, "reason", None) else "request failed"


def preview(text: str, lang: str, voice: str, cfg: dict | None = None,
            out_dir: Path | None = None, rate: float = 1.0, log=print) -> Path:
    """Render (or reuse) a short sample of one Google voice, so you can audition
    it before committing. Written into the previews folder and served like any
    Chatterbox preview. Cached by (voice, rate, text) so re-clicking is instant."""
    cfg = cfg or {}
    out_dir = Path(out_dir or (Path(__file__).resolve().parent.parent / "cache" / "previews"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t = _norm(text)
    out = out_dir / f"gc_prev_{_key(t, voice, lang, float(rate))}.wav"
    if not out.exists() or out.stat().st_size < 1024:
        _synth_one(t, voice, locale_for(lang, cfg), float(rate), out, cfg, log=log)
    return out


def synth(scenes, lang: str, ref_wav=None, cache: Path = CACHE,
          opts: dict | None = None, log=print, cfg: dict | None = None,
          reference: str = "") -> list[Path]:
    """One audio file per scene via Chirp 3 HD, cached like the other engines.

    `reference` is the Google voice NAME (e.g. en-US-Chirp3-HD-Kore). `ref_wav`
    is ignored — Chirp doesn't clone a clip — and is accepted only so the router
    can call every engine with the same signature.
    """
    o = {"speaking_rate": 1.0, **(opts or {})}
    cfg = cfg or {}
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)

    locale = locale_for(lang, cfg)
    voice = reference or default_voice(lang, cfg)
    if not voice:
        raise GTTSError(
            f"No Google voice chosen for '{lang}', and the voice catalogue "
            f"couldn't be read. Pick one in the Voices panel, and check the "
            f"Text-to-Speech API is enabled on your project.")
    rate = float(o.get("speaking_rate", 1.0))
    log(f"  Google Chirp 3 HD · {voice} · {locale}")

    out, made = [], 0
    for s in scenes:
        text = _norm(s.narration)
        p = cache / f"gc_{lang}_{s.n:03d}_{_key(text, voice, lang, rate)}.wav"
        if not p.exists() or p.stat().st_size < 1024:
            _synth_one(text, voice, locale, rate, p, cfg, log=log)
            made += 1
            log(f"S{s.n:>3} voiced  ({text[:52]}...)")
        out.append(p)
    log(f"Chirp: {made} generated, {len(scenes) - made} from cache (cloud).")
    return out
