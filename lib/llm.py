"""Where the pipeline talks to a large language model.

One door, several providers, chosen by config so you can pick the right one for
each machine — local on the RTX, a cloud key on a laptop — with no code change:

  - Gemini   (cloud, the default) — Google's API;
  - Ollama   (local) — a model on your own GPU: free, private, offline;
  - Grok     (cloud) — xAI, via its OpenAI-compatible API;
  - …any other OpenAI-compatible provider (Groq, OpenRouter, …) is one table
    entry away — see _OPENAI.

The generators in gemini.py don't care which one answers. They call
`gemini.call(prompt, schema, key, model, …)`, and the `model` string carries
everything needed to route the call, so it threads through the existing
(key, model) signatures with no shared state and no rewrite:

    gemini            "auto"  or  "gemini-2.5-flash"
    ollama            "ollama:<host>|<name>"
    openai-compatible "openai:<base_url>|<name>"     (Grok, Groq, OpenRouter…)

`model_for(cfg)` builds that string and `key_for(cfg)` returns the matching key;
`provider(cfg)` names the choice. Adding a provider means adding a row to _OPENAI,
nothing more.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
_OLLAMA = "ollama:"
_OPENAI_PREFIX = "openai:"
_VERTEX = "vertex:"
# Vertex AI (Gemini on Google Cloud) authenticates with a Cloud OAuth token, not
# an API key. The service-account JSON (or Application Default Credentials) is the
# permanent credential; google-auth mints and auto-refreshes the short-lived token
# from it, so nothing here is ever renewed by hand.
_VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TIMEOUT = 600          # a big local model on a long prompt is slow; be patient

# OpenAI-compatible cloud providers. name -> (base_url, key field, model field).
# Every one of these speaks POST {base}/chat/completions with Bearer auth and
# JSON-schema structured output, so they all share one backend below. To add
# Groq or OpenRouter, add a row — that is the whole change.
_OPENAI = {
    "grok": ("https://api.x.ai/v1", "grok_key", "grok_model"),
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter_key", "openrouter_model"),
    # "groq": ("https://api.groq.com/openai/v1", "groq_key", "groq_model"),
}


class LLMError(RuntimeError):
    pass


def _s(cfg: dict | None, key: str, default: str = "") -> str:
    v = (cfg or {}).get(key)
    return default if v in (None, "") else str(v)


def _json_loads(content: str):
    """Parse JSON a model returned, tolerating the mess weaker/free models add.

    Some models wrap the object in ```json fences, or prepend a sentence. Try the
    clean parse first, then strip fences, then fall back to the first {...} / [...]
    span. Raises ValueError if nothing parses, so callers can retry.
    """
    if not content:
        raise ValueError("empty response")
    try:
        return json.loads(content)
    except Exception:
        pass
    t = content.strip()
    if t.startswith("```"):                       # ```json … ```  or  ``` … ```
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.lstrip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        try:
            return json.loads(t.strip())
        except Exception:
            pass
    for open_c, close_c in (("{", "}"), ("[", "]")):    # first balanced-ish span
        i, j = content.find(open_c), content.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(content[i:j + 1])
            except Exception:
                continue
    raise ValueError("no JSON found in the response")


# ─────────────────────────────────────────────────────── provider selection

def provider(cfg: dict | None) -> str:
    """Which backend the config asks for. 'gemini' unless a known other is set."""
    p = _s(cfg, "llm", "gemini").lower()
    return p if (p in ("ollama", "vertex") or p in _OPENAI) else "gemini"


def host(cfg: dict | None) -> str:
    return _s(cfg, "ollama_host", DEFAULT_HOST).rstrip("/")


def available(cfg: dict | None) -> bool:
    """Is SOME model usable? Each provider needs its own thing configured."""
    p = provider(cfg)
    if p == "ollama":
        return bool(_s(cfg, "ollama_model"))
    if p == "vertex":
        # The service-account JSON is optional (ADC can supply credentials), so a
        # project and a model are the minimum. The token is proven at call time.
        return bool(_s(cfg, "vertex_project") and _s(cfg, "vertex_model"))
    if p in _OPENAI:
        _, kf, mf = _OPENAI[p]
        return bool(_s(cfg, kf) and _s(cfg, mf))
    return bool(_s(cfg, "gemini_key"))


def key_for(cfg: dict | None) -> str:
    """The credential to pass down. Empty for Ollama (local) and for Vertex when
    it uses Application Default Credentials; otherwise the API key, or — for
    Vertex — the PATH to the service-account JSON."""
    p = provider(cfg)
    if p == "ollama":
        return ""
    if p == "vertex":
        return _s(cfg, "vertex_service_account")     # "" => ADC / env var
    if p in _OPENAI:
        return _s(cfg, _OPENAI[p][1])
    return _s(cfg, "gemini_key")


def model_for(cfg: dict | None) -> str:
    """The self-routing model string the generators pass around. Empty means
    'nothing configured', which callers gate on."""
    p = provider(cfg)
    if p == "ollama":
        name = _s(cfg, "ollama_model")
        return f"{_OLLAMA}{host(cfg)}|{name}" if name else ""
    if p == "vertex":
        proj = _s(cfg, "vertex_project")
        loc = _s(cfg, "vertex_location", "us-central1") or "us-central1"
        name = _s(cfg, "vertex_model")
        return f"{_VERTEX}{proj}|{loc}|{name}" if (proj and name) else ""
    if p in _OPENAI:
        base, _, mf = _OPENAI[p]
        name = _s(cfg, mf)
        return f"{_OPENAI_PREFIX}{base}|{name}" if name else ""
    return _s(cfg, "gemini_model", "auto") or "auto"


def is_ollama(model: str | None) -> bool:
    return bool(model) and model.startswith(_OLLAMA)


def is_openai(model: str | None) -> bool:
    return bool(model) and model.startswith(_OPENAI_PREFIX)


def is_vertex(model: str | None) -> bool:
    return bool(model) and model.startswith(_VERTEX)


def _parse(model: str, prefix: str, default_head: str) -> tuple[str, str]:
    """'<prefix><head>|<name>' -> (head, name)."""
    rest = model[len(prefix):]
    head, _, name = rest.partition("|")
    return (head or default_head), (name or rest)


# ─────────────────────────────────────────────────────── the Ollama backend

def ollama_complete(model: str, prompt: str, schema: dict, system: str = "",
                    temperature: float = 0.4, retries: int = 2) -> dict:
    """One structured-JSON completion from a local Ollama model.

    Uses Ollama's native structured output: the JSON Schema goes in `format`, and
    the model is constrained to it. Same schema objects Gemini uses, so the
    generators are unchanged. Never returns junk — it raises a clear LLMError if
    Ollama isn't running, the model isn't pulled, or the reply isn't valid JSON.
    """
    h, name = _parse(model, _OLLAMA, DEFAULT_HOST)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = json.dumps({
        "model": name, "messages": messages, "stream": False,
        "format": schema, "options": {"temperature": temperature},
    }).encode("utf-8")

    last = ""
    for _ in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                f"{h}/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(
                f"Ollama returned {e.code}: {detail}\n"
                f"If the model isn't installed, run:  ollama pull {name}") from None
        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not reach Ollama at {h} ({e.reason}). Is it running? "
                f"Start it with `ollama serve`, then `ollama pull {name}`.") from None
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        content = (payload.get("message") or {}).get("content", "")
        try:
            return _json_loads(content)
        except Exception:
            last = "the model did not return valid JSON"
            continue
    raise LLMError(f"Ollama gave no usable JSON after {retries} tries. {last}")


def list_ollama(host_: str) -> list[str] | None:
    """Model names installed on an Ollama host, or None if it can't be reached."""
    try:
        with urllib.request.urlopen(f"{host_.rstrip('/')}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return None


# ───────────────────────────────────────── the OpenAI-compatible backend (Grok…)

def _openai_body(name, messages, schema, temperature, structured) -> bytes:
    b = {"model": name, "messages": messages, "temperature": temperature}
    if structured:
        # Full JSON-schema constraint — the reliable path on providers that
        # support it (xAI Grok does).
        b["response_format"] = {"type": "json_schema", "json_schema":
                                {"name": "response", "schema": schema, "strict": False}}
    else:
        # Fallback for endpoints that only do plain JSON mode: ask for JSON and
        # describe the shape in the prompt instead of constraining it.
        b["response_format"] = {"type": "json_object"}
    return json.dumps(b).encode("utf-8")


def openai_complete(model: str, key: str, prompt: str, schema: dict,
                    system: str = "", temperature: float = 0.4,
                    retries: int = 3) -> dict:
    """One structured-JSON completion from an OpenAI-compatible API (Grok, etc.).

    POSTs to {base}/chat/completions with Bearer auth and a json_schema
    response_format. If a provider rejects json_schema it retries once in plain
    JSON mode with the schema described in the prompt, so it works across
    providers. Raises a clear LLMError rather than returning junk.
    """
    base, name = _parse(model, _OPENAI_PREFIX, "")
    sys_msg = system
    structured = True
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
                   [{"role": "user", "content": prompt}]
        body = _openai_body(name, messages, schema, temperature, structured)
        try:
            req = urllib.request.Request(
                f"{base.rstrip('/')}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            if e.code in (400, 422) and structured:
                # Likely doesn't accept json_schema — drop to plain JSON mode and
                # spell the shape out in the prompt on the next try.
                structured = False
                sys_msg = (system + "\n\n" if system else "") + (
                    "Reply with a single JSON object that matches this JSON "
                    "schema exactly, and nothing else:\n" + json.dumps(schema))
                last = f"HTTP {e.code}: {detail}"
                continue
            if e.code == 429:
                last = "rate limited"
                continue
            raise LLMError(
                f"{name}: provider returned {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not reach {base} ({e.reason}). Check the network and "
                f"that the key/model are right.") from None
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        try:
            content = payload["choices"][0]["message"]["content"]
            return _json_loads(content)
        except Exception:
            last = "the model did not return valid JSON"
            continue
    raise LLMError(f"{name} gave no usable JSON after {retries} tries. {last}")


# ─────────────────────────────────────────────── the Vertex AI backend (Gemini)
#
# Gemini on Vertex is the same request/response shape as the AI Studio API — the
# only differences are the endpoint (a Cloud aiplatform host, with the project and
# region in the path) and the auth (a Cloud OAuth Bearer token instead of ?key=).
# So this reuses the same schema objects and JSON parsing; only the transport
# differs. Selecting "vertex" bills against the Google Cloud project (and its
# $300 free credit); the existing AI Studio "gemini" provider is untouched.

_VERTEX_CREDS: dict = {}          # credential-source -> cached google-auth creds


def _parse_vertex(model: str) -> tuple[str, str, str]:
    """'vertex:<project>|<location>|<model>' -> (project, location, model)."""
    project, _, rest = model[len(_VERTEX):].partition("|")
    location, _, name = rest.partition("|")
    return project, (location or "us-central1"), (name or "gemini-2.5-flash")


def _vertex_token(sa_path: str) -> str:
    """A valid Cloud access token, minted and auto-refreshed by google-auth.

    The permanent credential is the service-account JSON at `sa_path`; an empty
    path falls back to Application Default Credentials (the GOOGLE_APPLICATION_
    CREDENTIALS env var, or `gcloud auth application-default login`). The 1-hour
    token is refreshed here whenever it has expired, so nothing is renewed by hand.
    """
    try:
        import google.auth
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError:
        raise LLMError(
            "Vertex AI needs the google-auth library. Install it once:\n"
            "    pip install google-auth\n"
            "then set \"llm\": \"vertex\" in config.json.") from None

    cache_key = sa_path or "__adc__"
    creds = _VERTEX_CREDS.get(cache_key)
    if creds is None:
        if sa_path:
            if not os.path.exists(sa_path):
                raise LLMError(
                    f"Service-account file not found: {sa_path}\n"
                    f"Check \"vertex_service_account\" in config.json.")
            creds = service_account.Credentials.from_service_account_file(
                sa_path, scopes=[_VERTEX_SCOPE])
        else:
            creds, _ = google.auth.default(scopes=[_VERTEX_SCOPE])
        _VERTEX_CREDS[cache_key] = creds
    if not creds.valid:
        creds.refresh(Request())
    return creds.token


def vertex_complete(model: str, key: str, prompt: str, schema: dict,
                    system: str = "", temperature: float = 0.4,
                    retries: int = 3) -> dict:
    """One structured-JSON completion from Gemini on Vertex AI.

    `key` is the PATH to the service-account JSON (or "" for ADC). Same schema and
    JSON handling as the other providers; raises a clear LLMError rather than junk.
    """
    project, location, name = _parse_vertex(model)
    if not project:
        raise LLMError("Vertex AI needs a project — set \"vertex_project\".")
    host_ = ("aiplatform.googleapis.com" if location == "global"
             else f"{location}-aiplatform.googleapis.com")
    url = (f"https://{host_}/v1/projects/{project}/locations/{location}"
           f"/publishers/google/models/{name}:generateContent")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": temperature,
            "maxOutputTokens": 65536,
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    data = json.dumps(body).encode("utf-8")

    token = _vertex_token(key)
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            if e.code in (401, 403) and attempt == 1:
                # Token expired mid-run, or a stale cache — force a fresh mint.
                _VERTEX_CREDS.pop(key or "__adc__", None)
                token = _vertex_token(key)
                last = f"re-authenticated after {e.code}"
                continue
            if e.code == 429:
                last = "rate limited"
                time.sleep(8 * attempt)
                continue
            raise LLMError(
                f"Vertex AI returned {e.code} for {name} in {project}/{location}:\n"
                f"{detail}") from None
        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not reach Vertex AI ({e.reason}). Check the network and "
                f"that the region '{location}' is right.") from None
        except Exception as e:                        # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        cands = payload.get("candidates") or []
        if not cands:
            last = f"no candidates: {payload.get('promptFeedback', payload)}"
            continue
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        try:
            return _json_loads(text)
        except Exception:
            last = "the model did not return valid JSON"
            continue
    raise LLMError(f"Vertex AI gave no usable JSON after {retries} tries. {last}")


# Candidate Gemini models to try on Vertex, best-quality first. Availability
# varies by project and region and changes often, so this is only a list to
# PROBE — `faceless vertex-models` calls each one and reports what actually works
# for you. (id, one-line note.)
VERTEX_CANDIDATES: list[tuple[str, str]] = [
    ("gemini-3.1-pro",        "strongest reasoning (preview) — best quality, priciest"),
    ("gemini-3-pro",          "Gemini 3 Pro — top quality"),
    ("gemini-3.6-flash",      "newest Flash (GA) — great quality for the cost"),
    ("gemini-3-flash",        "Gemini 3 Flash"),
    ("gemini-3.5-flash-lite", "cheapest 3.x — fast and light"),
    ("gemini-2.5-pro",        "strong, but retires Oct 2026"),
    ("gemini-2.5-flash",      "reliable default — GA in us-central1"),
    ("gemini-2.5-flash-lite", "cheapest 2.5"),
    ("gemini-2.0-flash",      "older, cheap, widely available"),
]


def vertex_probe(project: str, location: str, model: str,
                 sa_path: str = "") -> tuple[bool, str]:
    """One tiny generateContent call to see if a model is enabled and available
    in a region for this project. Returns (ok, short reason). Never raises."""
    try:
        token = _vertex_token(sa_path)
    except LLMError as e:
        return False, str(e).splitlines()[0]
    host_ = ("aiplatform.googleapis.com" if location == "global"
             else f"{location}-aiplatform.googleapis.com")
    url = (f"https://{host_}/v1/projects/{project}/locations/{location}"
           f"/publishers/google/models/{model}:generateContent")
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                       "generationConfig": {"maxOutputTokens": 1}}).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json",
                                     "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return True, "ok"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        msg = detail
        try:
            msg = (json.loads(detail).get("error") or {}).get("message", detail)
        except Exception:
            pass
        return False, f"{e.code}: {msg.strip()[:90]}"
    except Exception as e:                            # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:70]}"


# ─────────────────────────────────────────────────────── status (doctor/UI)

def capability(cfg: dict | None = None) -> dict:
    """What the LLM layer can do here, for the doctor and Settings.

    {provider, ok, model, reason, host?, installed?}. Never raises; a probe that
    can't reach Ollama just reports it plainly.
    """
    p = provider(cfg)
    if p == "ollama":
        h, m = host(cfg), _s(cfg, "ollama_model")
        if not m:
            return {"provider": "ollama", "ok": False, "model": "", "host": h,
                    "reason": "set ollama_model in config.json (e.g. qwen3:14b)"}
        installed = list_ollama(h)
        if installed is None:
            return {"provider": "ollama", "ok": False, "model": m, "host": h,
                    "reason": f"Ollama not reachable at {h} — run `ollama serve`"}
        if m not in installed:
            return {"provider": "ollama", "ok": False, "model": m, "host": h,
                    "installed": installed,
                    "reason": f"model not pulled — run: ollama pull {m}"}
        return {"provider": "ollama", "ok": True, "model": m, "host": h,
                "installed": installed, "reason": "ready"}

    if p == "vertex":
        proj, m = _s(cfg, "vertex_project"), _s(cfg, "vertex_model")
        loc = _s(cfg, "vertex_location", "us-central1") or "us-central1"
        sa = _s(cfg, "vertex_service_account")
        if not proj:
            return {"provider": "vertex", "ok": False, "model": m,
                    "reason": "set vertex_project in config.json"}
        if not m:
            return {"provider": "vertex", "ok": False, "model": "",
                    "reason": "set vertex_model (e.g. gemini-2.5-flash)"}
        if sa and not os.path.exists(sa):
            return {"provider": "vertex", "ok": False, "model": m,
                    "reason": f"service-account file not found: {sa}"}
        cred = os.path.basename(sa) if sa else "application default credentials"
        return {"provider": "vertex", "ok": True, "model": m,
                "host": f"{proj} · {loc} · {cred}", "reason": "ready"}

    if p in _OPENAI:
        base, kf, mf = _OPENAI[p]
        key, m = _s(cfg, kf), _s(cfg, mf)
        if not key:
            return {"provider": p, "ok": False, "model": m,
                    "reason": f"set {kf} in config.json"}
        if not m:
            return {"provider": p, "ok": False, "model": "",
                    "reason": f"set {mf} in config.json (e.g. grok-4)"}
        return {"provider": p, "ok": True, "model": m, "reason": "ready"}

    has_key = bool(_s(cfg, "gemini_key"))
    return {"provider": "gemini", "ok": has_key,
            "model": _s(cfg, "gemini_model", "auto") or "auto",
            "reason": "ready" if has_key else "no gemini_key set"}
