"""Where the pipeline talks to a large language model.

Two providers behind one door:

  - Gemini (cloud, the default) — what we have always used;
  - Ollama (local) — a model on your own GPU, so writing the sheets, expanding
    image queries and drafting descriptions cost NOTHING and never leave the
    machine.

The generators in gemini.py don't care which one answers. They call
`gemini.call(prompt, schema, key, model, …)`, and when `model` names an Ollama
model that call is routed here instead of to Google. Which provider is used is a
single config setting (`llm`: gemini | ollama), so switching to free local
inference is a flag, not a rewrite.

The routing is stateless: the whole Ollama target — host and model name — is
encoded in the `model` string ("ollama:<host>|<name>"), so it threads through the
existing (key, model) call signatures untouched. `model_for(cfg)` builds it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
_PREFIX = "ollama:"
_TIMEOUT = 600          # a big local model on a long prompt is slow; be patient


class LLMError(RuntimeError):
    pass


def _s(cfg: dict | None, key: str, default: str = "") -> str:
    v = (cfg or {}).get(key)
    return default if v in (None, "") else str(v)


# ─────────────────────────────────────────────────────── provider selection

def provider(cfg: dict | None) -> str:
    """Which backend the config asks for — 'gemini' unless Ollama is chosen."""
    return "ollama" if _s(cfg, "llm", "gemini").lower() == "ollama" else "gemini"


def host(cfg: dict | None) -> str:
    return _s(cfg, "ollama_host", DEFAULT_HOST).rstrip("/")


def available(cfg: dict | None) -> bool:
    """Is SOME model usable at all? Gemini needs a key; Ollama needs a name."""
    if provider(cfg) == "ollama":
        return bool(_s(cfg, "ollama_model"))
    return bool(_s(cfg, "gemini_key"))


def key_for(cfg: dict | None) -> str:
    """The Gemini key to pass down (ignored on the Ollama path)."""
    return _s(cfg, "gemini_key")


def model_for(cfg: dict | None) -> str:
    """The model string the generators pass around.

    Ollama targets carry host + name so a single string routes the call with no
    shared state: "ollama:<host>|<name>". Gemini is just the model name (or
    'auto'). Empty means 'nothing configured', which callers gate on.
    """
    if provider(cfg) == "ollama":
        name = _s(cfg, "ollama_model")
        return f"{_PREFIX}{host(cfg)}|{name}" if name else ""
    return _s(cfg, "gemini_model", "auto") or "auto"


def is_ollama(model: str | None) -> bool:
    return bool(model) and model.startswith(_PREFIX)


def _parse(model: str) -> tuple[str, str]:
    """'ollama:<host>|<name>' -> (host, name)."""
    rest = model[len(_PREFIX):]
    host_, _, name = rest.partition("|")
    return (host_ or DEFAULT_HOST), (name or rest)


# ─────────────────────────────────────────────────────── the Ollama backend

def ollama_complete(model: str, prompt: str, schema: dict, system: str = "",
                    temperature: float = 0.4, retries: int = 2) -> dict:
    """One structured-JSON completion from a local Ollama model.

    Uses Ollama's native structured output: the JSON Schema goes in `format`, and
    the model is constrained to it. Same schema objects Gemini uses, so the
    generators are unchanged. Never returns junk — it raises a clear LLMError if
    Ollama isn't running, the model isn't pulled, or the reply isn't valid JSON.
    """
    h, name = _parse(model)
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
            return json.loads(content)
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


# ─────────────────────────────────────────────────────── status (doctor/UI)

def capability(cfg: dict | None = None) -> dict:
    """What the LLM layer can do here, for the doctor and Settings.

    {provider, ok, model, reason, host?, installed?}. Never raises; a probe that
    can't reach Ollama just reports it plainly.
    """
    if provider(cfg) == "ollama":
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

    has_key = bool(_s(cfg, "gemini_key"))
    return {"provider": "gemini", "ok": has_key,
            "model": _s(cfg, "gemini_model", "auto") or "auto",
            "reason": "ready" if has_key else "no gemini_key set"}
