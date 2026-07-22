#!/usr/bin/env python3
"""Freeze the LLM provider layer — Gemini (cloud) vs Ollama (local).

    python3 tools/test_llm.py

No network: the Ollama HTTP call and Gemini's model discovery are both mocked, so
this locks the routing (which provider answers), the config plumbing, the Ollama
request shape + JSON parse, and the capability reporting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import llm as LLM, gemini as G  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def read(self):
        return self._b


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<52}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    print("\n  provider selection + config plumbing:")
    check("default is gemini", LLM.provider({}), "gemini")
    check("gemini model_for -> auto", LLM.model_for({}), "auto")
    check("gemini available needs a key", LLM.available({"gemini_key": "K"}), True)
    check("no key -> not available", LLM.available({}), False)

    oll = {"llm": "ollama", "ollama_model": "qwen3:14b"}
    check("llm=ollama selects ollama", LLM.provider(oll), "ollama")
    check("ollama model_for encodes host+name", LLM.model_for(oll),
          "ollama:http://localhost:11434|qwen3:14b")
    check("ollama available needs a model", LLM.available(oll), True)
    check("ollama without a model is unavailable",
          LLM.available({"llm": "ollama"}), False)
    check("is_ollama detects the routed string", LLM.is_ollama(LLM.model_for(oll)), True)
    check("gemini key not required on the ollama path", LLM.key_for(oll), "")

    print("\n  a custom host rides along in the model string:")
    m = LLM.model_for({"llm": "ollama", "ollama_model": "glm-4.7-flash",
                       "ollama_host": "http://10.0.0.5:11434"})
    check("host encoded", m, "ollama:http://10.0.0.5:11434|glm-4.7-flash")

    print("\n  the Ollama call posts a schema and parses the JSON reply:")
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _Resp({"message": {"content": json.dumps({"scenes": [{"scene": 1}]})}})
    LLM.urllib.request.urlopen = fake_urlopen
    schema = {"type": "object", "properties": {"scenes": {"type": "array"}}}
    out = LLM.ollama_complete("ollama:http://localhost:11434|qwen3:14b",
                              "prompt here", schema, system="be terse",
                              temperature=0.2)
    check("hit /api/chat", captured["url"].endswith("/api/chat"))
    check("sent the JSON schema as format", captured["body"]["format"], schema)
    check("carried the model name", captured["body"]["model"], "qwen3:14b")
    check("included the system message",
          captured["body"]["messages"][0], {"role": "system", "content": "be terse"})
    check("parsed the model's JSON", out, {"scenes": [{"scene": 1}]})

    print("\n  gemini.call routes an ollama model to the local backend:")
    G.resolve_model = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("resolve_model must NOT run for a local model"))
    LLM.ollama_complete = lambda model, prompt, schema, system="", temperature=0.4: {"ok": model}
    r = G.call("p", {}, "KEY", "ollama:http://h|m", system="s", temperature=0.3)
    check("routed to Ollama, Gemini path untouched", r, {"ok": "ollama:http://h|m"})

    print("\n  capability reporting:")
    check("gemini without key -> not ok", LLM.capability({})["ok"], False)
    check("gemini with key -> ok", LLM.capability({"gemini_key": "K"})["ok"], True)
    LLM.list_ollama = lambda h: None                       # unreachable
    c = LLM.capability(oll)
    check("ollama unreachable is reported", c["ok"], False)
    check("  and says so", "reachable" in c["reason"] or "serve" in c["reason"], True)
    LLM.list_ollama = lambda h: ["llama3:8b"]              # reachable, wrong model
    check("model-not-pulled is caught", LLM.capability(oll)["ok"], False)
    LLM.list_ollama = lambda h: ["qwen3:14b", "llama3:8b"]
    check("ready when the model is installed", LLM.capability(oll)["ok"], True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
