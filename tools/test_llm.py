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

    print("\n  Grok (OpenAI-compatible) selection + routing:")
    grok = {"llm": "grok", "grok_key": "xai-abc", "grok_model": "grok-4"}
    check("llm=grok selects grok", LLM.provider(grok), "grok")
    check("grok model_for encodes the xAI base url", LLM.model_for(grok),
          "openai:https://api.x.ai/v1|grok-4")
    check("grok key_for returns the grok key", LLM.key_for(grok), "xai-abc")
    check("grok available needs key+model", LLM.available(grok), True)
    check("grok without a key is unavailable",
          LLM.available({"llm": "grok", "grok_model": "grok-4"}), False)
    check("is_openai detects the routed string", LLM.is_openai(LLM.model_for(grok)), True)

    print("\n  the OpenAI-compatible call posts a json_schema and parses the reply:")
    cap = {}

    def fake_oai(req, timeout=0):
        cap["url"] = req.full_url
        cap["auth"] = req.headers.get("Authorization")
        cap["body"] = json.loads(req.data)
        return _Resp({"choices": [{"message": {"content": json.dumps({"tags": ["a"]})}}]})
    LLM.urllib.request.urlopen = fake_oai
    out = LLM.openai_complete("openai:https://api.x.ai/v1|grok-4", "xai-abc",
                              "prompt", {"type": "object"}, system="sys", temperature=0.3)
    check("hit /chat/completions", cap["url"], "https://api.x.ai/v1/chat/completions")
    check("bearer auth set", cap["auth"], "Bearer xai-abc")
    check("used json_schema response_format",
          cap["body"]["response_format"]["type"], "json_schema")
    check("parsed the reply", out, {"tags": ["a"]})

    print("\n  gemini.call routes each foreign model to its backend:")
    G.resolve_model = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("resolve_model must NOT run for a non-Gemini model"))
    LLM.ollama_complete = lambda model, prompt, schema, system="", temperature=0.4: {"ok": "ollama"}
    LLM.openai_complete = lambda model, key, prompt, schema, system="", temperature=0.4: {"ok": "openai", "key": key}
    check("ollama model -> local backend",
          G.call("p", {}, "K", "ollama:http://h|m")["ok"], "ollama")
    r = G.call("p", {}, "xai-abc", "openai:https://api.x.ai/v1|grok-4")
    check("openai model -> openai backend, key threaded", (r["ok"], r["key"]),
          ("openai", "xai-abc"))

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
    check("grok ready with key+model", LLM.capability(grok)["ok"], True)
    check("grok missing key is reported",
          LLM.capability({"llm": "grok", "grok_model": "grok-4"})["ok"], False)
    check("grok capability names the provider",
          LLM.capability(grok)["provider"], "grok")

    print("\n  OpenRouter is a one-row addition on the same backend:")
    orc = {"llm": "openrouter", "openrouter_key": "or-1",
           "openrouter_model": "deepseek/deepseek-r1:free"}
    check("llm=openrouter selects it", LLM.provider(orc), "openrouter")
    check("routes to the openrouter base url", LLM.model_for(orc),
          "openai:https://openrouter.ai/api/v1|deepseek/deepseek-r1:free")
    check("available with key+model", LLM.available(orc), True)
    check("capability ready", LLM.capability(orc)["ok"], True)

    print("\n  Vertex AI (Gemini on Google Cloud) selection + routing:")
    vtx = {"llm": "vertex", "vertex_project": "my-proj",
           "vertex_location": "us-central1", "vertex_model": "gemini-2.5-flash",
           "vertex_service_account": "keys/sa.json"}
    check("llm=vertex selects vertex", LLM.provider(vtx), "vertex")
    check("vertex model_for encodes project|location|model", LLM.model_for(vtx),
          "vertex:my-proj|us-central1|gemini-2.5-flash")
    check("vertex key_for returns the SA path", LLM.key_for(vtx), "keys/sa.json")
    check("vertex available needs project+model", LLM.available(vtx), True)
    check("vertex without a project is unavailable",
          LLM.available({"llm": "vertex", "vertex_model": "gemini-2.5-flash"}), False)
    check("is_vertex detects the routed string", LLM.is_vertex(LLM.model_for(vtx)), True)
    check("_parse_vertex splits the three parts",
          LLM._parse_vertex("vertex:p|europe-west4|gemini-2.5-pro"),
          ("p", "europe-west4", "gemini-2.5-pro"))
    check("_parse_vertex defaults location + model",
          LLM._parse_vertex("vertex:p"), ("p", "us-central1", "gemini-2.5-flash"))

    print("\n  the Vertex call hits the aiplatform endpoint with a Bearer token:")
    LLM._vertex_token = lambda sa: "TOK-123"           # skip real google-auth
    vcap = {}

    def fake_vtx(req, timeout=0):
        vcap["url"] = req.full_url
        vcap["auth"] = req.headers.get("Authorization")
        vcap["body"] = json.loads(req.data)
        return _Resp({"candidates": [{"content": {"parts": [
            {"text": json.dumps({"scenes": [1]})}]}}]})
    LLM.urllib.request.urlopen = fake_vtx
    out = LLM.vertex_complete("vertex:my-proj|us-central1|gemini-2.5-flash",
                              "keys/sa.json", "prompt", {"type": "object"},
                              system="sys", temperature=0.2)
    check("hit the regional generateContent url", vcap["url"],
          "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/"
          "locations/us-central1/publishers/google/models/"
          "gemini-2.5-flash:generateContent")
    check("bearer token set", vcap["auth"], "Bearer TOK-123")
    check("sent responseSchema",
          vcap["body"]["generationConfig"]["responseSchema"], {"type": "object"})
    check("carried the system instruction",
          vcap["body"]["systemInstruction"]["parts"][0]["text"], "sys")
    check("parsed the candidate JSON", out, {"scenes": [1]})
    LLM.vertex_complete("vertex:my-proj|global|gemini-2.5-flash", "", "p", {}, temperature=0.1)
    check("global location drops the region prefix",
          vcap["url"].startswith("https://aiplatform.googleapis.com/v1/projects/"
                                 "my-proj/locations/global/"), True)

    print("\n  vertex capability + gemini.call dispatch:")
    check("capability ok with project+model via ADC",
          LLM.capability({"llm": "vertex", "vertex_project": "p",
                          "vertex_model": "gemini-2.5-flash"})["ok"], True)
    check("capability flags a missing service-account file",
          LLM.capability(vtx)["ok"], False)            # keys/sa.json doesn't exist
    check("capability names the provider",
          LLM.capability({"llm": "vertex", "vertex_project": "p",
                          "vertex_model": "m"})["provider"], "vertex")
    LLM.vertex_complete = lambda model, key, prompt, schema, system="", temperature=0.4: \
        {"ok": "vertex", "key": key}
    rv = G.call("p", {}, "keys/sa.json", "vertex:proj|us-central1|gemini-2.5-flash")
    check("vertex model -> vertex backend, SA path threaded",
          (rv["ok"], rv["key"]), ("vertex", "keys/sa.json"))

    print("\n  vertex_probe reports which models a project can call:")
    LLM._vertex_token = lambda sa: "TOK"               # skip real google-auth

    class _HTTPErr(Exception):
        def __init__(self, code, body):
            self.code = code
            self._b = body.encode()
        def read(self):
            return self._b
    LLM.urllib.error.HTTPError = _HTTPErr

    def probe_urlopen(req, timeout=0):
        if "gemini-2.5-flash" in req.full_url:
            return _Resp({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
        raise _HTTPErr(404, json.dumps({"error": {"message": "model not found in region"}}))
    LLM.urllib.request.urlopen = probe_urlopen
    ok, why = LLM.vertex_probe("proj", "us-central1", "gemini-2.5-flash", "")
    check("an available model probes ok", (ok, why), (True, "ok"))
    ok2, why2 = LLM.vertex_probe("proj", "us-central1", "gemini-3.6-flash", "")
    check("an unavailable model reports the 404 reason",
          ok2 is False and "404" in why2 and "not found" in why2, True)
    check("candidate list is non-empty and quality-ordered",
          LLM.VERTEX_CANDIDATES[0][0].startswith("gemini-3"), True)

    print("\n  name_real_people flips the search rule (biography vs faceless):")
    cap = {}

    def cap_call(prompt, schema, key, model, system="", temperature=0.4):
        cap["p"] = prompt
        return {"scenes": []}
    G.call = cap_call
    G.scenes_for_section("Elon Musk founded SpaceX.", {"name_people": True}, "k", "m")
    check("biography mode names the real person",
          "PUT THEIR NAME" in cap["p"] and "Never name a real living person" not in cap["p"])
    G.scenes_for_section("A rocket launches.", {"name_people": False}, "k", "m")
    check("faceless default never names a living person",
          "Never name a real living person" in cap["p"])

    print("\n  query expansion works on keyless providers (Ollama, Vertex-ADC):")
    # expand_queries used to bail on an empty key, which silently disabled it for
    # local Ollama and Vertex-via-ADC (both authenticate with no key). It must
    # route by the model string, not gate on the key.
    G.call = lambda prompt, schema, key, model, system="", temperature=0.4: \
        {"scenes": [{"scene": 1, "queries": ["dawn over a city", "sunrise skyline"]}]}
    got = G.expand_queries([{"n": 1, "query": "city at dawn", "narration": "morning"}],
                           key="", model="vertex:proj|us-central1|gemini-2.5-flash")
    check("empty key still expands (Vertex ADC / Ollama)", got, {1: ["dawn over a city", "sunrise skyline"]})

    print("\n  the parser tolerates fenced / prefixed JSON (free models):")
    check("plain JSON", LLM._json_loads('{"a": 1}'), {"a": 1})
    check("```json fenced", LLM._json_loads('```json\n{"a": 1}\n```'), {"a": 1})
    check("prose then object",
          LLM._json_loads('Sure! Here you go:\n{"a": [1, 2]}'), {"a": [1, 2]})

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
