"""Gemini client — turns a finished script into production sheets.

The model NEVER writes markdown. It returns structured JSON against a schema and
Python renders the files (see compose.py). That is the whole reason the output
format cannot drift: the model has no opportunity to get it wrong.

Every section of narration is verified against the source script word by word.
Anything that does not match is retried with the error fed back, then surfaced
to you as a diff rather than silently accepted.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"
API = BASE + "/models/{model}:generateContent"
LIST = BASE + "/models"
DEFAULT_MODEL = "auto"          # resolved against your key at run time
TIMEOUT = 180


class GeminiError(RuntimeError):
    pass


# ------------------------------------------------------- model discovery
# Google retires model names on short notice (gemini-2.5-flash started 404ing on
# 9 July 2026, months before its announced shutdown). Hardcoding a name just moves
# the breakage. So: ask the key what it can actually use, and pick the best fit.

_MODEL_CACHE: dict[str, str] = {}

_SKIP = ("embedding", "aqa", "image", "imagen", "veo", "tts", "audio",
         "vision", "live", "learnlm", "gemma", "robotics", "computer-use")


def list_models(key: str) -> list[dict]:
    out, token = [], ""
    for _ in range(6):
        url = f"{LIST}?key={key}&pageSize=200" + (f"&pageToken={token}" if token else "")
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url), timeout=30) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise GeminiError(
                f"Could not list models (HTTP {e.code}). Check the API key.")
        except Exception as e:
            raise GeminiError(f"Could not reach the Gemini API: {e}")
        out += d.get("models", [])
        token = d.get("nextPageToken", "")
        if not token:
            break
    return [m for m in out
            if "generateContent" in (m.get("supportedGenerationMethods") or [])]


def _score(name: str) -> tuple:
    """Rank candidate models. Flash is the sweet spot here: fast, cheap, and the
    work is structured extraction rather than deep reasoning."""
    n = name.lower()
    ver = 0.0
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
    if m:
        ver = float(m.group(1)) + (float(m.group(2)) / 10 if m.group(2) else 0)
    return (
        1 if ("flash" in n and "lite" not in n) else 0,   # flash preferred
        1 if "lite" not in n else 0,                      # lite is a fallback
        ver,                                              # newer wins
        1 if not re.search(r"preview|exp|-\d{3,}", n) else 0,   # stable wins
        -len(n),                                          # plain names win
    )


def resolve_model(key: str, preferred: str = "") -> str:
    """Return a model name this key can actually call."""
    if preferred and preferred != "auto":
        return preferred
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    names = [m["name"].split("/", 1)[-1] for m in list_models(key)]
    usable = [n for n in names if not any(s in n.lower() for s in _SKIP)]
    if not usable:
        raise GeminiError(
            "This API key has no usable text models. Check it at "
            "https://aistudio.google.com/apikey")
    best = sorted(usable, key=_score, reverse=True)[0]
    _MODEL_CACHE[key] = best
    return best


# --------------------------------------------------------------------- client

def call(prompt: str, schema: dict, key: str, model: str = DEFAULT_MODEL,
         system: str = "", temperature: float = 0.35, retries: int = 3,
         _redirected: bool = False) -> dict:
    # Another provider? The whole target is carried in `model`, so every
    # generator in this file works against a local model (Ollama) or an
    # OpenAI-compatible API (Grok, …) with no change. Lazy import avoids a cycle.
    if str(model).startswith("ollama:"):
        from . import llm
        return llm.ollama_complete(model, prompt, schema, system=system,
                                   temperature=temperature)
    if str(model).startswith("openai:"):
        from . import llm
        return llm.openai_complete(model, key, prompt, schema, system=system,
                                   temperature=temperature)
    if str(model).startswith("vertex:"):
        # Gemini on Vertex AI (Google Cloud). `key` is the path to the
        # service-account JSON (or "" for Application Default Credentials).
        from . import llm
        return llm.vertex_complete(model, key, prompt, schema, system=system,
                                   temperature=temperature)
    model = resolve_model(key, model)
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

    url = API.format(model=model) + f"?key={key}"
    data = json.dumps(body).encode("utf-8")
    last = ""

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            if e.code == 429:
                last = "rate limited"
                time.sleep(12 * attempt)
                continue
            if e.code == 404 and not _redirected:
                # The configured model has been retired. Google does this with
                # little notice, so find a live one and carry on rather than
                # dying halfway through a 115-scene job.
                _MODEL_CACHE.pop(key, None)
                fresh = resolve_model(key, "")
                if fresh != model:
                    return call(prompt, schema, key, fresh, system, temperature,
                                retries, _redirected=True)
                raise GeminiError(
                    f"Model '{model}' is unavailable and no replacement was found.\n"
                    f"Run: python3 make_video.py models\n{detail}")
            if e.code in (400, 403):
                raise GeminiError(
                    f"Gemini rejected the request ({e.code}). Usually a bad or "
                    f"missing API key.\n{detail}")
            last = f"HTTP {e.code}: {detail}"
            time.sleep(3 * attempt)
            continue
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(3 * attempt)
            continue

        cands = payload.get("candidates") or []
        if not cands:
            fb = payload.get("promptFeedback", {})
            raise GeminiError(f"Gemini returned nothing. {fb}")
        cand = cands[0]
        if cand.get("finishReason") == "MAX_TOKENS":
            raise GeminiError(
                "Gemini hit its output limit on one section. The script section "
                "is too long — reduce SECTION_WORDS in lib/gemini.py.")
        parts = cand.get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last = f"invalid JSON: {e}"
            time.sleep(2)

    raise GeminiError(f"Gemini failed after {retries} attempts — {last}")


# -------------------------------------------------------- publish metadata

_META_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "description", "tags"],
}

_META_SYSTEM = """\
You write YouTube metadata for a faceless narration channel. The audience is
general and skews older, so everything is clear and inviting — never clickbait,
never ALL CAPS, never emoji spam. It should read like a thoughtful person wrote
it about THIS specific video, not a template. Write everything in the language
you are told to use, and nothing else.

Return JSON with three fields: title, description, tags.

TITLE
- One line, honest and compelling, about 60 characters or fewer.
- No emoji, no ALL CAPS, no clickbait ("You won't believe…"), no year unless the
  script is genuinely about a specific year.

DESCRIPTION
- Open with a 1-2 sentence hook, drawn from what the video actually says, that
  makes someone want to watch.
- Then a short paragraph (2-4 sentences) on what the video covers — mention the
  real things it discusses, in plain, warm language.
- If, and only if, the subject calls for it, add ONE sentence of appropriate
  framing on its own line:
    * health, medicine, symptoms, treatment, nutrition, fitness, the body ->
      say plainly it is for general information and is not a substitute for
      professional or medical advice.
    * history, science, nature, space, "how/why" explainers, education ->
      say it is an informative, educational overview.
    * money, finance, law -> say it is general information, not professional advice.
  Skip this note entirely for light, purely entertaining, or everyday topics.
- Keep it human. No hashtag walls, no keyword stuffing, no fake urgency, no
  invented facts or links.

TAGS
- 10 to 18 short, lowercase phrases a real viewer might actually search, all
  relevant to the true content. No leading '#', no duplicates.
"""


def generate_metadata(narration: str, lang_name: str, topics: list[str],
                      title_hint: str, key: str, model: str = DEFAULT_MODEL) -> dict:
    """Write a YouTube title, description and tags for a finished video.

    Works from the narration in the target language, so the wording matches what
    the video actually says. `topics` is a hint (canonical subjects such as
    'medicine' or 'history') the model uses to decide whether a disclaimer or an
    'informative overview' note belongs — it still reads the narration itself.
    """
    subjects = ", ".join(topics) if topics else "general interest"
    prompt = (
        f"Language: {lang_name}\n"
        f"Subjects detected in this video: {subjects}\n"
        f"Project reference (not necessarily a good title): {title_hint}\n\n"
        f"This is the full spoken narration of the video:\n"
        f'"""\n{normalise(narration)[:6000]}\n"""\n\n'
        f"Write the title, description and tags in {lang_name}."
    )
    data = call(prompt, _META_SCHEMA, key, model, system=_META_SYSTEM,
                temperature=0.6)
    # Normalise the shape so callers never have to guard it.
    return {
        "title": (data.get("title") or "").strip(),
        "description": (data.get("description") or "").strip(),
        "tags": [t.strip().lstrip("#").strip()
                 for t in (data.get("tags") or []) if t and t.strip()],
    }


# --------------------------------------------------- visual query expansion

_EXPAND_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["scene", "queries"],
            },
        }
    },
    "required": ["scenes"],
}

_EXPAND_SYSTEM = (
    "You turn a line of narration into concrete, literal image-search phrases "
    "that a free stock library or photo archive would actually match. For each "
    "scene give THREE alternative phrases — different literal framings of the "
    "same idea, as a real photograph or piece of footage. Rules: 2-5 words each; "
    "only concrete things a camera can see; NO metaphors, NO abstract concepts, "
    "NO charts/diagrams/text unless the line is literally about one; prefer "
    "ordinary, photographable subjects. "
    "Example — line 'the real thief of your sleep is almost never the alarm' -> "
    "['person lying awake in bed', 'dark bedroom at night', 'glowing alarm clock 3am']."
)


def expand_queries(scenes: list[dict], key: str,
                   model: str = DEFAULT_MODEL) -> dict[int, list[str]]:
    """Concrete alternative image queries for each scene, batched.

    `scenes` is [{"n", "query", "narration"}, ...]. Returns {n: [phrase, ...]}.
    Best effort: any failure returns {} and the caller keeps the original
    queries. Chunked so one bad scene can't cost the whole video its expansions.

    Does NOT gate on `key`: local Ollama and Vertex-via-ADC both authenticate
    with an EMPTY key, and whether an LLM is configured was already decided by
    the caller (LLM.available). `call()` routes to the right backend by `model`.
    """
    if not scenes:
        return {}
    out: dict[int, list[str]] = {}
    CH = 40
    for i in range(0, len(scenes), CH):
        chunk = scenes[i:i + CH]
        lines = "\n".join(
            f"{s['n']}. line: {normalise(s.get('narration', ''))[:160]}"
            f"   (current search: {s.get('query', '')})"
            for s in chunk)
        prompt = ("Give alternative image-search phrases for each numbered "
                  "scene below.\n\n" + lines)
        try:
            data = call(prompt, _EXPAND_SCHEMA, key, model,
                        system=_EXPAND_SYSTEM, temperature=0.7)
        except Exception:
            continue
        for item in data.get("scenes", []):
            try:
                n = int(item["scene"])
            except (KeyError, TypeError, ValueError):
                continue
            qs = [q.strip() for q in (item.get("queries") or []) if q and q.strip()]
            if qs:
                out[n] = qs[:4]
    return out


_VIDEO_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "prompt": {"type": "string"},
                },
                "required": ["scene", "prompt"],
            },
        }
    },
    "required": ["scenes"],
}

_VIDEO_SYSTEM = (
    "You write prompts for an AI VIDEO model (Google Veo) that turns one line of "
    "narration into a single ~8 second cinematic clip. For each scene, write ONE "
    "vivid prompt of 30-55 words as a single flowing description a camera could "
    "actually film. Weave in, naturally: "
    "(1) SUBJECT — the concrete thing on screen with a couple of visual details; "
    "(2) ACTION/MOTION — one clear, CONTINUOUS motion that MATCHES the meaning of "
    "the narration line (what the subject is doing, or how the scene moves: "
    "drifting clouds, blowing spindrift, a slow reveal). This is the most "
    "important part — the motion must fit the line, never a random or theatrical "
    "gesture; (3) CAMERA — ONE deliberate move (slow dolly in, aerial push, "
    "tracking shot, gentle handheld); (4) LOOK — natural lighting and a grounded, "
    "realistic documentary style. "
    "Rules: photoreal documentary footage only; ONE continuous shot, no cuts; NO "
    "on-screen text, captions, subtitles, watermarks, logos or split screens; "
    "people's actions must be plausible and safe; present tense; describe only "
    "what is visible. Do not name real, identifiable people. "
    "Example — line 'K2 is one of the hardest and most dangerous mountains to "
    "climb', subject 'mountaineers climbing icy steep slope k2' -> 'A lone "
    "mountaineer in a red down suit kicks crampons into a near-vertical wall of "
    "blue glacial ice, ice axes biting as spindrift streams past in the wind; the "
    "camera tracks slowly upward alongside them, harsh high-altitude sun, thin "
    "cold air, realistic expedition documentary footage.'"
)


_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "splits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "narration": {"type": "string"},
                                "media": {"type": "string", "enum": ["IMAGE", "VIDEO"]},
                                "query": {"type": "string"},
                                "topic": {"type": "string"},
                                "fallback_query": {"type": "string"},
                            },
                            "required": ["narration", "query"],
                        },
                    },
                },
                "required": ["id", "parts"],
            },
        }
    },
    "required": ["splits"],
}

_SPLIT_SYSTEM = (
    "Each scene below bundles more than one thing a camera would show into a "
    "single shot. Split EACH into 2 or more smaller scenes, one per distinct "
    "visual beat. "
    "HARD RULE: the parts' narration, concatenated in order, must reproduce the "
    "input narration EXACTLY — same words, same order, verbatim. You ONLY choose "
    "the split points; never add, drop, translate, reword or re-punctuate a word. "
    "For each part give ONE concrete, literal search query (2-6 words, only things "
    "a camera can see) that is clearly DIFFERENT from the other parts' queries — "
    "the whole point is a different picture per beat. Set media to IMAGE or VIDEO "
    "and topic to the best fit. "
    "Example — narration 'in Walnüssen, Leinsamen und bestimmten Fischsorten' -> "
    "part 1 'in Walnüssen,' query 'bowl of shelled walnuts'; part 2 'Leinsamen' "
    "query 'bowl of brown flaxseeds'; part 3 'und bestimmten Fischsorten' query "
    "'fresh raw salmon fillet'."
)


def split_coarse_scenes(items: list[dict], key: str,
                        model: str = DEFAULT_MODEL) -> dict[int, list[dict]]:
    """Split scenes that bundle several pictures into per-beat sub-scenes.

    `items` is [{"id","narration","query","media"}, ...]. Returns
    {id: [part, ...]} where each part is a scene dict with its own query. Best
    effort — a scene the model can't split (or returns <2 parts for) is simply
    absent from the result, and the caller keeps it whole. Batched so one dense
    scene never costs the whole run."""
    if not items:
        return {}
    out: dict[int, list[dict]] = {}
    CH = 12
    for i in range(0, len(items), CH):
        chunk = items[i:i + CH]
        lines = "\n".join(
            f'id {it["id"]} [{(it.get("media") or "IMAGE")}]: '
            f'"{normalise(it.get("narration", ""))}"'
            f'   (current query: {it.get("query", "")})'
            for it in chunk)
        prompt = ("Split each scene below into tighter visual beats.\n\n" + lines)
        try:
            data = call(prompt, _SPLIT_SCHEMA, key, model,
                        system=_SPLIT_SYSTEM, temperature=0.4)
        except Exception:
            continue
        for item in data.get("splits", []):
            try:
                sid = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            parts = []
            for p in item.get("parts", []):
                narr = (p.get("narration") or "").strip()
                if not narr:
                    continue
                parts.append({
                    "narration": narr,
                    "media": (p.get("media") or "IMAGE").upper(),
                    "query": (p.get("query") or "").strip(),
                    "topic": (p.get("topic") or "").strip().lower(),
                    "fallback_query": (p.get("fallback_query") or "").strip(),
                })
            if len(parts) >= 2:
                out[sid] = parts
    return out


def video_prompts(scenes: list[dict], key: str,
                  model: str = DEFAULT_MODEL) -> dict[int, str]:
    """One cinematic Veo prompt per scene, batched. `scenes` is
    [{"n","query","narration"}, ...]; returns {n: prompt}. Best effort — any
    failure returns {} and the caller falls back to the plain prompt."""
    if not scenes:
        return {}
    out: dict[int, str] = {}
    CH = 20
    for i in range(0, len(scenes), CH):
        chunk = scenes[i:i + CH]
        lines = "\n".join(
            f"{s['n']}. line: {normalise(s.get('narration', ''))[:200]}"
            f"   (subject: {s.get('query', '')})"
            for s in chunk)
        prompt = ("Write one cinematic video prompt for each numbered scene "
                  "below.\n\n" + lines)
        try:
            data = call(prompt, _VIDEO_SCHEMA, key, model,
                        system=_VIDEO_SYSTEM, temperature=0.6)
        except Exception:
            continue
        for item in data.get("scenes", []):
            try:
                n = int(item["scene"])
            except (KeyError, TypeError, ValueError):
                continue
            pr = (item.get("prompt") or "").strip()
            if pr:
                out[n] = pr
    return out


# ─────────────────────────────────── AI still-image prompts (quality + consistency)

_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "prompt": {"type": "string"},
                },
                "required": ["scene", "prompt"],
            },
        }
    },
    "required": ["scenes"],
}

_BIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        }
    },
    "required": ["subjects"],
}

_BIBLE_SYSTEM = (
    "You are preparing a set of PHOTOREALISTIC images for one short video. Read ALL "
    "the narration lines, then list the people, objects and places that RECUR across "
    "several scenes and must look the SAME every time (e.g. a specific older woman, "
    "the narrator's phone, a kitchen, a delivery van). For each, give ONE short, "
    "concrete visual description to reuse verbatim — fixed details like age, build, "
    "hair, clothing colour, material, model, colour, era. Skip anything that appears "
    "only once. Keep each description under 20 words. Never name or depict a real, "
    "identifiable person — describe a generic stand-in instead.")

_IMAGE_SYSTEM = (
    "You write prompts for an AI IMAGE model that makes ONE photorealistic still per "
    "scene for a short video. For each numbered scene write ONE vivid prompt of "
    "25-45 words describing a single real photograph a camera could take. Weave in "
    "naturally: the concrete SUBJECT with a couple of specific details; the SETTING; "
    "the COMPOSITION and shot size (close-up, wide); the LIGHTING; and the MOOD. "
    "CONSISTENCY: a SHARED STYLE and a CONSISTENCY BIBLE may be given below — apply "
    "the shared style (palette, lighting, mood) to EVERY prompt, and whenever a "
    "scene involves something in the bible, describe it with the bible's exact "
    "wording so it looks identical across scenes. "
    "REALISM: photoreal documentary photography, natural light, realistic materials "
    "and skin with natural imperfections — avoid a glossy, over-smoothed 'AI' look. "
    "NO on-screen text, captions, watermarks, logos or collages; one clear subject; "
    "present tense; describe only what is visible. Never depict a real, identifiable "
    "person — use a generic stand-in.")


def _image_bible(scenes: list[dict], key: str, model: str) -> str:
    """A compact recurring-subjects 'bible' the per-scene prompts reuse, so a
    person/object/place is described the same way in every scene. '' on failure."""
    lines = "\n".join(
        normalise(s.get("narration", "") or s.get("query", "") or "")[:200]
        for s in scenes)
    if not lines.strip():
        return ""
    try:
        data = call("All narration lines for one video:\n\n" + lines,
                    _BIBLE_SCHEMA, key, model, system=_BIBLE_SYSTEM, temperature=0.2)
    except Exception:
        return ""
    out = []
    for it in (data.get("subjects") or []):
        nm = (it.get("name") or "").strip()
        de = (it.get("description") or "").strip()
        if nm and de:
            out.append(f"- {nm}: {de}")
    return "\n".join(out[:24])


def image_prompts(scenes: list[dict], key: str, model: str = DEFAULT_MODEL,
                  style: str = "") -> dict[int, str]:
    """One detailed photoreal image prompt per scene, batched, sharing the project
    `style` and a recurring-subject bible so the whole set stays consistent.

    `scenes` is [{"n","query","narration"}, ...]; returns {n: prompt}. Best effort
    — any failure returns {} and the caller falls back to the plain prompt."""
    if not scenes:
        return {}
    bible = _image_bible(scenes, key, model)
    ctx = ""
    if (style or "").strip():
        ctx += f"\n\nSHARED STYLE (apply to every image): {style.strip()}"
    if bible:
        ctx += f"\n\nCONSISTENCY BIBLE (reuse these exact descriptions):\n{bible}"
    out: dict[int, str] = {}
    CH = 18
    for i in range(0, len(scenes), CH):
        chunk = scenes[i:i + CH]
        lines = "\n".join(
            f"{s['n']}. line: {normalise(s.get('narration', ''))[:200]}"
            f"   (subject: {s.get('query', '')})"
            for s in chunk)
        prompt = ("Write one photorealistic image prompt for each numbered scene "
                  "below." + ctx + "\n\nScenes:\n" + lines)
        try:
            data = call(prompt, _IMAGE_SCHEMA, key, model,
                        system=_IMAGE_SYSTEM, temperature=0.5)
        except Exception:
            continue
        for item in data.get("scenes", []):
            try:
                n = int(item["scene"])
            except (KeyError, TypeError, ValueError):
                continue
            pr = (item.get("prompt") or "").strip()
            if pr:
                out[n] = pr
    return out


# ------------------------------------------------------------- text handling

SMART = {"’": "'", "‘": "'", "“": '"', "”": '"',
         "–": "-", "—": "-", "…": "...", " ": " "}


def normalise(s: str) -> str:
    for a, b in SMART.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalise(s).lower())


def split_sections(script: str, target: int = 700) -> list[str]:
    """Group paragraphs into sections of roughly `target` words.

    Sections are what get sent to Gemini one at a time. Paragraph boundaries are
    never broken, so a narration beat is never split across two requests.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", script) if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > target:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        out.append("\n\n".join(cur))
    return out


def diff_words(expected: str, got: str, context: int = 6) -> str:
    """A short, readable word-level diff for the first divergence."""
    import difflib
    a, b = words(expected), words(got)
    sm = difflib.SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        lo = max(0, i1 - context)
        before = " ".join(a[lo:i1])
        exp = " ".join(a[i1:i2]) or "(nothing)"
        act = " ".join(b[j1:j2]) or "(missing)"
        after = " ".join(a[i2:i2 + context])
        return (f"  ...{before}  [{exp}]  {after}...\n"
                f"  ...{before}  [{act}]  {after}...   ← Gemini")
    return "  (no word differences — only punctuation or spacing)"


def snap_to_script(section: str, scenes: list[dict]) -> tuple[list[dict], int] | None:
    """Force the scenes' narration to reproduce `section` WORD-FOR-WORD while
    keeping the model's scene boundaries.

    When it splits a script, the model occasionally mis-transcribes or drops a
    word (e.g. writes 'erfrishingen' for 'erfrischenden'). Rather than let a wrong
    word get spoken, we redistribute the AUTHOR'S exact text — with the author's
    own punctuation — across the same cut points the model chose. Returns
    (new_scenes, words_changed), or None if an exact match can't be guaranteed
    (the caller then keeps the model's version and warns).
    """
    import difflib
    S = section.split()                        # author tokens, punctuation kept
    per = [((s.get("narration", "") or "").split()) for s in scenes]
    J = [w for toks in per for w in toks]       # model tokens, flattened
    if not S or not J:
        return None
    scene_at = []                               # scene index for each J position
    for k, toks in enumerate(per):
        scene_at.extend([k] * len(toks))

    def norm(w):                                # compare on words, not punctuation
        return re.sub(r"[^a-z0-9']+", "", w.lower())

    sm = difflib.SequenceMatcher(None, [norm(w) for w in S], [norm(w) for w in J])
    owner: list = [None] * len(S)               # which scene each author token joins
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for t in range(i2 - i1):
                owner[i1 + t] = scene_at[j1 + t]
        elif tag in ("replace", "delete"):
            sc = scene_at[min(j1, len(J) - 1)]  # give these author words a home
            for i in range(i1, i2):
                owner[i] = sc
            changed += (i2 - i1)
        # 'insert' = words the model invented with no source → dropped
    if any(o is None for o in owner):
        return None
    new_narr: list = [[] for _ in scenes]
    for i, sc in enumerate(owner):
        new_narr[sc].append(S[i])
    out = []
    for k, s in enumerate(scenes):
        text = " ".join(new_narr[k]).strip()
        if not text:                            # a scene emptied out → unsafe, bail
            return None
        s2 = dict(s)
        s2["narration"] = text
        out.append(s2)
    if words(" ".join(x["narration"] for x in out)) != words(section):
        return None                             # never return an imperfect match
    return out, changed


# ------------------------------------------------------------------- schemas

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title_en": {"type": "string"},
        "acts": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "summary": {"type": "string"}},
            "required": ["name", "summary"]}},
        "recurring": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "look": {"type": "string"}},
            "required": ["name", "look"]}},
        "spine_phrase": {"type": "string"},
        "visual_style": {"type": "string"},
        "music_prompt": {"type": "string"},
        "thumbnail_prompt": {"type": "string"},
        "thumbnail_line1": {"type": "string"},
        "thumbnail_line2": {"type": "string"},
    },
    "required": ["title_en", "acts", "recurring", "spine_phrase", "visual_style",
                 "music_prompt", "thumbnail_prompt", "thumbnail_line1",
                 "thumbnail_line2"],
}

# The canonical topics live in lib/sources (single source of truth). The model
# buckets every scene into one of them, which is how routing scales to ANY
# subject without a fixed word list — it understands "beekeeping" or "Byzantine
# iconography" even though those exact words are in no vocabulary. The enum holds
# ONLY real topics: Gemini rejects an empty string inside an enum ("enum[0]:
# cannot be empty"), and with 17 broad topics there is always a closest one, so
# the model always picks a bucket rather than leaving it blank.
from . import sources as _sources                                  # noqa: E402
CANON_TOPICS = list(_sources.CANON_TOPICS)

SCENES_SCHEMA = {
    "type": "object",
    "properties": {"scenes": {"type": "array", "items": {"type": "object",
        "properties": {
            "narration": {"type": "string"},
            "media": {"type": "string", "enum": ["IMAGE", "VIDEO"]},
            "query": {"type": "string"},
            "domain": {"type": "string"},
            "topic": {"type": "string", "enum": CANON_TOPICS},
            "fallback_query": {"type": "string"},
            "safety_query": {"type": "string"},
            "note": {"type": "string"},
            "hero": {"type": "boolean"},
            "exact": {"type": "boolean"},
        },
        "required": ["narration", "media", "domain", "topic", "query",
                     "fallback_query", "safety_query", "note", "hero", "exact"]}}},
    "required": ["scenes"],
}

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["lines"],
}

YOUTUBE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "alt_titles": {"type": "array", "items": {"type": "string"}},
        "hook": {"type": "string"},
        "chapters": {"type": "array", "items": {"type": "object", "properties": {
            "scene": {"type": "integer"}, "label": {"type": "string"}},
            "required": ["scene", "label"]}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "thumbnail_line1": {"type": "string"},
        "thumbnail_line2": {"type": "string"},
    },
    "required": ["title", "alt_titles", "hook", "chapters", "tags",
                 "thumbnail_line1", "thumbnail_line2"],
}


# ------------------------------------------------------------------- prompts

SYSTEM = """You are a senior video producer for a faceless YouTube channel aimed at \
viewers aged 60 and over. The channel's voice is warm, calm and documentary — \
reassuring rather than alarming, respectful rather than patronising. You prepare \
production sheets that a small team executes literally, so precision matters more \
than flair. You never invent, rewrite, summarise or improve the writer's script."""

SPLIT_RULES = """
HOW TO SPLIT A SCRIPT INTO SCENES

════════════════════════════════════════════════════════════════════════
1. THE ONE RULE THAT OVERRIDES EVERYTHING: DO NOT CHANGE A WORD.
════════════════════════════════════════════════════════════════════════
Concatenating every `narration` value in order must reproduce the supplied text
EXACTLY: same words, same order, nothing added, dropped, reworded or summarised.
You are only deciding WHERE TO CUT. This is checked mechanically and a mismatch
is rejected.

════════════════════════════════════════════════════════════════════════
2. WHERE TO CUT:  ONE SCENE = ONE VISUAL, AND THE SCRIPT DECIDES HOW MANY
════════════════════════════════════════════════════════════════════════
NOT one noun = one scene. But ALSO NOT one sentence = one scene. Let the words
decide: cut EVERY time the picture would have to change, as many times as the
sentence demands. There is no fixed cadence and no target count — a sentence
that names three things is three scenes; a sentence that shows one thing is one.

A scene ends when the picture would have to change. Cut when any of these
changes:

  SUBJECT    the thing on screen becomes a different thing
             "ocean -> submarine -> deep-sea creatures" = 3 scenes

  ACTION     the same subject starts doing something meaningfully different
             "rocket launches -> reaches orbit -> deploys satellite" = 3 scenes

  LOCATION   where we are changes
             "Earth -> Moon -> Mars" = 3 scenes

  TIME       for history and biography, when the era moves
             "childhood -> adulthood -> discovery -> legacy" = 4 scenes

  CONCEPT    for explanation, when the argument moves to its next step
             "problem -> cause -> process -> result" = usually 3-4 scenes

If none of those change, DO NOT CUT — even at a full stop.

────────────────────────────────────────────────────────────────────────
ONE SENTENCE IS USUALLY SEVERAL SCENES — THIS IS THE #1 MISTAKE
────────────────────────────────────────────────────────────────────────
A single sentence that NAMES several things, or MOVES through several
subjects, places or eras, is several scenes — ONE PER VISUAL — even though it
is a single sentence with no full stop inside it. Leaving a compound sentence
on one static picture is the most common and most visible failure. Split it.

  "a journey full of great changes, astonishing discoveries and decisive
   turning points"
     -> 3 scenes:  something transforming / a moment of discovery /
                   a pivotal turning point

  "from the first civilisations in Egypt and Mesopotamia to mighty empires"
     -> 3 scenes:  Egyptian pyramids / Mesopotamian ziggurat / an empire's army

  "wars, inventions and scientific breakthroughs changed daily life"
     -> 3 scenes:  a battle / an inventor's workshop / a laboratory

Every item in a list of DIFFERENT things a camera would frame separately is
its own scene. Commas and the words "and / und / y / e / et / or / oder"
joining different things are your signal to cut.

────────────────────────────────────────────────────────────────────────
BUT DO NOT SPLIT WHAT SHARES ONE PICTURE
────────────────────────────────────────────────────────────────────────
A run of similar nouns a single camera catches in one frame stays ONE scene:

  "apples, bananas and oranges"  ->  one bowl of mixed fruit = 1 scene

Two clauses describing the SAME picture from different angles are ONE scene.
Never cut into single words or fragments that cannot be filmed on their own.

────────────────────────────────────────────────────────────────────────
NO STILL SITS TOO LONG
────────────────────────────────────────────────────────────────────────
Each scene gets ONE photo or ONE clip for its whole duration. A still that
would hold longer than about six seconds (~15 words) is dead air:

  - If it holds more than one visual -> SPLIT it (almost always the case).
  - If it is genuinely ONE thing but long -> use media "VIDEO"; motion holds
    attention where a static photo dies.
  - A very short scene is fine when it lands as a beat ("It was people.").

Most scenes land in the 4-14 word range. That is an OUTCOME of cutting on the
visuals the script names — never pad, never force a count, let the script lead.

════════════════════════════════════════════════════════════════════════
3. READ THE SCRIPT AND DECIDE WHAT KIND OF FILM THIS IS
════════════════════════════════════════════════════════════════════════
Before writing any query, work out the register from the writing itself. Do not
assume — a space documentary and a health video need completely different
pictures. Common registers:

  PEOPLE-LED     health, ageing, relationships, personal habit, advice
                 -> real people of the relevant age, doing ordinary things,
                    domestic and natural settings
  SUBJECT-LED    space, ocean, geology, weather, wildlife
                 -> the phenomenon itself; no people unless the script has them
  HISTORICAL     history, biography, archaeology
                 -> period-appropriate places, objects, artefacts, landscapes,
                    reenactment; be careful with named real individuals
  TECHNICAL      engineering, computing, industry, medicine-as-science
                 -> equipment, facilities, processes, close detail work
  ABSTRACT       learning, memory, emotion, economics, time
                 -> the hardest. Anchor to something filmable: a person doing
                    the thing, or a concrete metaphor. Never film the noun.

Infer the audience age and setting from the writing too. If the script speaks to
older readers about their own bodies, show people of that age. If it explains
tectonic plates, show no people at all.

════════════════════════════════════════════════════════════════════════
3b. TAG EACH SCENE WITH ITS SUBJECT
════════════════════════════════════════════════════════════════════════
`domain` decides WHICH LIBRARY is searched, so it must describe what the
PICTURE shows — not what the script is about overall.

Write one or two plain words. There is NO fixed list: use whatever actually
names the subject — astronomy, farming, surgery, shipping, mythology,
metalwork, insects, banking, monsoon, pottery. The pipeline maps your words
onto the libraries it has, and an unfamiliar word is handled gracefully, so
be accurate rather than trying to guess an approved category.

The one distinction that genuinely matters is MODERN versus HISTORICAL,
because it decides between a stock library and a museum archive:

  modern life        a present-day person, home, workplace, hospital, street
                     -> say `people`, `daily life`, `modern medicine`, `office`
  the past           artefacts, ruins, period scenes, old instruments
                     -> say `history`, `ancient rome`, `victorian`, `archaeology`

A researcher at a bench today is modern. A Victorian microscope is historical.
The same video will often contain both, and they must be tagged differently.

4. THE QUERY LADDER — THREE SEARCHES PER SCENE
════════════════════════════════════════════════════════════════════════
These queries are sent to free stock libraries (Pexels, Pixabay). They index
LITERAL, PHOTOGRAPHIC descriptions of what is visible. They do not understand
ideas, metaphors or feelings.

Every scene needs three, in decreasing specificity. The pipeline stays on
`query` and only steps down when it finds nothing usable, so all three must
show the SAME subject — a fallback that wanders off-topic just puts the wrong
picture on screen. Keep the subject fixed; relax only how hard it is to find:

  query           the shot you actually want. Specific, filmable.
  fallback_query  the SAME subject and action, with ONE hard-to-find detail
                  relaxed — a specific place, time of day or adjective dropped.
                  It must still read as the same scene, not a broader topic.
                  "a black hole" -> keep the black hole; don't become "space".
  safety_query    the plainest shot of that same subject that free stock is
                  certain to have. Still the same subject — just the common,
                  unmissable version of it. This one must never come back empty.

Worked examples:

  Narration: "supermassive black holes quietly consume matter"
    query           swirling accretion disk around a black hole in deep space
    fallback_query  glowing accretion disk around a black hole
    safety_query    bright ring of light around a dark centre in space

  Narration: "your muscles recover and your body releases important hormones"
    query           person sleeping deeply in a dark bedroom, calm breathing
    fallback_query  adult asleep in bed at night, soft light
    safety_query    person asleep in bed

  Narration: "engineers built aqueducts that carried fresh water"
    query           roman stone aqueduct arches across a dry landscape
    fallback_query  tall roman aqueduct arches in sunlight
    safety_query    old roman stone arches

  Narration: "repeated practice strengthens those pathways"
    query           person practising a musical instrument alone, concentrating
    fallback_query  person playing a musical instrument indoors
    safety_query    close-up of hands playing an instrument

RULES FOR ALL THREE QUERIES
  - ENGLISH ALWAYS, even when the narration is German or Spanish. Stock
    libraries index in English.
  - 4 to 12 words. Longer returns nothing.
  - Name what a CAMERA WOULD SEE: subject, what it is doing, where, and the
    light if it matters.
  - No abstractions, no "concept of", no feelings, no metaphors, no brand
    names, no requests for text or logos in the image.
  - (Naming real people is governed by the PEOPLE rule injected below.)
  - safety_query must be something free stock certainly has. When in doubt make
    it a plain landscape, texture, sky, room or hands.

  BAD   `the passage of time`          -> clip-art junk
  BAD   `feeling of loneliness`        -> nothing usable
  BAD   `neurons forming connections`  -> stylised nonsense
  GOOD  `elderly hands holding a warm mug by a window`
  GOOD  `waves breaking slowly on a dark rocky shore at dusk`

════════════════════════════════════════════════════════════════════════
5. DO NOT REPEAT YOURSELF
════════════════════════════════════════════════════════════════════════
Repeated footage is the clearest sign of a cheap video. Across the whole script:
  - No two scenes may share a `query`.
  - Avoid near-duplicates. Vary the subject, framing or setting, not just an
    adjective. "man walking on beach" and "person walking on beach" count as
    the same query.
  - Never more than two similar framings in a row. Alternate wide and close.

════════════════════════════════════════════════════════════════════════
6. MEDIA TYPE
════════════════════════════════════════════════════════════════════════
  IMAGE  the default, roughly 8 scenes in 10. Photos auto-size to the narration.
  VIDEO  only where motion carries the meaning, or where a long scene would
         otherwise sit still: water, weather, fire, machinery, crowds, hands
         working, walking, flying, launching, flowing.
         Never for a static concept or a portrait.

════════════════════════════════════════════════════════════════════════
7. NOTE FIELD
════════════════════════════════════════════════════════════════════════
A short editor note, or "" when there is nothing to say. Use it for:
  "title card", "key beat", "core line", "subscribe beat", "share beat",
  "next-episode tease", "disclaimer", "sign-off", or a recurring character name.

════════════════════════════════════════════════════════════════════════
8. HERO FLAG — BE STINGY
════════════════════════════════════════════════════════════════════════
hero=true means "a person must check this picture before it ships". Its whole
value is rarity. Flagging half the scenes makes it meaningless.

Set hero=true ONLY for:
  - a recurring named character appearing (so the same face is cast each time)
  - an on-screen title card
  - the single emotional payoff the video is built around
  - a medical or legal disclaimer, and the final sign-off shot

Set hero=false for everything else, INCLUDING scenes that merely feel
important, open a section, cite research or state a fact.

Target at most 1 scene in 6. If you have flagged more, unflag the weakest.

════════════════════════════════════════════════════════════════════════
9. EXACT FLAG — WHEN A GENERIC PHOTO WOULD BE WRONG
════════════════════════════════════════════════════════════════════════
exact=true means "a stock/search photo would NOT reliably show the RIGHT thing,
so this scene should be AI-GENERATED instead of searched." Judge it purely from
the narration + your query: would a keyword search return the SPECIFIC subject
named, or just a plausible look-alike?

Set exact=true when the shot must be one SPECIFIC, NAMEABLE, DESCRIBABLE thing a
generic search can't be trusted to get, for example (these are ONLY examples —
decide from THIS script):
  - a specific species / variety / object named in the line (a particular fruit,
    plant, animal, mineral, dish, tool, gadget) that ordinary stock would fake
    with a generic look-alike
  - a labelled diagram, chart, cutaway, map or how-it-works illustration
  - a specific anatomical part or medical/scientific depiction
  - a specific invention, artifact, structure or product described in detail
  - any "here is exactly X" moment where the WRONG-but-similar photo would mislead

Set exact=false for ordinary b-roll where a good representative photo is fine
(a person smiling, a city street, hands cooking, nature, generic mood shots) —
and for scenes about a REAL NAMED PERSON (those are found from photo archives,
not generated). Do not over-flag: only the shots that genuinely need the exact
subject. exact is independent of hero.
"""


def plan(script: str, key: str, model: str = DEFAULT_MODEL) -> dict:
    p = f"""Read this complete video script and produce a production plan for it.

The channel: faceless YouTube, audience 60+, warm calm documentary tone.

Give me:
- title_en: the working title, taken from the script's own subject
- acts: 3 to 5 narrative acts with a one-line summary each
- recurring: every named or implied recurring person in the script (e.g. a story
  character who reappears), with a short casting description so the same person can
  be cast in every shot. Empty array if there are none.
- spine_phrase: the single phrase the whole video turns on — the one that repeats and
  must be worded identically every time it appears
- visual_style: two or three sentences describing the palette and casting
- music_prompt: a background-music generation prompt. Calm, unobtrusive, no vocals,
  no drums, matching the emotional temperature of THIS script.
- thumbnail_prompt: a photorealistic AI image prompt, 16:9, leaving the LEFT third
  empty for text
- thumbnail_line1: 2-4 words, the biggest line
- thumbnail_line2: 3-6 words, the smaller line underneath

SCRIPT:
{script}"""
    return call(p, PLAN_SCHEMA, key, model, system=SYSTEM, temperature=0.5)


def scenes_for_section(section: str, ctx: dict, key: str, model: str,
                       feedback: str = "") -> list[dict]:
    extra = f"\n\nPREVIOUS ATTEMPT WAS REJECTED:\n{feedback}\nFix it exactly.\n" \
        if feedback else ""
    recurring = "\n".join(f"- {r['name']}: {r['look']}"
                          for r in ctx.get("recurring", [])) or "- (none)"
    topics = ", ".join(CANON_TOPICS)
    # Whether the search query may NAME a real person. Off (default) is the
    # faceless mode — a scene about a person becomes a generic filmable stand-in.
    # On is biography mode — a scene about a named person searches for THAT
    # person, so the picture is actually them (from photo archives).
    if ctx.get("name_people"):
        people_rule = """
NAMED PEOPLE — THIS IS A BIOGRAPHY OF A REAL PERSON
The viewer wants to SEE the real person — but NOT the same headshot every scene.
The fastest way to make a biography look cheap is to put the same portrait on
ten lines in a row. Sort each scene into ONE of two kinds and treat it that way:

  PERSON beat — the line is about the person themselves: who they are, what they
  did, a moment in their life. PUT THEIR NAME in the query, with a DIFFERENT
  context every time so a different photo is pulled. Vary the setting, era or
  action, and NEVER repeat a query across scenes:
    "Elon Musk speaking on stage"      "Elon Musk interview"
    "Elon Musk at Tesla factory"       "Elon Musk press conference"
  Keep it SHORT and findable — the full name plus one or two plain words. The
  photo archives match short phrases; a long staged description returns nothing.

  THING beat — the line is about a company, product, idea, industry or place, not
  the person (e.g. "the world's first trillionaire", "artificial intelligence",
  "digital payments", "electric vehicles", "aerospace"). Use plain topical b-roll
  with NO person in it:
    "stock market trading floor"       "server racks in a data center"
    "contactless card payment"         "electric car on a highway"
  Do NOT put the person in a THING beat — that repetition is exactly what makes a
  biography look like one face on a loop.

BALANCE: roughly alternate the two so the video breathes — a person shot, then a
thing, then the person again. Do not make every scene a portrait; only the beats
truly ABOUT the person should show their face.

FALLBACKS follow the beat:
  - a PERSON beat falls back to a simpler, DIFFERENT person query (last resort:
    the bare name, e.g. "Elon Musk"). Never reuse the same person query — and
    never "<name> portrait" — on more than one scene.
  - a THING beat falls back to a plainer version of the SAME b-roll, never a
    person.

If you are unsure which kind a beat is, PUT THEIR NAME only when the line names
them or their direct action; otherwise use the thing b-roll. Use the person's
full, correct name every time you name them. No two scenes may share a query."""
    else:
        people_rule = """
NAMED PEOPLE
Never name a real living person in a query; use a generic, filmable description
instead. For historical figures prefer the era, place or object over the face."""
    p = f"""Split the following SECTION of a video script into scenes.

VIDEO: {ctx.get('title_en', '')}
VISUAL STYLE: {ctx.get('visual_style', '')}
RECURRING PEOPLE (cast them consistently):
{recurring}
{SPLIT_RULES}{people_rule}
{extra}

CANONICAL TOPIC (`topic` field)
Set `topic` for each scene to the ONE canonical topic that best fits what its
PICTURE shows, chosen from EXACTLY this list:
  {topics}
Judge it by the IMAGE, not the overall video: a modern hospital scene is
`medicine`; a Victorian surgical kit is `history`; a rocket launch is `space`.

Use `people` ONLY when a PERSON is the real subject of the shot — a portrait, a
face, a close or medium shot where the person is what the picture is ABOUT. When
people are merely present in a place, landscape or activity, tag the PLACE or
ACTIVITY, not `people`:
  farmers in a wide field by a river  -> nature    (the field/river is the shot)
  a busy market street                -> culture   (the street is the shot)
  a surgeon mid-operation             -> medicine  (the operation is the shot)
  workers on a factory floor          -> tech      (the factory is the shot)
  an older couple close at home       -> people    (the couple IS the shot)
This matters: `people` routes to libraries of individual people and, in
biography mode, is treated as the video's NAMED subject — so a wide landscape
tagged `people` gets handled as if it were one specific person, which it is not.

Any subject on Earth fits one of these, so ALWAYS pick the closest single one —
never leave it blank. `domain` stays your free-text description; `topic` is the
bucket.

SECTION TO SPLIT (reproduce every word exactly, in order):
{section}"""
    out = call(p, SCENES_SCHEMA, key, model, system=SYSTEM, temperature=0.25)
    return out.get("scenes", [])


def segment_script(en_lines: list[str], script: str, lang_name: str, key: str,
                   model: str, feedback: str = "") -> list[str]:
    """Cut a pasted script into scene-sized narration aligned to the main script.

    The visuals are shared across languages, so line i in every language must
    describe the same moment as English line i. This does NOT translate: the
    caller has pasted this language's own words, and the only job here is to
    decide where each scene's words end. Concatenating the parts must reproduce
    the pasted script exactly — the caller verifies that word for word and
    retries with feedback if it drifts.
    """
    extra = f"\n\nPREVIOUS ATTEMPT WAS REJECTED: {feedback}\n" if feedback else ""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(en_lines))
    p = f"""A video is already cut into {len(en_lines)} scenes, shown below by their
English narration. You are given the SAME video's script in {lang_name}. Split
the {lang_name} script into EXACTLY {len(en_lines)} parts, so part i is the
narration for scene i and covers the same moment as English scene i.

HARD RULES:
- Return EXACTLY {len(en_lines)} parts, in order.
- DO NOT translate, rewrite, correct, reorder, add or drop a single word. Use
  the pasted {lang_name} text verbatim. You are ONLY choosing where each scene's
  words end.
- Every word of the pasted script must appear once, in order. Concatenating the
  parts back together must reproduce the pasted script exactly.
- A scene may end mid-sentence — English scenes do. Keep fragments as fragments.
- Match the English scene boundaries as closely as the {lang_name} wording allows,
  so each scene's words fit the picture chosen for it.
{extra}
ENGLISH SCENES (for alignment only — never output these):
{numbered}

{lang_name} SCRIPT TO SPLIT:
{script}"""
    out = call(p, TRANSLATE_SCHEMA, key, model, system=SYSTEM, temperature=0.2)
    return out.get("lines", [])


def youtube_package(narration: list[str], lang_name: str, ctx: dict, key: str,
                    model: str) -> dict:
    joined = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(narration))
    p = f"""Write the YouTube package for this video, in {lang_name}.

VIDEO: {ctx.get('title_en', '')}
SPINE: {ctx.get('spine_phrase', '')}

Give me:
- title: under 70 characters, curiosity without clickbait, suited to a 60+ audience
- alt_titles: three A/B alternatives
- hook: the opening 2-3 sentences of the description — what the video answers and why
  it is more hopeful/useful than the viewer expects
- chapters: 10 to 14 chapter markers. For each, the SCENE NUMBER it begins at (from the
  numbered narration below) and a short label. First chapter must be scene 1.
- tags: exactly 20 search tags, lowercase, no hashes
- thumbnail_line1 / thumbnail_line2: thumbnail text in {lang_name}

Do NOT write the disclaimer — that is added automatically.

NARRATION:
{joined}"""
    return call(p, YOUTUBE_SCHEMA, key, model, system=SYSTEM, temperature=0.55)
