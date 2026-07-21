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
from dataclasses import dataclass, field
from pathlib import Path

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

SCENES_SCHEMA = {
    "type": "object",
    "properties": {"scenes": {"type": "array", "items": {"type": "object",
        "properties": {
            "narration": {"type": "string"},
            "media": {"type": "string", "enum": ["IMAGE", "VIDEO"]},
            "query": {"type": "string"},
            "note": {"type": "string"},
            "hero": {"type": "boolean"},
        },
        "required": ["narration", "media", "query", "note", "hero"]}}},
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
HOW TO SPLIT THE SCRIPT INTO SCENES

1. ABSOLUTE RULE — DO NOT CHANGE ONE WORD.
   Concatenating every `narration` value in order must reproduce the supplied text
   EXACTLY: same words, same order, nothing added, nothing dropped, nothing reworded,
   nothing summarised. You are only deciding WHERE TO CUT.

2. Cut on natural breathing beats — where a narrator would pause. Usually a sentence
   end, sometimes a comma before a list or a turn of thought. A scene is typically
   one sentence, or one clause of a long sentence.

3. Aim for 2-8 seconds of speech per scene (roughly 6-25 words). Never let a scene run
   past ~30 words. Very short punch lines ("It was people.") deserve their own scene —
   that is a feature, not a problem.

4. Do NOT try to hit a scene-count target. Let the writing decide.

MEDIA TYPE
- Use IMAGE for about 9 scenes in 10. Images auto-size to the narration, so they are
  cheaper and safer.
- Use VIDEO only where motion genuinely carries meaning: hands doing something, walking,
  water, weather, a machine working, people laughing together. Never for a static concept.

ALT / SEARCH QUERIES — THIS IS WHERE MOST SHEETS FAIL
Stock libraries index literal, photographic descriptions. Write what a camera would see.

  GOOD  `alarm clock glowing in a dark bedroom at night, moody`
  GOOD  `elderly woman teaching piano to a young student, warm room`
  GOOD  `two older friends sitting together on a park bench talking`
  BAD   `the passage of time`               → returns clip-art
  BAD   `many clocks pattern`               → returns pink wallpaper
  BAD   `feeling of loneliness`             → returns nothing usable

Rules for queries:
- 5 to 12 words, English ALWAYS (even when the narration is in another language).
- Name the subject, their approximate age, what they are doing, and the setting.
- Add a lighting or mood word when it matters ("soft lamp light", "morning light").
- No brand names, no text-in-image requests, no "concept of", no abstractions.
- Subjects should mostly be 60-85 and look real, not like stock models.
- Vary the shots. Never more than two similar framings in a row.

NOTE FIELD
Short editor note, or "" if there is nothing to say. Use it to mark:
  "title card", "key beat", "core line", "subscribe beat", "share beat",
  "next-episode tease", "disclaimer", "sign-off", or a recurring character's name.

HERO FLAG — BE STINGY WITH THIS
hero=true means "a human must personally check this picture before it ships". Its
whole value is that it is RARE. Flagging half the scenes makes the flag useless.

Set hero=true ONLY for:
- a recurring named character appearing (so the same face is cast every time)
- an on-screen title card
- the single emotional payoff line the video is built around
- the medical disclaimer, and the final sign-off shot

Set hero=false for everything else, INCLUDING scenes that merely feel important,
introduce a section, cite research, or state a fact. Those are ordinary scenes.

Target: at most 1 scene in 6. In a 12-scene section that means 2 or fewer. If you
have flagged more than that, go back and unflag the weakest ones.
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
    p = f"""Split the following SECTION of a video script into scenes.

VIDEO: {ctx.get('title_en', '')}
VISUAL STYLE: {ctx.get('visual_style', '')}
RECURRING PEOPLE (cast them consistently):
{recurring}
{SPLIT_RULES}{extra}

SECTION TO SPLIT (reproduce every word exactly, in order):
{section}"""
    out = call(p, SCENES_SCHEMA, key, model, system=SYSTEM, temperature=0.25)
    return out.get("scenes", [])


def translate_section(lines: list[str], lang_name: str, ctx: dict, key: str,
                      model: str, feedback: str = "") -> list[str]:
    extra = f"\n\nPREVIOUS ATTEMPT WAS REJECTED: {feedback}\n" if feedback else ""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    p = f"""Translate these {len(lines)} narration lines into {lang_name}.

Rules:
- Return EXACTLY {len(lines)} lines, in the same order, one translation per input line.
- Never merge or split lines. Line 7 in must be line 7 out.
- This is spoken narration for listeners aged 60+. Warm, calm, natural to the ear —
  not literal, not stiff, not machine-sounding. Use the polite/formal register
  ("Sie" in German, "usted" in Spanish) throughout.
- The spine phrase of this video is "{ctx.get('spine_phrase', '')}". Translate it once,
  then reuse that exact wording every time it appears. Same for any running metaphor.
- Keep sentence fragments as fragments — many lines are half a sentence continuing
  from the previous line. Do not "fix" them into complete sentences.
- Numbers, names and medical terms must survive intact.
{extra}
LINES:
{numbered}"""
    out = call(p, TRANSLATE_SCHEMA, key, model, system=SYSTEM, temperature=0.35)
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
