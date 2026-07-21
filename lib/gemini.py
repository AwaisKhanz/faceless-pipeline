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
            "fallback_query": {"type": "string"},
            "safety_query": {"type": "string"},
            "note": {"type": "string"},
            "hero": {"type": "boolean"},
        },
        "required": ["narration", "media", "query", "fallback_query",
                     "safety_query", "note", "hero"]}}},
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
2. WHERE TO CUT:  ONE SCENE = ONE CLEAR VISUAL IDEA
════════════════════════════════════════════════════════════════════════
NOT one sentence = one scene. NOT one noun = one scene.

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
DO NOT OVER-SPLIT: group things that share one picture
────────────────────────────────────────────────────────────────────────
A list of similar nouns is ONE visual idea, or at most a few:

  "Apples, bananas, oranges, strawberries, blueberries, grapes, mangoes
   and watermelons..."

  WRONG  8 scenes, one per fruit
  RIGHT  2-3 scenes, grouped into shots a camera could actually take
         (a bowl of mixed fruit / berries close up / melons being cut)

Two sentences describing the same picture from different angles are ONE scene.

────────────────────────────────────────────────────────────────────────
DO NOT UNDER-SPLIT: a long passage holding one still image is dead air
────────────────────────────────────────────────────────────────────────
Each scene gets ONE photo or ONE clip for its whole duration.

  - Over ~28 words containing a second visual idea -> you MUST split it.
  - Genuinely one idea but long -> prefer media "VIDEO". Motion holds
    attention where a static photo dies.
  - Under ~5 words is fine when it lands as a beat ("It was people.").

Typical result: 6-25 words per scene. That is an OUTCOME of cutting on visual
ideas, not a target to hit. Never pad or force a scene count.

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
4. THE QUERY LADDER — THREE SEARCHES PER SCENE
════════════════════════════════════════════════════════════════════════
These queries are sent to free stock libraries (Pexels, Pixabay). They index
LITERAL, PHOTOGRAPHIC descriptions of what is visible. They do not understand
ideas, metaphors or feelings.

Every scene needs three, in decreasing specificity. The pipeline tries `query`
first and walks down until something returns:

  query           the shot you actually want. Specific, filmable.
  fallback_query  a looser version of the same idea. Drop the rarest element.
  safety_query    plain, common footage that still fits the topic and will
                  ALWAYS return results. This one must never come back empty.

Worked examples:

  Narration: "supermassive black holes quietly consume matter"
    query           swirling accretion disk around a black hole in deep space
    fallback_query  spiral galaxy slowly rotating against black starfield
    safety_query    stars and nebula in deep space

  Narration: "your muscles recover and your body releases important hormones"
    query           person sleeping deeply in a dark bedroom, calm breathing
    fallback_query  adult asleep in bed at night, soft light
    safety_query    empty bed in a quiet dark bedroom

  Narration: "engineers built aqueducts that carried fresh water"
    query           roman stone aqueduct arches across a dry landscape
    fallback_query  ancient stone arched bridge in sunlight
    safety_query    old stone ruins in the countryside

  Narration: "repeated practice strengthens those pathways"
    query           person practising a musical instrument alone, concentrating
    fallback_query  hands repeating a careful task at a desk
    safety_query    student studying at a table

RULES FOR ALL THREE QUERIES
  - ENGLISH ALWAYS, even when the narration is German or Spanish. Stock
    libraries index in English.
  - 4 to 12 words. Longer returns nothing.
  - Name what a CAMERA WOULD SEE: subject, what it is doing, where, and the
    light if it matters.
  - No abstractions, no "concept of", no feelings, no metaphors, no brand
    names, no requests for text or logos in the image.
  - Never name a real living person. For historical figures prefer the era,
    place or object over the face.
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
