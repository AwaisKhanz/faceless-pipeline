"""Script → production sheets.

Gemini returns structured data; every character of the file format is written
here, by Python. That is why the output cannot drift out of format.

Each section of narration is checked against the source script before anything is
written. Mismatches are retried with the error fed back, and only surfaced to you
if they survive that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import gemini as G

ROOT = Path(__file__).resolve().parent.parent
SECTION_WORDS = 700

LANG_NAMES = {"de": "German", "es": "Spanish", "fr": "French",
              "it": "Italian", "pt": "Portuguese"}


@dataclass
class Scene:
    n: int
    narration: str
    media: str
    query: str
    note: str = ""
    hero: bool = False
    # The routing signal. `generate()` builds Scenes with these two, and
    # render_main_script writes them into the sheet as `Domain:` and `Fallbacks:`
    # lines. They were added to the constructor and the renderer but not here,
    # so every sheet build crashed with "unexpected keyword argument 'domain'".
    # Defaulted, so an old caller that omits them still works.
    domain: str = ""
    fallbacks: list[str] = field(default_factory=list)
    # Canonical topic (one of sources.CANON_TOPICS), the model's bucket for this
    # scene. Routing prefers it over word-matching, which is what lets any subject
    # reach the right archive. "" for old sheets → routing falls back to words.
    topic: str = ""


@dataclass
class Result:
    files: dict[str, str] = field(default_factory=dict)   # filename -> content
    warnings: list[str] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    plan: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers

def esc(s: str) -> str:
    """Narration is wrapped in double quotes on one line — keep it single-line
    and free of the curly quotes that make later diffing painful."""
    return G.normalise(s).replace("\n", " ")


# ---------------------------------------------------------------- rendering

def render_main_script(plan: dict, scenes: list[Scene], pid: str, lang: str = "en") -> str:
    """The main script: only what the pipeline reads plus a little context.

    Deliberately lean. Everything here is either parsed by lib/sheet.py (the
    scene blocks) or a short human note; the old settings tables, act summaries,
    music/thumbnail prompts, checklists and count tables were decoration nothing
    read, so they're gone. One block per scene, in order.
    """
    L: list[str] = []
    add = L.append

    # The main script carries the structure language's own narration. Record
    # which language that is so the rest of the app never has to assume English —
    # a project can start in German or Spanish and still read correctly.
    add(f"<!-- main-lang: {lang} -->")
    add(f"# {plan.get('title_en', pid)}")
    add(f"_{len(scenes)} scenes · language: {lang} · "
        f"generated {datetime.now():%Y-%m-%d}_")
    add("")
    add("Each block is one scene = one narration line + one picture or clip. "
        "**ALT / search** is what gets searched for the visual; **Fallbacks** are "
        "tried only if it finds nothing. Edit any line, then re-source or "
        "re-render from the studio.")
    if plan.get("visual_style"):
        add("")
        add(f"**Visual style:** {plan['visual_style']}")
    add("")
    add("---")
    add("")

    for s in scenes:
        flag = f" ⚑ {s.note}" if getattr(s, "note", "") else ""
        add(f"**S{s.n} ⬜** · {s.media}{flag}")
        add(f"- Narration: \"{esc(s.narration)}\"")
        add(f"- ALT / search: `{s.query.strip().strip('`')}`")
        if getattr(s, "domain", ""):
            add(f"- Domain: {s.domain}")
        if getattr(s, "topic", ""):
            add(f"- Topic: {s.topic}")
        fb = [q.strip().strip('`') for q in getattr(s, "fallbacks", []) if q.strip()]
        if fb:
            # Its own line so sheets from before the ladder existed stay valid —
            # the parser treats Fallbacks as optional.
            add("- Fallbacks: " + " · ".join(f"`{q}`" for q in fb))
        add("")

    return "\n".join(L) + "\n"


def render_narration(scenes: list[Scene], lang: str, lines: list[str],
                       pid: str) -> str:
    """A per-language narration file: the main script's scenes, this language's words.

    Lean, like the main script. Only the `EN:` / `<LANG>:` lines are read (by
    lib/sheet.parse_narration); the reference line is labelled EN whatever the
    structure language is, because the parser keys on it. Titles/tags are written
    on demand (the Publish button), not baked in here."""
    code = lang.upper()
    name = LANG_NAMES.get(lang, lang.upper())
    L: list[str] = []
    add = L.append
    add(f"# {name} narration — {pid}")
    add(f"_{len(scenes)} scenes · same scene numbers as the main script, so the "
        f"pictures are shared_")
    add("")
    add("---")
    add("")
    for s, tr in zip(scenes, lines):
        add(f"**S{s.n}** · EN: \"{esc(s.narration)}\"")
        add(f"{code}: \"{esc(tr)}\"")
        add("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------- orchestration

def _flag_repeats(scenes, res) -> None:
    """Warn when two scenes would search for the same thing.

    The prompt forbids this, but a model will still do it occasionally, and
    repeated footage is the most visible sign of an automated video. Warning
    beats silently shipping it; the reviewer sees it on the approval sheet.

    Comparison ignores word order and a handful of filler words, so
    "man walking on the beach" and "a man walking on beach" are caught as the
    duplicates they are.
    """
    FILLER = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "to"}

    def shape(q: str) -> frozenset:
        return frozenset(w for w in re.findall(r"[a-z]+", q.lower())
                         if w not in FILLER)

    seen: dict[frozenset, int] = {}
    for s in scenes:
        key = shape(s.query)
        if not key:
            continue
        if key in seen:
            res.warnings.append(
                f"S{s.n}: searches for the same thing as S{seen[key]} "
                f"({s.query!r}) — swap one in the review step.")
        else:
            seen[key] = s.n


STRUCT_ORDER = ["en", "de", "es"]     # who defines the scenes when several arrive

# Conjunctions that, like commas, usually join DIFFERENT things a camera would
# frame separately. Multilingual because any language can be the structure
# language. Only a heuristic hint — never a hard rule.
_LIST_CONJ = {"and", "und", "y", "e", "et", "ed", "or", "oder", "sowie"}


def _under_split(narration: str) -> bool:
    """A rough, language-agnostic guess that a scene bundles several visuals.

    Not authoritative: it only decides whether to ASK the model to look again.
    A scene is flagged when it is very long, or long-ish AND clearly a list
    (commas and/or joining conjunctions). A short scene, or a long one that is
    plainly a single picture, is left alone — the model still makes the call.
    """
    words = narration.split()
    n = len(words)
    if n >= 20:
        return True
    breaks = narration.count(",") + narration.count(";")
    breaks += sum(1 for w in words if w.strip('.,;:!?"\'’”').lower() in _LIST_CONJ)
    return n >= 11 and breaks >= 2


def _finer_split_feedback(coarse: list[dict]) -> str:
    """Targeted retry note naming the scenes that still hold several pictures."""
    lines = "\n".join(f'  - "{s.get("narration", "")}"' for s in coarse[:6])
    return ("Some scenes still hold more than one picture. Split EACH of these "
            "on every change of visual — one scene per distinct thing named, "
            "keeping every word verbatim and in order:\n" + lines)


def split_into_scenes(script: str, plan: dict, key: str, model: str,
                      res: Result, tick, on_warn) -> list[Scene]:
    """Split ONE script into verified scenes. This is where the visuals come
    from, so it runs on the structure language's script only."""
    sections = G.split_sections(script, SECTION_WORDS)
    all_scenes: list[dict] = []
    for i, sec in enumerate(sections, start=1):
        tick(f"splitting section {i} of {len(sections)}")
        got, feedback = None, ""
        for attempt in range(1, 4):
            got = G.scenes_for_section(sec, plan, key, model, feedback)
            joined = " ".join(s.get("narration", "") for s in got)

            # Hard gate: the narration must reproduce the section word for word.
            if G.words(joined) != G.words(sec):
                feedback = ("Your narration did not reproduce the section exactly.\n"
                            + G.diff_words(sec, joined))
                if attempt == 3:
                    res.warnings.append(
                        f"Section {i}: narration still differs from the script after "
                        f"3 attempts.\n{G.diff_words(sec, joined)}")
                    on_warn(f"Section {i} did not match the script — see warnings")
                continue

            # Soft gate: push back on scenes that bundle several visuals into one
            # long shot, so a compound sentence becomes a beat per picture. The
            # model decides the real cuts; this only asks it to look again, and
            # never blocks a section that is genuinely word-accurate.
            coarse = [s for s in got if _under_split(s.get("narration", ""))]
            if coarse and attempt < 3:
                feedback = _finer_split_feedback(coarse)
                continue
            if coarse:
                res.warnings.append(
                    f"Section {i}: {len(coarse)} scene(s) may still hold more than "
                    f"one picture — check them in review and split if needed.")
            break
        all_scenes.extend(got or [])

    scenes = [Scene(n=i, narration=s.get("narration", ""),
                    media=(s.get("media") or "IMAGE").upper(),
                    query=(s.get("query") or "").strip(),
                    domain=(s.get("domain") or "").strip().lower(),
                    topic=(s.get("topic") or "").strip().lower(),
                    fallbacks=[q for q in ((s.get("fallback_query") or "").strip(),
                                           (s.get("safety_query") or "").strip())
                               if q],
                    note=s.get("note", "") or "",
                    hero=bool(s.get("hero")))
              for i, s in enumerate(all_scenes, start=1)]

    for s in scenes:
        if s.media not in ("IMAGE", "VIDEO"):
            s.media = "IMAGE"
        if not s.query.strip():
            if s.fallbacks:
                s.query, s.fallbacks = s.fallbacks[0], s.fallbacks[1:]
                res.warnings.append(f"S{s.n}: no primary query — used the fallback.")
            else:
                s.query = "calm natural scene, soft daylight"
                res.warnings.append(
                    f"S{s.n}: no search query at all — placeholder used, fix by hand.")

    _flag_repeats(scenes, res)
    return scenes


def segment_language(scenes: list[Scene], script: str, lang: str, key: str,
                     model: str, res: Result, on_warn) -> list[str]:
    """Cut a pasted script onto the shared scenes, verified word for word.

    Not a translation — the words are the user's own. Concatenating the parts
    must reproduce the pasted script; on failure it retries with feedback, and
    only pads (keeping scene numbering aligned) as a last resort.
    """
    name = LANG_NAMES.get(lang, lang.upper())
    en_lines = [s.narration for s in scenes]
    fb, seg = "", []
    for attempt in range(1, 4):
        seg = G.segment_script(en_lines, script, name, key, model, fb)
        same_words = G.words(" ".join(seg)) == G.words(script)
        if len(seg) == len(en_lines) and same_words:
            return seg
        if len(seg) != len(en_lines):
            fb = (f"You returned {len(seg)} parts but exactly {len(en_lines)} "
                  f"are required, one per scene.")
        else:
            fb = ("Concatenating the parts did not reproduce the pasted script "
                  "exactly. Do not translate or change any word.\n"
                  + G.diff_words(script, " ".join(seg)))
        if attempt == 3:
            res.warnings.append(
                f"{name}: could not split the script cleanly onto "
                f"{len(en_lines)} scenes — padded to keep numbering aligned. "
                f"Check the narration sheet.")
            on_warn(f"{name} split imperfectly — review its narration sheet")
    return (seg + [""] * len(en_lines))[:len(en_lines)]


def _language_sheet(scenes: list[Scene], pid: str, lang: str,
                    lines: list[str]) -> tuple[str, str]:
    """The narration file for one language. No YouTube call — titles/tags are
    generated on demand from the Publish button, not baked into the sheet."""
    name = LANG_NAMES.get(lang, lang.upper())
    return (f"{pid}_{name.upper()}_narration.md",
            render_narration(scenes, lang, lines, pid))


def generate(scripts: dict[str, str], pid: str, key: str,
             model: str = G.DEFAULT_MODEL, on_progress=lambda *_: None,
             on_warn=lambda *_: None) -> Result:
    """Per-language scripts in → main script + narration files out. No translation.

    `scripts` maps language code -> that language's pasted script. The first
    present language (English by preference) is the STRUCTURE language: its
    script is split into scenes, which define the shared visuals. Every other
    language's pasted script is segmented onto those same scenes, so all
    languages share one set of pictures and one scene numbering.
    """
    res = Result()
    present = ([l for l in STRUCT_ORDER if scripts.get(l, "").strip()]
               + [l for l in scripts if l not in STRUCT_ORDER and scripts.get(l, "").strip()])
    if not present:
        raise ValueError("No script was given for any language.")
    struct, others = present[0], present[1:]
    struct_script = scripts[struct].strip()
    struct_name = LANG_NAMES.get(struct, struct.upper())

    n_sections = len(G.split_sections(struct_script, SECTION_WORDS))
    total = 1 + n_sections + 1 + len(others) + 1
    step = [0]

    def tick(msg: str) -> None:
        step[0] += 1
        on_progress(step[0], total, msg)

    tick("reading the script")
    plan = G.plan(struct_script, key, model)
    res.plan = plan

    scenes = split_into_scenes(struct_script, plan, key, model, res, tick, on_warn)
    res.scenes = scenes

    # The structure language's narration IS the main script — no separate sheet
    # for it. Titles/descriptions/tags are written on demand (the Publish button),
    # so generating them here would be a wasted LLM call.
    tick(f"writing the {struct_name} main script")
    res.files[f"{pid}_main_script.md"] = render_main_script(plan, scenes, pid, struct)

    # Every other language: segment its pasted script onto the shared scenes.
    for lang in others:
        name = LANG_NAMES.get(lang, lang.upper())
        tick(f"splitting the {name} script onto {len(scenes)} scenes")
        lines = segment_language(scenes, scripts[lang].strip(), lang, key, model,
                                 res, on_warn)
        fn, c = _language_sheet(scenes, pid, lang, lines)
        res.files[fn] = c

    return res


def add_language(sheet: Path, lang: str, script: str, key: str,
                 model: str = G.DEFAULT_MODEL, on_progress=lambda *_: None,
                 on_warn=lambda *_: None) -> Result:
    """Compose ONE more language onto an existing project's shared scenes.

    Reads the finished main script (its scenes and visuals are fixed), segments
    the pasted script onto them, and returns just that language's narration sheet.
    Nothing about the main script or the other languages is touched.
    """
    import lib.sheet as sheetlib
    res = Result()
    pid = sheet.stem.replace("_main_script", "")
    scenes = [Scene(n=s.n, narration=s.narration, media=s.media, query=s.query,
                    domain=getattr(s, "domain", ""), hero=getattr(s, "hero", False))
              for s in sheetlib.parse_main_script(sheet)]
    res.scenes = scenes
    plan = {"title_en": pid, "spine_phrase": ""}

    name = LANG_NAMES.get(lang, lang.upper())
    on_progress(1, 3, f"reading {pid}")
    on_progress(2, 3, f"splitting the {name} script onto {len(scenes)} scenes")
    lines = segment_language(scenes, script.strip(), lang, key, model, res, on_warn)
    on_progress(3, 3, f"writing the {name} narration")
    fn, c = _language_sheet(scenes, pid, lang, lines)
    res.files[fn] = c
    return res


def write_files(res: Result, sheets_dir: Path, overwrite: bool = False) -> list[str]:
    sheets_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in res.files.items():
        p = sheets_dir / name
        if p.exists() and not overwrite:
            p = sheets_dir / f"{p.stem}__new{p.suffix}"
        p.write_text(content, encoding="utf-8")
        written.append(p.name)
    return written
