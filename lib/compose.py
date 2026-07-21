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
WORDS_PER_SEC = 2.5      # narration pace at the pipeline's default rate
GAP = 0.4                # silence between lines, matches TAIL - DISSOLVE

LANG_NAMES = {"de": "German", "es": "Spanish", "fr": "French",
              "it": "Italian", "pt": "Portuguese"}

DISCLAIMER = {
    "en": """⚠️ IMPORTANT — PLEASE READ
This video is for EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY. It is not medical advice, diagnosis, or treatment, and it is not a substitute for the guidance of a qualified healthcare professional. I am not a doctor. Everything here is a plain-language summary of published research, and it describes what tends to be found across large groups of people — not what is true for you personally. If you have any concern about your health, please speak with your own doctor, who knows your history. Never disregard professional medical advice or delay seeking it because of something you watched here.""",
    "de": """⚠️ WICHTIG — BITTE LESEN
Dieses Video dient AUSSCHLIESSLICH BILDUNGS- UND INFORMATIONSZWECKEN. Es ist keine medizinische Beratung, Diagnose oder Behandlung und ersetzt nicht den Rat einer qualifizierten medizinischen Fachperson. Ich bin kein Arzt. Alles hier ist eine allgemein verständliche Zusammenfassung veröffentlichter Forschung und beschreibt, was sich über große Gruppen von Menschen hinweg zeigt — nicht, was für Sie persönlich zutrifft. Wenn Sie sich um Ihre Gesundheit sorgen, sprechen Sie bitte mit Ihrem eigenen Arzt, der Ihre Vorgeschichte kennt. Ignorieren Sie ärztlichen Rat niemals und verzögern Sie ihn nicht wegen etwas, das Sie hier gesehen haben.""",
    "es": """⚠️ IMPORTANTE — POR FAVOR, LEA
Este vídeo tiene FINES EXCLUSIVAMENTE EDUCATIVOS E INFORMATIVOS. No es asesoramiento médico, ni diagnóstico, ni tratamiento, y no sustituye la orientación de un profesional sanitario cualificado. No soy médico. Todo lo que aquí se cuenta es un resumen en lenguaje sencillo de investigaciones publicadas, y describe lo que se observa en grandes grupos de personas, no lo que es cierto para usted en particular. Si le preocupa su salud, hable con su propio médico, que conoce su historial. Nunca ignore el consejo médico profesional ni retrase buscarlo por algo que haya visto aquí.""",
}


@dataclass
class Scene:
    n: int
    narration: str
    media: str
    query: str
    note: str = ""
    hero: bool = False
    # The routing signal. `generate()` builds Scenes with these two, and
    # render_master writes them into the sheet as `Domain:` and `Fallbacks:`
    # lines. They were added to the constructor and the renderer but not here,
    # so every sheet build crashed with "unexpected keyword argument 'domain'".
    # Defaulted, so an old caller that omits them still works.
    domain: str = ""
    fallbacks: list[str] = field(default_factory=list)


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


def ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def estimate_starts(scenes: list[Scene]) -> list[float]:
    t, out = 0.0, []
    for s in scenes:
        out.append(t)
        t += len(s.narration.split()) / WORDS_PER_SEC + GAP
    return out


def act_ranges(scenes: list[Scene], acts: list[dict]) -> list[tuple[dict, int, int]]:
    """Spread the acts evenly over the scenes. The model's act boundaries are a
    narrative judgement; the numbers have to be exact, so they are computed."""
    if not acts:
        return [({"name": "MAIN", "summary": ""}, 1, len(scenes))]
    n, k = len(scenes), len(acts)
    out, start = [], 1
    for i, a in enumerate(acts):
        end = n if i == k - 1 else round(n * (i + 1) / k)
        end = max(end, start)
        out.append((a, start, end))
        start = end + 1
    return out


# ---------------------------------------------------------------- rendering

def render_master(plan: dict, scenes: list[Scene], pid: str) -> str:
    vids = [s.n for s in scenes if s.media == "VIDEO"]
    heroes = sum(1 for s in scenes if s.hero)
    ranges = act_ranges(scenes, plan.get("acts", []))
    est = estimate_starts(scenes)
    total = est[-1] + len(scenes[-1].narration.split()) / WORDS_PER_SEC + 1.0

    L: list[str] = []
    add = L.append

    add(f"# 🎬 MASTER PRODUCTION SHEET — {pid}")
    add(f"## \"{plan.get('title_en', pid)}\"")
    add("")
    add(f"> **{len(scenes)} SCENES.** One scene = one narration line = one media "
        f"asset. Generated from the script on {datetime.now():%Y-%m-%d}.")
    add("")
    add("---")
    add("")
    add("## 0. PROJECT SETTINGS")
    add("")
    add("| Setting | Value |")
    add("|---|---|")
    add("| **Narrator voice** | Set per language in the studio |")
    add("| **Aspect** | Landscape 16:9 |")
    add(f"| **Total scenes** | **{len(scenes)}** |")
    add(f"| **Estimated runtime** | ~{int(total // 60)} min {int(total % 60)} s |")
    add("| **Audience** | 60+, warm reassuring documentary |")
    add("| **Captions** | Black-box white-text, burned in |")
    add("| **Music** | Prompt below, ~20% under narration |")
    add("| **Transitions** | Dissolve 0.6s |")
    add("")
    add("### ⭐ THE RULE")
    add("**One scene = one narration line = one media asset.**")
    add("")
    add("### Visual style")
    add(plan.get("visual_style", ""))
    if plan.get("recurring"):
        add("")
        add("**Recurring people — cast the same face every time:**")
        for r in plan["recurring"]:
            add(f"- **{r['name']}** — {r['look']}")
    if plan.get("spine_phrase"):
        add("")
        add(f"### ⚑ Spine phrase")
        add(f"**\"{plan['spine_phrase']}\"** — this is what the video turns on. "
            f"Keep it word-identical everywhere it appears, in every language.")
    add("")
    add("**Status:** `⬜ TODO` · `✅ DONE`")
    add("")

    for act, a, b in ranges:
        add("---")
        add("")
        add(f"# {act['name'].upper()} (S{a}–S{b})")
        if act.get("summary"):
            add("")
            add(f"_{act['summary']}_")
        add("")
        for s in scenes[a - 1:b]:
            flag = f" ⚑ {s.note}" if s.note else ""
            add(f"**S{s.n} ⬜** · {s.media}{flag}")
            add(f"- Narration: \"{esc(s.narration)}\"")
            add(f"- ALT / search: `{s.query.strip().strip('`')}`")
            fb = [q.strip().strip('`') for q in getattr(s, "fallbacks", []) if q.strip()]
            if getattr(s, "domain", ""):
                add(f"- Domain: {s.domain}")
            if fb:
                # Written on its own line so sheets from before the ladder
                # existed stay valid — the parser treats it as optional.
                add("- Fallbacks: " + " · ".join(f"`{q}`" for q in fb))
            add("")

    add("---")
    add("")
    add("# FINAL PASSES")
    add("")
    add("- [ ] **Visuals** — review every pick in the studio")
    add("- [ ] **Narration** — generate, spot-check the hero scenes")
    add("- [ ] **Render** — captions on, music under")
    add("- [ ] **Thumbnail** — prompt below")
    add("- [ ] **Upload** — description and tags from the language files")
    add("")
    add("---")
    add("")
    add("# 🎵 BACKGROUND MUSIC PROMPT")
    add(f"> \"{plan.get('music_prompt', '')}\"")
    add("")
    add("**Settings:** starts at **0:00** · spans the whole video · volume **~20%**")
    add("")
    add("---")
    add("")
    add("# 🖼️ THUMBNAIL PROMPT")
    add("")
    add("**AI image-gen prompt (background):**")
    add(f"> \"{plan.get('thumbnail_prompt', '')}\"")
    add("")
    add("**Text overlay (2 lines):**")
    add(f"- Line 1 (biggest, white/warm yellow): **\"{plan.get('thumbnail_line1', '')}\"**")
    add(f"- Line 2 (smaller, accent red): **\"{plan.get('thumbnail_line2', '')}\"**")
    add("")
    add("---")
    add("")
    add("### Scene count")
    add("| Act | Scenes | Count |")
    add("|---|---|---|")
    for act, a, b in ranges:
        add(f"| {act['name']} | S{a}–S{b} | {b - a + 1} |")
    add(f"| **Total** | | **{len(scenes)}** |")
    add("")
    add(f"**Media split:** {len(scenes) - len(vids)} IMAGE · {len(vids)} VIDEO"
        + (f" — {', '.join('S' + str(v) for v in vids)}" if vids else ""))
    add(f"**Hero scenes:** {heroes}")
    add("")
    return "\n".join(L) + "\n"


def render_translation(plan: dict, scenes: list[Scene], lang: str,
                       lines: list[str], yt: dict | None, pid: str) -> str:
    name = LANG_NAMES.get(lang, lang.upper())
    code = lang.upper()
    L: list[str] = []
    add = L.append

    add(f"# 🇩🇪 {name.upper()} NARRATION — {pid}" if lang == "de"
        else f"# {name.upper()} NARRATION — {pid}")
    add(f"## \"{plan.get('title_en', pid)}\"")
    if yt:
        add(f"### {name} title: \"{yt.get('title', '')}\"")
    add("")
    add(f"> Matches the {len(scenes)}-scene master sheet exactly. Scene numbers are "
        f"identical, so the pipeline reuses the same visuals.")
    add("")
    add("---")
    add("")
    for s, tr in zip(scenes, lines):
        add(f"**S{s.n}** · EN: \"{esc(s.narration)}\"")
        add(f"{code}: \"{esc(tr)}\"")
        add("")

    if yt:
        add("---")
        add("")
        add(f"## 📺 YOUTUBE PACKAGE ({name.upper()})")
        add("")
        add("### Title")
        add(f"> **{yt.get('title', '')}**")
        add("")
        if yt.get("alt_titles"):
            add("**A/B alternates:**")
            for t in yt["alt_titles"]:
                add(f"- {t}")
            add("")
        add("### Description")
        add("")
        add("```")
        add(yt.get("hook", ""))
        add("")
        starts = estimate_starts(scenes)
        for ch in yt.get("chapters", []):
            i = max(1, min(len(scenes), int(ch.get("scene", 1)))) - 1
            add(f"{ts(starts[i])} — {ch.get('label', '')}")
        add("")
        add(DISCLAIMER.get(lang, DISCLAIMER["en"]))
        add("```")
        add("")
        add("### Tags")
        add("")
        add("```")
        add(", ".join(yt.get("tags", [])))
        add("```")
        add("")
        add("### Thumbnail text")
        add(f"- **\"{yt.get('thumbnail_line1', '')}\"**")
        add(f"- _\"{yt.get('thumbnail_line2', '')}\"_")
        add("")

    add("---")
    add("")
    add("### Note")
    if plan.get("spine_phrase"):
        add(f"The spine of this video is **\"{plan['spine_phrase']}\"**. Its "
            f"{name} wording must stay identical everywhere it appears.")
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


def generate(script: str, pid: str, langs: list[str], key: str,
             model: str = G.DEFAULT_MODEL, on_progress=lambda *_: None,
             on_warn=lambda *_: None) -> Result:
    res = Result()
    sections = G.split_sections(script, SECTION_WORDS)
    total_steps = 1 + len(sections) + len(langs) * (len(sections) + 1) + 1

    step = [0]

    def tick(msg: str) -> None:
        step[0] += 1
        on_progress(step[0], total_steps, msg)

    # 1 — plan
    tick("reading the script")
    plan = G.plan(script, key, model)
    res.plan = plan

    # 2 — scenes, section by section, each verified against the source
    all_scenes: list[dict] = []
    section_sizes: list[int] = []
    for i, sec in enumerate(sections, start=1):
        tick(f"splitting section {i} of {len(sections)}")
        got, feedback = None, ""
        for attempt in range(1, 4):
            got = G.scenes_for_section(sec, plan, key, model, feedback)
            joined = " ".join(s.get("narration", "") for s in got)
            if G.words(joined) == G.words(sec):
                break
            feedback = ("Your narration did not reproduce the section exactly.\n"
                        + G.diff_words(sec, joined))
            if attempt == 3:
                res.warnings.append(
                    f"Section {i}: narration still differs from the script after "
                    f"3 attempts.\n{G.diff_words(sec, joined)}")
                on_warn(f"Section {i} did not match the script — see warnings")
        got = got or []
        all_scenes.extend(got)
        section_sizes.append(len(got))

    scenes = [Scene(n=i, narration=s.get("narration", ""),
                    media=(s.get("media") or "IMAGE").upper(),
                    query=(s.get("query") or "").strip(),
                    domain=(s.get("domain") or "").strip().lower(),
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
            # Promote a fallback rather than inventing a placeholder: a looser
            # query about the right subject beats a generic one about none.
            if s.fallbacks:
                s.query, s.fallbacks = s.fallbacks[0], s.fallbacks[1:]
                res.warnings.append(
                    f"S{s.n}: no primary query — used the fallback instead.")
            else:
                s.query = "calm natural scene, soft daylight"
                res.warnings.append(
                    f"S{s.n}: no search query at all — placeholder used, fix by hand.")

    _flag_repeats(scenes, res)
    res.scenes = scenes

    # 3 — English YouTube package
    tick("writing the English YouTube package")
    yt_en = G.youtube_package([s.narration for s in scenes], "English", plan, key, model)
    plan["thumbnail_line1"] = yt_en.get("thumbnail_line1") or plan.get("thumbnail_line1", "")
    plan["thumbnail_line2"] = yt_en.get("thumbnail_line2") or plan.get("thumbnail_line2", "")

    res.files[f"{pid}_MASTER_production_sheet.md"] = render_master(plan, scenes, pid)
    res.files[f"{pid}_ENGLISH_youtube.md"] = render_translation(
        plan, scenes, "en", [s.narration for s in scenes], yt_en, pid)

    # 4 — translations, sent in the same chunks the scenes were split in, so a
    #     scene can never end up under the wrong number
    per_section: list[list[str]] = []
    cursor = 0
    for size in section_sizes:
        per_section.append([s.narration for s in scenes[cursor:cursor + size]])
        cursor += size
    if cursor != len(scenes):        # should not happen; belt and braces
        per_section = [[s.narration for s in scenes[i:i + 25]]
                       for i in range(0, len(scenes), 25)]

    for lang in langs:
        if lang == "en":
            continue
        name = LANG_NAMES.get(lang, lang.upper())
        out: list[str] = []
        for i, chunk in enumerate(per_section, start=1):
            tick(f"translating into {name} — part {i} of {len(per_section)}")
            fb, lines = "", []
            for attempt in range(1, 4):
                lines = G.translate_section(chunk, name, plan, key, model, fb)
                if len(lines) == len(chunk):
                    break
                fb = (f"You returned {len(lines)} lines but exactly "
                      f"{len(chunk)} were required.")
                if attempt == 3:
                    res.warnings.append(
                        f"{name} part {i}: expected {len(chunk)} lines, got "
                        f"{len(lines)}. Padded to keep scene numbering aligned.")
                    on_warn(f"{name} part {i} returned the wrong number of lines")
                    lines = (lines + chunk[len(lines):])[:len(chunk)]
            out.extend(lines)

        tick(f"writing the {name} YouTube package")
        yt = G.youtube_package(out, name, plan, key, model)
        res.files[f"{pid}_{name.upper()}_narration.md"] = render_translation(
            plan, scenes, lang, out, yt, pid)

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
