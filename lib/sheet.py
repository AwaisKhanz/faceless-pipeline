"""Parse the master production sheet and translation narration files."""
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scene:
    n: int
    media: str            # "IMAGE" or "VIDEO"
    narration: str        # narration in the target language
    query: str            # stock search query (always English - stock sites index in English)
    # Looser searches to fall back on when `query` returns nothing. Free stock
    # simply does not have every shot you would want, and one query either hits
    # or ships junk; walking down a ladder keeps the scene on-topic instead.
    fallbacks: list[str] = field(default_factory=list)
    # Which library to ask. The sources barely overlap (see lib/sources.py),
    # so this is a routing decision, not a preference.
    domain: str = ""
    en_narration: str = ""
    note: str = ""        # e.g. "title card", "Arthur intro"
    hero: bool = False    # flagged recurring-character / title-card scene


# **S12 ⬜** · IMAGE   /  **S12 ✅** · VIDEO ⚑ title card
SCENE_RE = re.compile(r"^\*\*S(\d+)\s*[⬜✅]?\*\*\s*·\s*(IMAGE|VIDEO)(.*)$")
NARR_RE = re.compile(r'^-\s*Narration:\s*"(.*)"\s*$')
# The query is the first backtick-delimited span. Anything after it (e.g.
# '**+ on-screen text "SURFACING"**') is a note to the editor, not part of the
# search term - so this deliberately does not anchor to end of line.
ALT_RE = re.compile(r"^-\s*ALT\s*/\s*search:\s*`([^`]*)`")
ALT_LOOSE_RE = re.compile(r"^-\s*ALT\s*/\s*search:\s*(.+?)\s*$")
# Optional. Sheets written before the ladder existed have no such line, and
# must keep parsing exactly as they did.
FALLBACK_RE = re.compile(r"^-\s*Fallbacks?:\s*(.+?)\s*$")
DOMAIN_RE = re.compile(r"^-\s*Domain:\s*([a-z]+)\s*$", re.I)

# **S31** · EN: "..."   then next line   DE: "..."  / ES: "..."
TR_KEY_RE = re.compile(r'^\*\*S(\d+)\*\*\s*·\s*EN:\s*"(.*)"\s*$')
TR_VAL_RE = re.compile(r'^(DE|ES|EN|FR|IT|PT):\s*"(.*)"\s*$')

HERO_HINTS = ("title card", "Arthur", "piano teacher", "key beat",
              "core line", "sign-off", "disclaimer", "subscribe",
              "share beat", "next-episode", "motif")


def parse_master(path: Path) -> list[Scene]:
    """Read the master production sheet -> ordered list of Scenes (English)."""
    scenes: list[Scene] = []
    cur: dict | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m = SCENE_RE.match(line)
        if m:
            if cur:
                scenes.append(_finish(cur))
            tail = m.group(3) or ""
            cur = {"n": int(m.group(1)), "media": m.group(2),
                   "note": tail.replace("⚑", "").replace("*", "").strip(),
                   "narration": "", "query": "", "fallbacks": [],
                   "domain": ""}
            continue
        if cur is None:
            continue
        m = NARR_RE.match(line)
        if m:
            cur["narration"] = _unescape(m.group(1))
            continue
        m = ALT_RE.match(line)
        if m:
            cur["query"] = m.group(1).strip()
            continue
        m = DOMAIN_RE.match(line)
        if m:
            cur["domain"] = m.group(1).strip().lower()
            continue
        m = FALLBACK_RE.match(line)
        if m:
            # `a` · `b`  or  a | b  — accept either, ignore empties
            raw = m.group(1)
            parts = re.findall(r"`([^`]+)`", raw) or re.split(r"\s*[|·]\s*", raw)
            cur["fallbacks"] = [x.strip(" `*") for x in parts if x.strip(" `*")]
            continue
        m = ALT_LOOSE_RE.match(line)
        if m and not cur["query"]:
            # Backticks omitted - fall back to the rest of the line, minus any
            # bold editor note.
            cur["query"] = re.sub(r"\*\*.*?\*\*", "", m.group(1)).strip(" `*")
    if cur:
        scenes.append(_finish(cur))

    _validate(scenes, path)
    return scenes


def _finish(d: dict) -> Scene:
    note = d["note"]
    hero = any(h.lower() in note.lower() for h in HERO_HINTS)
    return Scene(n=d["n"], media=d["media"], narration=d["narration"],
                 query=d["query"], fallbacks=d.get("fallbacks") or [],
                 domain=d.get("domain") or "",
                 en_narration=d["narration"], note=note, hero=hero)


def _unescape(s: str) -> str:
    # curly quotes -> straight; the TTS engines handle both, but this keeps
    # SRT files and logs clean.
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'").strip())


def parse_translation(path: Path, lang: str) -> dict[int, str]:
    """Read a translation narration file -> {scene_number: narration}."""
    lang = lang.upper()
    out: dict[int, str] = {}
    pending: int | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = TR_KEY_RE.match(line)
        if m:
            pending = int(m.group(1))
            continue
        m = TR_VAL_RE.match(line)
        if m and pending is not None and m.group(1).upper() == lang:
            out[pending] = _unescape(m.group(2))
            pending = None
    return out


def load(master: Path, lang: str, translation: Path | None = None) -> list[Scene]:
    """Master sheet + optional translation -> scenes in the requested language."""
    scenes = parse_master(master)
    if lang.lower() == "en":
        return scenes
    if translation is None:
        raise SystemExit(
            f"Language '{lang}' needs a translation file. "
            f"Pass --translation path/to/videoNN_{lang.upper()}_narration.md"
        )
    tr = parse_translation(translation, lang)
    missing = [s.n for s in scenes if s.n not in tr]
    if missing:
        raise SystemExit(
            f"Translation file is missing {len(missing)} scenes: {missing[:12]}"
            f"{' ...' if len(missing) > 12 else ''}\n"
            f"Every scene in the master sheet needs a matching {lang.upper()}: line."
        )
    for s in scenes:
        s.narration = tr[s.n]
    return scenes


def _validate(scenes: list[Scene], path) -> None:
    if not scenes:
        raise SystemExit(f"No scenes found in {path}. Is this a master production sheet?")
    nums = [s.n for s in scenes]
    expected = list(range(1, len(scenes) + 1))
    if nums != expected:
        gaps = [e for e, a in zip(expected, nums) if e != a]
        raise SystemExit(
            f"Scene numbering is not continuous in {path}. "
            f"First mismatch at S{gaps[0] if gaps else '?'} "
            f"(found {len(scenes)} scenes)."
        )
    for s in scenes:
        if not s.narration:
            raise SystemExit(f"S{s.n} has no 'Narration:' line.")
        if not s.query:
            raise SystemExit(f"S{s.n} has no 'ALT / search:' line.")
