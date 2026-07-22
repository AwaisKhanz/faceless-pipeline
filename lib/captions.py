"""Subtitle styling and word-by-word (karaoke) caption rendering.

This module owns two things and nothing else:

  1. STYLE — a small, JSON-friendly description of how captions look (font,
     colours, the translucent bar, position, words per line...). A handful of
     named presets ship here, and a user's own style is just the same shape with
     different values, so "make your own" needs no new code.

  2. ASS OUTPUT — turning a scene's word timings + a style into an .ass subtitle
     file libass can burn. Each on-screen phrase keeps still while the word being
     spoken lights up in the accent colour, exactly like the reference clips.

Everything here is pure string/maths work: no ffmpeg, no models, no network, so
it runs and tests anywhere. Word *timings* come from lib/align.py; when those
aren't available a proportional split (heuristic_words) keeps the same look with
looser sync rather than falling back to a dead, whole-line caption.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, fields

# The frame the pipeline renders at. Kept in step with render.W/H; captions are
# positioned in these coordinates via the ASS PlayRes header.
W, H = 1920, 1080


# ─────────────────────────────────────────────────────────────── style model

@dataclass
class Style:
    """How captions look. Every field is plain JSON so a user style round-trips.

    Colours are "#RRGGBB". Opacity is 0..1 (1 = solid). Sizes are pixels in the
    1920x1080 frame.
    """
    name: str = "Custom"
    font: str = "Arial"
    size: int = 64
    bold: bool = True
    uppercase: bool = False

    text_color: str = "#FFFFFF"        # the words at rest
    active_color: str = "#B57BFF"      # the word currently being spoken
    outline_color: str = "#000000"
    outline: int = 3                   # px stroke around the letters
    shadow: int = 0

    bar: bool = True                   # the translucent pill behind the text
    bar_color: str = "#0A0A0F"
    bar_opacity: float = 0.55
    bar_radius: int = 24               # rounded-corner radius, px

    position: str = "bottom"           # bottom | center
    margin_v: int = 130                # gap from the frame edge, px

    max_words: int = 5                 # words shown together on one line
    karaoke: bool = True               # highlight the spoken word; off = plain
    active_scale: float = 1.0          # >1 pops the active word (may reflow)

    def merged(self, **over) -> "Style":
        d = asdict(self)
        d.update({k: v for k, v in over.items() if v is not None})
        return Style.from_dict(d)

    @staticmethod
    def from_dict(d: dict) -> "Style":
        keep = {f.name for f in fields(Style)}
        return Style(**{k: v for k, v in (d or {}).items() if k in keep})

    def to_dict(self) -> dict:
        return asdict(self)


# Ready-made looks. The first matches Awais's reference images (white bold text,
# dark translucent rounded pill, purple active word). The rest are common
# faceless-channel styles, all buildable from the same fields.
PRESETS: dict[str, Style] = {
    "reference": Style(
        name="Reference (purple pop)", font="Arial", size=66, bold=True,
        text_color="#FFFFFF", active_color="#B57BFF", outline=3,
        bar=True, bar_color="#0A0A0F", bar_opacity=0.55, bar_radius=26,
        position="bottom", max_words=5),
    "clean_white": Style(
        name="Clean White", font="Arial", size=64, bold=True,
        text_color="#FFFFFF", active_color="#FFFFFF", outline=4,
        bar=False, position="bottom", max_words=5),
    "bold_yellow": Style(
        name="Bold Yellow", font="Arial", size=70, bold=True, uppercase=True,
        text_color="#FFFFFF", active_color="#FFE100", outline=5,
        bar=False, position="bottom", max_words=4, active_scale=1.10),
    "night_cyan": Style(
        name="Night Cyan", font="Arial", size=64, bold=True,
        text_color="#FFFFFF", active_color="#39D6FF", outline=3,
        bar=True, bar_color="#05070D", bar_opacity=0.6, bar_radius=22,
        position="bottom", max_words=5),
    "minimal": Style(
        name="Minimal", font="Arial", size=58, bold=False,
        text_color="#F2F2F2", active_color="#FF5C8A", outline=2,
        bar=False, position="bottom", max_words=6),
    "green_pop": Style(
        name="Green Pop", font="Arial", size=68, bold=True, uppercase=True,
        text_color="#FFFFFF", active_color="#31E27B", outline=5,
        bar=False, position="center", max_words=4, active_scale=1.12),
}

DEFAULT_PRESET = "reference"


def preset_list() -> list[dict]:
    """Presets as plain dicts, for the picker."""
    return [{"id": k, **v.to_dict()} for k, v in PRESETS.items()]


def resolve_style(spec) -> Style:
    """Turn whatever is stored (a preset id, a dict, None) into a Style."""
    if spec is None:
        return PRESETS[DEFAULT_PRESET]
    if isinstance(spec, str):
        return PRESETS.get(spec, PRESETS[DEFAULT_PRESET])
    if isinstance(spec, dict):
        base = PRESETS.get(spec.get("template") or spec.get("preset"),
                           PRESETS[DEFAULT_PRESET])
        return base.merged(**{k: v for k, v in spec.items()
                              if k not in ("template", "preset")})
    return PRESETS[DEFAULT_PRESET]


# ─────────────────────────────────────────────────────────────── colour maths

def _clampi(x, lo, hi):
    return max(lo, min(hi, int(round(x))))


def hex_to_ass(hex_rgb: str, opacity: float = 1.0) -> str:
    """'#RRGGBB' + opacity(0..1) -> ASS '&HAABBGGRR'.

    ASS stores colour as alpha-blue-green-red, and its alpha is inverted:
    00 is fully opaque, FF fully transparent. So opacity 1 -> alpha 00.
    """
    s = (hex_rgb or "#FFFFFF").lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        s = "FFFFFF"
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    a = _clampi((1.0 - float(opacity)) * 255, 0, 255)
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def _ass_ts(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ───────────────────────────────────────────────────────── words → groups

_SENT_END = re.compile(r"[.!?…]$")


def chunk_words(words: list[dict], style: Style) -> list[dict]:
    """Break one scene's words into short on-screen groups.

    A group ends when it reaches max_words, when the line would get too long, at
    sentence punctuation, or across a real pause (so a breath starts a fresh
    line instead of a lopsided one). Each `word` is {word, start, end}.
    """
    groups: list[dict] = []
    cur: list[dict] = []
    max_words = max(1, int(style.max_words))
    max_chars = max(12, int(style.max_words) * 9)   # rough line-length guard

    def flush():
        if cur:
            groups.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                           "words": cur.copy()})
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        text_len = sum(len(x["word"]) + 1 for x in cur)
        end_here = len(cur) >= max_words or text_len >= max_chars
        if _SENT_END.search(w["word"]):
            end_here = True
        if i + 1 < len(words):
            gap = words[i + 1]["start"] - w["end"]
            if gap > 0.55 and len(cur) >= 2:        # a genuine breath
                end_here = True
        if end_here:
            flush()
    flush()
    return groups


def heuristic_words(text: str, start: float, dur: float) -> list[dict]:
    """Estimate per-word timings from a line when no aligner is available.

    Splits the line's duration across its words weighted by length, with a tiny
    gap between them. Not as tight as real alignment, but keeps the word-by-word
    look everywhere instead of collapsing to a static caption.
    """
    toks = [t for t in re.split(r"\s+", (text or "").strip()) if t]
    if not toks:
        return []
    weights = [max(1, len(re.sub(r"[^\w]", "", t))) for t in toks]
    total = sum(weights) or 1
    gap = min(0.04, dur / (len(toks) * 4)) if len(toks) > 1 else 0.0
    span = max(0.0, dur - gap * (len(toks) - 1))
    out, t = [], start
    for tok, wt in zip(toks, weights):
        d = span * (wt / total)
        out.append({"word": tok, "start": round(t, 3), "end": round(t + d, 3)})
        t += d + gap
    return out


# ─────────────────────────────────────────────────────────────── ASS output

def _round_rect(w: int, h: int, r: int) -> str:
    """An ASS vector path for a rounded rectangle, top-left origin at (0,0).

    Corners use the corner point as both bezier controls — a clean quarter-round
    at this size. Coordinates are integers so libass never chokes on the path.
    """
    r = max(0, min(int(r), w // 2, h // 2))
    w, h = int(w), int(h)
    return (f"m {r} 0 "
            f"l {w - r} 0 b {w} 0 {w} 0 {w} {r} "
            f"l {w} {h - r} b {w} {h} {w} {h} {w - r} {h} "
            f"l {r} {h} b 0 {h} 0 {h} 0 {h - r} "
            f"l 0 {r} b 0 0 0 0 {r} 0")


def _glyph_w(text: str, size: int, bold: bool) -> int:
    """Rough rendered width of a string. Enough to size the bar; not exact."""
    adv = size * (0.58 if bold else 0.50)
    # Narrow characters pull the average down a little.
    thin = sum(1 for c in text if c in "iIl.,:;'!|")
    return int(len(text) * adv - thin * size * 0.22)


def _esc(text: str) -> str:
    # Braces open override blocks in ASS; a literal brace in narration must be
    # escaped so it renders instead of being read as a tag.
    return text.replace("{", "(").replace("}", ")")


def ass_header(style: Style) -> str:
    prim = hex_to_ass(style.text_color)
    out = hex_to_ass(style.outline_color)
    bold = -1 if style.bold else 0
    # BorderStyle 1 = outline + shadow (no box); the bar is drawn separately so
    # its transparency never doubles up across the per-word events.
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{style.font},{style.size},{prim},{prim},{out},&H00000000,"
        f"{bold},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )


def _group_events(g: dict, style: Style) -> list[str]:
    """The Dialogue lines for one on-screen phrase.

    Layer 0 (optional): the translucent rounded bar, one event for the whole
    phrase. Layer 1: one event per word, the whole phrase drawn each time with
    just the active word recoloured, so the line holds still while the highlight
    moves across it.
    """
    words = g["words"]
    disp = [(_esc(w["word"].upper() if style.uppercase else w["word"]))
            for w in words]
    phrase = " ".join(disp)

    cx = W // 2
    cy = (H - style.margin_v) if style.position == "bottom" else (H // 2)

    # Inline \c overrides take 6-digit BGR with a trailing '&' (alpha is separate);
    # the 8-digit form is only for the [V4+ Styles] colour fields.
    prim = _rgb_only(style.text_color)
    act = _rgb_only(style.active_color)
    ev: list[str] = []

    # The bar, sized to the phrase, centred on (cx, cy). Drawn once for the whole
    # phrase (Layer 0) so its transparency never doubles across the per-word
    # events. Colour and alpha are set separately (\1c + \1a) so the fill matches
    # bar_color at exactly bar_opacity.
    if style.bar:
        bw = _glyph_w(phrase, style.size, style.bold) + int(style.size * 1.1)
        bh = int(style.size * 1.55)
        x0, y0 = cx - bw // 2, cy - bh // 2
        draw = _round_rect(bw, bh, style.bar_radius)
        ev.append(
            f"Dialogue: 0,{_ass_ts(g['start'])},{_ass_ts(g['end'])},Cap,,0,0,0,,"
            f"{{\\an7\\pos({x0},{y0})\\1c{_rgb_only(style.bar_color)}"
            f"\\1a{_alpha_only(style.bar_opacity)}\\bord0\\shad0\\p1}}"
            f"{draw}{{\\p0}}")

    # One event per word: recolour just that word, everything else at rest.
    scale = style.active_scale if style.active_scale and style.active_scale != 1 else None
    for i, w in enumerate(words):
        end = words[i + 1]["start"] if i + 1 < len(words) else g["end"]
        parts = []
        for j, token in enumerate(disp):
            if j == i:
                if scale:
                    parts.append(f"{{\\c{act}\\fscx{int(scale*100)}"
                                 f"\\fscy{int(scale*100)}}}{token}"
                                 f"{{\\c{prim}\\fscx100\\fscy100}}")
                else:
                    parts.append(f"{{\\c{act}}}{token}{{\\c{prim}}}")
            else:
                parts.append(token)
        body = " ".join(parts)
        ev.append(
            f"Dialogue: 1,{_ass_ts(w['start'])},{_ass_ts(end)},Cap,,0,0,0,,"
            f"{{\\an5\\pos({cx},{cy})}}{body}")
    return ev


def _rgb_only(hex_rgb: str) -> str:
    """ASS colour with no alpha byte, for use with a separate \\1a tag."""
    s = (hex_rgb or "#000000").lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _alpha_only(opacity: float) -> str:
    a = _clampi((1.0 - float(opacity)) * 255, 0, 255)
    return f"&H{a:02X}&"


def build_ass(scene_groups: list[dict], style: Style) -> str:
    """Full .ass text for a whole video.

    `scene_groups` is a flat list of on-screen groups (already in absolute
    time), each {start, end, words:[{word,start,end}]}. Callers build these from
    aligned (or heuristic) word timings per scene.
    """
    body = []
    for g in scene_groups:
        if not g.get("words"):
            continue
        if not style.karaoke:
            # Plain mode: the whole phrase, no moving highlight.
            g = {**g, "words": [{"word": " ".join(x["word"] for x in g["words"]),
                                 "start": g["start"], "end": g["end"]}]}
        body.extend(_group_events(g, style))
    return ass_header(style) + "\n".join(body) + "\n"


def groups_from_scenes(scene_words: list[list[dict]], style: Style) -> list[dict]:
    """Chunk each scene's words, keeping absolute times. Input is one word list
    per scene (already offset to the scene's place in the finished audio)."""
    out: list[dict] = []
    for words in scene_words:
        if words:
            out.extend(chunk_words(words, style))
    return out
