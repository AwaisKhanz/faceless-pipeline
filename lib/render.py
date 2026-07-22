"""ffmpeg assembly: scene clips -> dissolve chain -> audio -> captions -> MP4.

Timing model (this is the part that matters, so it is written down):

    TAIL      seconds of held picture after the narration line ends   (default 1.0)
    DISSOLVE  crossfade length between scenes                         (default 0.6)

    clip_i duration        d_i     = voice_i + TAIL
    clip_i start on final  start_i = sum_{j<i} (d_j - DISSOLVE)
                                   = sum_{j<i} (voice_j + TAIL - DISSOLVE)

    The audio track is simply  voice_1, gap, voice_2, gap, ...
    with  gap = TAIL - DISSOLVE.  Substituting shows the narration lands exactly
    on each clip start, and every dissolve happens during silence - so no word is
    ever crossfaded away. The finished video runs TAIL - DISSOLVE + DISSOLVE = TAIL
    seconds past the last word.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

W, H, FPS = 1920, 1080, 25


def run(cmd: list[str], quiet: bool = True, cwd: str | None = None) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-15:])
        where = f"\n  (in {cwd})" if cwd else ""
        raise RuntimeError(f"ffmpeg failed:\n  {' '.join(cmd[:9])} ...{where}\n{tail}")


_FILTERS: set[str] | None = None


def ffmpeg_fix_hint() -> str:
    """How to reinstall a full-featured ffmpeg, in the words of this OS."""
    import os
    import sys
    if os.name == "nt":
        return ("winget install Gyan.FFmpeg   (use the 'full' build — "
                "'essentials' has no text support)")
    if sys.platform == "darwin":
        # Homebrew's core 'ffmpeg' formula was slimmed and no longer bundles
        # libass (subtitles) or libfreetype (drawtext), so 'brew reinstall
        # ffmpeg' pours the same caption-less build straight back. 'ffmpeg-full'
        # is the batteries-included formula that carries both.
        return "brew install ffmpeg-full   (core 'ffmpeg' dropped libass/freetype)"
    return "sudo apt install --reinstall ffmpeg"


def available_filters() -> set[str]:
    """Which ffmpeg filters this build actually has.

    Parsed from `ffmpeg -filters`, whose lines look like:
        ` TSC acrossfade        AA->A       Cross fade two input audio streams.`
    Both stdout and stderr are read, and each candidate is confirmed by asking
    ffmpeg directly for its help — listing formats have shifted between major
    versions and a wrong answer here is expensive (it decides whether captions
    are even attempted).
    """
    global _FILTERS
    if _FILTERS is not None:
        return _FILTERS

    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                       capture_output=True, text=True)
    names: set[str] = set()
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        s = line.strip()
        if not s or s.startswith("Filters:"):
            continue
        # The legend at the top ("..C = Command support", "T.. = Timeline
        # support") looks exactly like a filter row apart from the " = ".
        # Excluding it by prefix instead threw away every filter whose flags
        # begin with the same characters — which was all three subtitle filters.
        if " = " in s:
            continue
        parts = s.split()
        # flags column is short, made of dots and letters: "TSC", "..C", "T.."
        if (len(parts) >= 2 and len(parts[0]) <= 4
                and all(c in "TSC." for c in parts[0])
                and parts[1].isidentifier()):
            names.add(parts[1])
    _FILTERS = names
    return _FILTERS


def filter_really_works(name: str) -> bool:
    """Ask ffmpeg directly. Definitive, and cheap enough to be worth it."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-h", f"filter={name}"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).lower()
    return r.returncode == 0 and "unknown filter" not in out and name in out


def has_filter(name: str) -> bool:
    """Listed OR confirmed by direct help lookup — the listing is the fast path,
    the lookup is the one that's actually authoritative."""
    if name in available_filters():
        return True
    return filter_really_works(name)


def caption_method() -> str:
    """How (or whether) this ffmpeg can burn captions.

    Homebrew's default build includes libass, but plenty of builds - conda,
    static downloads, some taps - do not, and then `ass` and `subtitles` simply
    do not exist. Rather than discover that after a 40 minute render, work it out
    up front and degrade gracefully.
    """
    if has_filter("ass"):
        return "ass"
    if has_filter("subtitles"):
        return "subtitles"
    if has_filter("drawtext") and has_filter("drawbox"):
        return "drawtext"
    return "none"


def pix_fmt_of(path: Path) -> str:
    """Pixel format of a clip, or "" if it can't be read."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def duration_of(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}")
    return float(json.loads(r.stdout)["format"]["duration"])


# ---------------------------------------------------------------- scene clips

def make_image_clip(src: Path, dur: float, out: Path, zoom: bool = True) -> None:
    """Still -> 1080p clip with a slow Ken Burns push (or a plain hold)."""
    frames = max(2, int(round(dur * FPS)))
    if zoom:
        # Pre-scale generously so the zoom never shows softness, then zoompan.
        vf = (
            f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
            f"crop={W*2}:{H*2},"
            f"zoompan=z='min(zoom+0.00035,1.12)':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS},"
            f"setsar=1,format=yuv420p"
        )
    else:
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},fps={FPS},setsar=1,format=yuv420p")
    run(["ffmpeg", "-y", "-loop", "1", "-i", str(src), "-t", f"{dur:.3f}",
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(out)])


def make_video_clip(src: Path, dur: float, out: Path) -> None:
    """Stock video -> 1080p clip trimmed (or looped) to exactly `dur`."""
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
          f"crop={W}:{H},fps={FPS},setsar=1,format=yuv420p")
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src), "-t", f"{dur:.3f}",
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(out)])


# ------------------------------------------------------------- dissolve chain

def _xfade_group(clips: list[tuple[Path, float]], T: float, out: Path) -> float:
    """Crossfade a handful of clips into one. Returns the resulting duration."""
    if len(clips) == 1:
        shutil.copy(clips[0][0], out)
        return clips[0][1]

    args: list[str] = ["ffmpeg", "-y"]
    for p, _ in clips:
        args += ["-i", str(p)]

    parts, prev, acc = [], "[0:v]", clips[0][1]
    for i in range(1, len(clips)):
        offset = acc - T
        label = f"[x{i}]"
        parts.append(f"{prev}[{i}:v]xfade=transition=fade:"
                     f"duration={T:.3f}:offset={offset:.3f}{label}")
        acc = acc + clips[i][1] - T
        prev = label

    # format=yuv420p on the way out AND -pix_fmt: xfade can negotiate itself up to
    # 4:4:4 from full-range JPEG sources, and libx264 then writes a High 4:4:4
    # Predictive stream that a lot of players and phones simply cannot decode.
    parts.append(f"{prev}format=yuv420p[vout]")
    args += ["-filter_complex", ";".join(parts), "-map", "[vout]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-an", str(out)]
    run(args)
    return acc


def dissolve_concat(clips: list[tuple[Path, float]], T: float, out: Path,
                    work: Path, group: int = 10) -> float:
    """Crossfade any number of clips, hierarchically so filter graphs stay small.

    A single 115-link xfade chain works but is slow and memory-hungry; grouping
    keeps each graph to ~10 links and is dramatically faster on a laptop.
    """
    if len(clips) <= group:
        return _xfade_group(clips, T, out)

    work.mkdir(parents=True, exist_ok=True)
    level, idx = clips, 0
    while len(level) > group:
        nxt: list[tuple[Path, float]] = []
        for i in range(0, len(level), group):
            chunk = level[i:i + group]
            gp = work / f"grp_{idx:04d}.mp4"
            idx += 1
            d = _xfade_group(chunk, T, gp)
            nxt.append((gp, d))
        level = nxt
    return _xfade_group(level, T, out)


# -------------------------------------------------------------------- audio

def build_audio(voices: list[Path], gaps: list[float], out: Path, work: Path,
                tail: float = 0.0) -> list[float]:
    """Concatenate narration with a per-scene gap. Returns each line's start time.

    `gaps[i]` is the silence inserted AFTER voice i. It is computed per scene from
    the clip's real (frame-rounded) duration rather than assumed - otherwise the
    rounding error accumulates and the narration slides off the picture by about
    a second over 115 scenes.
    """
    work.mkdir(parents=True, exist_ok=True)

    def silence(dur: float, path: Path) -> None:
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-t", f"{max(dur, 0.001):.4f}", str(path)])

    norm: list[Path] = []
    for i, v in enumerate(voices):
        n = work / f"v_{i:04d}.wav"
        run(["ffmpeg", "-y", "-i", str(v), "-ar", "48000", "-ac", "2", str(n)])
        norm.append(n)

    starts, t, lines = [], 0.0, []
    for i, n in enumerate(norm):
        starts.append(t)
        t += duration_of(n)
        lines.append(f"file '{n.resolve()}'")
        g = gaps[i] if i < len(gaps) else 0.0
        if g > 0.0005:
            sp = work / f"gap_{i:04d}.wav"
            silence(g, sp)
            lines.append(f"file '{sp.resolve()}'")
            t += duration_of(sp)

    if tail > 0.0005:
        tp = work / "tail.wav"
        silence(tail, tp)
        lines.append(f"file '{tp.resolve()}'")

    lst = work / "audio_concat.txt"
    lst.write_text("\n".join(lines), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c:a", "pcm_s16le", str(out)])
    return starts


def mix_music(voice: Path, music: Path, out: Path, level: float = 0.20) -> None:
    """Loop the music bed under the narration at a fixed level."""
    vdur = duration_of(voice)
    run(["ffmpeg", "-y", "-i", str(voice),
         "-stream_loop", "-1", "-i", str(music),
         "-filter_complex",
         f"[1:a]volume={level},atrim=0:{vdur:.3f},afade=t=in:st=0:d=2,"
         f"afade=t=out:st={max(0.0, vdur-3):.3f}:d=3[m];"
         f"[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
         "-map", "[a]", "-c:a", "pcm_s16le", str(out)])


# ------------------------------------------------------------------ captions

def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(texts: list[str], starts: list[float], durs: list[float],
              out: Path, max_chars: int = 42) -> None:
    """One cue per narration line, wrapped to two readable lines."""
    blocks = []
    for i, (txt, st, du) in enumerate(zip(texts, starts, durs), start=1):
        blocks.append(f"{i}\n{_ts(st)} --> {_ts(st + du)}\n{_wrap(txt, max_chars)}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    # Keep cues to two lines; longer narration simply gets a wider second line.
    if len(lines) > 2:
        half = math.ceil(len(words) / 2)
        lines = [" ".join(words[:half]), " ".join(words[half:])]
    return "\n".join(lines)


def _ass_ts(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(texts: list[str], starts: list[float], durs: list[float], out: Path,
              font: str = "Arial", size: int = 58, max_chars: int = 42,
              margin_v: int = 90) -> None:
    """Black-box / white-text captions, written as ASS with an explicit PlayRes.

    Going through SRT + force_style is unreliable: libass assumes PlayResY=288
    when the file does not declare one, so a 'FontSize=30' ends up rendering
    roughly four times too large on a 1080p frame. Declaring PlayRes here means
    `size` is real pixels. BorderStyle=3 draws its box from OutlineColour
    (BackColour is only the shadow), which is why Outline must be non-zero.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&HC8000000,&H00000000,-1,0,0,0,100,100,0,0,3,14,0,2,120,120,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for txt, st, du in zip(texts, starts, durs):
        body = _wrap(txt, max_chars).replace("\n", r"\N")
        lines.append(f"Dialogue: 0,{_ass_ts(st)},{_ass_ts(st + du)},Cap,,0,0,0,,{body}")
    out.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def _drawtext_filter(texts: list[str], starts: list[float], durs: list[float],
                     size: int, margin_v: int) -> str:
    """Caption chain built from drawtext + drawbox, for builds without libass.

    Not as pretty as ASS — no per-line box hugging the text — but it burns real
    captions with only filters that ship in almost every ffmpeg.
    """
    line_h = int(size * 1.35)
    parts = []
    for txt, st, du in zip(texts, starts, durs):
        lines = _wrap(txt, 42).split("\n")[:2]
        n = len(lines)
        box_h = line_h * n + 24
        y0 = H - margin_v - box_h
        en = f"between(t,{st:.3f},{st + du:.3f})"
        parts.append(f"drawbox=x=0:y={y0}:w={W}:h={box_h}:"
                     f"color=black@0.62:t=fill:enable='{en}'")
        for i, ln in enumerate(lines):
            safe = (ln.replace("\\", "\\\\").replace(":", r"\:")
                      .replace("'", r"'").replace("%", r"\%"))
            y = y0 + 12 + i * line_h
            parts.append(f"drawtext=text='{safe}':fontsize={size}:fontcolor=white:"
                         f"x=(w-text_w)/2:y={y}:enable='{en}'")
    return ",".join(parts)


def burn_captions(video: Path, subs: Path, out: Path,
                  texts: list[str] | None = None,
                  starts: list[float] | None = None,
                  durs: list[float] | None = None,
                  size: int = 58, margin_v: int = 90) -> str:
    """Burn captions into the picture. Returns the method actually used.

    Raises only if nothing at all worked — the caller is expected to fall back to
    an uncaptioned video rather than lose the whole render.
    """
    subs, video, out = Path(subs).resolve(), Path(video).resolve(), Path(out).resolve()
    method = caption_method()

    if method in ("ass", "subtitles"):
        # Run FROM the subtitle file's folder with a bare filename: absolute paths
        # need escaping inside a filter string and the rules differ between ffmpeg
        # versions (quoting that works on 4.x fails on 8.x).
        kind = "ass" if (method == "ass" and subs.suffix.lower() == ".ass") \
            else "subtitles"
        run(["ffmpeg", "-y", "-i", str(video),
             "-vf", f"{kind}=filename={subs.name}",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)],
            cwd=str(subs.parent))
        return kind

    if method == "drawtext" and texts and starts and durs:
        chain = _drawtext_filter(texts, starts, durs, size, margin_v)
        script = subs.parent / "captions_filter.txt"
        script.write_text(chain, encoding="utf-8")   # too long for a shell arg
        run(["ffmpeg", "-y", "-i", str(video),
             "-filter_script:v", str(script.name),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)],
            cwd=str(subs.parent))
        return "drawtext"

    raise RuntimeError(
        "This ffmpeg build has no subtitle filter (no 'ass', 'subtitles' or "
        f"'drawtext'). Reinstall with:  {ffmpeg_fix_hint()}")


def mux(video: Path, audio: Path, out: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)])
