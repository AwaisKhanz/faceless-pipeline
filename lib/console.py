"""Make text output survive the Windows console.

Two separate problems, both invisible on macOS:

1. Windows consoles default to a legacy codepage (cp1252 or cp850), not UTF-8.
   Printing a box-drawing character or an emoji raises UnicodeEncodeError and
   kills the program. PowerShell 7 happens to use UTF-8, cmd.exe does not — so
   the same script worked when typed by hand and crashed when launched from a
   .bat file.

2. Reading a file without naming an encoding uses the *locale* encoding, which
   on Windows is again cp1252. The production sheets contain emoji, so
   Path.read_text() cannot decode them at all there.

setup() fixes the printing side. The file side is fixed by passing
encoding="utf-8" at every call site — see read_text/write_text across lib/.
"""
from __future__ import annotations

import sys

# Fallbacks for consoles that genuinely cannot render the nice characters.
# The program stays readable rather than pretty; it never dies over decoration.
_FALLBACK = {
    "─": "-", "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "█": "#", "░": ".", "→": "->", "←": "<-", "⚠": "!", "·": "-",
    "—": "-", "–": "-", "…": "...", "✓": "ok", "⭐": "*", "⚑": "!",
    "⬜": "[ ]", "✅": "[x]",
}

unicode_ok = True


def setup() -> bool:
    """Point stdout/stderr at UTF-8 and never let encoding kill the program.

    errors="replace" is the important part: even if something slips through
    with a character the console truly cannot draw, you get a placeholder
    instead of a traceback. Losing a box-drawing character is not a reason to
    stop working.
    """
    global unicode_ok
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    enc = getattr(sys.stdout, "encoding", "") or ""
    unicode_ok = "utf" in enc.lower()
    return unicode_ok


def plain(text: str) -> str:
    """Strip decoration down to ASCII, for consoles that can't do better."""
    if unicode_ok:
        return text
    for fancy, simple in _FALLBACK.items():
        text = text.replace(fancy, simple)
    return text


def out(text: str = "") -> None:
    """print(), but it cannot fail on an unencodable character."""
    print(plain(text))
