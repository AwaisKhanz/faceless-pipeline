"""The `faceless` command.

Installed by setup.sh / setup.bat as a console entry point, so you can type

    faceless start

from anywhere in the project instead of `python make_video.py studio`. This is
the same idea as `npm start`: short verbs for the things you do every day.

Everything here is a thin forwarding layer. The actual work lives in
make_video.py, which stays runnable on its own — if this command ever breaks or
was never installed, `python make_video.py <step>` does exactly the same thing.
"""
from __future__ import annotations

import sys

# Friendly verbs on the left, the real step names on the right. Aliases exist
# because "start" and "build" are what people reach for, while the underlying
# steps are named after what they actually do.
ALIASES = {
    "start": "studio",       # npm start
    "panel": "studio",
    "ui": "studio",
    "check": "doctor",
    "bench": "benchtts",
    "new": "generate",       # new video from a script
    "visuals": "stock",
    "sheets": "list",
}

HELP = """
  faceless — turn a script into finished videos

  Every day
    faceless start                     open the control panel in your browser
    faceless check                     make sure this machine is set up right

  Working on a video
    faceless new    --script my.txt --id video06     script -> production sheets
    faceless visuals --sheet sheets/video06_main_script.md
    faceless voice   --sheet sheets/video06_main_script.md --lang en
    faceless render  --sheet sheets/video06_main_script.md --lang en --captions
    faceless all     --sheet sheets/video06_main_script.md --lang en --yes

  Voices
    faceless voices --lang en          list and choose reference clips
    faceless bench  --lang en          time the voice engine on this machine

  Anything else
    faceless sheets                    projects found in sheets/
    faceless models                    Gemini models your key can call
    faceless sources                   test each picture library, show routing
    faceless vertex-models             which Vertex Gemini models you can call

  Add --verbose to any command to see full errors.
  Everything is cached, so re-running is cheap and picks up where it stopped.

  This command is a shortcut. `python make_video.py <step>` always works too.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0

    if argv[0] in ("-V", "--version", "version"):
        print("faceless-studio 1.0.0")
        return 0

    # Translate the friendly verb, then hand the rest over untouched.
    argv[0] = ALIASES.get(argv[0], argv[0])
    sys.argv = ["make_video.py", *argv]

    # Imported here, not at module scope: make_video hands over to the project
    # venv on import, and doing that while this module is still being loaded
    # would re-enter the interpreter mid-import.
    import make_video

    verbose = "--verbose" in argv

    try:
        make_video.main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        return 130
    except Exception as e:
        # The pipeline raises these deliberately, with a message written for a
        # person. A traceback adds nothing and hides the sentence that matters.
        # Anything unexpected still gets the full traceback, because then the
        # traceback IS the information.
        expected = type(e).__name__ in (
            "GeminiError", "ChatterboxError", "StockError", "CaptionsSkipped")
        if not expected or verbose:
            raise
        print(f"\n  {e}")
        print("\n  Add --verbose to see the full technical detail.")
        return 1
    except SystemExit as e:
        # SystemExit's code is an int OR a message string — raise SystemExit("...")
        # is the normal way to abort with an explanation. Printing the message
        # and returning 1 is what the interpreter itself does; calling int() on
        # it throws ValueError and buries the real message under a traceback.
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
