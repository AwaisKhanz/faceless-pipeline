#!/usr/bin/env python3
"""Per-channel voice resolution, engine-aware.

A channel's voice is a single string whose MEANING depends on the engine: a Chirp
catalogue name under Chirp, a reference clip under Chatterbox/Higgs. The resolver
must only apply it when it fits the engine that will actually narrate — otherwise
a Chirp channel voice would break a local Chatterbox run (and vice-versa), so it
falls back to the language's own Settings voice. generate_voice, the dashboard
count and the file grouping must all agree on that one rule.

    python3 tools/test_channel_voice.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pipeline as pl        # noqa: E402
from lib import channels as ch        # noqa: E402
from lib import tts                   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<58}{'' if ok else repr(got)}")
        bad += not ok

    ch.FILE = Path(tempfile.mkdtemp()) / "channels.json"
    sheet = Path("/x/projects/Demo/sheets/Demo_main_script.md")
    pid = pl.project_id(sheet)
    ch.assign(pid, "Chan")
    saved_engine = tts.active_engine

    print("\n  a Chirp name vs a reference clip are told apart:")
    check("a Chirp catalogue name is NOT a clip",
          pl._is_clip_voice("de-DE-Chirp3-HD-Enceladus"), False)
    check("a folder/clip path IS a clip", pl._is_clip_voice("de/awais-male-1.mp3"), True)
    check("a bare .wav IS a clip", pl._is_clip_voice("voice.wav"), True)

    def resolve(engine, stored):
        ch.set_channel_voice("Chan", "de", stored)
        tts.active_engine = lambda cfg: engine
        return pl.channel_voice(sheet, "de", {})

    try:
        print("\n  the channel voice applies ONLY when the engine can speak it:")
        check("Chirp name under Chirp -> used",
              resolve("chirp", "de-DE-Chirp3-HD-Enceladus"), "de-DE-Chirp3-HD-Enceladus")
        check("Chirp name under Chatterbox -> '' (fall back to Settings)",
              resolve("chatterbox", "de-DE-Chirp3-HD-Enceladus"), "")
        check("clip under Chatterbox -> used",
              resolve("chatterbox", "de/awais-male-1.mp3"), "de/awais-male-1.mp3")
        check("clip under Higgs -> used",
              resolve("higgs", "de/awais-male-1.mp3"), "de/awais-male-1.mp3")
        check("clip under Chirp -> '' (fall back to Settings)",
              resolve("chirp", "de/awais-male-1.mp3"), "")
        check("no channel voice set -> '' (use Settings default)",
              (ch.set_channel_voice("Chan", "de", ""), pl.channel_voice(sheet, "de", {}))[1], "")
        check("project in no channel -> '' ",
              pl.channel_voice(Path("/x/projects/Lonely/sheets/Lonely_main_script.md"), "de", {}), "")

        print("\n  generate_voice hands the SAME resolved voice to the engine:")
        seen = {}
        tts.active_engine = lambda cfg: "chirp"
        ch.set_channel_voice("Chan", "de", "de-DE-Chirp3-HD-Enceladus")
        saved_synth = tts.synth
        tts.synth = lambda scenes, lang, cache, voice=None, log=None, **k: seen.setdefault("voice", voice) or []
        try:
            pl.generate_voice([], "de", sheet)                 # no explicit voice
            check("under Chirp, generate_voice uses the channel voice",
                  seen.get("voice"), "de-DE-Chirp3-HD-Enceladus")
            seen.clear()
            tts.active_engine = lambda cfg: "chatterbox"         # incompatible now
            pl.generate_voice([], "de", sheet)
            check("under Chatterbox, the Chirp channel voice is dropped (uses Settings)",
                  seen.get("voice"), None)
            seen.clear()
            tts.active_engine = lambda cfg: "chirp"
            pl.generate_voice([], "de", sheet, voice="explicit-override")
            check("an explicit per-run voice still wins over the channel voice",
                  seen.get("voice"), "explicit-override")
        finally:
            tts.synth = saved_synth
    finally:
        tts.active_engine = saved_engine

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
