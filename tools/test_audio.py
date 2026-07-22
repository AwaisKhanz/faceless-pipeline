#!/usr/bin/env python3
"""Freeze the audio mastering wiring — the ffmpeg graphs and the on/off logic.

    python3 tools/test_audio.py

The DSP itself (loudnorm hitting -14 LUFS, the sidechain duck) is verified with a
real ffmpeg render separately; this locks the parts that break silently: the
filter graphs we hand ffmpeg, the two-pass loudnorm command, the JSON parse, and
how config switches read as on/off. ffmpeg is mocked, so it runs anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import render as R, pipeline as pl  # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        bad += not ok
        print(f"  {'ok' if ok else '!!'}  {label:<54}{got}"
              f"{'' if ok else f'  (wanted {want})'}")

    print("\n  config switches read as on/off:")
    check("'auto' is on", pl._flag("auto"), True)
    check("'off' is off", pl._flag("off"), False)
    check("real False is off", pl._flag(False), False)
    check("missing falls back to default", pl._flag(None, True), True)
    check("'no' is off", pl._flag("no"), False)

    # Capture the ffmpeg command instead of running it.
    calls = []
    R.run = lambda cmd, *a, **k: calls.append(cmd)
    R.duration_of = lambda p: 30.0
    real_measure = R._loudnorm_measure       # kept for the JSON-parse test below

    print("\n  music mix ducks under the voice by default:")
    calls.clear()
    R.mix_music(Path("v.wav"), Path("m.wav"), Path("o.wav"), level=0.2, duck=True)
    fc = " ".join(calls[-1])
    check("uses sidechaincompress", "sidechaincompress" in fc)
    check("splits the voice (key + mix)", "asplit=2[v0][v1]" in fc)
    check("pins a stereo layout for the sidechain", "channel_layouts=stereo" in fc)

    print("\n  duck can be turned off (flat mix):")
    calls.clear()
    R.mix_music(Path("v.wav"), Path("m.wav"), Path("o.wav"), level=0.2, duck=False)
    fc = " ".join(calls[-1])
    check("no sidechain when off", "sidechaincompress" not in fc)
    check("still a real amix", "amix=inputs=2" in fc)

    print("\n  mastering: two-pass loudnorm to the target:")
    R._loudnorm_measure = lambda inp, I, TP, LRA: {
        "input_i": "-22.0", "input_tp": "-4.0", "input_lra": "6.0",
        "input_thresh": "-33.0", "target_offset": "0.3"}
    calls.clear()
    info = R.master_audio(Path("mix.wav"), Path("out.wav"), lufs=-14.0)
    af = " ".join(calls[-1])
    check("targets -14 LUFS, -1 dBTP", "loudnorm=I=-14.0:TP=-1.0" in af)
    check("feeds the measured pass-1 values", "measured_I=-22.0" in af and "linear=true" in af)
    check("reports it measured", info["measured"], True)

    print("\n  mastering still runs if the measure pass can't be parsed:")
    R._loudnorm_measure = lambda inp, I, TP, LRA: None
    calls.clear()
    info = R.master_audio(Path("mix.wav"), Path("out.wav"), lufs=-16.0)
    af = " ".join(calls[-1])
    check("single dynamic pass at the target", "loudnorm=I=-16.0" in af)
    check("no measured_ params in the fallback", "measured_I" not in af)
    check("reports it did not measure", info["measured"], False)

    print("\n  the pass-1 JSON is pulled out of ffmpeg's noisy stderr:")
    R._loudnorm_measure = real_measure       # test the real parser, not the stub
    fake_err = ("frame= 100 ...\nsome ffmpeg chatter\n"
                '{\n  "input_i" : "-20.50",\n  "input_tp" : "-3.10",\n'
                '  "input_lra" : "5.0",\n  "input_thresh" : "-30.0",\n'
                '  "target_offset" : "0.10"\n}\ntrailing line\n')
    R.subprocess = SimpleNamespace(
        run=lambda *a, **k: SimpleNamespace(stderr=fake_err, returncode=0))
    m = R._loudnorm_measure(Path("x.wav"), -14, -1, 11)
    check("parsed input_i from stderr", m and m["input_i"], "-20.50")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
