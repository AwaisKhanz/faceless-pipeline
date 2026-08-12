#!/usr/bin/env python3
"""The global Video/Image preference (media_mode) forces every scene's media
before sourcing — 'mixed' leaves the script's per-scene choice alone.

    python3 tools/test_media_mode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pipeline as pl        # noqa: E402
from lib import config_schema as CS   # noqa: E402
from lib.sheet import Scene           # noqa: E402


def _scenes():
    # A mix, as the writer would produce it.
    return [
        Scene(n=1, media="IMAGE", narration="a", query="a"),
        Scene(n=2, media="VIDEO", narration="b", query="b"),
        Scene(n=3, media="IMAGE", narration="c", query="c"),
    ]


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<52}{'' if ok else repr(got)}")
        bad += not ok

    print("\n  media_mode() normalises the setting:")
    check("default / empty -> mixed", pl.media_mode({}), "mixed")
    check("a bogus value -> mixed", pl.media_mode({"media_mode": "gif"}), "mixed")
    check("image kept", pl.media_mode({"media_mode": "image"}), "image")
    check("VIDEO (any case) -> video", pl.media_mode({"media_mode": "VIDEO"}), "video")

    print("\n  mixed leaves the writer's per-scene choice untouched:")
    got = [s.media for s in pl.apply_media_mode(_scenes(), {"media_mode": "mixed"})]
    check("mixed keeps IMAGE/VIDEO/IMAGE", got, ["IMAGE", "VIDEO", "IMAGE"])

    print("\n  image forces every scene to a still:")
    got = [s.media for s in pl.apply_media_mode(_scenes(), {"media_mode": "image"})]
    check("all IMAGE", got, ["IMAGE", "IMAGE", "IMAGE"])

    print("\n  video forces every scene to a clip:")
    got = [s.media for s in pl.apply_media_mode(_scenes(), {"media_mode": "video"})]
    check("all VIDEO", got, ["VIDEO", "VIDEO", "VIDEO"])

    print("\n  the Settings schema exposes it as a closed select:")
    F = {f["key"]: f for f in CS._FIELDS}
    check("media_mode is in the schema", "media_mode" in F, True)
    check("a valid option passes", CS._coerce(F["media_mode"], "video"), "video")
    check("an invalid option is rejected",
          "media_mode" in CS.validate_and_merge({}, {"media_mode": "nope"})[1], True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
