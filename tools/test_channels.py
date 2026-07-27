#!/usr/bin/env python3
"""Channels: assign/move, create-on-assign, rename, delete (keeps projects),
stale-pruning, and forget-on-delete.

    python3 tools/test_channels.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import channels as CH   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '}{label:<52}{'' if ok else repr(got)}")
        bad += not ok

    tmp = pathlib.Path(tempfile.mkdtemp())
    CH.FILE = tmp / "channels.json"
    CH.IMG_DIR = tmp / "channel_images"        # keep test avatars out of the real cache

    print("\n  assign / create-on-assign / list:")
    CH.create("Superfruits")
    CH.assign("v1", "Superfruits")
    CH.assign("v2", "History")                      # new channel made on assign
    check("channels listed (create + create-on-assign)", CH.names(), ["Superfruits", "History"])
    check("v1 -> Superfruits", CH.of("v1"), "Superfruits")
    check("v2 -> History", CH.of("v2"), "History")
    check("unassigned project -> ''", CH.of("v3"), "")

    print("\n  move and clear:")
    CH.assign("v1", "History")                      # move
    check("v1 moved to History", CH.of("v1"), "History")
    CH.assign("v1", "")                             # clear
    check("v1 cleared", CH.of("v1"), "")

    print("\n  rename keeps assignments:")
    CH.rename("History", "Old facts")
    check("channel renamed", "Old facts" in CH.names() and "History" not in CH.names(), True)
    check("v2 followed the rename", CH.of("v2"), "Old facts")

    print("\n  delete channel keeps its projects (just unassigns):")
    CH.delete = CH.remove
    CH.remove("Old facts")
    check("channel gone", "Old facts" not in CH.names(), True)
    check("v2 unassigned, not deleted", CH.of("v2"), "")

    print("\n  stale assignments are pruned to real projects:")
    CH.assign("ghost", "Superfruits")
    d = CH.data(valid_pids=["v1", "v2"])            # 'ghost' isn't real
    check("ghost pruned", "ghost" not in d["assign"], True)

    print("\n  forget_project on delete:")
    CH.assign("v1", "Superfruits")
    CH.forget_project("v1")
    check("assignment forgotten", CH.of("v1"), "")

    print("\n  channel profile (image / email / description):")
    CH.create("Superfruits")
    CH.set_meta("Superfruits", email="hi@fruit.tv", description="Fruit facts")
    m = CH.meta_of("Superfruits")
    check("email saved", m["email"], "hi@fruit.tv")
    check("description saved", m["description"], "Fruit facts")
    check("partial update keeps the other field",
          (CH.set_meta("Superfruits", email="new@fruit.tv"),
           CH.meta_of("Superfruits")["description"])[1], "Fruit facts")
    check("editing a not-yet-created channel creates it",
          (CH.set_meta("Fresh", description="x"), "Fresh" in CH.names())[1], True)

    print("\n  image save + rename carries the profile + delete drops it:")
    fn = CH.set_image("Superfruits", b"\x89PNGfake", "png")
    check("image filename recorded", CH.meta_of("Superfruits")["image"], fn)
    check("image file written to disk", (CH.IMG_DIR / fn).exists(), True)
    CH.rename("Superfruits", "Superfoods")
    check("rename carried the email", CH.meta_of("Superfoods")["email"], "new@fruit.tv")
    check("rename carried the image", CH.meta_of("Superfoods")["image"], fn)
    check("old name has no profile", CH.meta_of("Superfruits")["email"], "")
    CH.remove("Superfoods")
    check("delete dropped the profile", CH.meta_of("Superfoods")["email"], "")
    check("delete tidied the image file", (CH.IMG_DIR / fn).exists(), False)

    print("\n  an OLD file with no meta block still loads (backward compatible):")
    CH.FILE.write_text('{"channels":["Legacy"],"assign":{"v9":"Legacy"}}', encoding="utf-8")
    check("legacy channels load", "Legacy" in CH.names(), True)
    check("legacy assignment loads", CH.of("v9"), "Legacy")
    check("legacy meta is empty, no crash", CH.meta_of("Legacy")["email"], "")

    print("\n  tolerates a corrupt file:")
    CH.FILE.write_text("{ this is not json", encoding="utf-8")
    check("corrupt file -> empty, no crash", CH.names(), [])

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
