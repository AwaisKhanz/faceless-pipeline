#!/usr/bin/env python3
"""Confirm Openverse authentication is working, without a full sourcing run.

    python tools/openverse_test.py

Reads config.json, mints an access token from your openverse_client_id /
openverse_client_secret (exactly as the pipeline does), then runs one real
search. It tells you whether you are authenticated and whether real results come
back — so you know the setup is right before you source a whole video.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import pipeline as pl        # noqa: E402
from lib import sources as S          # noqa: E402


def main() -> int:
    cfg = pl.load_config()
    S.configure(cfg)

    have_creds = bool(cfg.get("openverse_client_id") and
                      cfg.get("openverse_client_secret"))
    have_token = bool(cfg.get("openverse_token"))
    if not (have_creds or have_token):
        print("No Openverse credentials in config.json.")
        print("Run:  python tools/openverse_register.py you@email.com")
        return 1

    print("Minting an access token from your credentials…")
    token = S._ov_access_token(cfg)
    if not token:
        print("\n✗ Could not get a token.")
        print("  Most likely: you have not clicked the verification link Openverse")
        print("  emailed you yet, or the client_id/secret is mistyped. Verify the")
        print("  email, then run this again.")
        return 1
    print(f"✓ Got a token ({token[:6]}…{token[-4:]}). You are authenticated.\n")

    print("Running one real search for \"Elon Musk\"…")
    try:
        hits = S.openverse("Elon Musk", "IMAGE", 5, cfg)
    except S.SourceError as e:
        print(f"\n✗ The search failed: {e}")
        return 1

    print(f"✓ Openverse returned {len(hits)} usable image(s).")
    for h in hits[:5]:
        who = (h.credit or "")[:40]
        print(f"    · {h.width}×{h.height}  {h.license:<22} {who}")
    if not hits:
        print("  (0 usable — try image_licenses: \"all\" in config for real people,")
        print("   since most photos of named people are CC BY, not CC0.)")
    print("\nAll set — re-source your project and Openverse will pull real people.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
