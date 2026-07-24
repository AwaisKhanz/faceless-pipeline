#!/usr/bin/env python3
"""Register a free Openverse API application and print the credentials.

Openverse (which mirrors Wikimedia Commons — i.e. photos of real, named people)
rate-limits anonymous callers hard. Registering lifts that limit. It is free,
takes one minute, and the credentials never expire.

    python tools/openverse_register.py you@email.com

It prints a client_id and client_secret. Paste them into config.json as
"openverse_client_id" and "openverse_client_secret", then click the verification
link Openverse emails you. That is all — the app mints and refreshes the short-
lived access token from these credentials on its own.

Nothing here touches config.json or stores anything: it makes one request and
prints the result, so you stay in control of where the secret goes.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

REGISTER = "https://api.openverse.org/v1/auth_tokens/register/"


def main(argv: list[str]) -> int:
    if len(argv) != 1 or "@" not in argv[0]:
        print("Usage: python tools/openverse_register.py you@email.com")
        return 2
    email = argv[0].strip()

    body = json.dumps({
        "name": f"faceless-studio-{email.split('@')[0]}",
        "description": "Personal faceless-video pipeline; sourcing CC/PD images.",
        "email": email,
    }).encode()
    req = urllib.request.Request(
        REGISTER, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "faceless-studio (openverse register)"})

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        print(f"Registration failed (HTTP {e.code}).\n{detail}")
        if e.code == 429:
            print("\nYou have registered too many times recently — wait a while.")
        return 1
    except Exception as e:
        print(f"Could not reach Openverse to register: {e}")
        print("If you are on a network that blocks it, try again on another one.")
        return 1

    cid = data.get("client_id", "")
    csec = data.get("client_secret", "")
    if not (cid and csec):
        print("Openverse did not return credentials. Full response:")
        print(json.dumps(data, indent=2))
        return 1

    print("\n  Openverse application registered.\n")
    print("  Paste these two lines into config.json:\n")
    print(f'    "openverse_client_id": "{cid}",')
    print(f'    "openverse_client_secret": "{csec}",')
    print("\n  Then check your inbox and click the verification link Openverse")
    print(f"  sent to {email} — the higher rate limit turns on once verified.")
    print("\n  Keep the client_secret private; anyone with it can use your quota.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
