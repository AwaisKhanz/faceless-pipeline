#!/usr/bin/env python3
"""Generate ONE image with Imagen — the cheapest way to confirm it works.

    python tools/imagen_test.py                          # a default prompt
    python tools/imagen_test.py "a busy stock exchange trading floor"

Reads config.json, generates a single 16:9 image (about $0.04 of your Google
Cloud credit — one image, nothing else), and saves it to gen_test.png in the
project folder so you can open it. Run it again for a fresh image.

If this works, image generation is ready: set "generate": "mixed" in config.json
and re-source, and weak scenes will be filled with generated images automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import imagen        # noqa: E402
from lib import pipeline as pl  # noqa: E402


def main(argv: list[str]) -> int:
    subject = argv[0] if argv else "a busy stock exchange trading floor"
    cfg = pl.load_config()

    if not imagen.available(cfg):
        print("Image generation needs \"vertex_project\" in config.json — the same")
        print("Vertex setup the LLM uses. Add it (and vertex_service_account), then")
        print("run this again.")
        return 1

    model = cfg.get("generate_model") or imagen.DEFAULT_MODEL
    location = cfg.get("generate_location") or imagen.DEFAULT_LOCATION
    prompt = imagen.prompt_for(subject, cfg)
    dest = ROOT / "gen_test.png"
    if dest.exists():
        dest.unlink()                      # always make a fresh image

    print(f"Generating one image for:\n  \"{prompt}\"")
    print(f"Model {model} · region {location}")
    print("This spends about $0.04 of your Google Cloud credit…\n")

    try:
        imagen.image(prompt, cfg, dest)
    except imagen.GenError as e:
        print(f"✗ Could not generate: {e}\n")
        print("Common causes:")
        print("  • A 404 usually means the model name — Google retired the old")
        print("    Imagen models in 2026. Use a Gemini image model like")
        print("    gemini-2.5-flash-image (the current default).")
        print("  • If \"global\" 404s, set generate_location to us-central1.")
        print("  • The service account lacks the 'Vertex AI User' role.")
        return 1

    size_kb = dest.stat().st_size // 1024
    print(f"✓ Saved {dest.name} ({size_kb} KB) in the project folder — open it.")
    print("\nHappy with it? Set \"generate\": \"mixed\" in config.json and re-source;")
    print("only scenes that search couldn't match well will be generated (capped by")
    print("generate_max). For the cheapest runs, a fast Imagen model costs about half.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
