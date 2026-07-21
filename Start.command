#!/bin/bash
# Double-click this file to open Faceless Studio.
cd "$(dirname "$0")"

# Prefer the project's own virtual environment. Homebrew's Python blocks
# system-wide pip installs (PEP 668), so this project's packages live in .venv.
if [ -x ".venv/bin/python3" ]; then
  PY="$PWD/.venv/bin/python3"
else
  PY="$(command -v python3)"
fi

if [ -z "$PY" ]; then
  echo ""
  echo "  Python isn't installed yet."
  echo "  Run this once in Terminal, then double-click Start again:"
  echo ""
  echo "    brew install python && bash setup.sh"
  echo ""
  read -n 1 -s -r -p "  Press any key to close."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ""
  echo "  ffmpeg isn't installed yet."
  echo "  Run this once in Terminal, then double-click Start again:"
  echo ""
  echo "    brew install ffmpeg"
  echo ""
  read -n 1 -s -r -p "  Press any key to close."
  exit 1
fi

if ! "$PY" -c "import chatterbox" >/dev/null 2>&1; then
  echo ""
  echo "  The voice engine isn't set up for this project yet."
  echo "  Run this once in Terminal, then double-click Start again:"
  echo ""
  echo "    bash setup.sh"
  echo ""
  read -n 1 -s -r -p "  Press any key to close."
  exit 1
fi

"$PY" studio.py

echo ""
read -n 1 -s -r -p "  Studio closed. Press any key to close this window."
