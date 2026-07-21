#!/bin/bash
# One-time setup for macOS. Run once, then never again.
#   bash install.sh
set -e

echo ""
echo "  Faceless video pipeline — setup"
echo "  ────────────────────────────────"
echo ""

# --- Homebrew -----------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  echo "  Homebrew is not installed. It's the standard way to install ffmpeg on a Mac."
  echo "  Install it by pasting this into Terminal, then run this script again:"
  echo ""
  echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  echo ""
  exit 1
fi
echo "  ✓ Homebrew found"

# --- ffmpeg -------------------------------------------------------------------
if command -v ffmpeg >/dev/null 2>&1; then
  echo "  ✓ ffmpeg found ($(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3))"
else
  echo "  → Installing ffmpeg (a few minutes)…"
  brew install ffmpeg
  echo "  ✓ ffmpeg installed"
fi

# --- Python -------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "  → Installing Python…"
  brew install python
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ✓ Python $PYV found"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit("  ✗ Python 3.10 or newer is required. Run: brew install python")
PY

# --- python packages ----------------------------------------------------------
# Delegated to setup.sh, which puts everything in a project-local .venv.
# Homebrew's Python refuses system-wide pip installs (PEP 668), and forcing past
# that with --break-system-packages can genuinely break Homebrew.
echo "  → Setting up the Python environment…"
bash setup.sh | sed 's/^/  /'

# --- config -------------------------------------------------------------------
if [ ! -f config.json ]; then
  cp config.example.json config.json
  echo "  ✓ Created config.json"
  NEEDKEYS=1
else
  echo "  ✓ config.json already exists"
fi

mkdir -p sheets cache/stock cache/voice work out music
chmod +x Start.command studio.py make_video.py 2>/dev/null || true
echo "  ✓ Folders ready"

echo ""
echo "  ────────────────────────────────"
if [ -n "$NEEDKEYS" ]; then
  echo "  ONE THING LEFT — free stock image keys:"
  echo ""
  echo "    1. https://www.pexels.com/api/     → sign up, copy your key"
  echo "    2. https://pixabay.com/api/docs/   → sign up, copy your key"
  echo "    3. Run this, and paste them in:    open -e config.json"
  echo ""
  echo "  Both are free forever. Pixabay is the backup for when Pexels has no match."
  echo ""
  echo "  ────────────────────────────────"
fi
echo "  When that's done:  double-click  Start.command"
echo ""
