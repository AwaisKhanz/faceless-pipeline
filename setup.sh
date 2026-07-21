#!/bin/bash
# Creates a private Python environment for this project and installs what it needs.
#
#   bash setup.sh              # everything the pipeline needs
#   bash setup.sh --cpu-only   # skip the CUDA build even if an NVIDIA card is there
#
# Homebrew's Python refuses system-wide pip installs (PEP 668). Rather than
# forcing past that with --break-system-packages, which really can break
# Homebrew, everything lives in .venv inside this folder. Delete .venv and
# re-run to start over; nothing else on your Mac is touched.
set -e
cd "$(dirname "$0")"

echo ""
echo "  Faceless Studio — Python setup"
echo "  ──────────────────────────────"

# The machine-learning stack lags new Python releases by a long way. Torch may
# publish a build for the newest Python while the packages around it (numba,
# transformers, diffusers) do not, and you get baffling runtime errors rather
# than a clean install failure. So: prefer a version the ecosystem has settled
# on, and only fall back to whatever is available.
PY=""
for v in 3.12 3.11 3.13; do
  if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
  if [ -x "/opt/homebrew/bin/python$v" ]; then PY="/opt/homebrew/bin/python$v"; break; fi
done
if [ -z "$PY" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "  Python 3 not found. Run: brew install python@3.12"
    exit 1
  fi
  PY="python3"
  echo "  ⚠ Using $($PY --version) — no 3.11/3.12/3.13 found."
  echo "    If the voice engine misbehaves, run:  brew install python@3.12"
  echo "    then delete .venv and run this script again."
else
  echo "  ✓ $($PY --version)  (chosen for ML-package compatibility)"
fi

if [ -d .venv ]; then
  HAVE=$(.venv/bin/python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")
  WANT=$($PY -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [ "$HAVE" != "$WANT" ]; then
    echo "  → .venv is Python $HAVE but $WANT is preferred — rebuilding it."
    rm -rf .venv
  fi
fi

if [ ! -d .venv ]; then
  echo "  → Creating .venv …"
  "$PY" -m venv .venv
fi
echo "  ✓ .venv ready"

# All the real work happens in Python — the same script Windows runs, so the
# install logic isn't written twice in two shell dialects that disagree about
# quoting, exit codes and everything else.
.venv/bin/python3 tools/install_deps.py "$@"

# The pip-installed `faceless` command only lands on your PATH when the .venv is
# activated, and almost nobody activates it — they open a terminal here and type
# a command. The ./faceless launcher in this folder needs no activation, and a
# one-line alias makes the bare word work from anywhere. Offer it rather than
# editing a shell file silently.
DIR="$(cd "$(dirname "$0")" && pwd)"
case "${SHELL##*/}" in
  zsh)  RC="$HOME/.zshrc" ;;
  bash) RC="$HOME/.bashrc" ;;
  *)    RC="$HOME/.profile" ;;
esac
LINE="alias faceless='$DIR/faceless'"
echo ""
echo "  ────────────────────────────────────────────────────────────"
echo "  Run it with:   ./faceless start        (from this folder)"
echo ""
if [ -f "$RC" ] && grep -qxF "$LINE" "$RC" 2>/dev/null; then
  echo "  Or just:       faceless start          (alias already set)"
else
  echo "  To type a bare 'faceless' from anywhere, add this alias once:"
  echo ""
  echo "      echo \"$LINE\" >> $RC && source $RC"
  echo ""
  echo "  Then:          faceless start"
fi
echo "  ────────────────────────────────────────────────────────────"
echo ""
