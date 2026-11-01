#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
SAM_DIR="$HOME/Library/Application Support/RoadTrafficAnalysis/Models/sam3.1-unsloth"

if [[ ! -x "$VENV/bin/python" ]]; then
  PYTHON=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        PYTHON="$(command -v "$candidate")"
        break
      fi
    fi
  done
  if [[ -z "$PYTHON" ]]; then
    echo "Install Python 3.11 or 3.12, then run this installer again."
    read -r -p "Press Return to close..."
    exit 1
  fi
  "$PYTHON" -m venv "$VENV"
fi

PY="$VENV/bin/python"
if ! "$PY" -c 'import av, cv2, numpy, onnxruntime, PIL, watchdog, tzdata' >/dev/null 2>&1; then
  "$PY" -m pip install --disable-pip-version-check -r "$ROOT/requirements-inference-macos.lock.txt"
fi

mkdir -p "$SAM_DIR"
echo "Installing anonymous SAM 3.1 BF16 weights and CPU runtime. This needs internet once and several GB of free disk space."
echo "No Hugging Face account, sign-in, or token is used. After setup, SAM loads from this Mac's local files."
"$PY" "$ROOT/scripts/setup-sam3.1.py" --destination "$SAM_DIR"

echo
echo "RoadLoom and SAM 3.1 are prepared at: $ROOT"
echo "Run bash scripts/Start-RoadLoom-Mac.command to launch RoadLoom."
echo "SAM uses CPU in this version; CUDA is not available on macOS."
read -r -p "Press Return to close..."
