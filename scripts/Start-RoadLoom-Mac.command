#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SAM_DIR="$HOME/Library/Application Support/RoadTrafficAnalysis/Models/sam3.1-unsloth"

if [[ ! -x "$PY" ]]; then
  echo "Run scripts/Install-RoadLoom-Mac.command first."
  read -r -p "Press Return to close..."
  exit 1
fi

export LOCALAPPDATA="${LOCALAPPDATA:-$HOME/Library/Application Support}"
export ROAD_TRAFFIC_SAM3_DIR="$SAM_DIR"
exec "$PY" "$ROOT/scripts/launch.py"
