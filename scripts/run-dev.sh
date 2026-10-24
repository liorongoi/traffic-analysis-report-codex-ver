#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
exec "${TRAFFIC_PYTHON:-python3}" -m road_traffic serve "$@"
