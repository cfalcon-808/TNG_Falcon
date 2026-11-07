#!/usr/bin/env bash
# Create tngviz/.venv and install the visualizer (Linux / macOS).
#   tngviz/setup.sh               board, perfect play and the trained models (CPU torch, about 200 MB)
#   tngviz/setup.sh --no-models   board and perfect play only
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=1
for arg in "$@"; do
    case "$arg" in
        --no-models) MODELS=0 ;;
        -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (use --no-models or --help)" >&2; exit 2 ;;
    esac
done
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv "$DIR/.venv"
PY="$DIR/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$DIR/requirements.txt"
if [ "$MODELS" = 1 ]; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install -r "$DIR/requirements-models.txt"
fi
echo "Done. Start it with: $DIR/run.sh"
