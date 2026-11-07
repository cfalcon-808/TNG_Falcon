#!/usr/bin/env bash
# Launch the Tigers & Goats visualizer with its virtualenv.
# Relative paths (--tables, --game) resolve against the directory you run it from.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Environment missing. Create it with:" >&2
    echo "  python3 -m venv \"$DIR/.venv\" && \"$DIR/.venv/bin/pip\" install -r \"$DIR/requirements.txt\"" >&2
    exit 1
fi
export PYTHONPATH="$(dirname "$DIR")${PYTHONPATH:+:$PYTHONPATH}"
# -P: never import modules from the caller's directory.
exec "$PY" -P -m tngviz "$@"
