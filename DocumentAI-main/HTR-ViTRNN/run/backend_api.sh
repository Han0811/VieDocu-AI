#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/mpeclab/torch-env/bin/python}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

exec "$PYTHON" -m uvicorn backend.api:app --host "$HOST" --port "$PORT"

