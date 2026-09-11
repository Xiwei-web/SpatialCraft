#!/usr/bin/env bash
# Offline preparation unless the caller explicitly adds --execute.
set -euo pipefail
umask 077
rag_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rag_python="${SPATIALCRAFT_PYTHON:-/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python}"
exec "$rag_python" "$rag_dir/run_rag.py" --model gpt-5.4-mini "$@"
