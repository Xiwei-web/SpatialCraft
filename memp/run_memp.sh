#!/usr/bin/env bash
# Default: preflight only. Pass --execute explicitly to make model/tool calls.
set -euo pipefail
umask 077
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=${MEMP_PYTHON:-/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python}
read -r -a datasets <<< "${MEMP_DATASETS:-robospatial erqa omni3d sat}"
args=(
  --preparation "${MEMP_PREPARATION:-/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1}"
  --benchmark-root "${MEMP_BENCHMARK_ROOT:-/l/users/xiwei.liu/benchmark}"
  --api-key-file "${MEMP_API_KEY_FILE:-/home/xiwei.liu/.config/spatialcraft/openai_api_key}"
  --datasets "${datasets[@]}"
  --model "${MEMP_MODEL:-gpt-5.4}"
  --reasoning-effort "${MEMP_REASONING_EFFORT:-none}"
  --max-output-tokens "${MEMP_MAX_OUTPUT_TOKENS:-4096}"
  --temperature "${MEMP_TEMPERATURE:-0}"
  --environment-rollouts "${MEMP_ENVIRONMENT_ROLLOUTS:-1}"
  --top-k "${MEMP_TOP_K:-3}"
  --memory-format "${MEMP_MEMORY_FORMAT:-proceduralization}"
  --embedding-model "${MEMP_EMBEDDING_MODEL:-text-embedding-3-large}"
  --stage "${MEMP_STAGE:-all}"
  --max-steps "${MEMP_MAX_STEPS:-50}"
)
if [[ -n ${MEMP_OUTPUT:-} ]]; then
  args+=(--output "$MEMP_OUTPUT")
fi
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_dir"
exec "$python_bin" "$script_dir/run_memp.py" "${args[@]}" "$@"
