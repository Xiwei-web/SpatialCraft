#!/usr/bin/env bash
# Default: preflight only. --variant R|GT|both and --output are required.
set -euo pipefail
umask 077
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=${MEMRL_PYTHON:-/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python}
read -r -a datasets <<< "${MEMRL_DATASETS:-robospatial erqa omni3d sat}"
args=(
  --preparation "${MEMRL_PREPARATION:-/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1}"
  --benchmark-root "${MEMRL_BENCHMARK_ROOT:-/l/users/xiwei.liu/benchmark}"
  --api-key-file "${MEMRL_API_KEY_FILE:-/home/xiwei.liu/.config/spatialcraft/openai_api_key}"
  --datasets "${datasets[@]}"
  --model "${MEMRL_MODEL:-gpt-5.4-mini}"
  --reasoning-effort "${MEMRL_REASONING_EFFORT:-medium}"
  --max-output-tokens "${MEMRL_MAX_OUTPUT_TOKENS:-16384}"
  --temperature "${MEMRL_TEMPERATURE:-0}"
  --environment-rollouts "${MEMRL_ENVIRONMENT_ROLLOUTS:-1}"
  --environment-passes "${MEMRL_ENVIRONMENT_PASSES:-1}"
  --max-steps "${MEMRL_MAX_STEPS:-50}"
  --top-k "${MEMRL_TOP_K:-3}"
  --candidate-k "${MEMRL_CANDIDATE_K:-10}"
  --utility-weight "${MEMRL_UTILITY_WEIGHT:-0.5}"
  --learning-rate "${MEMRL_LEARNING_RATE:-0.3}"
  --q-init "${MEMRL_Q_INIT:-0}"
  --threshold-quantile "${MEMRL_THRESHOLD_QUANTILE:-0.8}"
  --embedding-model "${MEMRL_EMBEDDING_MODEL:-text-embedding-3-large}"
  --stage "${MEMRL_STAGE:-all}"
)
if [[ -n ${MEMRL_OUTPUT:-} ]]; then
  args+=(--output "$MEMRL_OUTPUT")
fi
if [[ -n ${MEMRL_VARIANT:-} ]]; then
  args+=(--variant "$MEMRL_VARIANT")
fi
if [[ -n ${MEMRL_SIMILARITY_THRESHOLD:-} ]]; then
  args+=(--similarity-threshold "$MEMRL_SIMILARITY_THRESHOLD")
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_dir"
exec "$python_bin" "$script_dir/run_memrl.py" "${args[@]}" "$@"
