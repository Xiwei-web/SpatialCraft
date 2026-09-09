#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/home/xiwei.liu/spatialcraft
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
output=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1
snapshot=$output/code_revisions/output_budget_v1
export PYTHONPATH=$snapshot/src:/l/users/xiwei.liu/tool/env_overlays/qwen_fla_052
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/l/users/xiwei.liu/spatialcraftLog/runtime_cache/fla_triton
cd "$project"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
exec > "$output/launcher_logs/${stamp}-${SLURM_JOB_ID}-budget.log" 2>&1
"$python" scripts/inference/check_runtime_gpu.py --expected-gpus 2 --report "$output/launcher_logs/gpu-${SLURM_JOB_ID}.json"
"$python" scripts/inference/prepare_budget_recovery.py --run "$output" --check
exec "$python" -m spatialcraft.experiments.run_fast_qwen \
  --datasets robospatial --execute \
  --api-key-file /home/xiwei.liu/.config/spatialcraft/openai_api_key \
  --output "$output" --report "$output/preflight-budget.json"
