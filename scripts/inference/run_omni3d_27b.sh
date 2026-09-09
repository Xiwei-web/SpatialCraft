#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/home/xiwei.liu/spatialcraft
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
output=${SPATIALCRAFT_RUN_OUTPUT:-/l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_journalfix_v3}
key_file=${SPATIALCRAFT_API_KEY_FILE:-/home/xiwei.liu/.config/spatialcraft/openai_api_key}
snapshot=$output/code_snapshot
launcher=$output/launcher_code
config=$snapshot/configs/experiments/qwen36_27b_omni3d.yaml
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$snapshot/src"
if [[ -z ${SLURM_JOB_ID:-} || -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    echo 'Requires a Slurm allocation with one GPU' >&2
    exit 2
fi
mkdir -p "$output/launcher_logs"
"$python" "$launcher/freeze_run_source.py" --project "$snapshot" --output "$snapshot"
"$python" "$launcher/check_runtime_gpu.py" --expected-gpus 1 --report "$output/launcher_logs/gpu-${SLURM_JOB_ID}.json"
"$python" "$launcher/check_experiment_provider.py" --project "$snapshot" --config "$config" --report "$output/launcher_logs/provider-${SLURM_JOB_ID}.json"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
exec "$python" -m spatialcraft.experiments.run \
    --config "$config" --datasets omni3d --execute \
    --api-key-file "$key_file" --output "$output" \
    --report "$output/preflight.json" \
    > "$output/launcher_logs/${stamp}-${SLURM_JOB_ID}-execution.log" 2>&1
