#!/usr/bin/env bash
# Run again with the same output and unchanged code/config to resume commits.
set -euo pipefail
umask 077
project=/home/xiwei.liu/spatialcraft
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
output=${SPATIALCRAFT_RUN_OUTPUT:-/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1}
key_file=${SPATIALCRAFT_API_KEY_FILE:-/home/xiwei.liu/.config/spatialcraft/openai_api_key}
snapshot=${SPATIALCRAFT_CODE_SNAPSHOT:-$output/code_snapshot}
if [[ -f "$output/robospatial/code_patch.json" ]]; then
    patched_snapshot=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["patched_snapshot"])' "$output/robospatial/code_patch.json")
    if [[ -n ${SPATIALCRAFT_CODE_SNAPSHOT:-} && "$snapshot" != "$patched_snapshot" ]]; then
        echo 'Explicit snapshot conflicts with recorded bugfix revision' >&2
        exit 2
    fi
    snapshot=$patched_snapshot
fi
cd "$project"
mkdir -p "$output/launcher_logs"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z ${SLURM_JOB_ID:-} || -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    echo 'Run inside a Slurm GPU step, e.g. srun --jobid=YOUR_JOB --overlap bash scripts/inference/run_robospatial.sh' >&2
    exit 2
fi
"$python" scripts/inference/check_runtime_gpu.py --report "$output/launcher_logs/gpu-${SLURM_JOB_ID}.json"
"$python" scripts/inference/freeze_run_source.py --project "$project" --output "$snapshot"
export PYTHONPATH="$snapshot/src"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
exec "$python" -m spatialcraft.experiments.run \
    --datasets robospatial --execute \
    --api-key-file "$key_file" --output "$output" \
    --report "$output/preflight.json" \
    > "$output/launcher_logs/${stamp}-${SLURM_JOB_ID}-$$.log" 2>&1
