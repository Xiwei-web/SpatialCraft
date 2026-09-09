#!/usr/bin/env bash
# Run via bash inside a Slurm GPU allocation. This does not request GPUs itself.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Run this script inside a Slurm GPU allocation." >&2
    exit 1
fi

case "${1:-9b}" in
    9b)
        model_path="${QWEN35_9B_PATH:-/l/users/xiwei.liu/model/Qwen3.5-9B}"
        model_alias=qwen3.5-9b
        default_tp=1
        ;;
    27b)
        model_path="${QWEN36_27B_PATH:-/l/users/xiwei.liu/model/Qwen3.6-27B}"
        model_alias=qwen3.6-27b
        default_tp=2
        ;;
    *) echo "Usage: bash serve_qwen.sh [9b|27b]" >&2; exit 2 ;;
esac
if [[ ! -f "$model_path/config.json" ]]; then
    echo "Missing model config: $model_path/config.json" >&2
    exit 1
fi

tp_size="${SPATIALCRAFT_TP_SIZE:-$default_tp}"
if [[ ! "$tp_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "SPATIALCRAFT_TP_SIZE must be a positive integer." >&2
    exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#allocated_gpus[@]} < tp_size )); then
    echo "Need $tp_size allocated GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

# ERQA contains up to 16 images per question. This admission limit alone does not
# guarantee sufficient KV/vision memory: validate the chosen resolution/context.
max_images="${SPATIALCRAFT_MAX_IMAGES:-16}"
if [[ ! "$max_images" =~ ^[1-9][0-9]*$ ]]; then
    echo "SPATIALCRAFT_MAX_IMAGES must be a positive integer." >&2
    exit 2
fi

export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$model_alias" \
    --host 127.0.0.1 \
    --port "${SPATIALCRAFT_MODEL_PORT:-8000}" \
    --tensor-parallel-size "$tp_size" \
    --dtype bfloat16 \
    --max-model-len "${SPATIALCRAFT_MAX_MODEL_LEN:-8192}" \
    --max-num-seqs "${SPATIALCRAFT_MAX_NUM_SEQS:-2}" \
    --gpu-memory-utilization "${SPATIALCRAFT_GPU_MEMORY_UTILIZATION:-0.85}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --limit-mm-per-prompt "{\"image\": $max_images, \"video\": 0}" \
    --enforce-eager
