#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-fla-check
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
snapshot=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1/code_revisions/fla_dual_v2
export PYTHONPATH=$snapshot/src:/l/users/xiwei.liu/tool/env_overlays/qwen_fla_052
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/l/users/xiwei.liu/spatialcraftLog/runtime_cache/fla_triton
exec srun --ntasks=1 --gpus-per-task=2 /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python scripts/inference/check_fla_dualgpu.py \
  --project "$snapshot" \
  --run /l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1 \
  --output /l/users/xiwei.liu/spatialcraftLog/validation/fla-dualgpu-20260909/gpu-check
