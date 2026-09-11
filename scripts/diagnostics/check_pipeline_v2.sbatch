#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-v2-pipeline
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
exec /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python \
  scripts/diagnostics/check_pipeline_v2.py \
  --project /home/xiwei.liu/spatialcraft \
  --output "/home/xiwei.liu/spatialcraft/artifacts/v2_validation/pipeline_${SLURM_JOB_ID}" \
  --preparation /l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1 \
  --api-key-file /home/xiwei.liu/.config/spatialcraft/openai_api_key \
  --train-tasks 2
