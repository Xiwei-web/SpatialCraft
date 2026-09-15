#!/usr/bin/env bash
#SBATCH --job-name=memp-spatial
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
#SBATCH --output=slurm-memp-%j.out
set -euo pipefail
umask 077
: "${MEMP_OUTPUT:?Set MEMP_OUTPUT to the run directory before submitting}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# One GPU is reserved for local spatial tools; the agent and script LLM use API.
# Submitting this file opts into execution and potential API charges.
exec srun --ntasks=1 --gpus-per-task=1 \
  bash /home/xiwei.liu/spatialcraft/memp/run_memp.sh --execute "$@"
