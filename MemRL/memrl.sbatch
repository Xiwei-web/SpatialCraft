#!/usr/bin/env bash
#SBATCH --job-name=memrl-spatial
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --exclude=gpu-05,gpu-[49-56]
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
#SBATCH --output=slurm-memrl-%j.out
set -euo pipefail
umask 077
: "${MEMRL_OUTPUT:?Set MEMRL_OUTPUT to the run directory before submitting}"
project_dir=${MEMRL_PROJECT_ROOT:-/home/xiwei.liu/spatialcraft}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# GPU executes spatial tools; actor and reflection LLM calls use API.
# This batch script enables real execution, including API charges.
exec srun --ntasks=1 --gpus-per-task=a100-sxm4-40gb:1 \
  bash "$project_dir/MemRL/run_memrl.sh" --execute "$@"
