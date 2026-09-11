#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-v2-tools
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/xiwei.liu/spatialcraft/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
exec /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python scripts/diagnostics/check_tools_v2.py --output "/home/xiwei.liu/spatialcraft/artifacts/v2_validation/tools_${SLURM_JOB_ID}" --tool-timeout 240 --total-timeout 1500
