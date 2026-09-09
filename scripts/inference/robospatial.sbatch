#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-robo
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
# Submit only AFTER the independent training-only pilot passes.
set -euo pipefail
exec srun --ntasks=1 --gpus-per-task=1 bash scripts/inference/run_robospatial.sh
