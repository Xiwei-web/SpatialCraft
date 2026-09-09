#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-robo-budget2
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
exec srun --ntasks=1 --gpus-per-task=2 bash scripts/inference/run_robospatial_budget.sh
