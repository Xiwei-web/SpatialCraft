#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-omni27b-1gpu
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu-05,gpu-[49-56]
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
output=${SPATIALCRAFT_RUN_OUTPUT:-/l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_journalfix_v3}
exec srun --ntasks=1 --gpus-per-task=1 bash "$output/launcher_code/run_omni3d_27b.sh"
