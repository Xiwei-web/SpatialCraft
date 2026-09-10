#!/usr/bin/env bash
#SBATCH --job-name=gpt-mini-tool-react
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu-05,gpu-[49-56]
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
umask 077
: "${SPATIALCRAFT_RUN_OUTPUT:?Set the frozen baseline run directory}"
output=$SPATIALCRAFT_RUN_OUTPUT
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$output/code_snapshot/src"
"$python" "$output/launcher_code/freeze_run_source.py" --project "$output/code_snapshot" --output "$output/code_snapshot"
exec srun --ntasks=1 --gpus-per-task=1 "$python" -m spatialcraft.experiments.run_react_baseline \
    --project "$output/code_snapshot" --output "$output" \
    --datasets robospatial erqa omni3d sat \
    --model-config configs/models/gpt-5.4-mini-react.yaml --max-steps 50 --execute
