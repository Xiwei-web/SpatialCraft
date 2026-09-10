#!/usr/bin/env bash
#SBATCH --job-name=spatialcraft-gpt54-baseline
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
umask 077
: "${SPATIALCRAFT_RUN_OUTPUT:?Set the frozen baseline run directory}"
output=$SPATIALCRAFT_RUN_OUTPUT
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$output/code_snapshot/src"
"$python" "$output/launcher_code/freeze_run_source.py" --project "$output/code_snapshot" --output "$output/code_snapshot"
exec srun --ntasks=1 "$python" -m spatialcraft.experiments.run_api_baseline \
    --project "$output/code_snapshot" --output "$output" \
    --model-config configs/models/gpt-5.4-baseline.yaml --execute
