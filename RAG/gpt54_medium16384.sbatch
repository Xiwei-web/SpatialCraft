#!/usr/bin/env bash
#SBATCH --job-name=rag-gpt54-med16k
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
umask 077
: "${SPATIALCRAFT_RUN_OUTPUT:?Set the independent RAG output directory}"
rag_output="$SPATIALCRAFT_RUN_OUTPUT"
rag_snapshot="$rag_output/code_snapshot"
rag_python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$rag_snapshot/src:$rag_snapshot"
"$rag_python" "$rag_snapshot/RAG/freeze_run.py" --project "$rag_snapshot" --output "$rag_snapshot"
exec srun --ntasks=1 "$rag_python" "$rag_snapshot/RAG/run_rag.py" \
    --model gpt-5.4 --reasoning-effort medium --max-output-tokens 16384 \
    --environment-rollouts 4 --top-k 3 --embedding-model text-embedding-3-large \
    --datasets robospatial erqa omni3d sat --stage all \
    --output "$rag_output" --execute
