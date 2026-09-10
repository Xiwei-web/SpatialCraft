#!/usr/bin/env bash
#SBATCH --job-name=gpt54-mean-std
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/xiwei.liu/spatialcraft
set -euo pipefail
umask 077
: "${SPATIALCRAFT_REPEAT_GROUP:?Set the repeat group directory}"
group=$SPATIALCRAFT_REPEAT_GROUP
first=$group/run1
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
export PYTHONPATH="$first/code_snapshot/src"
exec srun --ntasks=1 "$python" "$group/launcher_code/aggregate_api_baselines.py" \
  --runs "$first" "$group/run2" "$group/run3" --output "$group/results"
