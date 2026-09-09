#!/usr/bin/env bash
# This explicit launcher never changes or stops another experiment.
set -euo pipefail
umask 077
project=/home/xiwei.liu/spatialcraft
python=/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python
output=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1
snapshot=$output/code_revisions/fla_dual_v2
validation=/l/users/xiwei.liu/spatialcraftLog/validation/fla-dualgpu-20260909
export PYTHONPATH=$snapshot/src:/l/users/xiwei.liu/tool/env_overlays/qwen_fla_052
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/l/users/xiwei.liu/spatialcraftLog/runtime_cache/fla_triton
cd "$project"
"$python" scripts/inference/check_runtime_gpu.py --expected-gpus 2 --report "$output/launcher_logs/gpu-${SLURM_JOB_ID}.json"
"$python" -c 'import sys,xml.etree.ElementTree as E; s=[x for p in sys.argv[1:] for x in E.parse(p).getroot().iter("testsuite")]; assert s and all(int(x.get("failures",0))==0 and int(x.get("errors",0))==0 for x in s)' "$validation/tests-final.xml" "$validation/frozen-tests.xml"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
exec > "$output/launcher_logs/${stamp}-${SLURM_JOB_ID}-fla.log" 2>&1
"$python" scripts/inference/prepare_fla_recovery.py --run "$output" --acceptance "$validation/gpu-check/acceptance.json"
exec "$python" -m spatialcraft.experiments.run_fast_qwen \
  --datasets robospatial --execute \
  --api-key-file /home/xiwei.liu/.config/spatialcraft/openai_api_key \
  --output "$output" --report "$output/preflight-fla.json"
