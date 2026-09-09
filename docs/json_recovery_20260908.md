# RoboSpatial knowledge JSON recovery (2026-09-08)

## Incident and bounded repair

Job 215987 stopped during `evolution/00054/skill_evolution`, after 220 training
rollouts and 55 Experience updates (54 completed evolution rounds). No deployment
rollout or final accuracy exists at this point.

The semantic-gradient response ended normally after 189 output tokens, but its
last `termination` string lacked a closing quote before a newline and the final
object brace. This was a syntax failure, not output-budget exhaustion.

`experiments/learning.py::_json_object` now:

- Parses valid JSON as before, including a complete Markdown JSON fence.
- Allows literal control characters inside strings without deleting their content.
- Repairs only a missing final value quote before an existing final object brace
  on its own line, and requires the entire repaired object to parse successfully.
- Rejects ambiguous/truncated structures, extra prose, duplicate keys and NaN/Infinity.
- Logs normalization operations plus raw-text and parsed-object digests.

There is no broad regex extraction, fabricated field, replacement diagnosis,
fallback reward, unlimited retry or extra LLM call in this repair. Raw model
responses remain immutable. Other malformed structures still stop safely rather
than silently creating invalid knowledge.

## Verification

Evidence: `/l/users/xiwei.liu/spatialcraftLog/validation/json-recovery-20260908/`.

- `replay.json`: the exact cached failed response now yields all four diagnosis
  fields; its raw file is unchanged. 1,498 historical valid JSON objects yield
  identical parsed content. This diagnostic uses no GPU/LLM/API calls.
- `tests.log`, `tests.xml`: full suite **98 passed**, including literal controls,
  missing terminal quotes, rejection cases and two-revision checkpoint lineage.
- `prepare.log`: existing-commit validation and preserved-record counts.

## Versioned continuation

Run root remains
`/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1`.
The next immutable snapshot is `code_revisions/json_output_v1/`. Its only changes
from `detect_bounds_v1` are `experiments/learning.py` and `experiments/journal.py`.
The detector fix remains included; no configs, prompts or hyperparameters change.

`robospatial/code_patch.json` becomes the active revision pointer and records the
ordered prior patch history. Its previous exact contents are archived as
`code_revisions/json_output_v1-parent_patch.json`. Original journal/source,
earlier snapshots, stage inputs/results and failed attempts are not rewritten.
`code_revisions/json_output_v1-before.json` records prior JSON checksums, excluding
only the active patch pointer that is deliberately updated.

The journal accepts inherited commits from their recorded code versions, rejects
broken revision chains and backwards commit versions, and still requires exact
data/model/settings/input matches. The launcher automatically selects the active
recorded snapshot. Do not edit frozen sources or submit duplicate writers.

The resumed evolution uses the cached malformed response through the corrected
parser; prior 220 rollouts and 55 Experience updates are not regenerated. New model
calls for the remaining evolution and subsequent tasks still use the confirmed
Qwen3.5-9B instruct / 4096 / thinking=false protocol and paid OpenAI embeddings.

This is an explicitly versioned bugfix continuation, not a claim that all earlier
trajectories were generated with the latest implementation.

## Current submission

Recovery job **217328**, submitted 2026-09-08 17:22:56 Asia/Dubai, is now RUNNING
on gpu-04. The CUDA kernel check passed and the cached malformed response was
successfully normalized at 17:24:31; subsequent model calls are committing.
All 28,371 pre-recovery JSON files were checked again after real resumption and
remain unchanged. Resources remain one A100 GPU, 8 CPUs, 64 GB host RAM and 48 hours.
9,061 old stage commits validated; 28,371 prior JSON files (excluding the archived
and updated active patch pointer) verified unchanged during revision installation.
There is no final test accuracy yet. Query live status rather than assuming that
submission means the interrupted evolution has already completed:

```bash
cd /home/xiwei.liu/spatialcraft
squeue -j 217328
/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python scripts/inference/status_robospatial.py /l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1
```

Only if that job has stopped and the final result is absent, submit a resume:

```bash
sbatch --export=ALL,SPATIALCRAFT_RUN_OUTPUT=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1 --exclude='gpu-05,gpu-[49-56]' --output=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1/launcher_logs/slurm-%j.log scripts/inference/robospatial.sbatch
```
