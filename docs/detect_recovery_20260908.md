# RoboSpatial detector recovery (2026-09-08)

Job 214256 exited at training task index 28, rollout 0, environment step 5.
It completed 112/700 training rollouts and 28 Experience updates; no deployment
result exists. This is an implementation failure, not a measured wrong answer.

## Cause and fix

The 1920×1440 training image produced a GroundingDINO box with bottom coordinate
1439.266566. Our inclusive pixel convention requires y ≤ 1439. The adapter
previously passed raw predictions directly to the strict overlay renderer.

`tools/real/detect.py` now clips finite, ordered model boxes to image bounds,
filters outside/degenerate/nonfinite boxes, and preserves an explicit
`bbox_postprocessing` audit. It does not reorder inverted boxes, change detector
confidence, loosen generic coordinate validation, or turn infrastructure failures
into reward zero. Filtering happens before the requested result limit.

Real GPU replay of the exact failed call reproduced the original exception and
then successfully persisted four detections and the corrected overlay. No LLM,
embedding, test labels or test images were used for this diagnostic. Evidence:
`/l/users/xiwei.liu/spatialcraftLog/validation/detect-recovery-20260908/real-detect.json`.
Full regression: 83 passed, including clipping/filtering and code-patch resumption.

## Preserved history and resume

Run root: `/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1`.

- Original `code_snapshot/`, `robospatial/journal.json` and failed attempts remain unchanged.
- `code_revisions/detect_bounds_v1/` is the new immutable source snapshot.
- Only `tools/real/detect.py` and `experiments/journal.py` differ from original source.
- `robospatial/code_patch.json` records the exact before/after code hashes and reason.
- 4,583 stage commits validated; 14,359 pre-existing JSON records preserved unchanged
  during patch preparation. Their hashes are in `code_revisions/detect_bounds_v1-before.json`.
- Journal accepts old stage results only under this explicit code-only lineage;
  all data, weights, dependencies, hyperparameters and stage inputs must still match.
- Newly committed results carry the new binding digest. Earlier results remain
  attributable to the old version; this is a bugfix continuation, not a fresh run
  entirely executed with patched code.
- The launcher automatically selects the revision in `code_patch.json` and rejects
  an explicit conflicting snapshot. Do not edit either frozen source or patch metadata.

Recovery job **215987** submitted with the same 48-hour GPU batch resources.
First attempt 215984 received system signal 53 on gpu-51 after three seconds,
before application logs were created; its scheduler evidence is preserved under
the validation directory. Current submission excludes gpu-05 and gpu-49 through
gpu-56. It is queued awaiting resources; submission is not proof of resumption
or benchmark completion. Check the current state:

```bash
cd /home/xiwei.liu/spatialcraft
squeue -j 215987
/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python scripts/inference/status_robospatial.py /l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1
```

Only after the current job stops, resume using:

```bash
sbatch --export=ALL,SPATIALCRAFT_RUN_OUTPUT=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1 --exclude='gpu-05,gpu-[49-56]' --output=/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1/launcher_logs/slurm-%j.log scripts/inference/robospatial.sbatch
```

Never submit duplicate writers. Existing completed work is reused, while the first
uncommitted detection stage and the remainder of the interrupted rollout are retried.
