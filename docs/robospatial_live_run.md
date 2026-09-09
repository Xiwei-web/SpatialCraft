# RoboSpatial / Qwen3.5-9B execution

Formal default output: `/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1`.
Training-only integration diagnostics are separate, under
`/l/users/xiwei.liu/spatialcraftLog/validation/robospatial-live-20260907/`.
Diagnostic snapshots are never imported into the formal experiment.

Protocol: fixed seed-42 prepared split, 175 environment tasks × four independent
rollouts, then 175 deployment tasks with the frozen final knowledge snapshot.
See `confirmed_protocol.md` for all sampling, evolution, thinking and reward rules.
Current image budget is 1,048,576 pixels. No ERQA/Omni3D execution in this run.

## Launch or resume

Inside the project directory, with an existing GPU allocation:

```bash
srun --jobid=YOUR_JOB_ID --overlap --nodes=1 --ntasks=1 \
  bash scripts/inference/run_robospatial.sh
```

Alternatively invoke the same script inside a GPU batch allocation. The script
loads only the explicit private file `~/.config/spatialcraft/openai_api_key`;
never paste keys into this document or experiment logs. The file must be owned by
the current user and inaccessible to group/others. A missing/empty key stops the
run. This is a paid OpenAI embedding call; generation/PPO run on the local Qwen.

Resumption uses the **same output directory and identical code, configurations,
weights and input data**. The launcher freezes `src/`, `configs/`, and `prompts/`
in `code_snapshot/` on first launch and executes that checked copy thereafter,
unless an explicit audited bugfix revision is recorded in `robospatial/code_patch.json`.
The launcher then validates and executes that separately preserved revision.
Workspace edits therefore do not silently change an existing experiment. Changes
to the frozen copy fail validation; never edit the journal to bypass it.
New launcher logs do not overwrite prior logs. Only this experiment
process may write the run at a time. GPU allocation expiry does not delete commits.

## Records

- `launcher_logs/`: per-launch stdout/stderr; START/COMMIT/FAILED stage markers.
- `code_snapshot/`: immutable source/config/prompts and checksums for resumption.
- `embedding_cache/`: reusable vectors and API usage records, no API keys.
- `robospatial/journal.json`: immutable input/code/model/protocol binding.
- `robospatial/stages/`: per-step model calls, exact inputs/outputs, tools,
  rewards, PPO scoring and committed knowledge updates; failed-attempt status.
- `robospatial/rollouts/`: complete training/deployment trajectories.
- `robospatial/experiences/`: task-level Experience Bank changes.
- `robospatial/skills/`: evolution rounds, candidates, gate decisions and queues.
- `robospatial/checkpoints/`: task/round snapshots, pending evolution queues,
  and final frozen deployment snapshot.
- `robospatial/results/deployment.json`: final aggregate accuracy and category
  metrics; only this completed formal deployment is a benchmark result.

Atomic stage commits are reused after interruption; an external request interrupted
before commit may be repeated and charged again. Uncommitted work is retried, not
silently recorded as incorrect. Infrastructure/model-format failures are not
treated as measured model accuracy. Do not equate an existing log with completion.

## Last verified status: 2026-09-08

17:26 update: job 217328 RUNNING on gpu-04; original JSON failure passed in the
resumed evolution. 28,371 prior JSON records rechecked unchanged after resumption.

Latest 17:23 Asia/Dubai: job 215987 advanced to 220 training rollouts and 55
Experience updates, then stopped on malformed semantic-gradient JSON. The exact
cached response now parses after bounded syntax normalization; 98 tests pass.
New job 217328 submitted (initially PENDING); active snapshot is
`code_revisions/json_output_v1`. See `json_recovery_20260908.md` for current
monitor/resume commands and chained source provenance. Older incident below:

Formal job 214256 stopped after 112 training rollouts due to a detector box
exceeding the image boundary. Exact-image GPU replay now passes after repair;
83 regression tests pass. Recovery job 215987 queued awaiting resources; first
recovery allocation 215984 failed before application startup on gpu-51.
All prior commits and the original source are preserved; see
`detect_recovery_20260908.md` for provenance and resumption. No full benchmark
result yet. Earlier job 214245 failed CUDA initialization before generation.

Historical initial thinking pilots stopped at
the first task decomposition because the entire 1,024-token output was consumed
by thinking (also after adding a concise-thinking instruction). No complete
trajectory, learned experience, evolved skill, or accuracy result was produced.
These diagnostic journals remain intact as `pilot-v1` and `pilot-v2`.

The latest user instruction supersedes the intervening 32,768-token thinking trial:
**instruct mode, enable_thinking=false, max_output_tokens=4096 and
skill_generation_max_tokens=4096**, uniformly across executor and knowledge calls.
PPO scores action tokens directly with no sampled thinking prefix. Old thinking
steps were stopped without cancelling the user's interactive allocation.
The latest `pilot-instruct4096-v2` completed 8 real trajectories, two Experience
updates, one six-trajectory evolution batch, three candidates and three real PPO
gate records. All candidates were rejected by the strict gate (not forced in).
Intentional interruption/resumption left all 165 prior committed files unchanged.
Acceptance is recorded in its `acceptance.json`. No diagnostic knowledge is
imported into the formal run. Full regression suite: 79 passed.

Verified independently:

- Private OpenAI credential file loaded successfully; one embedding probe returned
  1,536 dimensions and reported eight input tokens.
- 42 focused regression tests passed (`regression-after-tools.log`).
- All six neural tools passed actual `SerialToolExecutor` calls and artifact
  persistence on a public environment image (`tools-executor-v2.json` and `.log`):
  detect, segment, scale, reconstruct, pose, ocr. This is NOT combined-Qwen testing.
- Core torch 2.10.0 / transformers 5.5.4 / numpy 2.2.6 / Pillow 10.4.0 retained.
  Dependency installation logs and before/after environment records are in the
  same validation directory.

Compatibility fixes: local BERT path and old-BERT API adaptation for GroundingDINO;
upstream PyTorch CUDA deformable-attention fallback when the optional extension is
absent; SAM3 BF16 autocast; and the 722-output small orientation checkpoint's
matching decoder from [official historical inference.py](https://github.com/SpatialVision/Orient-Anything/blob/b62f5328505648be5fe9b24acb149a3df865365b/inference.py).
No test samples or labels were used for these diagnostics.
