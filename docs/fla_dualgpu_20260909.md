# Qwen3.5-9B: FLA and two-GPU continuation

## Current state (2026-09-09)

User explicitly requested efficient Qwen linear-attention kernels and two GPUs.
Implementation and dependencies are prepared; **GPU validation is still queued**.
Do not describe the kernels as GPU-validated or the full benchmark as completed.

- Validation job **220740**: 2 × A100-SXM4-40GB, 8 CPUs, 96GB host RAM, 1h limit.
  Scheduler estimate was 17:58:57 Asia/Dubai; this is not a guaranteed start time.
- Formal continuation job **220741**: same two GPUs and host resources, 48h limit;
  `afterok:220740` with cancellation on an invalid dependency. It must not start
  if validation fails. The launcher independently checks acceptance too.
- Earlier pending diagnostic submissions 220725 and 220732 were replaced before
  execution to validate the final immutable source; no other user job was cancelled.
- Existing RoboSpatial progress: 368 training trajectories, 92 Experience updates,
  no deployment result. Resume point: task index92 / rollout0 / step44.

## Isolation and unchanged scientific settings

Kernel overlay: `/l/users/xiwei.liu/tool/env_overlays/qwen_fla_052`.
Pinned `flash-linear-attention==0.5.2`, `fla-core==0.5.2`, and the official
`causal-conv1d==1.7.0` wheel for CPython3.11 / torch2.10 / CUDA12 / CXX11 ABI.
491 installation files are checksummed in `installation_manifest.json`.
Only jobs explicitly adding this directory to PYTHONPATH see these packages.
Shared conda packages were not upgraded; running Omni3D/27B job 220679 is untouched.

References: [FLA](https://github.com/fla-org/flash-linear-attention),
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d),
[Accelerate model placement](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference).

Run root: `/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1`.
Candidate snapshot: `code_revisions/fla_dual_v2`. It is based on the last frozen
RoboSpatial source, not the complete subsequently modified workspace. The unused
`fla_dual_v1` candidate is retained, not overwritten or used for formal results.
Only six selected backend/journal files are added or replaced. No dataset files,
prompts, model weights, quantization, image resolution or experiment YAML change.
BF16, 4096 output/Skill tokens, thinking=false, 50/8 steps, four rollouts/task and
all previously confirmed retrieval/evolution settings remain unchanged.

Placement uses `balanced` with weight-placement limits GPU0=28GiB / GPU1=34GiB.
These are placement budgets, NOT hard caps on runtime allocations. Both GPUs
must contain model modules; CPU/disk offloading and a linear-attention fallback
are rejected. Two cards provide 80GB aggregate, not a contiguous 80GB tensor pool.
Spatial tools remain serial and released between calls. GPU0 retains headroom.

## Validation gates

Evidence root: `/l/users/xiwei.liu/spatialcraftLog/validation/fla-dualgpu-20260909`.

- `tests-final.xml/.log`: 110 passed (workspace full suite).
- `frozen-tests.xml/.log`: 43 passed against the actual candidate snapshot.
- `gpu-check/acceptance.json`: written by GPU validation, initially absent/pending.
  Requires actual FLA-backed layers, both GPUs used, no CPU offload, a numerical
  delta-rule comparison, seeded multimodal replay, parent/candidate fixed-action
  scoring, exact previously OOM input generation with the original 4096 budget,
  and long-context scoring. Records peak allocated memory on each GPU.
- Exact response replay is diagnostic only: it is not added as a formal trajectory.
- Both sides of NEW PPO gates use the new backend. A kernel-tagged probability
  cache prevents mixing old-kernel baseline scores with new candidate scores.
  Earlier committed evolution results remain attributed to their original backend.

Kernel arithmetic is validated with tolerances, not a claim of bitwise identity
to the old implementation. All backend changes have explicit execution provenance.

## Activation, persistence and monitoring

GPU acceptance must match the frozen source manifest and kernel installation.
Only then does `prepare_fla_recovery.py --acceptance ...` validate old stage
commits, archive the old patch metadata, save prior-record checksums and switch
`robospatial/code_patch.json` to the new revision. Until that point the active
formal revision remains `json_output_v1`. All previous snapshots, raw model
responses, trajectories, experiences, skill queues and failed attempts are retained.

The dedicated `run_fast_qwen` entry changes only this process's runtime wiring;
default/other experiments are not automatically switched. Model-local placement
and kernel identity are explicitly bound, while other protocol changes are refused.

```bash
cd /home/xiwei.liu/spatialcraft
squeue -j 220740,220741
/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python scripts/inference/status_robospatial.py /l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1
```

New launcher: `scripts/inference/robospatial_fla.sbatch` →
`scripts/inference/run_robospatial_fla.sh`. It loads the existing explicit private
API key file for embeddings; no key is copied into dependencies, snapshots or logs.
Use the TWO-GPU launcher for subsequent resumes, not the earlier one-GPU launcher.
Never submit a duplicate writer. If GPU validation fails, inspect its log and
repair/revalidate; do not bypass the acceptance or dependency gate.
