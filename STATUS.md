# SpatialCraft implementation status

Latest 2026-09-09: FLA/two-GPU profile prepared in isolated dependency overlay;
110 workspace tests and 43 frozen-source tests passed. GPU acceptance job 220740
is queued; formal continuation 220741 depends on its success. No claim of GPU
acceptance or benchmark completion yet. Candidate `fla_dual_v2` is NOT activated
until exact-source/long-input/PPO validation passes. See `docs/fla_dualgpu_20260909.md`.

Last updated: 2026-09-08

Live update 17:26 Asia/Dubai: job 217328 RUNNING on gpu-04, CUDA precheck passed.
Original cached JSON failure passed during actual evolution resumption; subsequent
model calls are committing. All 28,371 prior JSON records verified unchanged again
after actual resumption. No final deployment result yet.

Latest (17:23 Asia/Dubai): job 215987 reached 220 training rollouts / 55 Experience
updates, then failed parsing a semantic-gradient JSON string missing its closing
quote. Exact cached-response replay now passes and 1,498 valid historical JSON
objects parse identically; 98 tests pass. New job 217328 submitted (initially
PENDING). Active snapshot `code_revisions/json_output_v1` includes both fixes.
9,061 commits and 28,371 prior JSON files verified unchanged; old patch metadata
archived before updating the revision pointer. No deployment results. Details:
`docs/json_recovery_20260908.md`. The recovery notes below are historical.

Current recovery: original job 214256 stopped after 112 training rollouts and
28 Experience updates due to a GroundingDINO bounding box exceeding the image
boundary. Exact-image GPU replay passes after repair; 83 full-suite tests pass.
Recovery job 215987 is queued (first allocation 215984 failed before application
startup). Original source, 4,583 stage commits and 14,359 prior JSON records were
preserved and validated. See `docs/detect_recovery_20260908.md`. No deployment
accuracy exists yet. The September 7 acceptance status below is historical.

## Current acceptance scope

- Current instruct protocol: four frozen rollouts/task; per-parent FIFO batches of six distinct related trajectories; at most two parents per task barrier; 50/8 environment steps; 4096 output/Skill-generation tokens; thinking disabled; action-only PPO; frozen deployment and resumable pending queues. This supersedes the historical thinking trials.
- New tests cover original-task advantages, parent/segment isolation, budget and fixed-thinking conditional action masking. See `spatialcraftLog/validation/protocol-v2-20260907/unit-tests.xml`. The older 62-test v1 report is retained separately.
- Real Qwen3.5-9B/A100 **thinking** checks passed: seeded sampling replay, multimodal conditional action scoring and native tool-call parsing (114/98 generated tokens). This is not a real all-tools benchmark run.
- OpenAI embedding probe passed (1536 dimensions). All six neural spatial tools passed real executor calls and persisted 12 artifacts on a training image. Latest instruct protocol: 79 full-suite tests passed; real two-task integration pilot completed eight rollouts, two Experience updates and three candidate PPO evaluations. An intentional interruption/resume preserved 165 committed files unchanged. Evidence: `spatialcraftLog/validation/robospatial-live-20260907/pilot-instruct4096-v2/acceptance.json`.
- Formal RoboSpatial run is in progress under Slurm job 214256 on gpu-03 (A100 40 GB). The first task's four rollouts have committed. Target: 175 environment tasks × four rollouts, followed by 175 frozen-knowledge deployment tasks. No final benchmark result yet. Current run: `/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1/`. Failed initial allocation 214245 is retained for audit and is not an accuracy result.
- Protocol, provisional defaults, log layout and resume command: `docs/confirmed_protocol.md`.

## Historical component milestones (2026-09-06)

The table below records component/interface tests at that date. “Complete” here does not certify the real full experiment or paper reproduction.

| Stage | Scope | Status | Verification |
|---|---|---|---|
| 1 | Schemas and storage | Complete | Existing schema/storage checks |
| 2 | Model providers and role configuration | Complete | Existing provider/config checks |
| 3 | Datasets and verification | Complete | `6 passed`; five real datasets inspected |
| 4 | Tool contracts and mock tools | Complete | `4 passed`; local/remote mock loop |
| 5 | Minimal agent and rollout loop | Complete | `2 passed`; multi-step/rollout JSONL replay |
| 6 | Real spatial tools | Complete | `2 passed`; 11-tool composed lazy registry |
| 7 | Experience branch | Complete | `2 passed`; retrieval/learning/snapshot |
| 8 | Skill execution and lifecycle | Complete | `2 passed`; selection, persistence, termination, vanilla equivalence |
| 9 | Skill learning and candidate generation | Complete | `2 passed`; credit, gradients, candidates, lineage, pruning |
| 10 | Strict NP-PPO gate | Complete | `2 passed`; hand-recomputed strict scoring and surrogate isolation |
| 11 | Accumulation and read-only deployment | Complete | `1 passed`; batch barrier, atomic snapshot, frozen deployment |
| 12 | Evaluation, baselines, and ablations | Complete | `2 passed`; metrics, eight baselines, all ablation axes |

## Historical component acceptance

- Stages 4–12 complete.
- Full suite: `25 passed`.
- Ruff lint and format checks: passed.
- Python bytecode compilation: passed.
