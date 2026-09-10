"""One held-out tool-only ReAct trajectory per task; durable, independent of learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from spatialcraft.agent import (
    ActionParser,
    ExecutionConfig,
    ExecutionLoop,
    StateBuilder,
)
from spatialcraft.agent.decision import (
    BUDGET_POLICY,
    FORCED_FINAL_TOKENS,
    RECOVERY_TOKENS,
)
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.rollout.reward import RewardComputer
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    sha256_file,
)
from spatialcraft.tools import ArtifactStore
from spatialcraft.tools.real import create_real_tool_registry
from spatialcraft.tools.real.common import SpatialToolPaths

from .journal import RunJournal, digest
from .protocol import public_task
from .react_api import (
    POLICY,
    SYSTEM,
    ReActResponsesProvider,
    ScopedToolExecutor,
    TaskMedia,
    ToolOnlyComposer,
)
from .rollout import JournaledRollout
from .run import _resource_files, load_api_key_file
from .run_api_baseline import COUNTS, DEFAULT_DATASETS, prepare, read_tasks, summarize


class PrivateReward(RewardComputer):
    def __init__(self, reference):
        super().__init__(binary=True)
        self.reference = reference

    def compute(self, task, action):
        if task.task_id != self.reference.task_id:
            raise ValueError("Verification task mismatch")
        return super().compute(self.reference, action)


def resource_binding(project):
    """Bind code, local spatial-tool sources/weights, and inference libraries."""
    paths = SpatialToolPaths.from_env()
    required = (
        paths.groundingdino_checkpoint,
        paths.groundingdino_config,
        paths.groundingdino_bert,
        paths.sam3_checkpoint,
        paths.moge_checkpoint,
        paths.da3_checkpoint,
        paths.orient_checkpoint,
        paths.orient_backbone,
        paths.easyocr_models,
    )
    repositories = (
        paths.groundingdino_repo,
        paths.sam3_repo,
        paths.moge_repo,
        paths.da3_repo,
        paths.orient_repo,
        paths.easyocr_repo,
    )
    if any(not p.exists() for p in (*required, *repositories)):
        raise ValueError("Required local spatial tool sources/weights are missing")
    source = _resource_files(
        (project / "src", project / "configs", project / "prompts")
    )
    print("Validating spatial tool source and checkpoint hashes...", flush=True)
    return {
        "code_sha256": digest(
            {str(p.relative_to(project)): sha256_file(p) for p in source}
        ),
        "tool_root": str(paths.root.resolve()),
        "tool_source_sha256": digest(
            {
                str(p.relative_to(paths.root)): sha256_file(p)
                for p in _resource_files(repositories)
                if p.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}
            }
        ),
        "tool_resources_sha256": {
            str(p): sha256_file(p) for p in _resource_files(required)
        },
        "library_versions": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "numpy",
                "Pillow",
                "openai",
                "safetensors",
                "timm",
                "scipy",
            )
        },
    }


def load_inputs(output, name, identity):
    data = output / "data" / name
    for kind in ("public", "private"):
        if sha256_file(data / f"{kind}.jsonl") != identity[f"{kind}_sha256"]:
            raise ValueError("Prepared task file changed")
    for uri, expected in identity["media"].items():
        if sha256_file(uri) != expected:
            raise ValueError("Prepared media changed")
    public, private = (
        read_tasks(data / "public.jsonl"),
        read_tasks(data / "private.jsonl"),
    )
    if (
        len(public) != identity["count"]
        or len(private) != len(public)
        or [t.task_id for t in public] != identity["task_ids"]
        or len({t.task_id for t in public}) != len(public)
    ):
        raise ValueError("Task IDs/count changed")
    for left, right in zip(public, private, strict=True):
        if left.dataset != name or digest(left.to_dict()) != digest(public_task(right)):
            raise ValueError("Public/private alignment or label sanitization failed")
    return public, private


def trajectory_row(task, trajectory, journal, prefix):
    usage = dict.fromkeys(
        ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens"), 0
    )
    models = set()
    for event in trajectory.metadata["generation_events"]:
        suffix = {
            "normal": "model",
            "recovery": "recovery/model",
            "forced_final": "forced_final/model",
        }[event["kind"]]
        wire = journal.read_committed(
            f"{prefix}/steps/{event['step_index']:04d}/{suffix}"
        )
        models.add(wire["model"])
        for key in usage:
            usage[key] += wire["usage"].get(key, 0) or 0
    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "question_type": task.metadata.get("question_type", "unknown"),
        "answer_type": task.answer_type.value,
        "source_answer_type": task.metadata.get("source_answer_type", "unknown"),
        "status": trajectory.status.value,
        "answer": trajectory.final_answer,
        "reward": trajectory.reward,
        "verifier": trajectory.verifier.to_dict() if trajectory.verifier else None,
        "failure_reason": trajectory.failure_reason,
        "model": next(iter(models)) if len(models) == 1 else ",".join(sorted(models)),
        "model_ids": sorted(models),
        "usage": usage,
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_prefix": prefix,
        "call_counts": trajectory.metadata["call_counts"],
        "tool_names": [
            call.tool_name
            for step in trajectory.transitions
            for call in step.action.tool_calls
        ],
        "environment_steps": trajectory.metadata["environment_steps"],
    }


def report_for(name, rows, total):
    from collections import Counter

    report = summarize(name, rows, total)
    report["policy"] = POLICY
    report["call_counts"] = {
        key: sum(row["call_counts"].get(key, 0) for row in rows)
        for key in (
            "normal_llm_calls",
            "recovery_calls",
            "forced_final_answers",
            "token_truncations",
            "tool_calls",
        )
    }
    report["tool_usage"] = dict(
        Counter(tool for row in rows for tool in row["tool_names"])
    )
    report["trajectories_with_tools"] = sum(bool(row["tool_names"]) for row in rows)
    report["model_ids"] = sorted({model for row in rows for model in row["model_ids"]})
    return report


def run_dataset(
    output,
    name,
    config,
    manifest,
    resources,
    *,
    max_steps=50,
    provider_factory=None,
    registry=None,
):
    identity = manifest["datasets"][name]
    public, private = load_inputs(output, name, identity)
    registry = registry or create_real_tool_registry()
    root = output / name
    journal = RunJournal(
        root,
        {
            **resources,
            "policy": POLICY,
            "dataset": name,
            "data_sha256": digest(identity),
            "model_id": config.model_id,
            "generation": asdict(config.generation),
            "endpoint": config.api.resolved_base_url(),
            "system_prompt": SYSTEM,
            "tool_schemas": [asdict(t) for t in registry.definitions()],
            "openai_tool_strict": False,
            "rollouts_per_task": 1,
            "passes": 1,
            "api_seed": None,
            "experience": False,
            "skill": False,
            "embedding": False,
            "learning": False,
            "image_policy": "original_bytes_all_images_original_order_detail_high",
            "trajectory_control": {
                "policy": BUDGET_POLICY,
                "max_tool_steps": max_steps,
                "actions_per_step": 1,
                "max_recovery_calls_per_step": 1,
                "recovery_max_tokens": RECOVERY_TOKENS,
                "forced_final_max_tokens": FORCED_FINAL_TOKENS,
            },
        },
    )
    layout = StorageLayout(root / "tool_store")
    store = ArtifactStore(layout, name)
    rows = []
    with file_lock(root / ".baseline.lock", timeout=0):
        try:
            for index, (task, reference) in enumerate(
                zip(public, private, strict=True)
            ):
                prefix = f"tasks/{index:05d}"
                media = TaskMedia(task, identity, layout)
                provider = (
                    provider_factory(config, media)
                    if provider_factory
                    else ReActResponsesProvider(config, media)
                )
                loop = ExecutionLoop(
                    provider=provider,
                    composer=ToolOnlyComposer(config, registry, media=media),
                    action_parser=ActionParser(registry),
                    tool_executor=ScopedToolExecutor(registry, store, media),
                    reward=PrivateReward(reference),
                    skill_controller=None,
                    config=ExecutionConfig(
                        max_steps=max_steps,
                        knowledge_snapshot_id="tool_only_no_knowledge",
                    ),
                )
                try:
                    trajectory = JournaledRollout(loop, journal).run(
                        task,
                        prefix=prefix,
                        initial_state=StateBuilder().initial(task),
                        rollout_index=0,
                        random_seed=None,
                    )
                finally:
                    if provider_factory is None and provider._client is not None:
                        provider._client.close()
                # Persist a compact per-task record; all detailed stages stay in journal.
                row = trajectory_row(task, trajectory, journal, prefix)
                rows.append(row)
                atomic_write_json(root / f"predictions/{index:05d}.json", row)
                report = report_for(name, rows, len(public))
                atomic_write_json(root / "progress.json", report)
                print(
                    json.dumps(
                        {
                            "dataset": name,
                            "completed": len(rows),
                            "total": len(public),
                            "correct": report["correct"],
                            "tool_calls": row["call_counts"]["tool_calls"],
                        }
                    ),
                    flush=True,
                )
        except Exception as exc:
            atomic_write_json(
                root / "progress.json",
                {
                    **report_for(name, rows, len(public)),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        report = report_for(name, rows, len(public))
        atomic_write_json(root / "results/deployment.json", report)
        atomic_write_bytes(
            root / "results/predictions.jsonl",
            b"".join(canonical_json_bytes(r) for r in rows),
            mode=0o600,
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--preparation",
        type=Path,
        default=Path(
            "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1"
        ),
    )
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("/l/users/xiwei.liu/benchmark")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/gpt-5.4-mini-react.yaml"),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/home/xiwei.liu/.config/spatialcraft/openai_api_key"),
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = prepare(
        args.output, args.preparation, args.benchmark_root, args.datasets
    )
    path = (
        args.model_config
        if args.model_config.is_absolute()
        else args.project / args.model_config
    )
    config = ModelConfig.from_dict(load_yaml(path))
    if config.model_id != "gpt-5.4-mini" or config.generation.seed is not None:
        raise ValueError("This baseline requires gpt-5.4-mini and no API seed")
    resources = resource_binding(args.project)
    for name in args.datasets:
        if manifest["datasets"][name]["count"] != COUNTS[name]:
            raise ValueError("Baseline must use the existing complete held-out split")
        load_inputs(args.output, name, manifest["datasets"][name])
    atomic_write_json(
        args.output / "preflight.json",
        {
            "status": "prepared",
            "policy": POLICY,
            "model": config.model_id,
            "generation": asdict(config.generation),
            "counts": {n: COUNTS[n] for n in args.datasets},
            "max_tool_steps": args.max_steps,
            "passes": 1,
            "rollouts_per_task": 1,
            "tool_names": [t.name for t in create_real_tool_registry().definitions()],
            "resources": resources,
            "data_sha256": digest(manifest),
        },
    )
    print("Offline preflight passed", flush=True)
    if not args.execute:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Real spatial tools require a GPU allocation")
    load_api_key_file(args.api_key_file)
    reports, failures = {}, []
    # One GPU, serial datasets and tools, releasing tool weights after each call.
    for name in args.datasets:
        try:
            reports[name] = run_dataset(
                args.output, name, config, manifest, resources, max_steps=args.max_steps
            )
        except Exception as exc:  # noqa: BLE001 -- independent datasets retain their results
            failures.append(name)
            print(f"FAILED {name}: {type(exc).__name__}", flush=True)
        atomic_write_json(
            args.output / "results/summary.json",
            {
                "status": "failed"
                if failures
                else "completed"
                if len(reports) == len(args.datasets)
                else "running",
                "model": config.model_id,
                "policy": POLICY,
                "datasets": reports,
                "failed_datasets": failures,
            },
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
