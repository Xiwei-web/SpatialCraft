"""Materialize auditable inputs, not an inference run or completed experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spatialcraft.datasets import create_default_registry
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import (
    atomic_write_bytes,
    canonical_json_bytes,
    file_lock,
    sha256_bytes,
    sha256_file,
)

from .protocol import public_task, split_category, stratified_halves

DATASET_ORDER = ("robospatial", "erqa", "omni3d")
EXPECTED_COUNTS = {"robospatial": 350, "erqa": 400, "omni3d": 501}
RESERVED_DIRECTORIES = (
    "rollouts",
    "experiences",
    "skills",
    "ppo",
    "artifacts",
    "checkpoints",
    "errors",
    "results",
)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _immutable(path: Path, data: bytes, *, private: bool = False) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Prepared input changed: {path}; use a new directory")
    else:
        atomic_write_bytes(
            path, data, overwrite=False, mode=0o600 if private else 0o644
        )


def _image_keys(tasks: Sequence[TaskSample]) -> set[str]:
    return {image.sha256 or image.uri for task in tasks for image in task.images}


def _content_keys(tasks: Sequence[TaskSample]) -> set[str]:
    # Unlike TaskSample.fingerprint, exclude source IDs for duplicate auditing.
    return {
        sha256_bytes(
            canonical_json_bytes(
                {
                    "question": task.question,
                    "images": [im.sha256 or im.uri for im in task.images],
                    "choices": task.choices,
                }
            )
        )
        for task in tasks
    }


def write_preparation(
    output: Path,
    datasets: Mapping[str, tuple[TaskSample, ...]],
    *,
    provenance: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Idempotent preparation; changed inputs fail closed without overwriting.

    The manifest is the final preparation commit. It deliberately does not choose
    unspecified training/rollout/embedding parameters or expose an inference CLI.
    """
    if set(datasets) != set(DATASET_ORDER):
        raise ValueError(f"Expected precisely these datasets: {DATASET_ORDER}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    payloads: dict[str, bytes] = {}
    counts: dict[str, Any] = {}
    for name in DATASET_ORDER:
        rows = datasets[name]
        if any(row.dataset != name for row in rows):
            raise ValueError(f"Unexpected dataset identity in {name}")
        environment, deployment = stratified_halves(rows, seed=seed)
        for split_name, split_rows in (
            ("environment", environment),
            ("deployment", deployment),
        ):
            payloads[f"{name}/splits/{split_name}.jsonl"] = _jsonl(
                [public_task(row) for row in split_rows]
            )
            # Deliberately separate evaluator-only GT/masks/depth from model input.
            payloads[f"{name}/verification/{split_name}.jsonl"] = _jsonl(
                [row.to_dict() for row in split_rows]
            )
        counts[name] = {
            "total": len(rows),
            "environment": len(environment),
            "deployment": len(deployment),
            "categories": {
                split_name: dict(
                    sorted(Counter(map(split_category, split_rows)).items())
                )
                for split_name, split_rows in (
                    ("all", rows),
                    ("environment", environment),
                    ("deployment", deployment),
                )
            },
            "answer_types": dict(
                sorted(Counter(t.answer_type.value for t in rows).items())
            ),
            "maximum_images_per_task": max(len(row.images) for row in rows),
            "task_id_overlap": sorted(
                {t.task_id for t in environment} & {t.task_id for t in deployment}
            ),
            "shared_image_count": len(
                _image_keys(environment) & _image_keys(deployment)
            ),
            "shared_question_image_choices_count": len(
                _content_keys(environment) & _content_keys(deployment)
            ),
        }
    manifest = {
        "schema_version": 1,
        "status": "prepared_not_run",
        "model": "Qwen3.5-9B",
        "dataset_order": DATASET_ORDER,
        "split_protocol": {
            "seed": seed,
            "unit": "task_instance_not_image",
            "environment_fraction": 0.5,
            "category_key": {
                "robospatial": "question_type",
                "erqa": "question_type",
                "omni3d": "source_answer_type",
            },
            "algorithm": "sorted categories; sorted task IDs; one Python random.Random(seed) per dataset; shuffle each category; alternate odd remainder starting environment",
            "sma_exact_split_id_reproduction_claimed": False,
        },
        "datasets": counts,
        "protocol_requires_confirmation": {
            "accumulation_passes": None,
            "rollouts_per_environment_task": None,
            "environment_batch_size": None,
            "maximum_agent_steps": None,
            "maximum_output_tokens": None,
            "generation_sampling": None,
            "deployment_repetitions": None,
            "embedding_model": None,
            "knowledge_builder_model": None,
            "independent_dataset_knowledge_banks": None,
            "ppo_and_retrieval_hyperparameters": None,
            "verifier_thresholds_and_omni_answer_protocol": None,
        },
        "deployment_requirement": "freeze selected knowledge before deployment; never update with deployment labels, scores, or trajectories",
        "runtime_readiness": {
            "full_spatialcraft_inference_started": False,
            "remaining_work": [
                "connect real multimodal executor, feedback, tool runtime and memory management",
                "wire experience/skill builders, exact target-action NP-PPO gate and frozen deployment",
                "integrate the stage journal throughout the runtime and fault-test a real rollout",
                "verify paper-aligned metrics and obtain the missing experiment parameter choices",
            ],
        },
        "provenance": dict(provenance),
        "input_files": {
            name: {"sha256": sha256_bytes(data), "size_bytes": len(data)}
            for name, data in sorted(payloads.items())
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    with file_lock(output / ".preparation.lock"):
        manifest_path = output / "manifest.json"
        # Detect binding changes before any existing prepared input is touched.
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Preparation binding changed; use a new output directory")
        for name in DATASET_ORDER:
            (output / name / "verification").mkdir(
                parents=True, exist_ok=True, mode=0o700
            )
            for directory in RESERVED_DIRECTORIES:
                (output / name / directory).mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            _immutable(output / name, data, private="/verification/" in name)
        _immutable(manifest_path, manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("/l/users/xiwei.liu/benchmark")
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("/l/users/xiwei.liu/model/Qwen3.5-9B")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    registry = create_default_registry(args.benchmark_root)
    datasets = {}
    sources = {}
    for name in DATASET_ORDER:
        adapter = registry.create(name)
        print(f"Loading {name} ...", flush=True)
        datasets[name] = adapter.load_split("test")
        if len(datasets[name]) != EXPECTED_COUNTS[name]:
            raise ValueError(
                f"Unexpected source count for {name}: {len(datasets[name])}"
            )
        sources[name] = [
            file_record(p) for p in sorted((adapter.root / "data").glob("*.parquet"))
        ]
        if not sources[name]:
            raise ValueError(f"No source Parquet provenance found for {name}")
    project = args.project_root.resolve(strict=True)
    tracked_inputs = sorted((project / "src/spatialcraft").rglob("*.py"))
    tracked_inputs += sorted((project / "configs").rglob("*.yaml"))
    tracked_inputs += [
        project / "CVPR (1).pdf",
        project
        / "Spatial Memory Agent- Experience-Grounded Procedure Memory for Spatial Intelligence.pdf",
    ]
    provenance = {
        "sources": sources,
        "project_inputs": [file_record(p) for p in tracked_inputs],
        "model_path": str(args.model_path.resolve(strict=True)),
        "model_config_inputs": [
            file_record(args.model_path / name)
            for name in (
                "config.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
                "preprocessor_config.json",
            )
        ],
        "model_weights_rehashed_by_this_command": False,
    }
    manifest = write_preparation(
        args.output, datasets, provenance=provenance, seed=args.seed
    )
    print(f"Prepared only (no inference): {args.output.resolve()}", flush=True)
    for name, values in manifest["datasets"].items():
        print(
            f"{name}: environment={values['environment']}, deployment={values['deployment']}, shared_images={values['shared_image_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
