"""Run explicitly adapted memory baselines; offline preflight unless --execute."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from spatialcraft.storage.atomic_io import atomic_write_json, read_json, sha256_file
from spatialcraft.tools.real.common import SpatialToolPaths

from .baseline_memory import (
    METHODS,
    MemoryBaselineConfig,
    MemoryBaselineLearner,
    MemoryBaselinePipeline,
    method_description,
)
from .prepare_v2 import DATASETS as V2_DATASETS
from .run import _resource_files, load_api_key_file, preflight
from .usage import cost_report


def build_memory_pipeline(shared, settings, config):
    """Bind independent baseline learning to the runtime's actual executor/tools."""
    if not settings.is_v2:
        raise ValueError(
            "Memory baselines require the spatialcraft_v2 operation-aware runtime"
        )
    learner = MemoryBaselineLearner(
        config,
        generate=shared.generate_knowledge,
        embedder=shared.embedder,
        artifact_resolver=shared.artifact_resolver,
        evolve_skills=shared.evolve_skills,
        skill_ratio_mode=settings.skill_options.get("ratio_mode", "sequence"),
        experience_config={
            "capacity": config.memory_capacity,
            "rollouts_per_task": settings.rollouts_per_task,
            "top_k_per_aspect": config.retrieval_top_k,
            **settings.experience_options,
        },
    )
    pipeline = MemoryBaselinePipeline(
        learner=learner,
        journal=shared.journal,
        rollout=shared.rollout,
        seed=settings.seed,
        rollouts_per_task=settings.rollouts_per_task,
        artifact_validator=shared.artifact_validator,
    )
    pipeline.metric_token_counter = getattr(shared, "metric_token_counter", None)
    pipeline.metric_tokenizer_id = getattr(shared, "metric_tokenizer_id", None)
    return pipeline


def bind_inference_resources(binding, datasets):
    media = {
        image.uri: image.sha256
        for splits in datasets.values()
        for rows in splits.values()
        for task in rows
        for image in task.images
    }
    for path, checksum in media.items():
        if checksum is None or sha256_file(path) != checksum:
            raise ValueError("Dataset image checksum changed")
    model_path = Path(binding["model_path"])
    shards = sorted(
        set(
            read_json(model_path / "model.safetensors.index.json")[
                "weight_map"
            ].values()
        )
    )
    binding["weights_sha256"] = {
        shard: sha256_file(model_path / shard) for shard in shards
    }
    paths = SpatialToolPaths.from_env()
    binding["tool_resources_sha256"] = {
        str(path.relative_to(paths.root)): sha256_file(path)
        for path in _resource_files(
            (
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
        )
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[3]
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=V2_DATASETS)
    parser.add_argument("--memory-capacity", type=int, default=100)
    parser.add_argument("--skill-capacity", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pilot-tasks", type=int)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.list_methods:
        print(
            json.dumps(
                [method_description(name) for name in METHODS],
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not all((args.method, args.config, args.preparation, args.output)):
        parser.error("--method, --config, --preparation and --output are required")
    config = MemoryBaselineConfig(
        args.method,
        memory_capacity=args.memory_capacity,
        skill_capacity=args.skill_capacity,
        retrieval_top_k=args.top_k,
        semantic_candidates=max(10, args.top_k),
    )
    settings, datasets, binding, report = preflight(
        project, args.preparation, args.config, args.datasets
    )
    if not settings.is_v2:
        parser.error("Use an explicitly versioned spatialcraft_v2 configuration")
    if settings.ablations:
        parser.error(
            "Baseline comparison uses its own method branch; use a configuration without unrelated ablations"
        )
    settings = replace(settings, experiment_name="baseline_" + args.method)
    descriptor = {**method_description(args.method), "config": asdict(config)}
    binding.update(
        memory_baseline=descriptor,
        mode="baseline_training_only_pilot" if args.pilot_tasks else "baseline_formal",
        pilot_tasks=args.pilot_tasks,
        settings=settings.to_dict(),
    )
    report = {
        **report,
        "memory_baseline": descriptor,
        "runtime_path": "MemoryBaselinePipeline -> shared real Runtime rollout callback",
        "result_path": str(
            args.output / "<dataset>" / "results" / "memory_baseline.json"
        ),
    }
    if args.pilot_tasks is not None and (
        args.pilot_tasks < 1
        or any(
            args.pilot_tasks >= len(split["environment"]) for split in datasets.values()
        )
    ):
        parser.error(
            "Pilot size must be positive and smaller than each environment split"
        )
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if args.api_key_file:
        load_api_key_file(args.api_key_file)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Execute inside a Slurm GPU allocation")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing; no inference started")
    bind_inference_resources(binding, datasets)
    from .runtime import ExperimentRuntime

    runtime = ExperimentRuntime(project, args.output, settings, binding)
    for name, splits in datasets.items():
        shared = runtime.dataset(name)
        pipeline = build_memory_pipeline(shared, settings, config)
        training = (
            splits["environment"][: args.pilot_tasks]
            if args.pilot_tasks
            else splits["environment"]
        )
        frozen = pipeline.accumulate(training)
        if args.pilot_tasks:
            print(name, "BASELINE_TRAINING_ONLY_PILOT_COMPLETED", flush=True)
            continue
        result = pipeline.deploy(splits["deployment"], frozen)
        atomic_write_json(
            shared.journal.root / "results" / "usage.json",
            cost_report(shared.journal.root, len(splits["deployment"])),
        )
        print(name, json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
