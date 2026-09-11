#!/usr/bin/env python3
"""Independent GPT RAG baseline; --execute is required for any API call."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from RAG.core import EMBEDDING_TEXT_POLICY, POLICY, PROMPT
from RAG.data import prepare_inputs
from RAG.runner import run_dataset
from spatialcraft.experiments.journal import digest
from spatialcraft.experiments.run import load_api_key_file
from spatialcraft.experiments.run_api_baseline import (
    COUNTS,
    DEFAULT_DATASETS,
    immutable,
)
from spatialcraft.models.providers.openai_embeddings import OpenAIEmbeddingsProvider
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.storage.atomic_io import (
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_file,
)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", choices=("gpt-5.4-mini", "gpt-5.4"), required=True)
    value.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Separate output directory for each model/protocol",
    )
    value.add_argument(
        "--datasets", nargs="+", choices=tuple(COUNTS), default=list(DEFAULT_DATASETS)
    )
    value.add_argument(
        "--stage", choices=("all", "environment", "deployment"), default="all"
    )
    value.add_argument(
        "--preparation",
        type=Path,
        default=Path(
            "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1"
        ),
    )
    value.add_argument(
        "--benchmark-root", type=Path, default=Path("/l/users/xiwei.liu/benchmark")
    )
    value.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help="Applied to both environment and deployment; thinking modes omit sampling overrides",
    )
    value.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Total generation cap including reasoning tokens, per call in both stages",
    )
    value.add_argument("--environment-rollouts", type=int, default=4)
    value.add_argument("--environment-temperature", type=float, default=0.7)
    value.add_argument("--top-k", type=int, default=3)
    value.add_argument(
        "--embedding-model",
        choices=("text-embedding-3-large", "text-embedding-3-small"),
        default="text-embedding-3-large",
    )
    value.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/home/xiwei.liu/.config/spatialcraft/openai_api_key"),
    )
    value.add_argument(
        "--execute", action="store_true", help="Enable model and embedding API calls"
    )
    return value


def configuration(args):
    config = ModelConfig.from_dict(
        load_yaml(PROJECT / f"configs/models/{args.model}-baseline.yaml")
    )
    if config.model_id != args.model or config.provider.value != "openai_responses":
        raise ValueError("Unexpected baseline model configuration")
    # Explicit overrides prevent model YAML defaults from silently enabling reasoning.
    settings = replace(
        config.generation,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        temperature=0.0 if args.reasoning_effort == "none" else None,
        top_p=None,
        seed=None,
        logprobs=False,
        top_logprobs=None,
        extra={"store": False},
    )
    deployment = replace(config, generation=settings)
    environment = replace(
        config,
        generation=replace(
            settings,
            temperature=args.environment_temperature
            if args.reasoning_effort == "none"
            else None,
        ),
    )
    embedding = ModelConfig.from_dict(
        load_yaml(PROJECT / f"configs/models/{args.embedding_model}.yaml")
    )
    return environment, deployment, embedding


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        cli.error("Dataset names must be distinct")
    if not 1 <= args.max_output_tokens <= 128000:
        cli.error(
            "max-output-tokens must be in [1, 128000] for the supported GPT models"
        )
    if args.environment_rollouts < 1 or args.top_k < 1:
        cli.error("Rollout count and top-k must be positive")
    if (
        not math.isfinite(args.environment_temperature)
        or not 0 <= args.environment_temperature <= 2
    ):
        cli.error("Environment temperature must be finite and in [0, 2]")
    args.output = args.output.resolve()
    environment, deployment, embedding_config = configuration(args)
    options = {
        "environment_rollouts": args.environment_rollouts,
        "environment_temperature": environment.generation.temperature,
        "top_k": args.top_k,
    }
    sources = {
        str(path.relative_to(PROJECT)): sha256_file(path)
        for folder in (PROJECT / "RAG", PROJECT / "src")
        for path in sorted(folder.rglob("*.py"))
    }
    with file_lock(args.output / ".rag_run.lock", timeout=0):
        # Reserve the run for exactly one model/settings before preparing any input.
        run_config = {
            "policy": POLICY,
            "model": args.model,
            "datasets": args.datasets,
            "options": options,
            "embedding_model": args.embedding_model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "deployment_temperature": deployment.generation.temperature,
            "sampling_policy": "explicit_temperature"
            if args.reasoning_effort == "none"
            else "provider_default_no_sampling_overrides",
            "deployment_rollouts": 1,
            "environment_retrieval": False,
            "deployment_memory_updates": False,
            "executor": "one_response_visual_qa",
            "tools": False,
            "correctness_filter": False,
            "reflections": False,
            "skills": False,
            "embedding_text_policy": EMBEDDING_TEXT_POLICY,
            "representative_policy": "first_completed_nonempty_rollout_per_prior_task",
            "retrieval_policy": "same_dataset_cosine_top_k_distinct_tasks_no_threshold",
            "prior_images_in_prompt": False,
            "exact_match_policy": "exclude_at_retrieval_preserve_split",
            "generation_seed": None,
            "code_sha256": digest(sources),
            "model_config_sha256": sha256_file(
                PROJECT / f"configs/models/{args.model}-baseline.yaml"
            ),
            "embedding_config_sha256": sha256_file(
                PROJECT / f"configs/models/{args.embedding_model}.yaml"
            ),
        }
        immutable(args.output / "run_config.json", canonical_json_bytes(run_config))
        manifest = prepare_inputs(
            args.output, args.preparation, args.benchmark_root, args.datasets
        )
        embedder = OpenAIEmbeddingsProvider(
            embedding_config, cache_dir=args.output / "embedding_cache"
        )
        report = {
            **run_config,
            "status": "prepared_not_executed",
            "requested_stage": args.stage,
            "counts": {
                name: {
                    "environment_tasks": item["environment_count"],
                    "environment_generation_calls": item["environment_count"]
                    * args.environment_rollouts,
                    "deployment_tasks": item["count"],
                    "deployment_generation_calls": item["count"],
                }
                for name, item in manifest["datasets"].items()
            },
            "embedding_identity": embedder.identity,
        }
        atomic_write_json(args.output / "preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not args.execute:
            return
        if args.stage == "deployment":
            for name in args.datasets:
                if not (args.output / name / "memory/snapshot.json").is_file():
                    raise ValueError(
                        f"Complete the environment stage before deployment: {name}"
                    )
        if not os.environ.get("OPENAI_API_KEY"):
            load_api_key_file(args.api_key_file)
        binding = {
            **run_config,
            "prompt": PROMPT,
            "data_manifest_sha256": digest(manifest),
            "model_endpoint": deployment.api.resolved_base_url(),
            "embedding_identity": embedder.identity,
            "openai_version": version("openai"),
            "numpy_version": version("numpy"),
            "pillow_version": version("Pillow"),
        }
        results, failed = {}, []
        for name in args.datasets:
            try:
                results[name] = run_dataset(
                    args.output,
                    name,
                    manifest["datasets"][name],
                    binding,
                    environment,
                    deployment,
                    options,
                    embedder,
                    args.stage,
                )
            except Exception as exc:  # noqa: BLE001 -- finish independent datasets, then fail the run
                failed.append({"dataset": name, "error_type": type(exc).__name__})
                print(
                    f"FAILED {name}: {type(exc).__name__}", file=sys.stderr, flush=True
                )
        embedding_usage = [
            read_json(path)
            for path in (args.output / "embedding_cache/usage").glob("*.json")
        ]
        summary = {
            "policy": POLICY,
            "model": args.model,
            "stage": args.stage,
            "status": "failed" if failed else "completed",
            "datasets": results,
            "failed_datasets": failed,
            "embedding_usage": {
                "recorded_requests": len(embedding_usage),
                "input_tokens": sum(row["input_tokens"] for row in embedding_usage),
            },
        }
        atomic_write_json(args.output / f"results/{args.stage}_summary.json", summary)
        atomic_write_json(args.output / "results/summary.json", summary)
        if args.stage in {"all", "deployment"}:
            lines = ["| Dataset | Correct / Total | Accuracy |", "|---|---:|---:|"]
            for name, result in results.items():
                row = result["deployment"]
                lines.append(
                    f"| {name} | {row['correct']} / {row['total']} | {row['accuracy']:.2%} |"
                )
            if failed:
                lines.extend(
                    [
                        "",
                        "Incomplete run; failed datasets: "
                        + ", ".join(row["dataset"] for row in failed),
                    ]
                )
            from spatialcraft.storage.atomic_io import atomic_write_bytes

            atomic_write_bytes(
                args.output / "results/accuracy.md", ("\n".join(lines) + "\n").encode()
            )
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
