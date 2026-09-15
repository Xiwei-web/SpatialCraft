#!/usr/bin/env python3
"""Prepare or execute MemRL-R / MemRL-GT spatial baselines with frozen deployment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]

from MemRL.runner import (
    POLICY,
    DatasetRunner,
    OpenAIEmbeddingsProvider,
    Settings,
    embedding_config,
    model_config,
)

from memp.data import load_split, prepare_inputs
from MemRL.reflection import ReflectionBuilder
from spatialcraft.experiments.journal import RunJournal, digest
from spatialcraft.experiments.run import load_api_key_file
from spatialcraft.experiments.run_react_baseline import resource_binding
from spatialcraft.storage.atomic_io import atomic_write_json, file_lock, sha256_file
from spatialcraft.tools.real import create_real_tool_registry
from spatialcraft.tools.real.common import SpatialToolPaths

DATASETS = ("robospatial", "erqa", "omni3d", "sat", "viewspatial")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--variant", choices=("R", "GT", "both"), required=True)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument(
        "--preparation",
        type=Path,
        default=Path(
            "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1"
        ),
    )
    result.add_argument(
        "--benchmark-root", type=Path, default=Path("/l/users/xiwei.liu/benchmark")
    )
    result.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/home/xiwei.liu/.config/spatialcraft/openai_api_key"),
    )
    result.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS[:-1])
    )
    result.add_argument(
        "--model", choices=("gpt-5.4", "gpt-5.4-mini"), default="gpt-5.4-mini"
    )
    result.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    result.add_argument("--max-output-tokens", type=int, default=16384)
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--environment-rollouts", type=int, default=1)
    result.add_argument("--environment-passes", type=int, default=1)
    result.add_argument("--max-steps", type=int, default=50)
    result.add_argument("--top-k", type=int, default=3)
    result.add_argument("--candidate-k", type=int, default=10)
    result.add_argument("--utility-weight", type=float, default=0.5)
    result.add_argument("--learning-rate", type=float, default=0.3)
    result.add_argument("--q-init", type=float, default=0.0)
    result.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Fixed cosine threshold; default calibrates from public environment queries only",
    )
    result.add_argument("--threshold-quantile", type=float, default=0.8)
    result.add_argument(
        "--embedding-model",
        choices=("text-embedding-3-large", "text-embedding-3-small"),
        default="text-embedding-3-large",
    )
    result.add_argument(
        "--stage", choices=("all", "environment", "deployment"), default="all"
    )
    result.add_argument(
        "--execute",
        action="store_true",
        help="Enable real API/tool execution; default only prepares and validates inputs",
    )
    return result


def source_binding(project):
    """Bind the MemRL implementation and all reused implementation/configuration files."""
    extensions = {
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".j2",
        ".jinja",
        ".jinja2",
        ".txt",
        ".md",
        ".sh",
        ".sbatch",
    }
    excluded = {"__pycache__", ".pytest_cache", ".ruff_cache", "runs", "tests"}
    files = []
    for directory in ("MemRL", "memp", "src", "configs", "prompts"):
        files.extend(
            path
            for path in (project / directory).rglob("*")
            if path.is_file()
            and path.suffix in extensions
            and not excluded.intersection(path.relative_to(project).parts)
        )
    files.extend([project / "RAG/__init__.py", project / "RAG/data.py"])
    checksums = {
        str(path.relative_to(project)): sha256_file(path) for path in sorted(set(files))
    }
    return {"code_sha256": digest(checksums), "source_files": checksums}


def tool_paths_preflight():
    """Check resource existence without loading tools or hashing model checkpoints."""
    paths = SpatialToolPaths.from_env()
    names = (
        "groundingdino_repo",
        "groundingdino_checkpoint",
        "groundingdino_config",
        "groundingdino_bert",
        "sam3_repo",
        "sam3_checkpoint",
        "moge_repo",
        "moge_checkpoint",
        "da3_repo",
        "da3_checkpoint",
        "orient_repo",
        "orient_checkpoint",
        "orient_backbone",
        "easyocr_repo",
        "easyocr_models",
    )
    items = {name: str(getattr(paths, name)) for name in names}
    return {
        "root": str(paths.root),
        "missing": [path for path in items.values() if not Path(path).exists()],
        "paths": items,
    }


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        settings = Settings(
            **{key: getattr(args, key) for key in Settings.__dataclass_fields__}
        )
    except ValueError as exc:
        arguments.error(str(exc))
    if len(set(args.datasets)) != len(args.datasets):
        arguments.error("datasets must be unique")
    variants = ["R", "GT"] if args.variant == "both" else [args.variant]
    output = args.output.resolve()
    if output == PROJECT or output.is_relative_to(PROJECT):
        arguments.error(
            "Use a run directory outside the source checkout, e.g. "
            "/home/xiwei.liu/spatialcraftRuns/memrl_mini_v1"
        )
    config = model_config(PROJECT, settings)
    embedding = embedding_config(PROJECT, settings.embedding_model)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".memrl_run.lock", timeout=0):
        manifest = prepare_inputs(
            output, args.preparation, args.benchmark_root, args.datasets
        )
        sources = source_binding(PROJECT)
        tool_check = tool_paths_preflight()
        registry = create_real_tool_registry()
        binding = {
            "policy": POLICY,
            "variants": variants,
            **sources,
            "data_sha256": digest(manifest),
            "settings": asdict(settings),
            "generation": asdict(config.generation),
            "endpoint": config.api.resolved_base_url(),
            "embedding_endpoint": embedding.api.resolved_base_url(),
            "tool_paths": tool_check["paths"],
            "tool_schemas": [asdict(tool) for tool in registry.definitions()],
        }
        # Stage is excluded so environment -> deployment resumes under one identity.
        RunJournal(output / "run_identity", binding)
        preflight = {
            "status": "prepared"
            if not tool_check["missing"]
            else "prepared_but_tool_resources_missing",
            **binding,
            "counts": {
                name: {
                    "environment": manifest["datasets"][name]["environment_count"],
                    "deployment": manifest["datasets"][name]["count"],
                }
                for name in args.datasets
            },
            "requested_stage": args.stage,
            "tools": tool_check,
            "executes_api": False,
            "protocol": "sequential_environment_learning_frozen_deployment",
            "fidelity": "memrl_sma_spatial_adaptation",
            "threshold_calibration": "public_environment_queries_only",
            "sampling_policy": "temperature_sent_only_for_reasoning_none; no_API_seed",
        }
        atomic_write_json(output / "preflight.json", preflight)
        print(
            json.dumps(
                {
                    key: preflight[key]
                    for key in (
                        "status",
                        "variants",
                        "counts",
                        "settings",
                        "requested_stage",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not args.execute:
            return 0
        if tool_check["missing"]:
            raise ValueError(
                "Required local spatial tool resources missing; inspect preflight.json"
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("MemRL spatial tool execution needs an allocated GPU")
        resources = {**resource_binding(PROJECT), **sources}
        # Credentials and clients are accessed only after explicit --execute.
        load_api_key_file(args.api_key_file)
        reports = {}
        for variant in variants:
            variant_output = output / variant
            reports[variant] = {}
            for name in args.datasets:
                identity = manifest["datasets"][name]
                embedder = OpenAIEmbeddingsProvider(
                    embedding, cache_dir=variant_output / name / "embedding_cache"
                )
                builder = ReflectionBuilder(config, variant=variant)
                try:
                    runner = DatasetRunner(
                        variant_output,
                        name,
                        identity,
                        settings,
                        config,
                        resources,
                        variant=variant,
                        embedder=embedder,
                        builder=builder,
                        registry=registry,
                    )
                    reports[variant][name] = runner.run(
                        args.stage,
                        environment=load_split(output, name, identity, "environment"),
                        deployment=load_split(output, name, identity, "deployment"),
                    )
                except Exception as exc:
                    atomic_write_json(
                        output / f"results/{args.stage}_summary.json",
                        {
                            "status": "failed",
                            "stage": args.stage,
                            "policy": POLICY,
                            "failed_variant": variant,
                            "failed_dataset": name,
                            "error_type": type(exc).__name__,
                            "variants": reports,
                        },
                    )
                    raise
                finally:
                    builder.close()
                    if embedder._client is not None:
                        embedder._client.close()
                completed = sum(
                    len(dataset_reports) for dataset_reports in reports.values()
                )
                atomic_write_json(
                    output / f"results/{args.stage}_summary.json",
                    {
                        "status": "completed"
                        if completed == len(variants) * len(args.datasets)
                        else "running",
                        "stage": args.stage,
                        "policy": POLICY,
                        "variants": reports,
                    },
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
