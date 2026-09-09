"""Offline preflight by default. --execute explicitly enables paid embeddings."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.schemas import TaskSample, TaskSplit
from spatialcraft.storage.atomic_io import atomic_write_json, read_json, sha256_file
from spatialcraft.tools.real.common import SpatialToolPaths

from .journal import digest
from .prepare import DATASET_ORDER
from .protocol import public_task
from .settings import ExperimentSettings


def _resource_files(paths):
    """Only inference inputs; omit download caches and VCS metadata."""
    files = set()
    for root in paths:
        candidates = (root,) if root.is_file() else root.rglob("*")
        files.update(
            p
            for p in candidates
            if p.is_file()
            and not {".cache", ".git", "__pycache__"}.intersection(p.parts)
        )
    return sorted(files)


def load_api_key_file(path: Path) -> None:
    """Read only an explicitly supplied, owner-private regular credential file."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("API key file must be a regular file owned by this user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("API key file must not be accessible to group/others")
        if info.st_size > 4096:
            raise ValueError("API key file is unexpectedly large")
        key = os.read(fd, 4096).decode().strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("API key file is empty or contains whitespace")
        os.environ["OPENAI_API_KEY"] = key
    finally:
        os.close(fd)



def check_runtime_initialization(project: Path, settings, binding, names):
    """Exercise real runtime/journal setup without GPU loading or paid calls."""
    from .runtime import ExperimentRuntime

    checks = {}
    with tempfile.TemporaryDirectory(prefix="spatialcraft-preflight-") as folder:
        runtime = ExperimentRuntime(project, Path(folder), settings, binding)
        for name in names:
            pipeline = runtime.dataset(name)
            payload = {"runtime_initialization": "passed"}
            pipeline.journal.execute("preflight", {"dataset": name}, lambda payload=payload: payload)
            restored = runtime.dataset(name).journal.read_committed("preflight")
            if restored != payload:
                raise ValueError("Runtime journal commit/readback mismatch")
            checks[name] = "journal_create_commit_reopen_passed"
        if runtime.local._model is not None or runtime.embedding._client is not None:
            raise RuntimeError("Offline initialization must not load models or API clients")
    return checks


def preflight(project: Path, preparation: Path, config_path: Path, names=None):
    settings = ExperimentSettings.load(config_path)
    manifest = read_json(preparation / "manifest.json")
    datasets = {}
    selected = tuple(DATASET_ORDER if names is None else names)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in DATASET_ORDER for name in selected)
    ):
        raise ValueError("Select unique supported benchmarks")
    for name in selected:
        splits = {}
        for split in ("environment", "deployment"):
            relative = f"{name}/verification/{split}.jsonl"
            if (
                sha256_file(preparation / relative)
                != manifest["input_files"][relative]["sha256"]
            ):
                raise ValueError(f"Prepared labels changed: {relative}")
            rows = tuple(
                TaskSample.from_dict(json.loads(line))
                for line in (preparation / relative).read_text().splitlines()
            )
            if len(rows) != manifest["datasets"][name][split]:
                raise ValueError("Prepared sample count mismatch")
            if any(row.dataset != name or row.reference_answer is None for row in rows):
                raise ValueError("Dataset identity/reference answer missing")
            expected_split = (
                TaskSplit.TRAIN if split == "environment" else TaskSplit.TEST
            )
            if any(row.split is not expected_split for row in rows) or len(
                {r.task_id for r in rows}
            ) != len(rows):
                raise ValueError("Prepared split identity/uniqueness mismatch")
            public_relative = f"{name}/splits/{split}.jsonl"
            if (
                sha256_file(preparation / public_relative)
                != manifest["input_files"][public_relative]["sha256"]
            ):
                raise ValueError("Prepared public inputs changed")
            public_rows = tuple(
                TaskSample.from_dict(json.loads(line))
                for line in (preparation / public_relative).read_text().splitlines()
            )
            if any(row.reference_answer is not None for row in public_rows) or [
                public_task(r) for r in rows
            ] != [public_task(r) for r in public_rows]:
                raise ValueError("Prepared public/private task correspondence mismatch")
            splits[split] = rows
        if {r.task_id for r in splits["environment"]} & {
            r.task_id for r in splits["deployment"]
        }:
            raise ValueError("Training/deployment task IDs overlap")
        datasets[name] = splits
    model = ModelConfig.from_dict(
        load_yaml(project / f"configs/models/{settings.backbone}.yaml")
    )
    if model.local is None:
        raise ValueError("This experiment requires the local Qwen backbone")
    model_path = Path(model.local.path)
    index_path = model_path / "model.safetensors.index.json"
    weight_index = read_json(index_path)
    for shard in set(weight_index["weight_map"].values()):
        if not (model_path / shard).is_file():
            raise ValueError(f"Missing model shard: {shard}")
    embedding = ModelConfig.from_dict(
        load_yaml(project / "configs/models/text-embedding-3-small.yaml")
    )
    if embedding.model_id != settings.embedding_model:
        raise ValueError("Embedding model mismatch")
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
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise ValueError(f"Missing spatial tool resources: {missing}")
    repositories = (
        paths.groundingdino_repo,
        paths.sam3_repo,
        paths.moge_repo,
        paths.da3_repo,
        paths.orient_repo,
        paths.easyocr_repo,
    )
    if any(not p.is_dir() for p in repositories):
        raise ValueError("Missing spatial tool source repository")
    versions = {}
    for package in (
        "torch",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
        "openai",
        "tiktoken",
        "numpy",
        "Pillow",
        "safetensors",
        "torchvision",
        "iopath",
        "ftfy",
        "python-bidi",
        "moviepy",
        "utils3d_moge",
        "plyfile",
        "pycolmap",
        "trimesh",
        "evo",
        "e3nn",
        "pillow-heif",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f"Missing inference dependency: {package}") from exc
    files = sorted((project / "src/spatialcraft").rglob("*.py")) + sorted(
        (project / "configs").rglob("*.yaml")
    )
    files += _resource_files(
        (project / "src/spatialcraft/resources", project / "prompts")
    )
    binding = {
        "datasets": list(selected),
        "preparation_sha256": sha256_file(preparation / "manifest.json"),
        "code_sha256": digest(
            {str(p.relative_to(project)): sha256_file(p) for p in files}
        ),
        "model_path": str(model_path.resolve()),
        "model_index_sha256": sha256_file(index_path),
        "model_auxiliary_sha256": {
            str(p.relative_to(model_path)): sha256_file(p)
            for p in _resource_files((model_path,))
            if p.suffix in {".json", ".jinja", ".txt", ".model"}
        },
        "tool_root": str(paths.root.resolve()),
        "tool_source_sha256": digest(
            {
                str(p.relative_to(paths.root)): sha256_file(p)
                for p in _resource_files(repositories)
                if p.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}
            }
        ),
        "library_versions": versions,
        "settings": settings.to_dict(),
        "evolution_protocol": {
            "batch_unit": "distinct_trajectories_per_parent_version",
            "batch_size": settings.evolution_batch_trajectories,
            "max_parents_per_round": settings.max_parent_skills_per_round,
            "round_boundary": "after_all_four_task_rollouts",
            "tail": "pending_not_forced",
            "thinking_score": settings.ppo_thinking_mode,
            "output_budget_includes_thinking": True,
        },
    }
    initialization = check_runtime_initialization(project, settings, binding, selected)
    report = {
        "status": "offline_preflight_passed_not_api_or_full_tool_validation",
        "runtime_initialization": initialization,
        "paid_api_called": False,
        "settings": settings.to_dict(),
        "counts": {
            name: {
                "training": len(s["environment"]),
                "training_rollouts": len(s["environment"]) * 4,
                "deployment": len(s["deployment"]),
            }
            for name, s in datasets.items()
        },
        "runtime_models": {
            "executor": settings.backbone,
            "knowledge_builder": settings.backbone,
            "ppo_scorer": settings.backbone,
            "embedding": settings.embedding_model,
        },
        "provisional_engineering_defaults": load_yaml(config_path).get(
            "provisional_engineering_defaults", []
        ),
        "binding": binding,
        "limitations": [
            "OpenAI account/network not tested",
            "full spatial-tool inference not tested",
            "existing verifier tolerances are explicit project defaults, not claimed identical to SMA",
        ],
    }
    return settings, datasets, binding, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "configs/experiments/qwen35_9b_spatialcraft.yaml",
    )
    parser.add_argument(
        "--preparation",
        type=Path,
        default=Path(
            "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/l/users/xiwei.liu/spatialcraftLog/runs/qwen35_9b_protocol_v2"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument(
        "--pilot-tasks",
        type=int,
        help="Training-only diagnostic subset; never a formal benchmark result",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize paid OpenAI embeddings and real GPU/tool inference",
    )
    args = parser.parse_args()
    settings, datasets, binding, report = preflight(
        project, args.preparation, args.config, args.datasets
    )
    if args.pilot_tasks is not None:
        if args.pilot_tasks < 1 or any(
            args.pilot_tasks >= len(s["environment"]) for s in datasets.values()
        ):
            parser.error("Pilot size must be positive and smaller than training split")
        binding["mode"] = "training_only_pilot"
        binding["pilot_tasks"] = args.pilot_tasks
    else:
        binding["mode"] = "formal"
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    if args.api_key_file:
        load_api_key_file(args.api_key_file)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Execute inside a Slurm GPU allocation")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing; no inference started")
    # Before the first external call, bind actual media and model weights too.
    media = {
        im.uri: im.sha256
        for splits in datasets.values()
        for rows in splits.values()
        for t in rows
        for im in t.images
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
    print(
        "Binding spatial-tool checkpoint hashes before any external API call...",
        flush=True,
    )
    binding["tool_resources_sha256"] = {
        str(p.relative_to(paths.root)): sha256_file(p)
        for p in _resource_files(
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
    from .runtime import ExperimentRuntime

    runtime = ExperimentRuntime(project, args.output, settings, binding)
    for name in datasets:
        pipeline = runtime.dataset(name)
        training = datasets[name]["environment"]
        if args.pilot_tasks is not None:
            training = training[: args.pilot_tasks]
        frozen = pipeline.accumulate(training)
        if args.pilot_tasks is not None:
            print(name, "TRAINING_ONLY_PILOT_COMPLETED_NO_TEST_EVALUATION", flush=True)
            continue
        result = pipeline.deploy(datasets[name]["deployment"], frozen)
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
