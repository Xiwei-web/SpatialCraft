"""Reuse held-out inputs and prepare a separate, label-free environment corpus."""

from pathlib import Path

from PIL import Image

from spatialcraft.datasets import create_default_registry
from spatialcraft.datasets.normalizer import ImageMaterializer
from spatialcraft.experiments.journal import digest
from spatialcraft.experiments.prepare_v2 import sat_splits
from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.run_api_baseline import (
    immutable,
    read_tasks,
)
from spatialcraft.experiments.run_api_baseline import (
    prepare as prepare_deployment,
)
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import (
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_file,
)

ENVIRONMENT_COUNTS = {
    "robospatial": 175,
    "erqa": 200,
    "omni3d": 251,
    "sat": 300,
    "viewspatial": 2856,
}


def task_content_key(task, media):
    """Exclude source IDs and labels when checking exact task overlap."""
    return digest(
        {
            "question": task.question,
            "choices": task.choices,
            "images": [media[image.uri] for image in task.images],
        }
    )


def validate_splits(name, environment, deployment, media, *, allow_exact_overlap=False):
    for rows in (environment, deployment):
        if not rows or len({row.task_id for row in rows}) != len(rows):
            raise ValueError(f"Empty split or duplicate task IDs: {name}")
        if any(row.dataset != name for row in rows):
            raise ValueError(f"Dataset mismatch: {name}")
        if any(digest(row.to_dict()) != digest(public_task(row)) for row in rows):
            raise ValueError(f"Public split contains private annotations: {name}")
    if {t.task_id for t in environment} & {t.task_id for t in deployment}:
        raise ValueError(f"Environment/deployment task IDs overlap: {name}")
    overlap = {task_content_key(t, media) for t in environment} & {
        task_content_key(t, media) for t in deployment
    }
    if overlap and not allow_exact_overlap:
        raise ValueError(f"Exact question/image/choices overlap across splits: {name}")
    # Existing splits are task-level: sharing an image alone is permitted and audited.
    return len(
        {media[im.uri] for t in environment for im in t.images}
        & {media[im.uri] for t in deployment for im in t.images}
    )


def _verified_source(root, name):
    relative = f"{name}/splits/environment.jsonl"
    manifest = read_json(root / "manifest.json")
    source = root / relative
    expected = manifest["input_files"][relative]["sha256"]
    if sha256_file(source) != expected:
        raise ValueError(f"Environment source checksum mismatch: {relative}")
    return read_tasks(source), {"path": str(source), "sha256": expected}


def prepare_inputs(output, preparation, benchmark_root, names):
    """Offline preparation only. Existing baseline deployment tasks are preserved."""
    manifest_path = output / "data/rag_manifest.json"
    with file_lock(output / ".rag_prepare.lock", timeout=0):
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if list(names) != manifest["dataset_order"]:
                raise ValueError(
                    "Dataset selection changed; use a new output directory"
                )
            if manifest["source_preparation"] != str(preparation.resolve()):
                raise ValueError("Preparation path changed; use a new output directory")
            if manifest["benchmark_root"] != str(benchmark_root.resolve()):
                raise ValueError("Benchmark root changed; use a new output directory")
            verify_inputs(output, manifest)
            return manifest

        deployment = prepare_deployment(output, preparation, benchmark_root, names)
        result = {
            "schema_version": 1,
            "policy": "rag_examples_data_v1",
            "dataset_order": list(names),
            "source_preparation": str(preparation.resolve()),
            "benchmark_root": str(benchmark_root.resolve()),
            "split_seed": 42,
            "datasets": {},
        }
        for name in names:
            identity = deployment["datasets"][name]
            data = output / "data" / name
            public = read_tasks(data / "public.jsonl")
            if name == "sat":
                # Reuse a supplied five-dataset preparation when available.
                if (preparation / "sat/splits/environment.jsonl").is_file():
                    environment, source = _verified_source(preparation, name)
                else:
                    source_path = benchmark_root / "SAT/SAT_val.parquet"
                    source = {
                        "path": str(source_path),
                        "sha256": sha256_file(source_path),
                        "selection": "sorted_validation_ids_random_sample_300_seed42",
                    }
                    adapter = create_default_registry(benchmark_root).create("sat")
                    adapter.images = ImageMaterializer(
                        output / "data/sat/environment_media"
                    )
                    try:
                        validation = adapter.load_split("validation")
                    finally:
                        if hasattr(adapter, "close"):
                            adapter.close()
                    selected, _ = sat_splits(validation, public, seed=42)
                    environment = tuple(
                        TaskSample.from_dict(public_task(t)) for t in selected
                    )
            else:
                root = (
                    Path(identity["split_preparation_path"])
                    if name == "viewspatial"
                    else preparation
                )
                environment, source = _verified_source(root, name)

            if len(environment) != ENVIRONMENT_COUNTS[name]:
                raise ValueError(f"Unexpected environment count: {name}")
            if any(digest(t.to_dict()) != digest(public_task(t)) for t in environment):
                raise ValueError("Environment inputs must already be label-free")
            immutable(
                data / "environment.jsonl",
                b"".join(canonical_json_bytes(t.to_dict()) for t in environment),
            )
            media = dict(identity["media"])
            for task in environment:
                for image in task.images:
                    if image.uri not in media:
                        with Image.open(image.uri) as decoded:
                            decoded.verify()
                        media[image.uri] = sha256_file(image.uri)
                    if image.sha256 and image.sha256 != media[image.uri]:
                        raise ValueError("Environment image checksum mismatch")
            shared = validate_splits(
                name, environment, public, media, allow_exact_overlap=True
            )
            overlap = {task_content_key(t, media) for t in environment} & {
                task_content_key(t, media) for t in public
            }
            result["datasets"][name] = {
                **identity,
                "environment_count": len(environment),
                "environment_sha256": sha256_file(data / "environment.jsonl"),
                "environment_task_ids": [t.task_id for t in environment],
                "environment_source": source,
                "shared_image_count": shared,
                "exact_task_overlap_count": len(overlap),
                "exact_match_policy": "exclude_at_retrieval_preserve_split",
                "media": media,
            }
        immutable(manifest_path, canonical_json_bytes(result))
        verify_inputs(output, result)
        return result


def verify_inputs(output, manifest):
    """Recheck committed data and media on every resume, without API calls."""
    for name, identity in manifest["datasets"].items():
        root = output / "data" / name
        for kind in ("public", "private", "environment"):
            if sha256_file(root / f"{kind}.jsonl") != identity[f"{kind}_sha256"]:
                raise ValueError(f"Frozen RAG input changed: {name}/{kind}")
        for path, checksum in identity["media"].items():
            if sha256_file(path) != checksum:
                raise ValueError(f"Frozen RAG media changed: {path}")
        environment = read_tasks(root / "environment.jsonl")
        public, private = (
            read_tasks(root / "public.jsonl"),
            read_tasks(root / "private.jsonl"),
        )
        if (
            [t.task_id for t in environment] != identity["environment_task_ids"]
            or len(environment) != identity["environment_count"]
            or [t.task_id for t in public] != identity["task_ids"]
            or len(public) != identity["count"]
            or len(private) != len(public)
        ):
            raise ValueError("Frozen RAG task count/order changed")
        for left, right in zip(public, private, strict=True):
            if digest(left.to_dict()) != digest(public_task(right)):
                raise ValueError("Public/private deployment alignment mismatch")
        validate_splits(
            name, environment, public, identity["media"], allow_exact_overlap=True
        )
