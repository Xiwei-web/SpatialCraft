"""Frozen public tasks and separate private scoring references for MemP.

The existing RAG preparation defines the five benchmark splits and image identities.
This module adds environment references for offline reward computation. Callers must
give only the public half returned by ``load_split`` to the agent; private tasks are
for the scorer. No reference is copied into a public task or its metadata.
"""

from pathlib import Path

from RAG import data as rag_data
from spatialcraft.experiments.journal import digest
from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.run_api_baseline import immutable, read_tasks
from spatialcraft.storage.atomic_io import (
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_file,
)

DATA_POLICY = "memp_private_environment_v1"


def _aligned_private(name, public, private):
    """Align source references by ID, then require identical public observations."""
    ids = [task.task_id for task in public]
    references = {task.task_id: task for task in private}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or len(references) != len(private)
        or set(ids) != set(references)
    ):
        raise ValueError(f"Public/private task ID mismatch: {name}")
    ordered = tuple(references[task_id] for task_id in ids)
    for exposed, reference in zip(public, ordered, strict=True):
        if exposed.dataset != name or reference.dataset != name:
            raise ValueError(f"Public/private dataset mismatch: {name}")
        if digest(exposed.to_dict()) != digest(public_task(exposed)):
            raise ValueError(f"Public split contains private annotations: {name}")
        if digest(exposed.to_dict()) != digest(public_task(reference)):
            raise ValueError(
                f"Public/private content mismatch: {name}/{exposed.task_id}"
            )
        if reference.reference_answer is None:
            raise ValueError(f"Missing private reference: {name}/{exposed.task_id}")
    return ordered


def _environment_references(output, preparation, benchmark_root, name, identity):
    if name == "sat" and not (preparation / "sat/splits/environment.jsonl").is_file():
        # Use the same materialization path and selection as RAG's public snapshot.
        source = dict(identity["environment_source"])
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("SAT environment source checksum mismatch")
        adapter = rag_data.create_default_registry(benchmark_root).create("sat")
        adapter.images = rag_data.ImageMaterializer(
            output / "data/sat/environment_media"
        )
        try:
            validation = adapter.load_split("validation")
        finally:
            if hasattr(adapter, "close"):
                adapter.close()
        selected, _ = rag_data.sat_splits(
            validation, read_tasks(output / "data/sat/public.jsonl"), seed=42
        )
        return selected, source

    root = (
        Path(identity["split_preparation_path"])
        if name == "viewspatial"
        else preparation
    )
    relative = f"{name}/verification/environment.jsonl"
    source = root / relative
    manifest = read_json(root / "manifest.json")
    try:
        expected = manifest["input_files"][relative]["sha256"]
    except KeyError as error:
        raise ValueError(f"Unbound private environment source: {source}") from error
    if sha256_file(source) != expected:
        raise ValueError(f"Private environment source checksum mismatch: {name}")
    return read_tasks(source), {"path": str(source), "sha256": expected}


def prepare_inputs(
    output: Path, preparation: Path, benchmark_root: Path, names: list[str]
):
    """Prepare immutable public/private pairs without any model or API calls."""
    output, preparation, benchmark_root = map(
        Path, (output, preparation, benchmark_root)
    )
    manifest_path = output / "data/memp_manifest.json"
    with file_lock(output / ".memp_prepare.lock", timeout=0):
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (
                list(names) != manifest["dataset_order"]
                or str(preparation.resolve()) != manifest["source_preparation"]
                or str(benchmark_root.resolve()) != manifest["benchmark_root"]
            ):
                raise ValueError(
                    "MemP preparation inputs changed; use a new output directory"
                )
            verify_inputs(output, manifest)
            return manifest

        base = rag_data.prepare_inputs(output, preparation, benchmark_root, names)
        manifest = {
            **base,
            "memp_data_policy": DATA_POLICY,
            "public_data_manifest_file": "data/rag_manifest.json",
            "public_data_manifest_sha256": sha256_file(
                output / "data/rag_manifest.json"
            ),
            "datasets": {},
        }
        for name in names:
            identity = base["datasets"][name]
            public = read_tasks(output / "data" / name / "environment.jsonl")
            private, source = _environment_references(
                output, preparation, benchmark_root, name, identity
            )
            private = _aligned_private(name, public, private)
            relative = f"data/{name}/environment_private.jsonl"
            immutable(
                output / relative,
                b"".join(canonical_json_bytes(task.to_dict()) for task in private),
            )
            manifest["datasets"][name] = {
                **identity,
                "environment_private_file": relative,
                "environment_private_sha256": sha256_file(output / relative),
                "environment_private_source": source,
            }
        immutable(manifest_path, canonical_json_bytes(manifest))
        verify_inputs(output, manifest)
        return manifest


def load_split(output, name, identity, phase="environment"):
    """Return aligned ``(public_tasks, private_tasks)`` after integrity checks.

    Image bytes are checked by ``verify_inputs`` at preparation/resume. Here image
    membership and declared checksums are checked against that verified identity.
    """
    root = Path(output) / "data" / name
    if phase == "environment":
        if (
            identity["environment_private_file"]
            != f"data/{name}/environment_private.jsonl"
        ):
            raise ValueError(f"Unexpected private environment path: {name}")
        public_path, private_path = (
            root / "environment.jsonl",
            root / "environment_private.jsonl",
        )
        public_hash, private_hash = "environment_sha256", "environment_private_sha256"
        expected_ids, count = (
            identity["environment_task_ids"],
            identity["environment_count"],
        )
    elif phase == "deployment":
        public_path, private_path = root / "public.jsonl", root / "private.jsonl"
        public_hash, private_hash = "public_sha256", "private_sha256"
        expected_ids, count = identity["task_ids"], identity["count"]
    else:
        raise ValueError(f"Unknown MemP phase: {phase}")
    for path, key in ((public_path, public_hash), (private_path, private_hash)):
        if sha256_file(path) != identity[key]:
            raise ValueError(f"Frozen MemP input changed: {name}/{path.name}")
    public, private = read_tasks(public_path), read_tasks(private_path)
    if [task.task_id for task in public] != expected_ids or len(public) != count:
        raise ValueError(f"Frozen MemP task count/order changed: {name}/{phase}")
    aligned = _aligned_private(name, public, private)
    if [task.task_id for task in private] != expected_ids:
        raise ValueError(f"Frozen MemP private task order changed: {name}/{phase}")
    for task in public:
        for image in task.images:
            expected = identity["media"].get(image.uri)
            if expected is None or (image.sha256 and image.sha256 != expected):
                raise ValueError(f"Unbound MemP task media: {name}/{task.task_id}")
    return public, aligned


def verify_inputs(output, manifest):
    """Validate frozen files, image bytes, IDs, and public/private separation."""
    output = Path(output)
    if manifest.get("memp_data_policy") != DATA_POLICY:
        raise ValueError("Unsupported MemP data policy")
    if manifest.get("public_data_manifest_file") != "data/rag_manifest.json":
        raise ValueError("Unexpected public data manifest path")
    if (
        sha256_file(output / "data/rag_manifest.json")
        != manifest["public_data_manifest_sha256"]
    ):
        raise ValueError("Frozen public data manifest changed")
    # JSON object key order is irrelevant; execution order is the explicit list.
    order = manifest["dataset_order"]
    if (
        not order
        or len(set(order)) != len(order)
        or set(manifest["datasets"]) != set(order)
    ):
        raise ValueError("MemP dataset selection mismatch")
    rag_data.verify_inputs(output, manifest)
    for name in manifest["dataset_order"]:
        identity = manifest["datasets"][name]
        load_split(output, name, identity, "environment")
        load_split(output, name, identity, "deployment")
