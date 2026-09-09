"""Stage a narrowly scoped FLA revision; activate only after exact-source GPU acceptance."""

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from freeze_run_source import freeze, manifest
from prepare_detect_recovery import code_digest

from spatialcraft.experiments.fast_qwen import fast_local_config, kernel_identity
from spatialcraft.experiments.journal import RunJournal, _execution_revision
from spatialcraft.models.registry import LocalModelConfig
from spatialcraft.storage.atomic_io import (
    atomic_copy,
    atomic_write_json,
    file_lock,
    read_json,
    sha256_file,
)

SELECTED = {
    "src/spatialcraft/experiments/fast_qwen.py",
    "src/spatialcraft/experiments/run_fast_qwen.py",
    "src/spatialcraft/experiments/journal.py",
    "src/spatialcraft/experiments/runtime.py",
    "src/spatialcraft/models/providers/transformers_local.py",
    "src/spatialcraft/models/providers/offload.py",
}


def stage(project, run):
    parent = read_json(run / "robospatial/code_patch.json")
    source = Path(parent["patched_snapshot"])
    if manifest(source) != read_json(source / "source_manifest.json"):
        raise ValueError("Previous frozen source changed")
    destination = run / "code_revisions/fla_dual_v2"
    with tempfile.TemporaryDirectory(prefix="spatialcraft-fla-source-") as folder:
        folder = Path(folder)
        for name in ("src", "configs", "prompts"):
            shutil.copytree(
                source / name,
                folder / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for name in SELECTED:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, target)
        before, after = manifest(source), manifest(folder)
        changed = {
            name
            for name in before.keys() | after.keys()
            if before.get(name) != after.get(name)
        }
        if not changed <= SELECTED:
            raise ValueError("Unreviewed snapshot changes")
        freeze(folder, destination)
        if manifest(destination) != after:
            raise ValueError("Staged FLA source differs from reviewed source")
    print(destination, flush=True)


def activate(run, acceptance):
    root = run / "robospatial"
    snapshot = run / "code_revisions/fla_dual_v2"
    passed = read_json(acceptance)
    if passed.get("status") != "passed" or passed.get(
        "source_manifest_sha256"
    ) != sha256_file(snapshot / "source_manifest.json"):
        raise ValueError("Exact frozen-source GPU acceptance is required")
    if manifest(snapshot) != read_json(snapshot / "source_manifest.json"):
        raise ValueError("FLA snapshot changed after acceptance")
    identity = kernel_identity()
    if identity != passed["kernel_runtime"]:
        raise ValueError("Kernel identity changed after acceptance")
    original = read_json(root / "journal.json")
    patch_path = root / "code_patch.json"
    parent = read_json(patch_path)
    if parent["patched_snapshot"] == str(snapshot):
        current = {
            **original["binding"],
            **parent["execution_overrides"],
            "code_sha256": parent["new_code_sha256"],
        }
        RunJournal(root, current)
        print("Existing FLA execution revision verified", flush=True)
        return
    before_binding = {**original["binding"], "code_sha256": parent["new_code_sha256"]}
    journal = RunJournal(root, before_binding)
    count = 0
    for path in (root / "stages").rglob("result.json"):
        journal.read_committed(str(path.parent.relative_to(root / "stages")))
        count += 1
    local = fast_local_config(LocalModelConfig(**original["binding"]["model_local"]))
    overrides = json.loads(
        json.dumps({"model_local": asdict(local), "kernel_runtime": identity})
    )
    current = _execution_revision(before_binding, overrides)
    source = Path(parent["patched_snapshot"])
    before, after = manifest(source), manifest(snapshot)
    changed = {
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    }
    if not changed <= SELECTED or code_digest(source) != parent["new_code_sha256"]:
        raise ValueError("Unexpected parent or source delta")
    patch = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "User approved FLA kernels and two-GPU model sharding after single-GPU long-context OOM. Inputs, BF16 precision, sampling and protocol unchanged.",
        "old_binding_sha256": original["binding_sha256"],
        "parent_code_sha256": parent["new_code_sha256"],
        "new_code_sha256": code_digest(snapshot),
        "original_snapshot": str(run / "code_snapshot"),
        "patched_snapshot": str(snapshot),
        "prior_patches": [
            *parent.get("prior_patches", []),
            {k: v for k, v in parent.items() if k != "prior_patches"},
        ],
        "execution_overrides": overrides,
        "source_changes": {
            name: {"before": before.get(name), "after": after.get(name)}
            for name in sorted(changed)
        },
        "gpu_acceptance_sha256": sha256_file(acceptance),
    }
    current["code_sha256"] = patch["new_code_sha256"]
    with tempfile.TemporaryDirectory(prefix="spatialcraft-fla-binding-") as folder:
        folder = Path(folder)
        atomic_write_json(folder / "journal.json", original)
        atomic_write_json(folder / "code_patch.json", patch)
        RunJournal(folder, current)
    with file_lock(root / ".manifest.lock"):
        if read_json(patch_path) != parent:
            raise ValueError("Another process changed the execution revision")
        preserved = {
            str(p.relative_to(root)): sha256_file(p)
            for p in root.rglob("*.json")
            if p != patch_path
        }
        atomic_write_json(
            run / "code_revisions/fla_dual_v2-before.json", preserved, overwrite=False
        )
        atomic_copy(
            patch_path,
            run / "code_revisions/fla_dual_v2-parent_patch.json",
            overwrite=False,
        )
        atomic_write_json(patch_path, patch)
    print(
        {
            "validated_commits": count,
            "preserved_json_files": len(preserved),
            "active_snapshot": str(snapshot),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args()
    if args.acceptance:
        activate(args.run.resolve(), args.acceptance.resolve())
    else:
        stage(args.project.resolve(), args.run.resolve())
