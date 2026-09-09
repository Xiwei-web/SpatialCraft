"""Append an audited JSON parser revision to a STOPPED detector-recovery run."""

import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from freeze_run_source import freeze, manifest
from prepare_detect_recovery import code_digest

from spatialcraft.experiments.journal import RunJournal
from spatialcraft.storage.atomic_io import (
    atomic_copy,
    atomic_write_json,
    file_lock,
    read_json,
    sha256_file,
)

ALLOWED = {
    "src/spatialcraft/experiments/learning.py",
    "src/spatialcraft/experiments/journal.py",
}


def prepare(project, run):
    root = run / "robospatial"
    patch_path = root / "code_patch.json"
    parent = read_json(patch_path)
    source = Path(parent["patched_snapshot"])
    old_manifest = read_json(source / "source_manifest.json")
    current = manifest(project)
    if (
        manifest(source) != old_manifest
        or code_digest(source) != parent["new_code_sha256"]
    ):
        raise ValueError("Previous revision was modified")
    changed = {
        name
        for name in old_manifest.keys() | current.keys()
        if old_manifest.get(name) != current.get(name)
    }
    if changed != ALLOWED:
        raise ValueError(f"Unexpected source changes: {sorted(changed)}")
    original = read_json(root / "journal.json")
    before_binding = {**original["binding"], "code_sha256": parent["new_code_sha256"]}
    old_journal = RunJournal(root, before_binding)
    count = 0
    for path in (root / "stages").rglob("result.json"):
        old_journal.read_committed(str(path.parent.relative_to(root / "stages")))
        count += 1
    print(f"Verified {count} existing commits", flush=True)
    with file_lock(root / ".manifest.lock"):
        if read_json(patch_path) != parent:
            raise ValueError("Patch metadata changed during validation")
        snapshot = freeze(project, run / "code_revisions/json_output_v1")
        if manifest(snapshot) != current:
            raise ValueError("New snapshot differs from reviewed source")
        patch = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": "Normalize literal JSON string controls and a missing terminal value quote without regenerating or inventing knowledge. Resume failed Skill semantic-gradient stage with cached response.",
            "old_binding_sha256": original["binding_sha256"],
            "new_code_sha256": code_digest(snapshot),
            "parent_code_sha256": parent["new_code_sha256"],
            "original_snapshot": str(run / "code_snapshot"),
            "patched_snapshot": str(snapshot),
            "prior_patches": [
                *parent.get("prior_patches", []),
                {k: v for k, v in parent.items() if k != "prior_patches"},
            ],
            "source_changes": {
                name: {"before": old_manifest[name], "after": current[name]}
                for name in sorted(changed)
            },
        }
        # Validate the new chain without touching the live metadata pointer.
        with tempfile.TemporaryDirectory(
            prefix="spatialcraft-patch-check-"
        ) as temporary:
            temp = Path(temporary)
            atomic_write_json(temp / "journal.json", original)
            atomic_write_json(temp / "code_patch.json", patch)
            RunJournal(
                temp, {**original["binding"], "code_sha256": patch["new_code_sha256"]}
            )
        preserved = {
            str(p.relative_to(root)): sha256_file(p)
            for p in root.rglob("*.json")
            if p != patch_path
        }
        audit = run / "code_revisions/json_output_v1-before.json"
        atomic_write_json(audit, preserved, overwrite=False)
        backup = run / "code_revisions/json_output_v1-parent_patch.json"
        atomic_copy(patch_path, backup, overwrite=False)
        if any(
            sha256_file(root / name) != checksum for name, checksum in preserved.items()
        ):
            raise ValueError("Existing records changed during preparation")
        atomic_write_json(patch_path, patch)
    print(
        {
            "snapshot": str(snapshot),
            "validated_commits": count,
            "preserved_json_files": len(preserved),
            "prior_patch_backup": str(backup),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project.resolve(), args.run.resolve())
