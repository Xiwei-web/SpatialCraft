"""Install an audited detector bugfix revision; run ONLY while experiment is stopped.

Original source, journal and every prior commit remain byte-for-byte intact.
The whitelist intentionally forbids unrelated source/config changes.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from freeze_run_source import freeze, manifest

from spatialcraft.experiments.journal import RunJournal, digest
from spatialcraft.experiments.run import _resource_files
from spatialcraft.storage.atomic_io import (
    atomic_write_json,
    file_lock,
    read_json,
    sha256_file,
)

ALLOWED = {
    "src/spatialcraft/tools/real/detect.py",
    "src/spatialcraft/experiments/journal.py",
}


def code_digest(project):
    files = sorted((project / "src/spatialcraft").rglob("*.py")) + sorted(
        (project / "configs").rglob("*.yaml")
    )
    files += _resource_files(
        (project / "src/spatialcraft/resources", project / "prompts")
    )
    return digest({str(p.relative_to(project)): sha256_file(p) for p in files})


def prepare(project, run):
    original = run / "code_snapshot"
    old_manifest = read_json(original / "source_manifest.json")
    if manifest(original) != old_manifest:
        raise ValueError("Original frozen source was modified")
    current = manifest(project)
    changed = {
        name
        for name in old_manifest.keys() | current.keys()
        if old_manifest.get(name) != current.get(name)
    }
    if changed != ALLOWED:
        raise ValueError(f"Unexpected source changes: {sorted(changed)}")
    root = run / "robospatial"
    old = read_json(root / "journal.json")
    if code_digest(original) != old["binding"]["code_sha256"]:
        raise ValueError("Original source does not match journal")
    with file_lock(root / ".manifest.lock"):
        if (root / "code_patch.json").exists():
            raise ValueError(
                "Patch already installed; validate and resume, do not overwrite"
            )
        snapshot = freeze(project, run / "code_revisions/detect_bounds_v1")
        if manifest(snapshot) != current:
            raise ValueError("Recovery snapshot does not match reviewed code")
        new_code = code_digest(snapshot)
        patch = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": "GroundingDINO predicted boxes exceeded inclusive image bounds; normalize model output before overlay. Preserve earlier outcomes; resume first uncommitted stage.",
            "old_binding_sha256": old["binding_sha256"],
            "new_code_sha256": new_code,
            "original_snapshot": str(original),
            "patched_snapshot": str(snapshot),
            "source_changes": {
                name: {"before": old_manifest[name], "after": current[name]}
                for name in sorted(changed)
            },
        }
        preserved = {
            str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*.json")
        }
        atomic_write_json(
            run / "code_revisions/detect_bounds_v1-before.json",
            preserved,
            overwrite=False,
        )
        atomic_write_json(root / "code_patch.json", patch, overwrite=False)
    journal = RunJournal(root, {**old["binding"], "code_sha256": new_code})
    count = 0
    for path in (root / "stages").rglob("result.json"):
        journal.read_committed(str(path.parent.relative_to(root / "stages")))
        count += 1
    if any(
        sha256_file(root / name) != checksum for name, checksum in preserved.items()
    ):
        raise ValueError("Pre-existing records changed during patch preparation")
    print(
        {
            "snapshot": str(snapshot),
            "validated_commits": count,
            "preserved_json_files": len(preserved),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project.resolve(), args.run.resolve())
