"""Stage/test/activate an output-budget revision without rewriting prior commits."""

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from freeze_run_source import freeze, manifest
from prepare_detect_recovery import code_digest
from spatialcraft.experiments.journal import RunJournal, _execution_revision
from spatialcraft.storage.atomic_io import atomic_copy, atomic_write_json, file_lock, read_json, sha256_file

REVISION = "output_budget_v1"
SELECTED = {
    "src/spatialcraft/experiments/learning.py",
    "src/spatialcraft/experiments/rollout.py",
    "src/spatialcraft/experiments/output_budget.py",
}


def binding(original, patch):
    value = original["binding"]
    for revision in [*patch.get("prior_patches", []), patch]:
        value = {**_execution_revision(value, revision.get("execution_overrides", {})),
                 "code_sha256": revision["new_code_sha256"]}
    return value


def checked_source(source):
    value = manifest(source)
    if value != read_json(source / "source_manifest.json"):
        raise ValueError("Frozen source changed")
    return value


def stage(project, run):
    parent = read_json(run / "robospatial/code_patch.json")
    source = Path(parent["patched_snapshot"])
    checked_source(source)
    with tempfile.TemporaryDirectory(prefix="spatialcraft-budget-source-") as folder:
        folder = Path(folder)
        for name in ("src", "configs", "prompts"):
            shutil.copytree(source / name, folder / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in SELECTED:
            shutil.copy2(project / name, folder / name)
        expected = manifest(folder)
        snapshot = freeze(folder, run / "code_revisions" / REVISION)
        if checked_source(snapshot) != expected:
            raise ValueError("Existing candidate differs from reviewed code")
    print(snapshot, flush=True)


def check(run):
    root = run / "robospatial"
    patch = read_json(root / "code_patch.json")
    snapshot = run / "code_revisions" / REVISION
    checked_source(snapshot)
    if patch["patched_snapshot"] != str(snapshot) or code_digest(snapshot) != patch["new_code_sha256"]:
        raise ValueError("Output-budget revision is not active")
    tests = patch["offline_validation"]
    if sha256_file(tests["report"]) != tests["report_sha256"]:
        raise ValueError("Validation report changed")
    if sha256_file(snapshot / "source_manifest.json") != tests["source_manifest_sha256"]:
        raise ValueError("Validated source changed")
    RunJournal(root, binding(read_json(root / "journal.json"), patch))
    print("Output-budget execution revision verified", flush=True)


def activate(run, report):
    root = run / "robospatial"
    snapshot = run / "code_revisions" / REVISION
    after = checked_source(snapshot)
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    if not suites or any(int(s.get("failures", 0)) or int(s.get("errors", 0)) for s in suites):
        raise ValueError("Passing frozen-source tests required")
    # The immutable RoboSpatial source deliberately excludes the unrelated
    # later Omni3D/27B runner APIs. Test that branch only against its own source.
    if sum(int(s.get("tests", 0)) for s in suites) < 100:
        raise ValueError("RoboSpatial regression suite required")
    names = {t.get("name", "") for s in suites for t in s.iter("testcase")}
    if not {"test_skipped_updates_continue_next_task_and_resume",
            "test_knowledge_forced_completion_and_cached_resume",
            "test_evolution_fallback_keeps_pending_trajectories"} <= names:
        raise ValueError("Output-budget regression tests missing")
    patch_path = root / "code_patch.json"
    with file_lock(root / ".pipeline.lock", timeout=0):
        parent = read_json(patch_path)
        if parent["patched_snapshot"] == str(snapshot):
            check(run)
            return
        source = Path(parent["patched_snapshot"])
        before = checked_source(source)
        changed = {n for n in before.keys() | after.keys() if before.get(n) != after.get(n)}
        if changed != SELECTED or code_digest(source) != parent["new_code_sha256"]:
            raise ValueError(f"Unexpected source changes: {sorted(changed)}")
        original = read_json(root / "journal.json")
        current = binding(original, parent)
        journal = RunJournal(root, current)
        count = 0
        for path in (root / "stages").rglob("result.json"):
            journal.read_committed(str(path.parent.relative_to(root / "stages")))
            count += 1
            if count % 2000 == 0:
                print(f"Verified {count} historical commits", flush=True)
        patch = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": "User approved one forced-completion call after output limit; on repeated exhaustion continue with audited truncation/skipped knowledge update. Preserve four-rollout groups, BF16, FLA two-GPU backend and all existing commits.",
            "old_binding_sha256": original["binding_sha256"],
            "parent_code_sha256": parent["new_code_sha256"],
            "new_code_sha256": code_digest(snapshot),
            "original_snapshot": str(run / "code_snapshot"),
            "patched_snapshot": str(snapshot),
            "prior_patches": [*parent.get("prior_patches", []),
                              {k: v for k, v in parent.items() if k != "prior_patches"}],
            "source_changes": {n: {"before": before.get(n), "after": after[n]} for n in sorted(changed)},
            "offline_validation": {"report": str(report), "report_sha256": sha256_file(report),
                                   "source_manifest_sha256": sha256_file(snapshot / "source_manifest.json")},
        }
        with tempfile.TemporaryDirectory(prefix="spatialcraft-budget-binding-") as folder:
            folder = Path(folder)
            atomic_write_json(folder / "journal.json", original)
            atomic_write_json(folder / "code_patch.json", patch)
            RunJournal(folder, binding(original, patch))
        preserved = {str(p.relative_to(root)): sha256_file(p)
                     for p in root.rglob("*.json") if p != patch_path}
        with file_lock(root / ".manifest.lock"):
            if read_json(patch_path) != parent:
                raise ValueError("Active execution revision changed")
            atomic_write_json(run / f"code_revisions/{REVISION}-before.json", preserved, overwrite=False)
            atomic_copy(patch_path, run / f"code_revisions/{REVISION}-parent_patch.json", overwrite=False)
            atomic_write_json(patch_path, patch)
        check(run)
        print({"validated_commits": count, "preserved_json_files": len(preserved),
               "snapshot": str(snapshot)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(args.run.resolve())
    elif args.tests:
        activate(args.run.resolve(), args.tests.resolve())
    else:
        stage(args.project.resolve(), args.run.resolve())
