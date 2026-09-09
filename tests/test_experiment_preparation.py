from dataclasses import replace

import pytest

from spatialcraft.experiments import RunJournal, stratified_halves
from spatialcraft.experiments.prepare import write_preparation
from spatialcraft.experiments.protocol import public_task, split_category
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import atomic_write_json, read_json


def tasks():
    return tuple(
        TaskSample(
            dataset="robospatial",
            task_id=f"{category}-{i}",
            question="q",
            reference_answer="yes",
            metadata={"question_type": category, "mask_uri": "/secret/mask.png"},
        )
        for category, n in (("a", 3), ("b", 5), ("c", 4))
        for i in range(n)
    )


def test_split_deterministic_disjoint_alternating():
    rows = tasks()
    env, dep = stratified_halves(rows)
    assert len(env) == len(dep) == 6
    assert [r.task_id for r in env] == [
        r.task_id for r in stratified_halves(tuple(reversed(rows)))[0]
    ]
    assert sum(split_category(r) == "a" for r in env) == 2
    assert sum(split_category(r) == "b" for r in env) == 2
    assert not {r.task_id for r in env} & {r.task_id for r in dep}
    assert public_task(env[0])["reference_answer"] is None
    assert "mask_uri" not in public_task(env[0])["metadata"]


def test_omni_uses_source_type_and_duplicate_ids_rejected():
    row = replace(
        tasks()[0], dataset="omni3d", metadata={"source_answer_type": "float"}
    )
    assert split_category(row) == "float"
    with pytest.raises(ValueError):
        stratified_halves((row, row))


def test_resume_committed_stage_and_fail_closed(tmp_path):
    journal = RunJournal(tmp_path, {"model": "qwen", "split_sha256": "a"})
    result, reused = journal.execute(
        "d/pass-1/task-1", {"seed": 42}, lambda: {"reward": 1}
    )
    assert result == {"reward": 1} and not reused

    def must_not_run():
        raise AssertionError("Repeated committed inference")

    assert journal.execute("d/pass-1/task-1", {"seed": 42}, must_not_run) == (
        result,
        True,
    )
    with pytest.raises(ValueError):
        journal.execute("d/pass-1/task-1", {"seed": 0}, must_not_run)
    with pytest.raises(ValueError):
        RunJournal(tmp_path, {"model": "other"})
    path = tmp_path / "stages/d/pass-1/task-1/result.json"
    commit = read_json(path)
    commit["result"]["reward"] = 0
    atomic_write_json(path, commit)
    with pytest.raises(ValueError):
        journal.execute("d/pass-1/task-1", {"seed": 42}, must_not_run)


def test_interrupted_stage_retries_without_replaying_committed_prefix(tmp_path):
    journal = RunJournal(tmp_path, {"run": "test"})
    journal.execute("first", {}, lambda: {"output": "saved"})

    def interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        journal.execute("second", {}, interrupt)
    assert not (tmp_path / "stages/second/result.json").exists()
    restored = RunJournal(tmp_path, {"run": "test"})
    assert restored.execute("first", {}, interrupt)[1]
    assert restored.execute("second", {}, lambda: {"output": "resumed"}) == (
        {"output": "resumed"},
        False,
    )
    assert len(list((tmp_path / "stages/second/attempts").glob("*.json"))) == 2


def test_journal_rejects_traversal(tmp_path):
    journal = RunJournal(tmp_path, {})
    with pytest.raises(ValueError):
        journal.execute("../outside", {}, dict)


def test_journal_normalizes_sequences_and_copies_binding(tmp_path):
    binding = {"models": ["qwen"], "seeds": (42,)}
    journal = RunJournal(tmp_path, binding)
    binding["models"].append("other")
    assert journal.binding["models"] == ["qwen"]
    result, _ = journal.execute("task", {"ids": (1, 2)}, lambda: {"scores": (1, 0)})
    assert result == {"scores": [1, 0]}
    restored = RunJournal(tmp_path, {"models": ["qwen"], "seeds": (42,)})
    assert restored.execute("task", {"ids": [1, 2]}, dict) == (result, True)


def test_preparation_is_idempotent_private_and_not_a_run(tmp_path):
    datasets = {
        name: tuple(
            replace(
                row,
                dataset=name,
                metadata={**row.metadata, "source_answer_type": "str"},
            )
            for row in tasks()
        )
        for name in ("robospatial", "erqa", "omni3d")
    }
    manifest = write_preparation(tmp_path, datasets, provenance={"source": "fixed"})
    assert manifest["status"] == "prepared_not_run"
    assert manifest["protocol_requires_confirmation"]["accumulation_passes"] is None
    path = tmp_path / "manifest.json"
    original_mtime = path.stat().st_mtime_ns
    assert (
        write_preparation(tmp_path, datasets, provenance={"source": "fixed"})
        == manifest
    )
    assert path.stat().st_mtime_ns == original_mtime
    public = (tmp_path / "robospatial/splits/environment.jsonl").read_text()
    private_path = tmp_path / "robospatial/verification/environment.jsonl"
    assert '"reference_answer":null' in public and "/secret/mask.png" not in public
    assert '"reference_answer":"yes"' in private_path.read_text()
    assert private_path.stat().st_mode & 0o077 == 0
    assert not list((tmp_path / "robospatial/results").iterdir())
    with pytest.raises(ValueError):
        write_preparation(tmp_path, datasets, provenance={"source": "changed"})
    split_path = tmp_path / "robospatial/splits/environment.jsonl"
    split_path.write_text("tampered")
    with pytest.raises(ValueError):
        write_preparation(tmp_path, datasets, provenance={"source": "fixed"})


def test_journal_normalizes_mixed_device_keys_and_resumes(tmp_path):
    binding = {"model_local": {"max_memory": {0: "22GiB", "cpu": "90GiB"}}}
    journal = RunJournal(tmp_path, binding)
    result, reused = journal.execute(
        "task", {"devices": {1: "gpu", "cpu": "host"}},
        lambda: {"memory": {0: 22, "cpu": 90}},
    )
    assert not reused and result == {"memory": {"0": 22, "cpu": 90}}
    assert 0 in binding["model_local"]["max_memory"]
    normalized = {"model_local": {"max_memory": {"cpu": "90GiB", "0": "22GiB"}}}
    restored = RunJournal(tmp_path, normalized)
    assert restored.binding_digest == journal.binding_digest
    assert restored.execute("task", {"devices": {"cpu": "host", "1": "gpu"}}, dict) == (result, True)
    changed = {"model_local": {"max_memory": {0: "24GiB", "cpu": "90GiB"}}}
    with pytest.raises(ValueError, match="binding changed"):
        RunJournal(tmp_path, changed)


def test_journal_rejects_key_collisions_and_nonfinite_values(tmp_path):
    with pytest.raises(ValueError, match="Duplicate journal JSON key"):
        RunJournal(tmp_path, {"nested": {0: "gpu", "0": "duplicate"}})
    with pytest.raises(ValueError):
        RunJournal(tmp_path, {"nested": {0: float("nan")}})
