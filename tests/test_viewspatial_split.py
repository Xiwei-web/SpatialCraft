"""ViewSpatial per-category halves and deployment-only baseline scope."""

from collections import Counter
from pathlib import Path

import pytest

from spatialcraft.experiments import run_api_baseline as baseline
from spatialcraft.experiments import viewspatial_split as split
from spatialcraft.experiments.protocol import public_task
from spatialcraft.schemas import AnswerType, TaskSample
from spatialcraft.storage.atomic_io import atomic_write_json


@pytest.fixture
def tasks():
    # Even categories between odd categories must not consume the alternation.
    return tuple(
        TaskSample(
            dataset="viewspatial",
            task_id=f"{category}-{i}",
            question="Pick one",
            choices=("left", "right"),
            reference_answer="A",
            answer_type=AnswerType.MULTIPLE_CHOICE,
            metadata={
                "question_type": category,
                "choice_labels": ["A", "B"],
                "reference_answer_text": "PRIVATE_REFERENCE",
            },
        )
        for category, count in (("a", 4), ("b", 3), ("c", 4), ("d", 2), ("e", 5))
        for i in range(count)
    )


def test_exact_odd_remainders_idempotent_and_no_labels(tmp_path, tasks):
    manifest = split.write_split(tmp_path, tasks, {"source": "fixed"})
    assert manifest["environment"] == manifest["deployment"] == 9
    assert manifest["categories"]["environment"] == {
        "a": 2,
        "b": 2,
        "c": 2,
        "d": 1,
        "e": 2,
    }
    assert manifest["categories"]["deployment"] == {
        "a": 2,
        "b": 1,
        "c": 2,
        "d": 1,
        "e": 3,
    }
    assert not set(manifest["task_ids"]["environment"]) & set(
        manifest["task_ids"]["deployment"]
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json*")}
    assert (
        split.write_split(tmp_path, tuple(reversed(tasks)), {"source": "fixed"})
        == manifest
    )
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (
        split.validate_split(tmp_path, {"source": "fixed"}, expected_total=18)
        == manifest
    )
    public = split.rows(tmp_path / "viewspatial/splits/deployment.jsonl")
    private = split.rows(tmp_path / "viewspatial/verification/deployment.jsonl")
    assert all(
        t.reference_answer is None and "reference_answer_text" not in t.metadata
        for t in public
    )
    assert [public_task(t) for t in private] == [t.to_dict() for t in public]


def test_reject_changed_source_and_tampered_file(tmp_path, tasks):
    split.write_split(tmp_path, tasks, {"source": "fixed"})
    with pytest.raises(ValueError, match="source/protocol/count"):
        split.validate_split(tmp_path, {"source": "changed"}, expected_total=18)
    path = tmp_path / "viewspatial/splits/deployment.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        split.validate_split(tmp_path, {"source": "fixed"}, expected_total=18)


def test_duplicate_source_ids_rejected(tmp_path, tasks):
    with pytest.raises(ValueError, match="Duplicate task IDs"):
        split.write_split(tmp_path, (tasks[0], tasks[0]), {})


def test_baseline_copies_only_deployment_and_reuses_preparation(
    tmp_path, tasks, monkeypatch
):
    preparation = tmp_path / "preparation/unused"
    root = preparation.parent / "viewspatial_seed42_v1"
    manifest = split.write_split(root, tasks, {"source": "fixed"})
    calls = []

    def prepared(path, benchmark_root):
        assert path == root
        calls.append(path)
        return split.validate_split(root, {"source": "fixed"}, expected_total=18)

    monkeypatch.setattr(split, "prepare_viewspatial", prepared)
    monkeypatch.setitem(baseline.COUNTS, "viewspatial", 9)

    def forbidden(*args, **kwargs):
        raise AssertionError("Other datasets should not be loaded")

    monkeypatch.setattr(baseline, "create_default_registry", forbidden)
    output = tmp_path / "run"
    result = baseline.prepare(
        output, preparation, tmp_path / "bench", names=["viewspatial"]
    )
    assert set(result["datasets"]) == {"viewspatial"}
    identity = result["datasets"]["viewspatial"]
    assert identity["count"] == 9 and identity["evaluation_split"] == "deployment"
    assert identity["task_ids"] == manifest["task_ids"]["deployment"]
    assert not set(identity["task_ids"]) & set(manifest["task_ids"]["environment"])
    assert Counter(
        t.metadata["experiment_split"]
        for t in baseline.read_tasks(output / "data/viewspatial/public.jsonl")
    ) == {"deployment": 9}
    assert (
        baseline.prepare(output, preparation, tmp_path / "bench", names=["viewspatial"])
        == result
    )
    assert len(calls) == 1


def test_old_full_dataset_cache_rejected_before_model_calls(tmp_path):
    atomic_write_json(
        tmp_path / "data/manifest.json", {"datasets": {"viewspatial": {"count": 5712}}}
    )
    with pytest.raises(ValueError, match="2856 tasks"):
        baseline.prepare(
            tmp_path, Path("/unused"), Path("/unused"), names=["viewspatial"]
        )
    with pytest.raises(ValueError, match="2856 tasks"):
        baseline.run_dataset(
            tmp_path,
            "viewspatial",
            None,
            {"datasets": {"viewspatial": {"count": 5712}}},
            "code",
            None,
        )


def test_original_default_datasets_unchanged():
    assert baseline.DEFAULT_DATASETS == ("robospatial", "erqa", "omni3d", "sat")
    assert baseline.COUNTS["viewspatial"] == 2856
