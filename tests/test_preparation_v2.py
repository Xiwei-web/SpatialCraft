import json
from dataclasses import replace

import pytest

from spatialcraft.experiments.prepare_v2 import sat_splits, write_preparation
from spatialcraft.schemas import TaskSample, TaskSplit


def task(index, split):
    return TaskSample(
        task_id=f"{split}-{index}",
        dataset="sat",
        question=f"Question {index}",
        reference_answer="PRIVATE",
        split=TaskSplit.VALIDATION if split == "val" else TaskSplit.TEST,
        metadata={
            "question_type": "direction",
            "source_split": split,
            "secret": "not_public",
        },
    )


def test_sat_validation_sampling_keeps_complete_official_test_and_is_label_independent(
    tmp_path,
):
    validation = tuple(task(i, "val") for i in range(8))
    test = tuple(task(i, "test") for i in range(3))
    train, deploy = sat_splits(validation, test)
    assert len(train) == len(deploy) == 3
    assert [r.task_id for r in deploy] == [r.task_id for r in test]
    assert not {r.task_id for r in train} & {r.task_id for r in deploy}
    modified = tuple(replace(r, reference_answer="OTHER") for r in validation)
    assert [r.task_id for r in sat_splits(modified, test)[0]] == [
        r.task_id for r in train
    ]
    report = write_preparation(tmp_path, {"sat": test}, sat_validation=validation)
    assert report["datasets"]["sat"]["deployment"] == 3
    assert report["split_protocol"]["test_labels_used_for_sampling"] is False
    public = [
        json.loads(line)
        for line in (tmp_path / "sat/splits/environment.jsonl").read_text().splitlines()
    ]
    assert all(
        r["reference_answer"] is None and "secret" not in r["metadata"] for r in public
    )
    assert (
        write_preparation(tmp_path, {"sat": test}, sat_validation=validation) == report
    )
    with pytest.raises(ValueError, match="binding changed"):
        write_preparation(tmp_path, {"sat": test}, sat_validation=validation, seed=43)


def test_sat_rejects_overlap_and_small_validation_pool():
    with pytest.raises(ValueError, match="at least"):
        sat_splits((task(0, "val"),), tuple(task(i, "test") for i in range(2)))
    with pytest.raises(ValueError, match="overlap"):
        sat_splits((task(0, "test"),), (task(0, "test"),))
