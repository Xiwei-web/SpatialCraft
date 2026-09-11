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


def duplicate_pair(tmp_path, *, different_question=False):
    from spatialcraft.schemas import ImageInput
    from spatialcraft.storage.atomic_io import sha256_file

    image = tmp_path / "first.png"
    copy = tmp_path / "copy.png"
    image.write_bytes(b"same-public-image-content")
    copy.write_bytes(image.read_bytes())
    a = TaskSample(
        task_id="different-id-A",
        dataset="robospatial",
        question="Which is left?",
        choices=("A", "B"),
        reference_answer="A",
        images=(ImageInput(uri=str(image), sha256=sha256_file(image)),),
        metadata={"question_type": "direction"},
    )
    b = replace(
        a,
        task_id="different-id-B",
        images=(ImageInput(uri=str(copy), sha256=sha256_file(copy)),),
        question="Which is nearer?" if different_question else a.question,
    )
    return a, b


def test_strict_preparation_rejects_cross_id_exact_task_duplicates_before_writes(
    tmp_path,
):
    rows = duplicate_pair(tmp_path)
    output = tmp_path / "strict"
    with pytest.raises(ValueError, match="Exact public task content overlaps"):
        write_preparation(
            output,
            {"robospatial": rows},
            exact_duplicate_policy="reject_cross_split_v1",
        )
    assert not output.exists()


def test_strict_policy_allows_shared_images_with_distinct_questions_and_checks_actual_bytes(
    tmp_path,
):
    from spatialcraft.experiments.prepare_v2 import validate_content_isolation

    rows = duplicate_pair(tmp_path, different_question=True)
    output = tmp_path / "strict"
    manifest = write_preparation(
        output, {"robospatial": rows}, exact_duplicate_policy="reject_cross_split_v1"
    )
    assert manifest["datasets"]["robospatial"]["shared_image_count"] == 1
    assert (
        manifest["datasets"]["robospatial"]["shared_question_image_choices_count"] == 0
    )
    assert manifest["split_protocol"]["repartitioned_for_duplicates"] is False
    audit = validate_content_isolation(manifest, "robospatial", (rows[0],), (rows[1],))
    assert audit["shared_image_count"] == 1
    assert len(audit["training_task_content_sha256"]) == 1
    from pathlib import Path

    Path(rows[1].images[0].uri).write_bytes(b"changed-after-preparation")
    with pytest.raises(ValueError, match="image checksum changed"):
        validate_content_isolation(manifest, "robospatial", (rows[0],), (rows[1],))


def test_default_manifest_keeps_report_only_policy_and_existing_split_bytes(tmp_path):
    from spatialcraft.experiments.prepare_v2 import validate_content_isolation

    rows = duplicate_pair(tmp_path)
    output = tmp_path / "existing"
    report = write_preparation(output, {"robospatial": rows})
    before = {str(path): path.read_bytes() for path in output.rglob("*.json*")}
    assert "exact_duplicate_policy" not in report["split_protocol"]
    assert (
        validate_content_isolation(report, "robospatial", (rows[0],), (rows[1],))
        is None
    )
    assert write_preparation(output, {"robospatial": rows}) == report
    with pytest.raises(ValueError, match="Exact public task content overlaps"):
        write_preparation(
            output,
            {"robospatial": rows},
            exact_duplicate_policy="reject_cross_split_v1",
        )
    assert {str(path): path.read_bytes() for path in output.rglob("*.json*")} == before


def test_preflight_recomputes_strict_duplicate_claim_instead_of_trusting_zero_count(
    tmp_path,
):
    from pathlib import Path

    from spatialcraft.experiments.prepare_v2 import (
        CONTENT_FINGERPRINT,
        EXACT_DUPLICATE_POLICY,
    )
    from spatialcraft.experiments.run import preflight
    from spatialcraft.storage.atomic_io import atomic_write_json

    rows = duplicate_pair(tmp_path)
    output = tmp_path / "prep"
    manifest = write_preparation(output, {"robospatial": rows})
    manifest["split_protocol"].update(
        exact_duplicate_policy=EXACT_DUPLICATE_POLICY,
        content_fingerprint=CONTENT_FINGERPRINT,
    )
    manifest["datasets"]["robospatial"]["shared_question_image_choices_count"] = 0
    atomic_write_json(output / "manifest.json", manifest)
    project = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="Exact public task content overlaps"):
        preflight(
            project,
            output,
            project / "configs/experiments/qwen35_9b_spatialcraft_v2.yaml",
            ["robospatial"],
        )


def test_preflight_shared_image_distinct_question_passes_strict_dataset_validation(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from spatialcraft.experiments import run

    rows = duplicate_pair(tmp_path, different_question=True)
    output = tmp_path / "prep"
    write_preparation(
        output, {"robospatial": rows}, exact_duplicate_policy="reject_cross_split_v1"
    )
    project = Path(__file__).resolve().parents[1]

    def reached_models(*args):
        raise RuntimeError("passed-strict-dataset-validation-before-models")

    monkeypatch.setattr(run, "load_yaml", reached_models)
    with pytest.raises(RuntimeError, match="passed-strict-dataset"):
        run.preflight(
            project,
            output,
            project / "configs/experiments/qwen35_9b_spatialcraft_v2.yaml",
            ["robospatial"],
        )
