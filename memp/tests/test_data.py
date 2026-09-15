"""Local fixtures exercise real split preparation and frozen-data checks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from memp import data
from spatialcraft.experiments.protocol import public_task
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import canonical_json_bytes, sha256_file


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row.to_dict()) for row in rows))


def _task(name, task_id, image):
    return TaskSample.from_dict(
        {
            "task_id": task_id,
            "dataset": name,
            "question": f"Question {task_id}?",
            "reference_answer": "PRIVATE_ANSWER",
            "metadata": {
                "private_note": "PRIVATE_METADATA",
                "question_type": "spatial",
            },
            "images": [
                {
                    "image_id": "fixture-image",
                    "uri": str(image),
                    "sha256": sha256_file(image),
                    "metadata": {"private_note": "PRIVATE_IMAGE_METADATA"},
                }
            ],
        }
    )


def _public(task):
    return TaskSample.from_dict(public_task(task))


def _fixture(tmp_path, monkeypatch, name="robospatial", *, sat_fallback=False):
    output, preparation, benchmark = (
        tmp_path / item for item in ("output", "preparation", "benchmark")
    )
    source_root = (
        tmp_path / "view_preparation" if name == "viewspatial" else preparation
    )
    image = tmp_path / "image.png"
    Image.new("RGB", (3, 2), color="blue").save(image)
    environment = tuple(_task(name, f"env-{i}", image) for i in range(2))
    deployment = (_task(name, "test-0", image),)
    monkeypatch.setitem(
        data.rag_data.ENVIRONMENT_COUNTS, name, 1 if sat_fallback else 2
    )

    def prepare_deployment(out, prep, bench, names):
        assert (out, prep, bench, names) == (output, preparation, benchmark, [name])
        root = out / "data" / name
        public = tuple(_public(task) for task in deployment)
        _write_rows(root / "public.jsonl", public)
        _write_rows(root / "private.jsonl", deployment)
        identity = {
            "task_ids": [task.task_id for task in public],
            "count": len(public),
            "public_sha256": sha256_file(root / "public.jsonl"),
            "private_sha256": sha256_file(root / "private.jsonl"),
            "media": {str(image): sha256_file(image)},
        }
        if name == "viewspatial":
            identity["split_preparation_path"] = str(source_root)
        return {"datasets": {name: identity}}

    monkeypatch.setattr(data.rag_data, "prepare_deployment", prepare_deployment)
    source_public = source_root / name / "splits/environment.jsonl"
    source_private = source_root / name / "verification/environment.jsonl"

    def bind_source(private=environment):
        _write_rows(source_public, tuple(_public(task) for task in environment))
        _write_rows(source_private, tuple(reversed(private)))
        (source_root / "manifest.json").write_bytes(
            canonical_json_bytes(
                {
                    "input_files": {
                        str(path.relative_to(source_root)): {
                            "sha256": sha256_file(path)
                        }
                        for path in (source_public, source_private)
                    }
                }
            )
        )

    adapters = []
    if sat_fallback:
        parquet = benchmark / "SAT/SAT_val.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"fixture adapter supplies validation rows")

        class Adapter:
            closed = False

            def load_split(self, split):
                assert split == "validation"
                return environment

            def close(self):
                self.closed = True

        def create(dataset):
            assert dataset == "sat"
            adapter = Adapter()
            adapters.append(adapter)
            return adapter

        monkeypatch.setattr(
            data.rag_data,
            "create_default_registry",
            lambda root: SimpleNamespace(create=create),
        )
    else:
        bind_source()
    return SimpleNamespace(
        output=output,
        preparation=preparation,
        benchmark=benchmark,
        name=name,
        source_private=source_private,
        environment=environment,
        bind_source=bind_source,
        adapters=adapters,
        prepare=lambda: data.prepare_inputs(output, preparation, benchmark, [name]),
    )


@pytest.mark.parametrize(
    "name", ["robospatial", "erqa", "omni3d", "viewspatial", "sat"]
)
def test_private_references_stay_separate_and_resume(tmp_path, monkeypatch, name):
    fixture = _fixture(tmp_path, monkeypatch, name)
    manifest = fixture.prepare()
    identity = manifest["datasets"][name]
    assert identity["environment_private_source"]["path"] == str(fixture.source_private)
    for phase in ("environment", "deployment"):
        public, private = data.load_split(fixture.output, name, identity, phase)
        assert [task.task_id for task in public] == [task.task_id for task in private]
        assert all(task.reference_answer is None for task in public)
        assert all(task.reference_answer == "PRIVATE_ANSWER" for task in private)
        assert all(task.metadata.get("private_note") is None for task in public)
        assert all(not image.metadata for task in public for image in task.images)
        assert [task.to_dict() for task in public] == [
            public_task(task) for task in private
        ]
    private_path = fixture.output / identity["environment_private_file"]
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert (
        b"PRIVATE_"
        not in (fixture.output / "data" / name / "environment.jsonl").read_bytes()
    )
    assert fixture.prepare() == manifest
    data.verify_inputs(fixture.output, manifest)


def test_sat_fallback_replays_validation_selection_with_private_labels(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch, "sat", sat_fallback=True)
    manifest = fixture.prepare()
    identity = manifest["datasets"]["sat"]
    public, private = data.load_split(fixture.output, "sat", identity)
    assert len(public) == len(private) == 1
    assert private[0].reference_answer == "PRIVATE_ANSWER"
    assert public[0].metadata["experiment_split"] == "environment"
    assert identity["environment_source"] == identity["environment_private_source"]
    assert len(fixture.adapters) == 2
    assert all(adapter.closed for adapter in fixture.adapters)
    fixture.prepare()
    assert len(fixture.adapters) == 2


@pytest.mark.parametrize(
    "mutation", ["id", "question", "missing", "duplicate", "no_reference"]
)
def test_reject_source_reference_misalignment(tmp_path, monkeypatch, mutation):
    fixture = _fixture(tmp_path, monkeypatch)
    first, second = fixture.environment
    private = {
        "id": (replace(first, task_id="different-id"), second),
        "question": (replace(first, question="A different question?"), second),
        "missing": (first,),
        "duplicate": (first, first),
        "no_reference": (replace(first, reference_answer=None), second),
    }[mutation]
    fixture.bind_source(private)
    with pytest.raises(ValueError, match="mismatch|Missing private reference"):
        fixture.prepare()


def test_reject_private_source_checksum_change(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    with fixture.source_private.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(
        ValueError, match="Private environment source checksum mismatch"
    ):
        fixture.prepare()


def test_reject_frozen_private_tamper_in_verify_and_load(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = fixture.prepare()
    identity = manifest["datasets"][fixture.name]
    with (fixture.output / identity["environment_private_file"]).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="Frozen MemP input changed"):
        data.verify_inputs(fixture.output, manifest)
    with pytest.raises(ValueError, match="Frozen MemP input changed"):
        data.load_split(fixture.output, fixture.name, identity)


def test_alignment_checked_even_when_private_checksum_matches(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = fixture.prepare()
    identity = dict(manifest["datasets"][fixture.name])
    private_path = fixture.output / identity["environment_private_file"]
    private = data.read_tasks(private_path)
    _write_rows(
        private_path, (replace(private[0], question="Wrong question"), private[1])
    )
    identity["environment_private_sha256"] = sha256_file(private_path)
    with pytest.raises(ValueError, match="Public/private content mismatch"):
        data.load_split(fixture.output, fixture.name, identity)


def test_zero_reference_is_valid_and_public_labels_are_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture.bind_source(
        tuple(replace(task, reference_answer=0) for task in fixture.environment)
    )
    manifest = fixture.prepare()
    identity = dict(manifest["datasets"][fixture.name])
    public, private = data.load_split(fixture.output, fixture.name, identity)
    assert private[0].reference_answer == 0
    public_path = fixture.output / "data" / fixture.name / "environment.jsonl"
    _write_rows(public_path, (replace(public[0], reference_answer=0), public[1]))
    identity["environment_sha256"] = sha256_file(public_path)
    with pytest.raises(ValueError, match="Public split contains private annotations"):
        data.load_split(fixture.output, fixture.name, identity)


def test_reject_changed_preparation_and_unknown_phase(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = fixture.prepare()
    with pytest.raises(ValueError, match="preparation inputs changed"):
        data.prepare_inputs(
            fixture.output, tmp_path / "different", fixture.benchmark, [fixture.name]
        )
    with pytest.raises(ValueError, match="Unknown MemP phase"):
        data.load_split(
            fixture.output, fixture.name, manifest["datasets"][fixture.name], "train"
        )
