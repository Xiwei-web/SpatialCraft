"""Frozen source provenance, real v2 deployment wiring and offline transfer checks."""

from dataclasses import replace
from pathlib import Path

import pytest
from test_runtime_v2 import Embeddings, ScriptedModel, tasks

from spatialcraft.experiments.accumulation import KnowledgeState
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.run_frozen_transfer import (
    FrozenTransferDeployment,
    main,
    read_frozen_source,
    target_contract,
    validate_target_location,
    validate_target_tasks,
)
from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.experience.index import ModelEmbedder
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.schemas import ExperienceItem, TaskSplit
from spatialcraft.storage.atomic_io import atomic_write_json, read_json, sha256_file
from spatialcraft.tools import create_mock_tool_registry

PROJECT = Path(__file__).resolve().parents[1]


def source_journal(path):
    state = KnowledgeState(
        ExperienceBank(
            (
                ExperienceItem(
                    experience_id="E1",
                    condition="When comparing observer directions",
                    action="Identify observer axes first.",
                ),
            )
        ).freeze(),
        SeedCatalog.pool().freeze(),
    )
    binding = {
        "dataset": "robospatial",
        "code_sha256": "source-code",
        "settings": {
            "protocol_version": "spatialcraft_v2",
            "backbone": "qwen3.5-9b",
            "roles": {"executor": "qwen3.5-9b", "knowledge_builder": "qwen3.5-9b"},
        },
    }
    journal = RunJournal(path, binding)
    journal.execute(
        "initial",
        {"task_ids": ["source-train"]},
        lambda: {
            **state.to_dict(),
            "dataset": "robospatial",
            "training_task_ids": ["source-train"],
        },
    )
    journal.execute(
        "frozen_deployment_snapshot", {"snapshot_id": state.snapshot_id}, state.to_dict
    )
    return state


def files(path):
    return {
        str(p.relative_to(path)): (sha256_file(p), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file()
    }


def test_source_reader_verifies_commits_without_creating_or_touching_files(tmp_path):
    source_path = tmp_path / "source"
    state = source_journal(source_path)
    before = files(source_path)
    source = read_frozen_source(source_path, state.snapshot_id)
    assert source.knowledge.snapshot_id == state.snapshot_id
    assert source.provenance["source_executor"] == "qwen3.5-9b"
    assert source.provenance["source_knowledge_builder"] == "qwen3.5-9b"
    source.verify_unchanged()
    assert files(source_path) == before


@pytest.mark.parametrize(
    "component",
    [
        "journal.json",
        "stages/initial/result.json",
        "stages/frozen_deployment_snapshot/result.json",
    ],
)
def test_source_checksum_tampering_is_rejected(tmp_path, component):
    source_path = tmp_path / "source"
    source_journal(source_path)
    path = source_path / component
    value = read_json(path)
    if component == "journal.json":
        value["binding"]["dataset"] = "erqa"
    else:
        value["result"]["tampered"] = True
    atomic_write_json(path, value)
    with pytest.raises(ValueError, match="checksum"):
        read_frozen_source(source_path)


def test_partial_source_and_wrong_expected_snapshot_are_rejected(tmp_path):
    source_path = tmp_path / "source"
    source_journal(source_path)
    with pytest.raises(ValueError, match="requested identity"):
        read_frozen_source(source_path, "knowledge_wrong")
    (source_path / "stages/frozen_deployment_snapshot/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        read_frozen_source(source_path)


def test_source_and_target_are_separate_and_source_training_ids_cannot_transfer(
    tmp_path,
):
    source_path = tmp_path / "source"
    source_journal(source_path)
    source = read_frozen_source(source_path)
    task = replace(
        tasks(tmp_path)[0],
        dataset="robospatial",
        task_id="source-train",
        split=TaskSplit.TEST,
    )
    with pytest.raises(ValueError, match="overlaps"):
        validate_target_tasks(source, (task,))
    with pytest.raises(ValueError, match="source benchmark"):
        validate_target_tasks(source, (replace(task, dataset="erqa", task_id="other"),))
    with pytest.raises(ValueError, match="deployment/test"):
        validate_target_tasks(
            source, (replace(task, split=TaskSplit.TRAIN, task_id="other"),)
        )
    with pytest.raises(ValueError, match="separate"):
        validate_target_location(source, source_path / "target")


def test_target_tokenizer_checks_frozen_skills_without_rewriting(tmp_path):
    source_path = tmp_path / "source"
    source_journal(source_path)
    source = read_frozen_source(source_path)
    before = source.knowledge.to_dict()
    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2", backbone="qwen3.6-27b-local"
    )
    contract = target_contract(PROJECT, settings, source, token_counter=lambda text: 55)
    assert contract["target_executor"] == "qwen3.6-27b-local"
    assert contract["target_reasoning_mode"] == "instruct"
    assert set(contract["target_skill_token_counts"].values()) == {55}
    assert contract["target_tokenizer"]["files_sha256"]
    with pytest.raises(ValueError, match="will not be rewritten"):
        target_contract(PROJECT, settings, source, token_counter=lambda text: 1025)
    assert source.knowledge.to_dict() == before


def test_transfer_uses_production_v2_retrieval_and_rollout_without_training(tmp_path):
    source_path = tmp_path / "source"
    state = source_journal(source_path)
    source = read_frozen_source(source_path)
    before = files(source_path)
    model = ScriptedModel()
    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2",
        backbone="qwen3.6-27b-local",
        embedding_model="text-embedding-3-large",
        experiment_name="cross_model_frozen_deployment",
    )
    target = ExperimentRuntime(
        PROJECT, tmp_path / "target", settings, {"test": "frozen-transfer"}
    )
    target.local = model
    target.embedding = Embeddings()
    target.embedder = ModelEmbedder(target.embedding)
    target.tools = create_mock_tool_registry()
    pipeline = target.dataset("robospatial")
    contract = target_contract(PROJECT, settings, source, token_counter=lambda text: 55)
    runner = FrozenTransferDeployment(source, pipeline, contract)
    heldout = replace(
        tasks(tmp_path)[0],
        dataset="robospatial",
        task_id="new-test",
        split=TaskSplit.TEST,
    )
    result = runner.deploy((heldout,))
    assert result["accuracy"] == 1
    assert result["transfer"]["source_snapshot_id"] == state.snapshot_id
    assert result["transfer"]["target_executor"] == "qwen3.6-27b-local"
    assert files(source_path) == before
    assert not (pipeline.journal.root / "stages/initial").exists()
    assert not (pipeline.journal.root / "stages/tasks").exists()
    assert not (pipeline.journal.root / "stages/frozen_deployment_snapshot").exists()
    assert (pipeline.journal.root / "results/frozen_transfer.json").is_file()
    operations = [request.metadata["operation"] for request in model.requests]
    assert "retrieval.decomposition" in operations and "retrieval.rewrite" in operations
    assert "execution.deployment" in operations
    assert not any(operation.startswith("experience.") for operation in operations)
    assert not any(
        operation
        in {
            "execution.accumulation",
            "skill.semantic_gradient",
            "skill.candidate_generation",
        }
        for operation in operations
    )
    for request in model.requests:
        assert "SECRET" not in str(request)
        assert request.model_alias == "qwen3.6-27b-local"
    count = len(model.requests)
    assert runner.deploy((heldout,)) == result
    assert len(model.requests) == count


def test_source_changed_after_preflight_is_rejected_before_execution(tmp_path):
    source_path = tmp_path / "source"
    source_journal(source_path)
    source = read_frozen_source(source_path)
    with (source_path / "journal.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed after validation"):
        source.verify_unchanged()


def test_transfer_cli_preflight_does_not_construct_execution_runtime(
    monkeypatch, tmp_path, capsys
):
    from spatialcraft.experiments import run_frozen_transfer as module

    source_path = tmp_path / "source"
    source_journal(source_path)
    heldout = replace(
        tasks(tmp_path)[0],
        dataset="robospatial",
        task_id="target-test",
        split=TaskSplit.TEST,
    )
    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2", backbone="qwen3.6-27b-local"
    )
    called = []

    def preflight(*args):
        called.append(args[-1])
        return (
            settings,
            {"robospatial": {"deployment": (heldout,), "environment": ()}},
            {},
            {},
        )

    original_contract = module.target_contract
    monkeypatch.setattr(module, "preflight", preflight)
    monkeypatch.setattr(
        module,
        "target_contract",
        lambda project, config, source, **kwargs: original_contract(
            project, config, source, token_counter=lambda text: 10
        ),
    )
    report_path = tmp_path / "report.json"
    main(
        [
            "--source-journal",
            str(source_path),
            "--preparation",
            str(tmp_path / "prep"),
            "--config",
            str(tmp_path / "target-config.yaml"),
            "--output",
            str(tmp_path / "target"),
            "--report",
            str(report_path),
        ]
    )
    report = read_json(report_path)
    assert report["status"] == "preflight_passed_not_run"
    assert report["accumulation_required"] is False
    assert report["api_target_executor_supported"] is False
    assert called == [["robospatial"]]
    assert not (tmp_path / "target").exists()
    assert "preflight_passed_not_run" in capsys.readouterr().out


def test_strict_source_binding_rejects_same_public_task_under_different_id(tmp_path):
    from spatialcraft.experiments.prepare_v2 import (
        CONTENT_FINGERPRINT,
        EXACT_DUPLICATE_POLICY,
        public_content_fingerprints,
    )
    from spatialcraft.schemas import TaskSample

    source_path = tmp_path / "source"
    source_journal(source_path)
    source_task = TaskSample(
        task_id="source-train",
        dataset="robospatial",
        question="Same public question",
        choices=("A", "B"),
    )
    manifest = read_json(source_path / "journal.json")
    old_binding = manifest["binding_sha256"]
    manifest["binding"]["content_isolation"] = {
        "robospatial": {
            "policy": EXACT_DUPLICATE_POLICY,
            "fingerprint": CONTENT_FINGERPRINT,
            "training_task_content_sha256": public_content_fingerprints((source_task,)),
        }
    }
    from spatialcraft.experiments.journal import digest

    manifest["binding_sha256"] = digest(manifest["binding"])
    atomic_write_json(source_path / "journal.json", manifest)
    for path in (source_path / "stages").rglob("*.json"):
        value = read_json(path)
        if value.get("binding_sha256") == old_binding:
            value["binding_sha256"] = manifest["binding_sha256"]
            atomic_write_json(path, value)
    source = read_frozen_source(source_path)
    heldout = replace(source_task, task_id="brand-new-id", split=TaskSplit.TEST)
    with pytest.raises(ValueError, match="exact public task content overlaps"):
        validate_target_tasks(source, (heldout,))
    validate_target_tasks(
        source, (replace(heldout, question="Different public question"),)
    )
