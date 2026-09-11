"""Tiny files prove independent KB identity and resume checks; no model loading."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_v2 import Embeddings, ScriptedModel

from spatialcraft.experiments import model_resources
from spatialcraft.experiments.runtime_v2 import build_dataset
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.tools import create_mock_tool_registry

PROJECT = Path(__file__).resolve().parents[1]


def tiny_model(root, alias):
    weights = root / "weights"
    processor = root / "processor"
    weights.mkdir(parents=True)
    processor.mkdir()
    (weights / "model.safetensors").write_bytes(b"tiny deterministic weight bytes")
    (weights / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model.safetensors"}})
    )
    (weights / "config.json").write_text('{"model_type":"fixture"}')
    for name in ("tokenizer.json", "processor_config.json"):
        (processor / name).write_text('{"fixture":1}')
    (processor / "chat_template.jinja").write_text("{{ messages }}")
    value = load_yaml(PROJECT / "configs/models/qwen3.5-9b.yaml")
    value.update(alias=alias, model_id=alias)
    value["local"].update(
        path=str(weights), tokenizer_path=str(processor), revision="fixed-revision"
    )
    value["metadata"]["experiment_auto_placement"] = True
    return ModelConfig.from_dict(value), value


def fixture_models(tmp_path):
    _executor, executor_wire = tiny_model(tmp_path / "executor", "qwen3.5-9b")
    kb, kb_wire = tiny_model(tmp_path / "kb", "qwen3.6-27b-local")
    project = tmp_path / "project"
    configs = project / "configs/models"
    configs.mkdir(parents=True)
    for model in (executor_wire, kb_wire):
        (configs / (model["alias"] + ".yaml")).write_text(json.dumps(model))
    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2", roles={"knowledge_builder": kb.alias}
    )
    models = model_resources.resolve_role_models(project, settings)
    return project, settings, models


def binding_for(models):
    binding = {
        "settings": {"protocol_version": "spatialcraft_v2"},
        "model_path": models["executor"].local.path,
        "model_resources": model_resources.role_model_resources(models),
    }
    model_resources.complete_model_resource_binding(binding)
    return binding


def test_preflight_skips_weights_and_execute_hashes_unique_shared_paths_once(
    tmp_path, monkeypatch
):
    model, _ = tiny_model(tmp_path / "model", "qwen3.5-9b")
    # Hugging Face snapshots can link model.safetensors to a hash-named blob.
    weight_link = Path(model.local.path) / "model.safetensors"
    blob = tmp_path / "hashnamedblob"
    weight_link.rename(blob)
    weight_link.symlink_to(blob)
    symlink = tmp_path / "same-checkpoint"
    symlink.symlink_to(model.local.path, target_is_directory=True)
    alias = replace(
        model, alias="shared-alias", local=replace(model.local, path=str(symlink))
    )
    models = {"executor": model, "scorer": model, "knowledge_builder": alias}
    original = model_resources.sha256_file
    calls = Counter()

    def count(path):
        calls[str(Path(path).resolve())] += 1
        return original(path)

    monkeypatch.setattr(model_resources, "sha256_file", count)
    manifest = model_resources.role_model_resources(models)
    assert manifest["weight_verification"] == "pending_execute"
    assert len(manifest["resources"]) == 1
    assert len({row["resource_id"] for row in manifest["roles"].values()}) == 1
    assert str(blob.resolve()) not in calls
    calls.clear()
    binding = {"model_resources": manifest}
    model_resources.complete_model_resource_binding(binding)
    assert binding["model_resources"]["weight_verification"] == "complete"
    assert set(calls.values()) == {1}
    assert calls[str(weight_link.resolve())] == 1
    assert binding["weights_sha256"]["model.safetensors"]


@pytest.mark.parametrize(
    "resource",
    ["weights", "tokenizer.json", "processor_config.json", "chat_template.jinja"],
)
def test_kb_resource_change_rejects_old_runtime_and_cached_journal(tmp_path, resource):
    project, settings, models = fixture_models(tmp_path)
    binding = binding_for(models)
    runtime = SimpleNamespace(
        settings=settings,
        model=models["executor"],
        local=ScriptedModel(),
        project=project,
        output=tmp_path / "run",
        binding=binding,
        embedder=Embeddings(),
        tools=create_mock_tool_registry(),
    )
    pipeline = build_dataset(runtime, "fixture")
    pipeline.journal.execute(
        "saved_kb_output", {"input": 1}, lambda: {"lesson": "before mutation"}
    )
    assert build_dataset(runtime, "fixture").journal.read_committed(
        "saved_kb_output"
    ) == {"lesson": "before mutation"}
    kb = models["knowledge_builder"].local
    path = (
        Path(kb.path) / "model.safetensors"
        if resource == "weights"
        else Path(kb.tokenizer_path) / resource
    )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="resources changed"):
        build_dataset(runtime, "fixture")
    refreshed = binding_for(models)
    executor_id = binding["model_resources"]["roles"]["executor"]["resource_id"]
    assert (
        refreshed["model_resources"]["resources"][executor_id]
        == binding["model_resources"]["resources"][executor_id]
    )
    runtime.binding = refreshed
    with pytest.raises(ValueError, match="binding"):
        build_dataset(runtime, "fixture")
    assert runtime.local.requests == []


def test_external_processor_path_revision_and_file_inventory_are_bound(tmp_path):
    _, _, models = fixture_models(tmp_path)
    binding = binding_for(models)
    kb = models["knowledge_builder"]
    changed = {
        **models,
        "knowledge_builder": replace(
            kb, local=replace(kb.local, revision="another-revision")
        ),
    }
    with pytest.raises(ValueError, match="identity differs"):
        model_resources.validate_runtime_model_resources(changed, binding)
    (Path(kb.local.tokenizer_path) / "added_tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="resources changed"):
        model_resources.validate_runtime_model_resources(models, binding)


def test_preflight_change_and_missing_kb_shard_cannot_be_finalized(tmp_path):
    _, _, models = fixture_models(tmp_path)
    binding = {"model_resources": model_resources.role_model_resources(models)}
    kb_path = Path(models["knowledge_builder"].local.path)
    (kb_path / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed since preflight"):
        model_resources.complete_model_resource_binding(binding)
    (kb_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"missing": "absent.safetensors"}})
    )
    with pytest.raises(ValueError, match="Missing local role model shard"):
        model_resources.role_model_resources(models)


def test_v2_execute_rejects_executor_only_binding():
    with pytest.raises(ValueError, match="every local role model"):
        model_resources.complete_model_resource_binding(
            {"settings": {"protocol_version": "spatialcraft_v2"}}
        )
