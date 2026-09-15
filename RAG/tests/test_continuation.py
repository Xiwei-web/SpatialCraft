"""Offline checks for reduced-budget continuation without repeating paid calls."""

import json
import shutil
from dataclasses import replace

import pytest

from RAG.continuation import continuation_spec, model_key, rollout_counts
from RAG.runner import run_dataset
from RAG.tests.test_rag import FakeEmbeddings, FakeProvider, response, task
from spatialcraft.experiments.protocol import public_task
from spatialcraft.models.interfaces import ModelRequestError
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.storage.atomic_io import canonical_json_bytes


@pytest.fixture
def interrupted(tmp_path):
    source = tmp_path / "source"
    data = source / "data/sat"
    data.mkdir(parents=True)
    previous = [task(f"prior-{i}") for i in range(3)]
    current = replace(task("held-out"), reference_answer="B")
    for filename, rows in [
        ("environment", [public_task(t) for t in previous]),
        ("public", [public_task(current)]),
        ("private", [current.to_dict()]),
    ]:
        (data / f"{filename}.jsonl").write_bytes(
            b"".join(canonical_json_bytes(row) for row in rows)
        )
    from pathlib import Path

    config = ModelConfig.from_dict(
        load_yaml(
            Path(__file__).resolve().parents[2]
            / "configs/models/gpt-5.4-mini-baseline.yaml"
        )
    )
    options = {"environment_rollouts": 4, "top_k": 3}
    binding = {
        "policy": "offline-test",
        "code_sha256": "original",
        "datasets": ["sat"],
        "model": config.model_id,
        "options": options,
    }
    identity = {"media": {}}

    class InterruptedProvider(FakeProvider):
        def generate(self, request):
            if len(self.requests) == 7:
                raise ModelRequestError("synthetic interruption")
            self.requests.append(request)
            return response(f"Final Answer: original-{len(self.requests)}")

    with pytest.raises(ModelRequestError):
        run_dataset(
            source,
            "sat",
            identity,
            binding,
            config,
            config,
            options,
            FakeEmbeddings(),
            "environment",
            provider=InterruptedProvider(),
        )
    target = tmp_path / "target"
    shutil.copytree(source / "data", target / "data")
    new_options = {
        **options,
        "environment_rollouts": 1,
        "environment_resume": continuation_spec(source, ["sat"], 7),
    }
    new_binding = {**binding, "code_sha256": "continuation", "options": new_options}
    return source, target, identity, new_binding, config, new_options


def invoke(case, provider=None, **kwargs):
    _source, target, identity, binding, config, options = case
    return run_dataset(
        target,
        "sat",
        identity,
        binding,
        config,
        config,
        options,
        FakeEmbeddings(),
        "all",
        provider=provider or FakeProvider(),
        **kwargs,
    )


def test_preserve_all_prior_outputs_and_only_generate_unstarted_task(interrupted):
    source, target, *_ = interrupted
    before = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    provider = FakeProvider()
    report = invoke(interrupted, provider)
    assert len(provider.requests) == 2  # one new environment call + one deployment
    assert report["environment"]["record_count"] == 8
    assert report["environment"]["rollouts_per_task"] is None
    assert report["environment"]["prior_model_calls"] == 7
    assert report["environment"]["new_model_calls"] == 1
    assert report["environment"]["rollout_count_histogram"] == {4: 1, 3: 1, 1: 1}
    assert report["deployment"]["evaluated"] == 1
    for i in range(7):
        key = model_key(*divmod(i, 4))
        prior = json.loads((source / "sat/stages" / key / "result.json").read_text())
        imported = json.loads((target / "sat/stages" / key / "result.json").read_text())
        assert imported["result"] == prior["result"]
        assert imported["binding_sha256"] != prior["binding_sha256"]
    assert before == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    invoke(interrupted, provider)
    assert (
        len(provider.requests) == 2
    )  # resumed continuation reuses its own commits too


def test_corrupt_source_result_fails_before_new_model_calls(interrupted):
    source = interrupted[0]
    p = source / "sat/stages" / model_key(0, 0) / "result.json"
    data = json.loads(p.read_text())
    data["result"]["text"] = "corrupt output"
    p.write_text(json.dumps(data))
    provider = FakeProvider()
    with pytest.raises(ValueError, match="checksum"):
        invoke(interrupted, provider)
    assert provider.requests == []


def test_nonbudget_setting_change_is_rejected(interrupted):
    source, target, identity, binding, config, options = interrupted
    changed = (
        source,
        target,
        identity,
        {**binding, "model": "different-model"},
        config,
        options,
    )
    provider = FakeProvider()
    with pytest.raises(ValueError, match="non-budget"):
        invoke(changed, provider)
    assert provider.requests == []


def test_missing_prefix_commit_is_rejected(interrupted):
    source = interrupted[0]
    (source / "sat/stages" / model_key(0, 2) / "result.json").unlink()
    provider = FakeProvider()
    with pytest.raises(ValueError, match="contiguous prefix"):
        invoke(interrupted, provider)
    assert provider.requests == []


def test_partial_import_retries_without_repeating_model_calls(interrupted, monkeypatch):
    from spatialcraft.experiments.journal import RunJournal

    original = RunJournal.execute
    fired = []

    def interrupt(self, key, inputs, operation):
        if self.root.parent.name == "target" and key == model_key(1, 0) and not fired:
            fired.append(True)
            raise RuntimeError("local copy interrupted")
        return original(self, key, inputs, operation)

    monkeypatch.setattr(RunJournal, "execute", interrupt)
    provider = FakeProvider()
    with pytest.raises(RuntimeError, match="copy interrupted"):
        invoke(interrupted, provider)
    assert provider.requests == []
    report = invoke(interrupted, provider)
    assert report["environment"]["record_count"] == 8
    assert len(provider.requests) == 2


def test_actual_229405_schedule():
    counts = rollout_counts(
        2856,
        {
            "environment_rollouts": 1,
            "environment_resume": {
                "source_environment_rollouts": 4,
                "completed_calls": 375,
            },
        },
    )
    assert counts == [4] * 93 + [3] + [1] * 2762
    assert sum(counts) == 3137
