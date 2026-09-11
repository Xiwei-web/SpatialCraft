"""Provider-boundary and hand-calculated cost regressions from the second audit."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from spatialcraft.evaluation.cost_report_v2 import build_cost_report
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.knowledge_generator import KnowledgeGenerator
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.experiments.usage import AuditedProviderV2
from spatialcraft.models import ModelProvider, ModelResponse, RequestBuilder, TokenUsage
from spatialcraft.models.interfaces import ModelRequestError
from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.storage.atomic_io import atomic_write_json

PROJECT = Path(__file__).resolve().parents[1]


def config():
    return ModelConfig.from_dict(load_yaml(PROJECT / "configs/models/gpt-5.4.yaml"))


class Client:
    def __init__(self):
        self.responses = self
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        return {
            "id": "fixture-response",
            "model": "gpt-5.4-fixture-version",
            "system_fingerprint": "fixture-fingerprint",
            "output_text": '{"ok":true}',
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }


def test_real_scoped_knowledge_request_sends_bounded_metadata_and_effective_parameters(
    tmp_path,
):
    model, client = config(), Client()
    journal = RunJournal(tmp_path, {"fixture": "scoped-api-no-network"})
    provider = AuditedProviderV2(
        OpenAIResponsesProvider(model, client=client), journal, "knowledge_builder"
    )
    generator = KnowledgeGenerator(
        ExperimentSettings(protocol_version="spatialcraft_v2"),
        {"knowledge_builder": model},
        {"knowledge_builder": provider},
        PROJECT,
        journal,
    )
    generator.scope = {
        "phase": "accumulation_learning",
        "task_id": "x" * 800,
        "snapshot_id": "snapshot",
        "rollout_index": 2,
        "step_index": 3,
        "private_nested_audit": {"text": "z" * 1000},
    }
    assert generator("experience.summary", {"evidence": "fixture"}) == {"ok": True}
    payload = client.payloads[0]
    assert len(generator.audit[0]) > 16
    assert payload["reasoning"] == {"effort": "medium"}
    assert "temperature" not in payload and "top_p" not in payload
    assert 0 < len(payload["metadata"]) <= 16
    assert all(len(k) <= 64 and len(v) <= 512 for k, v in payload["metadata"].items())
    assert payload["metadata"]["task_id"].startswith("sha256:")
    assert "private_nested_audit" not in payload["metadata"]
    assert (
        generator.audit[0]["private_nested_audit"]
        == generator.scope["private_nested_audit"]
    )
    audit = generator.audit[0]["provider_request_parameters"]
    assert audit["requested_temperature"] == 0.6
    assert (
        audit["effective_temperature"] is None and audit["sampling_parameters_omitted"]
    )
    events = [json.loads(p.read_text()) for p in (tmp_path / "usage").glob("*.json")]
    assert events[0]["effective_top_p"] is None
    assert events[0]["server_model"] == "gpt-5.4-fixture-version"
    assert events[0]["server_system_fingerprint"] == "fixture-fingerprint"
    assert generator("experience.summary", {"evidence": "fixture"}) == {"ok": True}
    assert len(client.payloads) == 1
    generator("skill.selection_judge", {"evidence": "fixture"})
    assert client.payloads[-1]["reasoning"] == {"effort": "none"}
    assert client.payloads[-1]["temperature"] == 0.0
    assert client.payloads[-1]["top_p"] == 1.0


@pytest.mark.parametrize("model_id", ["gpt-5.4", "gpt-5.4-2026-03-05"])
def test_reasoning_sampling_cannot_be_reintroduced_through_extra(model_id):
    model = replace(config(), model_id=model_id)
    provider = OpenAIResponsesProvider(model)
    request = (
        RequestBuilder(model)
        .user("fixture")
        .settings(reasoning_effort="medium", temperature=0.6, top_p=1.0)
        .build()
    )
    assert "temperature" not in provider.build_payload(request)
    request = replace(
        request, settings=replace(request.settings, extra={"temperature": 0.6})
    )
    with pytest.raises(ModelRequestError, match="managed fields"):
        provider.build_payload(request)


def write_events(root, events):
    for index, event in enumerate(events):
        atomic_write_json(
            root / "usage" / f"{index}.json",
            {"schema_version": 2, "phase": "deployment", "task_id": "t", **event},
        )


def test_mixed_token_work_is_78_and_counts_each_kind_without_reasoning_double_charge(
    tmp_path,
):
    events = [
        {
            "kind": "generation",
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 7,
        },
        {"kind": "embedding", "input_tokens": 5, "requests": 1},
        {
            "kind": "fixed_target_score",
            "input_tokens": 40,
            "target_tokens": 3,
            "output_tokens": 0,
        },
    ]
    write_events(tmp_path, events + [{**e, "cache_reused": True} for e in events])
    report = build_cost_report(tmp_path, 1)
    online = report["online_deployment"]
    assert report["schema_version"] == 3
    assert online["total_tokens"]["total"] == 78
    assert online["input_tokens"]["total"] == 55
    assert online["output_tokens"]["total"] == 20
    for field, count in {
        "generation_input_tokens": 10,
        "generation_output_tokens": 20,
        "generation_total_tokens": 30,
        "embedding_input_tokens": 5,
        "scoring_prefix_tokens": 40,
        "scoring_target_tokens": 3,
    }.items():
        assert online[field]["total"] == count
    assert report["token_work_is_equivalent_cost"] is False
    assert online["full_call_latency_ms"]["total"] is None


@pytest.mark.parametrize("missing_kind", ["embedding", "fixed_target_score"])
def test_missing_auxiliary_tokens_make_total_incomplete(tmp_path, missing_kind):
    write_events(
        tmp_path,
        [
            {"kind": "generation", "input_tokens": 10, "output_tokens": 20},
            {"kind": missing_kind, "input_tokens": None, "target_tokens": 3},
        ],
    )
    online = build_cost_report(tmp_path)["online_deployment"]
    assert online["total_tokens"]["total"] is None
    assert online["total_tokens"]["known_sum"] == 30
    assert online["total_tokens"]["unknown_records"] == 1


def test_provider_latency_and_wrapper_elapsed_are_distinct(tmp_path, monkeypatch):
    class Provider(ModelProvider):
        def generate(self, request):
            return ModelResponse(
                provider="fixture",
                model="fixture",
                text="ok",
                latency_ms=3,
                raw={"model_version": "fixture-resolved-version"},
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )

    times = iter([10.0, 10.025])
    monkeypatch.setattr(
        "spatialcraft.experiments.usage.perf_counter", lambda: next(times)
    )
    provider = AuditedProviderV2(
        Provider(), RunJournal(tmp_path, {"fixture": True}), "executor"
    )
    provider.generate(RequestBuilder(config()).user("fixture").build())
    event = json.loads(next((tmp_path / "usage").glob("*.json")).read_text())
    assert event["server_model_version"] == "fixture-resolved-version"
    assert event["provider_latency_ms"] == 3
    assert event["full_call_latency_ms"] == pytest.approx(25)
    assert event["latency_ms"] == pytest.approx(25)
