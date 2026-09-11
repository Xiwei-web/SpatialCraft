from dataclasses import asdict

import pytest

from spatialcraft.evaluation.cost_report_v2 import build_cost_report
from spatialcraft.models import TokenUsage
from spatialcraft.models.response_parser import (
    parse_openai_chat_response,
    parse_openai_responses,
)
from spatialcraft.storage.atomic_io import atomic_write_json


def test_missing_api_usage_remains_unknown_and_known_counts_derive_total():
    response = parse_openai_responses(
        {"output_text": "A"}, provider="mock", model="mock"
    )
    assert asdict(response.usage) == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
    }
    response = parse_openai_chat_response(
        {
            "choices": [{"message": {"content": "A"}}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
        provider="mock",
        model="mock",
    )
    assert response.usage.total_tokens == 6
    assert response.usage.cached_input_tokens == 0
    assert response.usage.reasoning_tokens is None
    assert TokenUsage(input_tokens=3).total_tokens is None
    assert TokenUsage(total_tokens=5).input_tokens is None
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=-1)


def test_cost_report_does_not_recharge_cache_or_hide_missing_usage(tmp_path):
    events = [
        {
            "phase": "accumulation_learning",
            "kind": "generation",
            "role": "knowledge_builder",
            "model": "m",
            "operation": "experience.summary",
            "actual_call": True,
            "status": "completed",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "latency_ms": 10,
        },
        {
            "phase": "accumulation_learning",
            "kind": "generation",
            "role": "knowledge_builder",
            "model": "m",
            "operation": "experience.summary",
            "actual_call": False,
            "cache_reused": True,
            "status": "reused",
            "input_tokens": 100,
            "output_tokens": 20,
            "latency_ms": 10,
        },
        {
            "phase": "accumulation_learning",
            "kind": "generation",
            "role": "knowledge_builder",
            "model": "m",
            "operation": "experience.summary",
            "actual_call": True,
            "status": "failed",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": 2,
        },
        {
            "phase": "deployment",
            "kind": "generation",
            "role": "executor",
            "model": "m",
            "operation": "execution.deployment",
            "actual_call": True,
            "status": "completed",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 4,
        },
        {
            "phase": "accumulation_learning",
            "kind": "embedding",
            "role": "embedding",
            "model": "e",
            "status": "failed",
            "requests": None,
            "input_tokens": None,
            "latency_ms": 3,
        },
        {
            "phase": "deployment",
            "kind": "tool",
            "role": "tools",
            "actual_call": True,
            "status": "completed",
            "latency_ms": 7,
        },
    ]
    for index, event in enumerate(events):
        atomic_write_json(
            tmp_path / "usage" / f"{index}.json", {"schema_version": 2, **event}
        )
    report = build_cost_report(tmp_path, deployment_count=2)
    summary = next(
        r for r in report["groups"] if r["operation"] == "experience.summary"
    )
    assert summary["actual_calls"] == 2 and summary["cache_reuses"] == 1
    assert summary["input_tokens"]["known_sum"] == 100
    assert summary["input_tokens"]["unknown_records"] == 1
    assert summary["input_tokens"]["total"] is None
    assert summary["reasoning_tokens"]["total"] is None
    assert report["offline_amortized_per_deployment_task"]["input_tokens"] is None
    assert report["offline_latency_ms_amortized_per_deployment_task"] == 7.5
    embed = next(r for r in report["groups"] if r["kind"] == "embedding")
    assert (
        embed["actual_calls_total"] is None
        and embed["actual_call_unknown_records"] == 1
    )
    tool = next(r for r in report["groups"] if r["kind"] == "tool")
    assert tool["input_tokens"]["not_applicable_records"] == 1
    assert report["online_deployment"]["output_tokens"]["total"] == 5


def _write_events(root, events):
    for index, event in enumerate(events):
        atomic_write_json(
            root / "usage" / f"{index}.json", {"schema_version": 2, **event}
        )


def _generation(task_id, phase, input_tokens, output_tokens, latency_ms, **kwargs):
    return {
        "task_id": task_id,
        "phase": phase,
        "kind": "generation",
        "role": "executor",
        "model": "m",
        "operation": "execution",
        "actual_call": True,
        "status": "completed",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_ms": latency_ms,
        **kwargs,
    }


def test_per_task_cost_and_amortization_have_independent_hand_calculated_denominators(
    tmp_path,
):
    # One shared call plus four rollouts; cache replay cannot multiply either.
    events = [_generation("train", "accumulation_execution", 100, 10, 5)]
    for index in range(4):
        events.append(
            _generation(
                "train",
                "accumulation_execution",
                10,
                2,
                2,
                rollout_index=index,
                rollout_prefix="train/rollouts",
                trajectory_id=f"tr-{index}",
            )
        )
    events.append(
        {
            **events[1],
            "cache_reused": True,
            "actual_call": False,
            "input_tokens": 999,
            "output_tokens": 999,
            "total_tokens": 1998,
            "latency_ms": 999,
        }
    )
    events.append(
        {
            "task_id": "train",
            "phase": "accumulation_execution",
            "kind": "embedding",
            "requests": 2,
            "input_tokens": 5,
            "latency_ms": 3,
        }
    )
    events.extend(
        [
            _generation("a", "deployment", 20, 4, 4),
            _generation("a", "deployment", 30, 6, 6),
            {
                "task_id": "a",
                "phase": "deployment",
                "kind": "tool",
                "actual_call": True,
                "latency_ms": 3,
            },
            _generation("b", "deployment", 10, 2, 2),
            {
                "task_id": "b",
                "phase": "deployment",
                "kind": "embedding",
                "requests": None,
                "input_tokens": None,
                "status": "failed",
                "latency_ms": 1,
            },
        ]
    )
    _write_events(tmp_path, events)
    report = build_cost_report(tmp_path, deployment_count=10)
    training = next(row for row in report["per_task"] if row["task_id"] == "train")
    assert training["input_tokens"]["total"] == 145
    assert training["output_tokens"]["total"] == 18
    assert training["latency_ms"]["total"] == 16
    assert training["call_counts"]["all_calls"]["total"] == 7
    assert training["call_counts"]["model_calls"]["total"] == 5
    assert training["call_counts"]["embedding_requests"]["total"] == 2
    assert len(training["rollout_refs"]) == 4
    assert training["records_without_rollout_identity"] == 2
    assert training["cache_reuses"] == 1
    task_a = next(row for row in report["per_task"] if row["task_id"] == "a")
    task_b = next(row for row in report["per_task"] if row["task_id"] == "b")
    assert task_a["input_tokens"]["total"] == 50
    assert task_a["latency_ms"]["total"] == 13
    assert task_a["call_counts"]["tool_calls"]["total"] == 1
    assert task_b["input_tokens"]["known_sum"] == 10
    assert task_b["input_tokens"]["total"] is None
    mean = report["online_mean_per_deployment_task"]
    assert mean["denominator"] == 2  # Observed tasks, never the future M=10.
    assert mean["input_tokens"]["known_sum_per_task"] == 30
    assert mean["input_tokens"]["unknown_records"] == 1
    assert mean["input_tokens"]["total"] is None
    assert mean["output_tokens"]["total"] == 6
    assert mean["latency_ms"]["total"] == 8
    assert mean["call_counts"]["model_calls"]["total"] == 1.5
    assert mean["call_counts"]["tool_calls"]["total"] == 0.5
    assert mean["call_counts"]["all_calls"]["total"] is None
    assert mean["call_counts"]["all_calls"]["unknown_records"] == 1
    combined = report["amortized_total_per_deployment_task"]
    assert combined["input_tokens"]["total"] is None
    assert combined["input_tokens"]["known_sum_per_task"] == 44.5
    assert combined["output_tokens"]["total"] == pytest.approx(7.8)  # 18/10 + 12/2.
    assert combined["latency_ms"]["total"] == pytest.approx(9.6)  # 16/10 + 16/2.
    assert combined["call_counts"]["model_calls"]["total"] == 2  # 5/10 + 3/2.
    assert combined["call_counts"]["all_calls"]["total"] is None
    assert combined["reasoning_tokens"]["total"] is None
    assert report["prices_included"] is False
    assert report["gpu_resource_cost_included"] is False


def test_unattributed_online_and_empty_logs_never_invent_task_denominators(tmp_path):
    _write_events(
        tmp_path,
        [
            _generation("a", "deployment", 10, 2, 3),
            _generation(None, "deployment", 20, 4, 5),
        ],
    )
    report = build_cost_report(tmp_path, deployment_count=100)
    assert report["online_deployment"]["input_tokens"]["total"] == 30
    assert report["deployment_task_attribution"]["unattributed_records"] == 1
    assert report["online_mean_per_deployment_task"]["input_tokens"]["total"] is None
    assert (
        report["online_mean_per_deployment_task"]["input_tokens"]["known_sum_per_task"]
        is None
    )
    assert report["amortized_total_per_deployment_task"]["latency_ms"]["total"] is None
    assert any(row["task_id"] is None for row in report["per_task"])
    empty = build_cost_report(tmp_path / "empty", deployment_count=0)
    assert empty["online_mean_per_deployment_task"]["latency_ms"]["total"] is None
    assert empty["amortized_total_per_deployment_task"]["input_tokens"]["total"] is None


def test_actual_audit_preserves_rollout_identity_without_leaking_task_text(tmp_path):
    import json
    from types import SimpleNamespace

    from spatialcraft.experiments.journal import RunJournal
    from spatialcraft.experiments.usage import AuditedProviderV2
    from spatialcraft.models import (
        MessageRole,
        ModelMessage,
        ModelRequest,
        ModelResponse,
    )
    from spatialcraft.storage.atomic_io import read_json

    calls = []

    def generate(request):
        calls.append(request)
        return ModelResponse(
            provider="fixture",
            model="m",
            text="A",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            latency_ms=3,
        )

    audited = AuditedProviderV2(
        SimpleNamespace(generate=generate),
        RunJournal(tmp_path, {"run_id": "test"}),
        "executor",
    )
    request = ModelRequest(
        model_alias="m",
        messages=(ModelMessage.text(MessageRole.USER, "PRIVATE_TASK_TEXT"),),
        metadata={
            "phase": "deployment",
            "task_id": "task-1",
            "rollout_prefix": "deployment/task-1",
            "rollout_index": 0,
            "trajectory_id": "trajectory-1",
            "operation": "execution.deployment",
            "requested_reasoning_mode": "instruct",
            "effective_reasoning_mode": "instruct",
        },
    )
    audited.generate(request)
    audited.generate(request)
    assert len(calls) == 1
    events = [read_json(path) for path in (tmp_path / "usage").glob("*.json")]
    assert len(events) == 2
    for event in events:
        assert event["rollout_index"] == 0
        assert event["rollout_prefix"] == "deployment/task-1"
        assert event["trajectory_id"] == "trajectory-1"
        assert event["requested_reasoning_mode"] == "instruct"
        assert "PRIVATE_TASK_TEXT" not in json.dumps(event)
    report = build_cost_report(tmp_path, deployment_count=1)
    assert report["online_mean_per_deployment_task"]["input_tokens"]["total"] == 10
    assert report["per_task"][0]["actual_calls_total"] == 1
    assert "PRIVATE_TASK_TEXT" not in json.dumps(report)
