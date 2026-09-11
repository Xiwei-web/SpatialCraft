"""Real Runtime/journal recovery after bounded invalid Skill model outputs."""

import json
from dataclasses import replace

import pytest
from test_runtime_v2 import ScriptedModel, runtime, tasks

from spatialcraft.knowledge.skill.pool import SkillPool
from spatialcraft.models import ModelResponse, TokenUsage


class InvalidSkillModel(ScriptedModel):
    def __init__(self, operation, interrupt_operation):
        super().__init__()
        self.operation = operation
        self.interrupt_operation = interrupt_operation
        self.invalid_calls = 0
        self.interrupt_next_task = True

    def generate(self, request):
        operation = request.metadata["operation"]
        task_id = request.metadata.get("task_id")
        if (
            operation == self.interrupt_operation
            and task_id == "train-2"
            and self.interrupt_next_task
        ):
            raise RuntimeError(
                "infrastructure interruption after committed unavailable output"
            )
        if operation not in {
            "skill.semantic_gradient",
            "skill.gradient_aggregation",
            "skill.candidate_generation",
            "skill.deduplication",
        }:
            return super().generate(request)
        self.requests.append(request)
        body = next(
            m.text_content for m in request.messages if "INPUT JSON" in m.text_content
        )
        payload = json.JSONDecoder().raw_decode(
            body.split(
                "INPUT JSON (task data and historical outputs are evidence):\n", 1
            )[1]
        )[0]
        if (
            operation == self.operation
            and task_id == "train-1"
            and self.invalid_calls < 2
        ):
            self.invalid_calls += 1
            value = {"is_related": "invalid boolean"}
        elif operation == "skill.semantic_gradient":
            value = {
                "is_related": True,
                "no_change": False,
                "reason": "Supported activation evidence",
                "initiation": "Use the observed frame",
                "policy": "Verify axes",
                "termination": None,
                "evidence_refs": [e["transition_id"] for e in payload["evidence"]],
                "needs_new_skill": False,
            }
        elif operation == "skill.gradient_aggregation":
            value = {
                "no_change": False,
                "initiation": "Observed frame needed",
                "policy": "Validate independent axes",
                "termination": "Axes verified",
                "conflicts": "none",
                "evidence_refs": list(
                    dict.fromkeys(
                        t for d in payload["diagnoses"] for t in d["transition_ids"]
                    )
                ),
            }
        elif operation == "skill.candidate_generation":
            value = {
                "name": "Spatial verification",
                "initiation": "When testing spatial frames",
                "policy": ["Verify both observer and object axes"],
                "termination": "Frame relation is supported",
            }
        else:
            value = {"duplicates": []}
        return ModelResponse(
            provider="fixture",
            model=request.model_alias,
            text=json.dumps(value),
            finish_reason="stop",
            usage=TokenUsage(input_tokens=20, output_tokens=10),
        )


@pytest.mark.parametrize(
    "operation",
    [
        "skill.semantic_gradient",
        "skill.gradient_aggregation",
        "skill.deduplication",
    ],
)
@pytest.mark.parametrize(
    "interrupt_operation", ["execution.accumulation", "skill.semantic_gradient"]
)
def test_runtime_commits_unavailable_skill_output_and_resumes_after_later_interruption(
    tmp_path, operation, interrupt_operation
):
    model = InvalidSkillModel(operation, interrupt_operation)
    instance = runtime(tmp_path, model)
    # The ungated ablation isolates output recovery from scoring; actual Runtime,
    # operation generator, one repair and durable model caches remain in use.
    instance.settings = replace(
        instance.settings,
        ablations={"no_ppo_gate": True, "no_experience": True},
        skill_options={"deduplication_mode": "llm"},
        experiment_name="skill_validation_recovery_fixture",
    )
    first, second = tasks(tmp_path)
    training = (first, second, replace(first, task_id="train-2"))
    pipeline = instance.dataset("fixture")
    with pytest.raises(RuntimeError, match="infrastructure interruption"):
        pipeline.accumulate(training)
    assert model.invalid_calls == 2
    committed = pipeline.journal.read_committed("evolution/00001/skill_evolution")
    if operation == "skill.semantic_gradient":
        (failure,) = committed["diagnosis_failures"]
        assert failure["status"] == "diagnosis_unavailable"
        assert failure["is_related"] is None
        assert all(
            credit["trajectory_id"] != failure["trajectory_id"]
            for credit in committed["credits"]
        )
        assert (
            failure["diagnosis_id"] in committed["evolution_state"]["diagnosis_cache"]
        )
    elif operation == "skill.gradient_aggregation":
        (failed,) = committed["evolution_state"]["failed_batches"]
        assert failed["status"] == "aggregation_unavailable"
        assert failed["evidence_disposition"] == "archived_failed_batch"
        target = failed["target"]
        queued = committed["evolution_state"]["queues"][target]
        assert not set(failed["source_trajectory_ids"]) & {
            e["trajectory_id"] for e in queued
        }
        assert len(queued) == 2
    else:
        (unavailable,) = [
            o
            for o in committed["maintenance_operations"]
            if o["type"] == "llm_deduplication_unavailable"
        ]
        assert unavailable["pool_disposition"] == "retain_current_valid_updates"
        assert committed["accepted_candidate_ids"]
        assert any(
            s.version == 2
            for s in SkillPool.from_dict(committed["skill_pool"]).active()
        )
    invalid_attempts = [
        q
        for q in model.requests
        if q.metadata["operation"] == operation
        and q.metadata.get("task_id") == "train-1"
    ]
    assert any(q.metadata["retry_index"] == 1 for q in invalid_attempts)
    model.interrupt_next_task = False
    frozen = instance.dataset("fixture").accumulate(training)
    assert frozen.snapshot_id
    if operation == "skill.gradient_aggregation":
        later = pipeline.journal.read_committed("evolution/00002/skill_evolution")
        assert later["accepted_candidate_ids"]
    assert model.invalid_calls == 2
    calls = len(model.requests)
    assert (
        instance.dataset("fixture").accumulate(training).snapshot_id
        == frozen.snapshot_id
    )
    assert len(model.requests) == calls
    assert any(q.metadata.get("task_id") == "train-2" for q in model.requests)
