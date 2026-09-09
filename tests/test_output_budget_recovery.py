import pytest
from test_protocol_pipeline import completed
from test_stage5_agent_rollout import _agent

from spatialcraft.experiments.accumulation import KnowledgeState, ProtocolPipeline
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.learning import LearningBuilders
from spatialcraft.experiments.output_budget import (
    OutputBudgetExhausted,
    budget_safe_learning,
)
from spatialcraft.experiments.rollout import JournaledRollout
from spatialcraft.experiments.runtime import AuditedProvider
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import ModelResponse
from spatialcraft.schemas import SpatialState, TaskSample, TaskSplit, TrajectoryStatus


class Responses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


def response(text="partial", reason="length"):
    return ModelResponse(provider="test", model="test", text=text, finish_reason=reason)


def builders(tmp_path, provider):
    agent, _ = _agent(tmp_path)
    return LearningBuilders(
        ExperimentSettings(),
        agent.execution_loop.composer.model,
        provider,
        None,
        lambda p: p,
    )


def test_knowledge_forced_completion_and_cached_resume(tmp_path):
    source = Responses(response(), response('{"terminate":true}', "stop"))
    provider = AuditedProvider(source, RunJournal(tmp_path / "journal", {"test": 1}))
    builder = builders(tmp_path, provider)
    for _ in range(2):
        assert (
            builder.text('Return JSON {"terminate":true/false}') == '{"terminate":true}'
        )
    assert len(source.calls) == 2
    original, recovery = source.calls
    assert (
        recovery.settings.max_output_tokens
        == original.settings.max_output_tokens
        == 4096
    )
    assert recovery.settings.temperature == 0
    assert recovery.messages[:-1] == original.messages
    assert recovery.metadata["chat_template_kwargs"] == {"enable_thinking": False}
    assert recovery.metadata["output_budget_recovery"]["purpose"] == "knowledge"


@pytest.mark.parametrize(
    "last", [response(), response('{"a":', "stop"), response("", "stop")]
)
def test_recovery_failure_is_typed_and_bounded(tmp_path, last):
    source = Responses(response(), last)
    with pytest.raises(OutputBudgetExhausted):
        builders(tmp_path, source).text("Summarize")
    assert len(source.calls) == 2


def test_infrastructure_error_not_swallowed(tmp_path):
    source = Responses(response(), RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        builders(tmp_path, source).text("Summarize")


@pytest.mark.parametrize("second", [response("Final Answer: A", "stop"), response()])
def test_rollout_action_recovery_and_resume(tmp_path, second):
    agent, _ = _agent(tmp_path)
    source = Responses(response(), second)
    agent.execution_loop.provider = source
    runner = JournaledRollout(
        agent.execution_loop, RunJournal(tmp_path / "journal", {"test": 1})
    )
    task = TaskSample(
        task_id="t", dataset="test", question="Answer A or B?", reference_answer="A"
    )
    state = SpatialState(task_id="t", step_index=0)
    kwargs = {
        "prefix": "rollout",
        "initial_state": state,
        "rollout_index": 0,
        "random_seed": 42,
    }
    row = runner.run(task, **kwargs)
    replay = runner.run(task, **kwargs)
    assert replay.to_dict() == row.to_dict()
    assert len(source.calls) == 2
    assert source.calls[-1].tools == source.calls[0].tools
    assert source.calls[-1].settings.max_output_tokens <= 1024
    assert "reference_answer" not in str(source.calls[-1].messages)
    if second.finish_reason == "stop":
        assert row.status is TrajectoryStatus.COMPLETED
        assert row.final_answer == "Final Answer: A" and row.reward == 1
        assert row.transitions[-1].metadata["call_kind"] == "recovery"
        assert row.metadata["call_counts"]["forced_final_answers"] == 0
    else:
        assert row.status is TrajectoryStatus.FAILED
        assert row.final_answer is None and row.reward == 0
        assert row.metadata["failure_kind"] == "action_recovery_failed"


def test_failed_experience_update_preserves_bank(tmp_path):
    knowledge = KnowledgeState(ExperienceBank().freeze(), SeedCatalog.pool().freeze())
    task = TaskSample(task_id="t", dataset="test", question="q")
    rows = tuple(completed(task, knowledge, i, i) for i in range(4))
    result = builders(tmp_path, Responses(response(), response())).update_experiences(
        knowledge, rows
    )
    assert result["experience_bank"] == knowledge.experiences.to_dict()
    assert result["updates"] == []
    assert result["output_budget_recovery"]["status"] == "skipped_output_limit"


def test_evolution_fallback_keeps_pending_trajectories():
    class FailedEvolution:
        @budget_safe_learning
        def evolve(self, knowledge, rows, batch_index, evolution_state=None):
            raise OutputBudgetExhausted(response(), response())

    knowledge = KnowledgeState(ExperienceBank().freeze(), SeedCatalog.pool().freeze())
    task = TaskSample(task_id="t", dataset="test", question="q")
    rows = tuple(completed(task, knowledge, i, i) for i in range(4))
    result = FailedEvolution().evolve(knowledge=knowledge, rows=rows, batch_index=0)
    assert result["skill_pool"] == knowledge.skills.to_dict()
    assert len(result["evolution_state"]["seen_trajectory_ids"]) == 4
    assert sum(result["pending_counts"].values()) == 4


def test_retrieval_and_termination_fallbacks():
    class Failed:
        @budget_safe_learning
        def retrieve(self):
            raise OutputBudgetExhausted(response(), response())

        @budget_safe_learning
        def terminate(self):
            raise OutputBudgetExhausted(response(), response())

    assert Failed().retrieve() == ()
    assert Failed().terminate() is True


def test_skipped_updates_continue_next_task_and_resume(tmp_path):
    tasks = tuple(
        TaskSample(task_id=f"t{i}", dataset="test", question="q", split=TaskSplit.TRAIN)
        for i in range(2)
    )
    source = Responses(*(response() for _ in range(4)))
    builder = builders(tmp_path, source)
    seen = []

    def rollout(task, knowledge, refs, index, seed, prefix, deployment):
        seen.append((task.task_id, index))
        return completed(task, knowledge, index, seed)

    class FailedEvolution:
        @budget_safe_learning
        def evolve(self, knowledge, rows, batch_index, evolution_state=None):
            raise OutputBudgetExhausted(response(), response())

    pipeline = ProtocolPipeline(
        ExperimentSettings(),
        RunJournal(tmp_path / "journal", {"test": "continue"}),
        prepare_experiences=lambda task, knowledge: (),
        rollout=rollout,
        update_experiences=builder.update_experiences,
        evolve_skills=FailedEvolution().evolve,
    )
    frozen = pipeline.accumulate(tasks)
    assert len(seen) == 8 and seen[-1] == ("t1", 3)
    assert len(source.calls) == 4
    assert pipeline.accumulate(tasks).snapshot_id == frozen.snapshot_id
    assert len(seen) == 8 and len(source.calls) == 4
    import json

    audit = json.loads((tmp_path / "journal/skills/round-00001.json").read_text())
    assert sum(audit["pending_counts"].values()) == 8
