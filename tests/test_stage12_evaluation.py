from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spatialcraft.evaluation import (
    BaselineName,
    BaselineRunner,
    accuracy,
    efficiency,
    estimate_pass_at_k,
    experience_metrics,
    model_pair_ablations,
    pass_at_k,
    pool_size_ablations,
    rollout_count_ablations,
    skill_metrics,
    standard_ablations,
    standard_baselines,
    tool_metrics,
    transfer_metrics,
)
from spatialcraft.schemas import (
    ActiveSkillRef,
    AgentAction,
    AnswerType,
    RetrievedExperienceRef,
    SpatialState,
    TaskSample,
    ToolCall,
    ToolResult,
    ToolStatus,
    Trajectory,
    TrajectoryStatus,
    Transition,
    VerifierOutcome,
)


def _trajectory(task, reward, index):
    active = ActiveSkillRef(
        skill_id="skill-eval", version=2, activated_at_step=0, age_steps=0
    )
    experience = RetrievedExperienceRef(
        experience_id="experience-eval",
        version=1,
        retrieval_score=0.9,
        original_text="Use geometry.",
    )
    first = SpatialState(
        task_id=task.task_id,
        step_index=0,
        active_skill=active,
        retrieved_experiences=(experience,),
    )
    call = ToolCall(tool_name="geometry", call_id=f"call-{index}")
    tool_action = AgentAction.tool(call)
    result = ToolResult(
        tool_call_id=call.call_id,
        tool_name="geometry",
        status=ToolStatus.SUCCEEDED,
        text="measured",
    )
    second = first.next_step(
        active_skill=ActiveSkillRef(
            skill_id="skill-eval", version=2, activated_at_step=0, age_steps=1
        ),
        token_count=10,
    )
    final_action = AgentAction.final("A" if reward else "B")
    transitions = (
        Transition(
            step_index=0,
            state_before=first,
            action=tool_action,
            tool_results=(result,),
            state_after=second,
            active_skill=active,
            used_experience_ids=("experience-eval",),
        ),
        Transition(
            step_index=1,
            state_before=second,
            action=final_action,
            state_after=second.next_step(active_skill=None, token_count=20),
            active_skill=second.active_skill,
            used_experience_ids=("experience-eval",),
            reward=reward,
            done=True,
        ),
    )
    now = datetime.now(timezone.utc)
    return Trajectory(
        task=task,
        rollout_index=index,
        executor_model="mock",
        knowledge_snapshot_id="snapshot-eval",
        status=TrajectoryStatus.COMPLETED,
        transitions=transitions,
        reward=reward,
        verifier=VerifierOutcome(
            verifier_name="test", score=reward, is_correct=bool(reward)
        ),
        started_at=now,
        finished_at=now,
        metadata={"cost": 0.1},
    )


def test_metrics_cover_accuracy_tools_memory_skill_efficiency_and_transfer() -> None:
    task = TaskSample(
        dataset="target",
        task_id="eval-task",
        question="Which relation?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        choices=("left", "right"),
        reference_answer="A",
    )
    rows = tuple(
        _trajectory(task, reward, index) for index, reward in enumerate((1.0, 0.0, 1.0))
    )
    assert accuracy(rows).accuracy == 2 / 3
    assert estimate_pass_at_k(3, 2, 1) == pytest.approx(2 / 3)
    assert pass_at_k(rows, 2) == 1.0
    tools = tool_metrics(rows)
    assert tools.total_calls == 3 and tools.success_rate == 1.0
    experiences = experience_metrics(rows, vanilla=(rows[1],))
    assert experiences.use_rate == 1.0
    assert experiences.unique_experiences_used == 1
    assert experiences.reward_lift == pytest.approx(2 / 3)
    skills = skill_metrics(rows)
    assert skills.activation_rate == 1.0
    assert skills.mean_active_lifetime == 2.0
    costs = efficiency(rows)
    assert costs.mean_steps == 2.0
    assert costs.mean_tool_calls == 1.0
    assert costs.mean_state_tokens == 20.0
    transfer = transfer_metrics(
        (rows[0], rows[2]),
        (rows[0], rows[1]),
        source_dataset="source",
        target_dataset="target",
    )
    assert transfer.target_accuracy == 1.0
    assert transfer.vanilla_target_accuracy == 0.5
    assert transfer.relative_error_reduction == 1.0


def test_eight_baselines_and_complete_ablation_axes_are_runnable() -> None:
    configs = standard_baselines()
    assert len(configs) == 8
    assert {item.name for item in configs} == set(BaselineName)
    assert next(
        item for item in configs if item.name is BaselineName.MEMRL_R_GT
    ).variants == ("R", "GT")
    calls = []

    def execute(config, tasks, seed):
        calls.append((config.name, len(tasks), seed))
        return ()

    results = BaselineRunner(execute).run((), seed=11)
    assert set(results) == set(BaselineName)
    assert len(calls) == 8
    fixed = standard_ablations()
    assert {item.name for item in fixed} == {
        "no_experience",
        "no_skill",
        "no_task_decomposition",
        "no_experience_rewrite",
        "no_visual_summary",
        "no_cross_rollout_critique",
        "no_semantic_gradient",
        "no_ppo_gate",
        "no_skill_score_pruning",
        "static_seed_skills",
    }
    assert len(pool_size_ablations((10, 20), (6, 12))) == 4
    assert len(rollout_count_ablations((1, 4, 8))) == 3
    pairs = model_pair_ablations(
        ("gpt-5.4", "qwen3.6-27b"), ("gpt-5.4", "gpt-5.4-mini")
    )
    assert len(pairs) == 4
    assert all("models.executor" in item.overrides for item in pairs)
