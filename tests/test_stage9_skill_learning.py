from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from spatialcraft.knowledge.skill import (
    GradientAggregator,
    SeedCatalog,
    SemanticGradientEngine,
    SkillCandidateGenerator,
    SkillCreditAssigner,
    SkillLineage,
    SkillMaintenance,
    SkillPool,
    SkillStatistics,
)
from spatialcraft.schemas import (
    ActiveSkillRef,
    AgentAction,
    AnswerType,
    SkillEvolutionType,
    SkillItem,
    SkillStats,
    SpatialState,
    TaskSample,
    Trajectory,
    TrajectoryStatus,
    Transition,
)


def _trajectory(
    task: TaskSample,
    reward: float,
    rollout_index: int,
    active: ActiveSkillRef | None,
) -> Trajectory:
    state = SpatialState(task_id=task.task_id, step_index=0, active_skill=active)
    action = AgentAction.final("A" if reward > 0 else "B")
    transition = Transition(
        step_index=0,
        state_before=state,
        action=action,
        state_after=state.next_step(active_skill=None),
        active_skill=active,
        reward=reward,
        done=True,
    )
    now = datetime.now(timezone.utc)
    return Trajectory(
        task=task,
        rollout_index=rollout_index,
        executor_model="mock",
        knowledge_snapshot_id="snapshot-0",
        status=TrajectoryStatus.COMPLETED,
        transitions=(transition,),
        reward=reward,
        started_at=now,
        finished_at=now,
    )


def test_credit_statistics_gradients_candidates_and_lineage() -> None:
    task = TaskSample(
        dataset="test",
        task_id="shared-task",
        question="Which spatial relation is correct?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        choices=("left", "right"),
        reference_answer="A",
    )
    pool = SeedCatalog.pool()
    parent = pool.active()[0]
    active = ActiveSkillRef(
        skill_id=parent.skill_id,
        version=parent.version,
        activated_at_step=0,
    )
    trajectories = (
        _trajectory(task, 1.0, 0, active),
        _trajectory(task, 0.0, 1, active),
        _trajectory(task, 0.5, 2, None),
    )
    assigner = SkillCreditAssigner()
    advantages = assigner.advantages(trajectories)
    assert {round(item.advantage, 4) for item in advantages} == {0.5, -0.5, 0.0}
    credits = assigner.assign(trajectories)
    assert len(credits) == 2
    assert {item.skill_reference for item in credits} == {parent.reference}

    updated = SkillStatistics().update(pool, credits, iteration=1)
    stats = updated.get(parent.reference).stats
    assert stats.frequency == 2
    assert stats.average_gain == 0.0
    assert stats.maturity == 1

    advantage_map = {item.trajectory_id: item.advantage for item in advantages}
    gradients = SemanticGradientEngine().diagnose(
        trajectories, advantage_map, per_outcome=2
    )
    assert all(item.metadata["post_rollout_only"] for item in gradients)
    assert any(item.is_related for item in gradients)
    assert any(not item.is_related for item in gradients)
    aggregates = GradientAggregator().aggregate(gradients)
    candidates = SkillCandidateGenerator().generate(aggregates, updated)
    assert {item.evolution_type for item in candidates} == {
        SkillEvolutionType.REFINE,
        SkillEvolutionType.NEW,
    }
    refined = next(
        item for item in candidates if item.evolution_type is SkillEvolutionType.REFINE
    )
    assert refined.skill.skill_id == parent.skill_id
    assert refined.skill.version == parent.version + 1
    assert refined.skill.parent_skill_ref == parent.reference
    assert refined.source_gradient_ids and refined.source_trajectory_ids
    lineage = SkillLineage().record(refined)
    assert lineage.ancestors(refined.skill.reference) == (parent.reference,)


def test_duplicate_low_quality_and_stale_pruning() -> None:
    base = SkillItem(
        skill_id="skill-duplicate-a",
        name="Duplicate A",
        initiation="When comparing two spatial objects",
        policy=("Detect both objects then compare their centers",),
        termination="Stop after the comparison",
        stats=SkillStats(
            frequency=4,
            total_gain=-1.0,
            average_gain=-0.25,
            maturity=2,
            last_evolved_iteration=1,
        ),
    )
    duplicate = replace(
        base,
        skill_id="skill-duplicate-b",
        name="Duplicate B",
        stats=SkillStats(
            frequency=5,
            total_gain=2.0,
            average_gain=0.4,
            maturity=2,
            success_count=2,
            last_evolved_iteration=9,
        ),
    )
    pool = SkillPool((base, duplicate))
    maintenance = SkillMaintenance()
    pruned = maintenance.prune_references(
        pool, duplicate_threshold=0.8, current_iteration=12, stale_after=10
    )
    assert base.reference in pruned
    result = maintenance.apply(pool, pruned)
    assert {item.reference for item in result.active()} == {duplicate.reference}


def test_reactivated_skill_has_separate_windows_and_per_activation_statistics():
    pool = SeedCatalog.pool()
    skill = pool.active()[0]
    task = TaskSample(task_id="windows", dataset="test", question="Compare objects")
    refs = [
        ActiveSkillRef(
            skill_id=skill.skill_id,
            version=skill.version,
            activated_at_step=start,
            age_steps=i - start,
        )
        for i, start in enumerate((0, 0, 2, 2))
    ]
    states = [
        SpatialState(task_id=task.task_id, step_index=i, active_skill=ref)
        for i, ref in enumerate((*refs, None))
    ]
    rows = tuple(
        Transition(
            step_index=i,
            state_before=states[i],
            state_after=states[i + 1],
            action=AgentAction.final("A") if i == 3 else AgentAction.noop(),
            active_skill=refs[i],
            done=i == 3,
        )
        for i in range(4)
    )
    trajectory = replace(_trajectory(task, 1.0, 0, refs[0]), transitions=rows)
    credits = SkillCreditAssigner().assign((trajectory,), baselines={task.task_id: 0.0})
    assert [
        (c.activation_start_step, c.activation_steps, c.advantage) for c in credits
    ] == [(0, 2, 0.25), (2, 2, 0.25)]
    stats = SkillStatistics().update(pool, credits).get(skill.reference).stats
    assert stats.frequency == 2 and stats.average_gain == 0.25
    seen = []

    def diagnose(segment, advantage, references):
        seen.append((segment.start_step, segment.end_step, references))
        assert len(segment.transitions) == 2
        return {
            "diagnosis": "window only",
            "initiation": "when relevant",
            "policy": "observe",
            "termination": "evidence sufficient",
        }

    gradients = SemanticGradientEngine(diagnose).diagnose(
        (trajectory,), {trajectory.trajectory_id: 1.0}
    )
    assert seen == [(0, 2, (skill.reference,)), (2, 4, (skill.reference,))]
    assert len(gradients) == 2


def test_strict_critique_preserves_rewards_and_all_failures():
    from spatialcraft.knowledge.experience import CrossRolloutCritic
    from spatialcraft.knowledge.experience.visual_summarizer import (
        VisualTrajectorySummarizer,
    )

    task = TaskSample(task_id="failures", dataset="test", question="Where is the cup?")
    rows = tuple(_trajectory(task, 0.0, i, None) for i in range(4))
    seen = []

    def critique(prompt):
        seen.append(prompt)
        return "CONDITION: When locating a cup\nACTION: Check visible evidence."

    result = CrossRolloutCritic(
        summarizer=VisualTrajectorySummarizer(
            generator=lambda _: "Visual summary without a score"
        ),
        generator=critique,
        strict=True,
    ).critique(rows)
    assert result.best_trajectory_ids == ()
    assert len(result.failed_trajectory_ids) == 4
    assert seen[0].count("verified reward: 0.0") == 4
    with pytest.raises(ValueError, match="no heuristic fallback"):
        CrossRolloutCritic(generator=lambda _: "invalid output", strict=True).critique(
            rows
        )
