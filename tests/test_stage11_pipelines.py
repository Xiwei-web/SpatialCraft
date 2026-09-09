from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spatialcraft.knowledge import (
    ExperienceBank,
    HashingEmbedder,
    KnowledgeConflictError,
    KnowledgeCoordinator,
    SeedCatalog,
)
from spatialcraft.pipelines import AccumulationPipeline, ReadOnlyDeploymentPipeline
from spatialcraft.schemas import (
    AgentAction,
    AnswerType,
    ExperienceItem,
    ExperienceOperationType,
    ExperienceUpdate,
    SkillCandidate,
    SkillEvolutionType,
    SkillItem,
    SpatialState,
    TaskSample,
    Trajectory,
    TrajectoryStatus,
    Transition,
)
from spatialcraft.storage import SnapshotManifestStore, StorageLayout


def _trajectory(task, snapshot_id, index, reward=None):
    state = SpatialState(task_id=task.task_id, step_index=0)
    action = AgentAction.final("A")
    transition = Transition(
        step_index=0,
        state_before=state,
        action=action,
        state_after=state.next_step(),
        reward=reward,
        done=True,
    )
    now = datetime.now(timezone.utc)
    return Trajectory(
        task=task,
        rollout_index=index,
        executor_model="mock",
        knowledge_snapshot_id=snapshot_id,
        status=TrajectoryStatus.COMPLETED,
        transitions=(transition,),
        reward=reward,
        started_at=now,
        finished_at=now,
    )


def test_batch_barrier_joint_atomic_commit_and_read_only_deployment(tmp_path) -> None:
    manifests = SnapshotManifestStore(StorageLayout(tmp_path / "store"))
    coordinator = KnowledgeCoordinator(manifests, HashingEmbedder(32))
    initial = coordinator.initialize(
        run_id="joint-run",
        experience_bank=ExperienceBank(),
        skill_pool=SeedCatalog.pool(),
    )
    task = TaskSample(
        dataset="test",
        task_id="batch-task",
        question="Which relation?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        choices=("left", "right"),
        reference_answer="A",
    )
    events = []

    def rollout_batch(tasks, frozen, num_rollouts, seed):
        events.append("rollouts")
        assert frozen.snapshot.snapshot_id == initial.snapshot.snapshot_id
        assert frozen.experience_bank.frozen and frozen.skill_pool.frozen
        assert frozen.experience_index.frozen
        return tuple(
            _trajectory(tasks[0], frozen.snapshot.snapshot_id, index)
            for index in range(num_rollouts)
        )

    def reward_function(trajectory):
        events.append("reward")
        return 1.0 - trajectory.rollout_index * 0.25

    learned = ExperienceItem(
        condition="When testing the batch barrier",
        action="Commit knowledge only after every rollout has completed.",
    )

    def experience_builder(frozen, trajectories):
        events.append("experience")
        assert all(item.reward is not None for item in trajectories)
        return (
            ExperienceUpdate(
                operation=ExperienceOperationType.ADD,
                rationale="Learned after the reward barrier.",
                proposed_experience=learned,
                source_trajectory_ids=tuple(
                    item.trajectory_id for item in trajectories
                ),
            ),
        )

    new_skill = SkillItem(
        skill_id="skill-batch-barrier",
        name="BatchBarrierSkill",
        initiation="When coordinating parallel knowledge accumulation.",
        policy=("Freeze, rollout, reward, update, then commit.",),
        termination="After atomic snapshot publication.",
        evolution_type=SkillEvolutionType.NEW,
    )

    def skill_builder(frozen, trajectories):
        events.append("skill")
        assert learned.experience_id not in {
            item.experience_id for item in frozen.experience_bank.active()
        }
        return (
            SkillCandidate(
                candidate_id="accepted-batch-skill",
                skill=new_skill,
                evolution_type=SkillEvolutionType.NEW,
                source_trajectory_ids=tuple(
                    item.trajectory_id for item in trajectories
                ),
            ),
        )

    result = AccumulationPipeline(
        coordinator,
        rollout_batch,
        reward_function=reward_function,
        experience_builder=experience_builder,
        skill_builder=skill_builder,
    ).run(run_id="joint-run", tasks=(task,), num_rollouts=2, seed=7)
    assert events == ["rollouts", "reward", "reward", "experience", "skill"]
    assert result.phases == (
        "freeze_snapshot",
        "rollouts_complete",
        "rewards_complete",
        "experience_updates_built",
        "skill_updates_built",
        "snapshot_committed",
    )
    current = coordinator.load_current("joint-run")
    assert current.snapshot.parent_snapshot_id == initial.snapshot.snapshot_id
    assert current.snapshot.iteration == 1
    assert len(current.experience_bank.active()) == 1
    assert current.skill_pool.get(new_skill.reference) == new_skill
    assert current.experience_bank.frozen and current.skill_pool.frozen
    assert current.experience_index.frozen
    with pytest.raises(ValueError):
        current.experience_index.vectors[0, 0] = 9.0
    with pytest.raises(KnowledgeConflictError):
        coordinator.commit_batch(initial, source_trajectory_ids=())

    trajectory_sink = []
    metrics_sink = []
    deployed = ReadOnlyDeploymentPipeline(
        current,
        lambda sample, frozen: _trajectory(
            sample, frozen.snapshot.snapshot_id, 0, reward=1.0
        ),
    ).run(
        (task,),
        trajectory_sink=trajectory_sink.append,
        metrics_sink=metrics_sink.append,
    )
    assert deployed == tuple(trajectory_sink)
    assert metrics_sink[0]["mean_reward"] == 1.0
    with pytest.raises(RuntimeError):
        current.skill_pool.add(new_skill)
