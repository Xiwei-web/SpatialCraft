from dataclasses import replace
from pathlib import Path

import pytest

from spatialcraft.experiments.evolution_queue import EvolutionQueue
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.schemas import (
    ActiveSkillRef,
    AgentAction,
    SpatialState,
    TaskSample,
    TaskSplit,
    Trajectory,
    TrajectoryStatus,
    Transition,
)


def task_rows(task_index, parents, rewards=(1.0, 0.0, 1.0, 0.0)):
    task = TaskSample(
        task_id=f"task-{task_index}",
        dataset="test",
        question="Compare objects",
        split=TaskSplit.TRAIN,
    )
    rows = []
    for i, reward in enumerate(rewards):
        transitions = []
        # Repeat the first parent in another activation window deliberately.
        for step, parent in enumerate((*parents, parents[0])):
            active = ActiveSkillRef(
                skill_id=parent.skill_id, version=parent.version, activated_at_step=step
            )
            state = SpatialState(
                task_id=task.task_id, step_index=step, active_skill=active
            )
            transitions.append(
                Transition(
                    step_index=step,
                    state_before=state,
                    state_after=state.next_step(),
                    action=AgentAction.noop(),
                    active_skill=active,
                )
            )
        rows.append(
            Trajectory(
                task=task,
                trajectory_id=f"row-{task_index}-{i}",
                executor_model="test",
                knowledge_snapshot_id=f"snapshot-{task_index}",
                rollout_index=i,
                reward=reward,
                transitions=tuple(transitions),
                status=TrajectoryStatus.TRUNCATED,
            )
        )
    return tuple(rows)


def test_six_distinct_related_trajectories_fifo_two_parents_and_original_baselines():
    parents = SeedCatalog.pool().active()[:3]
    refs = {s.reference for s in parents}
    queue = EvolutionQueue()
    first = task_rows(0, parents)
    queue.enqueue(first, task_index=0, active_references=refs)
    assert all(len(entries) == 4 for entries in queue.state["queues"].values())
    assert not queue.take_round()  # Four trajectories, not eight activation windows.
    queue = EvolutionQueue(queue.state)  # Simulate restart.
    second = task_rows(1, parents, rewards=(0.0, 0.0, 0.0, 1.0))
    queue.enqueue(second, task_index=1, active_references=refs)
    batches = queue.take_round()
    assert len(batches) == 2
    expected = [r.trajectory_id for r in (*first, *second[:2])]
    for entries in batches.values():
        assert [e["trajectory_id"] for e in entries] == expected
        assert [e["advantage"] for e in entries] == [0.5, -0.5, 0.5, -0.5, -0.25, -0.25]
    assert sorted(len(e) for e in queue.state["queues"].values()) == [2, 2, 8]
    selected = next(iter(batches))
    discarded = queue.discard_inactive(refs - {selected})
    assert len(discarded[selected]) == 2  # Do not reattribute old-version leftovers.
    assert selected not in queue.state["queues"]
    with pytest.raises(ValueError, match="twice"):
        queue.enqueue(second, task_index=1, active_references=refs)


def test_tail_is_not_forced_and_only_related_skill_is_counted():
    parents = SeedCatalog.pool().active()[:2]
    queue = EvolutionQueue()
    queue.enqueue(
        task_rows(0, parents[:1]),
        task_index=0,
        active_references={s.reference for s in parents},
    )
    assert set(queue.state["queues"]) == {parents[0].reference}
    assert queue.take_round() == {}
    assert len(EvolutionQueue(queue.state).state["queues"][parents[0].reference]) == 4


def test_confirmed_v2_defaults_and_no_legacy_task_batch():
    settings = ExperimentSettings()
    assert (
        settings.evolution_batch_trajectories,
        settings.max_parent_skills_per_round,
    ) == (6, 2)
    assert (settings.max_steps, settings.skill_max_lifetime) == (50, 8)
    assert settings.max_output_tokens == settings.skill_generation_max_tokens == 4096
    assert settings.training_temperature == 0.7 and settings.enable_thinking is False
    assert settings.ppo_thinking_mode == "action_only"
    with pytest.raises(TypeError):
        ExperimentSettings(batch_size=6)
    with pytest.raises(ValueError):
        replace(settings, skill_generation_max_tokens=4097)
    for values in (
        {"evolution_batch_trajectories": 5},
        {"max_parent_skills_per_round": 3},
        {"max_steps": 51},
        {"skill_max_lifetime": 9},
        {"max_output_tokens": 4097},
        {"enable_thinking": True},
        {"ppo_thinking_mode": "fixed_sampled_prefix"},
    ):
        with pytest.raises(ValueError):
            replace(settings, **values)


def test_confirmed_sampling_gate_archive_and_only_image_budget_pending():
    from spatialcraft.models.registry import load_yaml

    path = (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/qwen35_9b_spatialcraft.yaml"
    )
    settings = ExperimentSettings.load(path)
    assert settings.training_top_p == 0.9
    assert settings.deployment_temperature == settings.auxiliary_temperature == 0
    assert settings.ppo_epsilon == 0.2 and settings.ppo_positive_margin == 0
    config = load_yaml(path)
    assert config["provisional_engineering_defaults"] == ["image_max_pixels"]
    assert (
        config["protocol"]["old_version_remainder"]
        == "archive_after_parent_refinement_or_pruning"
    )
    with pytest.raises(ValueError):
        replace(settings, auxiliary_temperature=0.7)
