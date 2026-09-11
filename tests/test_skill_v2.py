from dataclasses import replace
from datetime import datetime, timezone
from math import exp
from types import SimpleNamespace

import numpy as np
import pytest

from spatialcraft.agent.skill_controller import SkillController
from spatialcraft.agent.termination_controller import TerminationController
from spatialcraft.knowledge.skill.learning_v2 import SkillLearningV2
from spatialcraft.knowledge.skill.lifecycle import SkillLifecyclePolicy
from spatialcraft.knowledge.skill.pool import SkillPool
from spatialcraft.knowledge.skill.ppo_gate import (
    SequenceLikelihoodExample,
    SequenceLikelihoodGate,
)
from spatialcraft.knowledge.skill.selector import SkillSelector
from spatialcraft.models import MessageRole, ModelMessage, ModelRequest
from spatialcraft.models.scoring import ScoringMode, with_skill_prompt
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import (
    ActiveSkillRef,
    AgentAction,
    LogProbTrace,
    SkillCandidate,
    SkillEvolutionType,
    SkillItem,
    SkillStats,
    SpatialState,
    TaskSample,
    Trajectory,
    TrajectoryStatus,
    Transition,
)


class Embedder:
    def embed(self, texts):
        return np.asarray([[1.0, float(i + 1)] for i, _ in enumerate(texts)])


class Scorer:
    mode = ScoringMode.STRICT_TEACHER_FORCED
    model_alias = "mock"

    def __init__(self, delta=0.1, token_ids=(1, 2)):
        self.delta, self.token_ids, self.calls = delta, token_ids, []

    def score(self, request, text):
        self.calls.append((request, text))
        improved = any(
            "improved" in m.text_content
            for m in request.messages
            if m.metadata.get("ppo_skill_slot")
        )
        value = -2 + (self.delta * (1 if text == "A" else -1) if improved else 0)
        return LogProbTrace(
            model_alias="mock",
            target_text=text,
            token_ids=self.token_ids,
            token_logprobs=(value,) * len(self.token_ids),
            prompt_token_count=10,
            tokenizer_id="mock",
        )


def skill():
    return SkillItem(
        skill_id="K",
        name="spatial",
        initiation="when comparing frames",
        policy=("check frame axes",),
        termination="relation supported",
    )


def candidate(parent, improved=True):
    proposed = replace(
        parent,
        version=parent.version + 1,
        initiation="improved" if improved else parent.initiation,
        parent_skill_ref=parent.reference,
        evolution_type=SkillEvolutionType.REFINE,
    )
    return SkillCandidate(skill=proposed, evolution_type=SkillEvolutionType.REFINE)


def request(parent):
    return with_skill_prompt(
        ModelRequest(
            model_alias="mock",
            messages=(
                ModelMessage.text(MessageRole.SYSTEM, "system"),
                ModelMessage.text(MessageRole.USER, "public task"),
            ),
        ),
        parent,
    )


def example(parent, trajectory_id="T", transition_id="t", reward=1, text="A"):
    return SequenceLikelihoodExample(
        trajectory_id=trajectory_id,
        transition_id=transition_id,
        base_request=request(parent),
        target_text=text,
        target_token_ids=(1, 2),
        advantage=reward,
    )


def rows(parent, task_id="task", rewards=(1, 0, 1, 0)):
    task = TaskSample(dataset="test", task_id=task_id, question="Compare frames")
    result = []
    for index, reward in enumerate(rewards):
        active = (
            ActiveSkillRef(
                skill_id=parent.skill_id, version=parent.version, activated_at_step=0
            )
            if parent
            else None
        )
        state = SpatialState(task_id=task_id, step_index=0, active_skill=active)
        transition = Transition(
            step_index=0,
            state_before=state,
            state_after=state.next_step(),
            action=AgentAction.final("A" if reward else "B"),
            active_skill=active,
            done=True,
            transition_id=f"{task_id}-{index}-step",
            metadata={
                "model_request": request_to_dict(request(parent)),
                "action_target": {
                    "text": "A" if reward else "B",
                    "token_ids": [1, 2],
                    "prefix": "",
                    "prefix_token_ids": [],
                },
            },
        )
        now = datetime.now(timezone.utc)
        result.append(
            Trajectory(
                started_at=now,
                finished_at=now,
                task=task,
                rollout_index=index,
                executor_model="mock",
                knowledge_snapshot_id="snapshot",
                trajectory_id=f"{task_id}-{index}",
                status=TrajectoryStatus.COMPLETED,
                transitions=(transition,),
                reward=reward,
            )
        )
    return tuple(result)


def generate_factory(calls, *, related=True, discovery=False, no_change=False):
    def generate(operation, payload, media=(), validator=None):
        calls.append((operation, payload))
        if operation == "skill.semantic_gradient":
            output = {
                "is_related": related,
                "no_change": no_change,
                "reason": "supported observation",
                "initiation": "refine viewpoint condition",
                "policy": "check independent axes",
                "termination": "supported relation",
                "evidence_refs": [e["transition_id"] for e in payload["evidence"]],
                "needs_new_skill": discovery,
                "discovery_key": "reference_frames" if discovery else None,
                "discovery_need": "resolve observer reference frames"
                if discovery
                else None,
            }
        elif operation == "skill.gradient_aggregation":
            output = {
                "no_change": no_change,
                "initiation": "observer differs from camera",
                "policy": "check independent axes",
                "termination": "supported relation",
                "conflicts": "none",
                "evidence_refs": list(
                    dict.fromkeys(
                        t for d in payload["diagnoses"] for t in d["transition_ids"]
                    )
                ),
            }
        elif operation == "skill.candidate_generation":
            output = {
                "name": "new spatial",
                "initiation": "improved",
                "policy": ["validate axes"],
                "termination": "relation supported",
            }
        elif operation == "skill.deduplication":
            output = {"duplicates": []}
        else:
            raise AssertionError(operation)
        if validator:
            validator(output)
        return output

    return generate


def learner(calls, *, config=None, **kwargs):
    return SkillLearningV2(
        generate_factory(calls, **kwargs),
        Embedder(),
        Scorer(),
        lambda text: len(text.split()),
        config={"deduplication_mode": "exact", **(config or {})},
    )


def test_sequence_gate_uses_sum_and_equal_trajectory_weights():
    parent = skill()
    examples = (
        example(parent, "long", "a", 1),
        example(parent, "long", "b", 1),
        example(parent, "short", "c", -1, "B"),
    )
    result = SequenceLikelihoodGate(Scorer()).select(
        (candidate(parent),), parent=parent, examples=examples
    )
    record = result.records[0]
    assert record["candidate_objective"] == pytest.approx((1.2 - exp(-0.2)) / 2)
    assert record["trajectory_scores"][0]["importance_ratio"] == pytest.approx(exp(0.2))
    assert record["metadata"]["old_objective"] == 0
    assert result.accepted_candidate_id


def test_identical_candidate_and_zero_signal_are_never_accepted():
    parent = skill()
    gate = SequenceLikelihoodGate(Scorer())
    result = gate.select(
        (candidate(parent, improved=False),), parent=parent, examples=(example(parent),)
    )
    assert not result.accepted_candidate_id
    assert result.records[0]["metadata"]["gain"] == 0
    result = gate.select(
        (candidate(parent),), parent=parent, examples=(example(parent, reward=0),)
    )
    assert result.records[0]["reason"] == "no_relative_scoring_signal"


def test_numerical_overflow_and_recorded_token_mismatch_reject():
    parent = skill()
    for scorer in (Scorer(delta=-1000), Scorer(token_ids=(3, 4))):
        result = SequenceLikelihoodGate(scorer).select(
            (candidate(parent),),
            parent=parent,
            examples=(example(parent, reward=-1, text="B"),),
        )
        assert not result.accepted_candidate_id
        assert result.records[0]["candidate_objective"] is None
        assert result.records[0]["reason"] == "invalid_likelihood_score"


def test_none_behavior_cannot_be_fabricated_by_removing_parent():
    parent = skill()
    new = SkillCandidate(
        skill=replace(skill(), skill_id="new", evolution_type=SkillEvolutionType.NEW),
        evolution_type=SkillEvolutionType.NEW,
    )
    with pytest.raises(ValueError, match="recorded parent/NONE"):
        SequenceLikelihoodGate(Scorer()).select(
            (new,), parent=None, examples=(example(parent),)
        )


def test_skill_slot_replacement_preserves_historical_position_and_messages():
    parent = skill()
    original = request(parent)
    messages = (
        original.messages[0],
        original.messages[2],
        original.messages[1],
        ModelMessage.text(MessageRole.ASSISTANT, "historical observation"),
    )
    original = replace(original, messages=messages)
    modified = with_skill_prompt(original, candidate(parent).skill)
    assert modified.messages[0] == original.messages[0]
    assert modified.messages[1] == original.messages[1]
    assert modified.messages[3] == original.messages[3]
    assert modified.messages[2].metadata["ppo_skill_slot"]


def test_full_refine_path_calls_real_operations_and_retires_old_queue():
    calls = []
    parent = skill()
    learning = learner(calls)
    first = rows(parent, "task1")
    first_result = learning.evolve(
        SimpleNamespace(skills=SkillPool((parent,))), first, 0
    )
    assert first_result["pending_counts"] == {parent.reference: 4}
    saved = {f"tasks/00000/rollouts/{r.rollout_index:02d}/complete": r for r in first}
    learning.load_trajectory = saved.__getitem__
    result = learning.evolve(
        SimpleNamespace(skills=SkillPool.from_dict(first_result["skill_pool"])),
        rows(parent, "task2"),
        1,
        first_result["evolution_state"],
    )
    assert len(result["candidates"]) == 3
    assert result["accepted_candidate_ids"]
    assert result["evolution_batches"][0]["reward_counts"] == {"0": 3, "1": 3}
    assert len(result["discarded_inactive_version_entries"][parent.reference]) == 2
    active = SkillPool.from_dict(result["skill_pool"]).active()
    assert len(active) == 1 and active[0].reference == "K@2"
    assert [o for o, _ in calls].count("skill.gradient_aggregation") == 1
    candidates = [p for o, p in calls if o == "skill.candidate_generation"]
    assert len({p["parent"]["version"] for p in candidates}) == 1
    assert len({p["seed"] for p in candidates}) == 3


def test_false_relevance_does_not_enter_refine_or_discovery():
    result = learner([], related=False).evolve(
        SimpleNamespace(skills=SkillPool((skill(),))), rows(skill()), 0
    )
    assert result["pending_counts"] == {}
    assert all(d["is_related"] is False for d in result["semantic_gradients"])
    assert not result["credits"]


def test_discovery_uses_true_none_and_adds_new_identity():
    calls = []
    learning = learner(
        calls,
        related=False,
        discovery=True,
        config={
            "batch_trajectories": 4,
            "preferred_low_reward": 2,
            "preferred_high_reward": 2,
        },
    )
    result = learning.evolve(SimpleNamespace(skills=SkillPool()), rows(None), 0)
    pool = SkillPool.from_dict(result["skill_pool"])
    assert len(pool.active()) == 1
    assert pool.active()[0].parent_skill_ref is None
    assert pool.active()[0].skill_id.startswith("learned_")
    assert all(r["parent_skill_ref"] == "NONE" for r in result["ppo_records"])


def test_task_group_all_success_does_not_delete_neutral_skill():
    learning = learner([])
    parent = skill()
    result = learning.evolve(
        SimpleNamespace(skills=SkillPool((parent,))),
        rows(parent, rewards=(1, 1, 1, 1)),
        0,
    )
    active = SkillPool.from_dict(result["skill_pool"]).active()
    assert active[0].stats.success_count == 4
    assert active[0].stats.average_gain == 0
    assert not result["pruned_skill_references"]
    negative = replace(
        parent,
        stats=SkillStats(frequency=3, total_gain=-1.5, average_gain=-0.5),
        metadata={"quality_nonzero_trajectory_ids": ["a", "b", "c"]},
    )
    cleaned, removed, _ = learning._maintain(SkillPool((negative,)))
    assert not cleaned.active() and removed == [parent.reference]


def test_selector_can_return_none_from_nonempty_pool_and_honors_threshold():
    parent = skill()
    task = TaskSample(dataset="test", task_id="task", question="unrelated")
    state = SpatialState(task_id=task.task_id, step_index=0)
    selector = SkillSelector(
        embedder=Embedder(), applicability_judge=lambda skills, task, state: ()
    )
    assert selector.select(SkillPool((parent,)), task, state) is None
    selector = SkillSelector(embedder=Embedder(), minimum_score=1.01)
    assert selector.select(SkillPool((parent,)), task, state) is None
    selector = SkillSelector(
        embedder=Embedder(), applicability_judge=lambda *args: ("UNKNOWN@1",)
    )
    with pytest.raises(ValueError, match="unknown"):
        selector.select(SkillPool((parent,)), task, state)


def test_termination_receives_task_and_keeps_deterministic_shortcuts():
    parent = skill()
    task = TaskSample(dataset="test", task_id="task", question="observable condition")
    calls = []
    policy = SkillLifecyclePolicy(
        task_condition_evaluator=lambda t, k, s, a: calls.append(t) or True
    )
    controller = SkillController(
        SkillPool((parent,)), termination=TerminationController(policy)
    )
    state = controller.before_step(
        task, SpatialState(task_id=task.task_id, step_index=0)
    )
    result = controller.after_step(task, state, AgentAction.noop(), state.next_step())
    assert calls == [task] and result.active_skill is None
    calls.clear()
    controller.after_step(task, state, AgentAction.final("A"), state.next_step())
    assert calls == []


def test_ablation_switches_change_actual_calls_and_commit_behavior():
    calls = []
    learning = learner(
        calls,
        config={
            "no_semantic_gradient": True,
            "no_gradient_aggregation": True,
            "no_ppo_gate": True,
            "no_score_pruning": True,
            "batch_trajectories": 4,
            "preferred_low_reward": 2,
            "preferred_high_reward": 2,
        },
    )
    result = learning.evolve(
        SimpleNamespace(skills=SkillPool((skill(),))),
        rows(skill(), rewards=(1, 1, 1, 1)),
        0,
    )
    assert [op for op, _ in calls] == ["skill.candidate_generation"] * 3
    assert result["accepted_candidate_ids"]
    assert result["ppo_records"][0]["metadata"]["scoring_mode"] == "ungated_ablation"
    assert (
        result["semantic_gradients"][0]["eligibility_mode"]
        == "executed_segment_ablation"
    )


def test_unavailable_action_span_rejects_only_the_selected_batch():
    calls = []
    learning = learner(
        calls,
        config={
            "batch_trajectories": 4,
            "preferred_low_reward": 2,
            "preferred_high_reward": 2,
        },
    )
    group = list(rows(skill()))
    bad = group[0].transitions[0]
    group[0] = replace(
        group[0],
        transitions=(
            replace(bad, metadata={"model_request": bad.metadata["model_request"]}),
        ),
    )
    result = learning.evolve(
        SimpleNamespace(skills=SkillPool((skill(),))), tuple(group), 0
    )
    assert result["evolution_batches"][0]["status"] == "batch_rejected_missing_span"
    assert not result["candidates"]
    assert len(SkillPool.from_dict(result["skill_pool"]).active()) == 1


def test_provider_configuration_failure_is_not_a_candidate_rejection():
    parent = skill()

    class FailingScorer(Scorer):
        def score(self, request, text):
            raise ValueError("provider configuration is invalid")

    with pytest.raises(ValueError, match="provider configuration"):
        SequenceLikelihoodGate(FailingScorer()).select(
            (candidate(parent),), parent=parent, examples=(example(parent),)
        )
