"""Behavioral coverage of v2 Experience transactions and visual retrieval."""

import numpy as np
import pytest

from spatialcraft.experiments.accumulation import KnowledgeState
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.experience.index import ExperienceIndex
from spatialcraft.knowledge.experience.learning_v2 import ExperienceLearningV2
from spatialcraft.knowledge.skill import SkillPool
from spatialcraft.models import ContentKind
from spatialcraft.schemas import (
    ActionType,
    AgentAction,
    ExperienceItem,
    ImageInput,
    RetrievedExperienceRef,
    SpatialState,
    TaskSample,
    Trajectory,
    Transition,
)


class ConstantEmbedder:
    identity = "test-semantic-space-v2"

    def embed(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts]).reshape(-1, 2)


def item(name="E1", **kwargs):
    return ExperienceItem(
        experience_id=name,
        condition="When comparing observer-relative directions",
        action="First identify the observer's coordinate axes.",
        **kwargs,
    )


def knowledge(*items):
    return KnowledgeState(ExperienceBank(tuple(items)).freeze(), SkillPool().freeze())


def rows(state, *, rewards=(1.0, 0.0, 1.0, 0.0), injected=None, task=None):
    task = task or TaskSample(
        task_id="task-1",
        dataset="test",
        question="Which object is left?",
        reference_answer="PRIVATE_REFERENCE",
        images=(ImageInput(uri="image.png"),),
    )
    selected = state.experiences.active() if injected is None else injected
    refs = tuple(
        RetrievedExperienceRef(
            experience_id=value.experience_id,
            version=value.version,
            retrieval_score=1.0,
            original_text=value.prompt_text,
            contextualized_text=value.prompt_text,
        )
        for value in selected
    )
    return tuple(
        Trajectory(
            task=task,
            rollout_index=index,
            trajectory_id=f"trajectory-{index}",
            executor_model="test",
            knowledge_snapshot_id=state.snapshot_id,
            reward=reward,
            transitions=(
                Transition(
                    step_index=0,
                    state_before=SpatialState(
                        task_id=task.task_id, step_index=0, retrieved_experiences=refs
                    ),
                    action=AgentAction(action_type=ActionType.FINAL, final_answer="A"),
                    done=True,
                ),
            ),
        )
        for index, reward in enumerate(rewards)
    )


def add(
    condition="When directions depend on viewpoint",
    action="Resolve the observer first.",
):
    return {
        "type": "add",
        "condition": condition,
        "action": action,
        "evidence_refs": ["trajectory-0:step-0"],
    }


class Generator:
    def __init__(self, operations=(), *, merge=None, manage=None, rewrite=None):
        self.operations = list(operations)
        self.merge = merge
        self.manage = manage
        self.rewrite = rewrite
        self.calls = []

    def __call__(self, operation, payload, *, media=(), validator=None):
        self.calls.append((operation, payload, media))
        if operation == "experience.summary":
            result = {
                "summary": "Observed a final decision and verified its result.",
                "observed_facts": ["A was answered."],
                "inferred_causes": [],
                "evidence_refs": [payload["trajectory_id"]],
            }
            if "operations" in payload["expected_output"]:
                result["operations"] = [
                    {
                        "type": "add",
                        "condition": "When observer axes are unclear",
                        "action": "Resolve the observer before comparing directions.",
                        "evidence_refs": [payload["trajectory_id"]],
                    }
                ]
        elif operation == "experience.critique":
            result = {"operations": self.operations}
        elif operation == "experience.merge":
            result = (
                self.merge(payload)
                if self.merge
                else {"decision": "keep_separate", "reason": "Different conditions."}
            )
        elif operation == "experience.manage":
            result = self.manage(payload)
        elif operation == "retrieval.decomposition":
            result = {
                "aspects": [
                    {"type": "reference_frame", "query": "Resolve observer axes"}
                ]
            }
        elif operation == "retrieval.rewrite":
            result = (
                self.rewrite(payload)
                if self.rewrite
                else {
                    "items": [
                        {
                            "source_ref": value["reference"],
                            "decision": "keep",
                            "condition": value["condition"],
                            "action": value["action"],
                        }
                        for value in payload["experiences"]
                    ]
                }
            )
        else:
            raise AssertionError(operation)
        if validator:
            validator(result)
        return result


def test_local_llm_merge_below_capacity_preserves_lineage_and_bank():
    state = knowledge(item())
    original = state.to_dict()
    generator = Generator(
        [add()],
        merge=lambda payload: {
            "decision": "merge",
            "source_refs": ["E1@1"],
            "condition": "When comparing viewpoints",
            "action": "Resolve the requested observer's frame first.",
            "reason": "Compatible observer guidance.",
        },
    )
    learner = ExperienceLearningV2(generator, ConstantEmbedder())
    result = learner.update(state, rows(state))
    bank = ExperienceBank.from_dict(result["experience_bank"])
    assert result["status"] == "completed"
    assert len(bank.active()) == 1
    assert len(bank.all()) == 3  # old item, incoming proposal, merged version
    merged = bank.active()[0]
    assert "E1@1" in merged.source_experience_refs
    assert set(merged.provenance.trajectory_ids) == {
        f"trajectory-{index}" for index in range(4)
    }
    assert "statistics_v2" not in merged.metadata
    assert state.to_dict() == original
    assert [call[0] for call in generator.calls].count("experience.merge") == 1
    assert not any(call[0] == "experience.manage" for call in generator.calls)


def test_merge_threshold_is_strict_and_no_candidate_skips_llm(monkeypatch):
    state = knowledge(item())
    monkeypatch.setattr(
        ExperienceIndex, "search", lambda *args, **kwargs: (("E1@1", 0.70),)
    )
    generator = Generator([add()])
    result = ExperienceLearningV2(generator, ConstantEmbedder()).update(
        state, rows(state)
    )
    assert result["status"] == "completed"
    assert len(ExperienceBank.from_dict(result["experience_bank"]).active()) == 2
    assert not any(call[0] == "experience.merge" for call in generator.calls)


def test_high_similarity_can_keep_opposite_conditions_separate():
    state = knowledge(item())
    generator = Generator(
        [
            add(
                "When the observer is the object rather than camera",
                "Use object-relative directions.",
            )
        ]
    )
    result = ExperienceLearningV2(generator, ConstantEmbedder()).update(
        state, rows(state)
    )
    assert len(ExperienceBank.from_dict(result["experience_bank"]).active()) == 2
    assert result["merge_audits"][0]["decision"]["decision"] == "keep_separate"


def test_modify_precedes_add_and_local_index_uses_updated_version():
    state = knowledge(item())
    modify = {
        **add(),
        "type": "modify",
        "target_ref": "E1@1",
        "action": "Check updated observer axes.",
    }

    def merge(payload):
        assert [candidate["reference"] for candidate in payload["candidates"]] == [
            "E1@2"
        ]
        assert payload["candidates"][0]["action"] == "Check updated observer axes."
        return {
            "decision": "merge",
            "source_refs": ["E1@2"],
            "condition": "When the frame is ambiguous",
            "action": "Resolve observer axes before comparing directions.",
            "reason": "Compatible correction.",
        }

    result = ExperienceLearningV2(
        Generator([add(), modify], merge=merge), ConstantEmbedder()
    ).update(state, rows(state))
    assert result["status"] == "completed"
    assert [value["operation"] for value in result["updates"]] == [
        "modify",
        "add",
        "merge",
    ]
    assert (
        "E1@2"
        in ExperienceBank.from_dict(result["experience_bank"])
        .active()[0]
        .source_experience_refs
    )


def test_unknown_modify_rejects_entire_critique_before_add():
    state = knowledge(item())
    bad = {**add(), "type": "modify", "target_ref": "invented@1"}
    result = ExperienceLearningV2(Generator([add(), bad]), ConstantEmbedder()).update(
        state, rows(state)
    )
    assert result["status"] == "skipped_validation_failure"
    assert result["experience_bank"] == state.experiences.to_dict()
    assert result["updates"] == []


def test_capacity_manager_sees_all_101_items_and_commits_atomically():
    state = knowledge(*(item(f"E{index:03d}") for index in range(100)))

    def manage(payload):
        assert len(payload["experiences"]) == 101
        assert payload["capacity"] == 100
        return {
            "operations": [
                {
                    "type": "delete",
                    "target_ref": "E099@1",
                    "reason": "Redundant frame instruction.",
                }
            ]
        }

    generator = Generator([add()], manage=manage)
    result = ExperienceLearningV2(generator, ConstantEmbedder()).update(
        state, rows(state, injected=())
    )
    bank = ExperienceBank.from_dict(result["experience_bank"])
    assert result["status"] == "completed"
    assert len(bank.active()) == 100
    assert bank.get("E099@1").status.value == "archived"
    assert [call[0] for call in generator.calls].count("experience.manage") == 1


@pytest.mark.parametrize(
    "manage",
    [
        lambda payload: {"operations": []},
        lambda payload: {
            "operations": [
                {"type": "delete", "target_ref": "unknown@1", "reason": "bad"}
            ]
        },
        lambda payload: {
            "operations": [
                {"type": "delete", "target_ref": "E1@1", "reason": "duplicate"},
                {
                    "type": "merge",
                    "source_refs": ["E1@1", payload["experiences"][1]["reference"]],
                    "condition": "When comparing",
                    "action": "Check frames",
                    "reason": "conflict",
                },
            ]
        },
    ],
)
def test_invalid_manage_rolls_back_add_and_statistics(manage):
    state = knowledge(item())
    learner = ExperienceLearningV2(
        Generator([add()], manage=manage), ConstantEmbedder(), {"capacity": 1}
    )
    learner.last_retrieval_audit = {
        "task_id": "task-1",
        "snapshot_id": state.snapshot_id,
        "retrieved_refs": ["E1@1"],
    }
    result = learner.update(state, rows(state))
    assert result["status"] == "skipped_validation_failure"
    assert result["failed_stage"] == "manage"
    assert result["experience_bank"] == state.experiences.to_dict()
    assert result["updates"] == []


def test_descriptive_shared_statistics_count_one_retrieval_four_injections():
    state = knowledge(item())
    learner = ExperienceLearningV2(Generator(), ConstantEmbedder())
    learner.last_retrieval_audit = {
        "task_id": "task-1",
        "snapshot_id": state.snapshot_id,
        "retrieved_refs": ["E1@1"],
    }
    result = learner.update(state, rows(state))
    assert result["status"] == "no_content_update"
    saved = ExperienceBank.from_dict(result["experience_bank"]).get("E1@1")
    stats = saved.metadata["statistics_v2"]
    assert stats["retrieval_count"] == 1
    assert stats["injection_count"] == 4
    assert stats["outcome_association"]["mean_reward"] == 0.5
    assert saved.stats.cumulative_reward_gain == 0  # legacy field is not used


def test_retrieval_passes_images_and_safe_task_and_records_skipped_ref():
    state = knowledge(item("E1"), item("E2"))
    task = rows(state)[0].task

    def rewrite(payload):
        return {
            "items": [
                {
                    "source_ref": "E1@1",
                    "decision": "keep",
                    "condition": "When observer axes are ambiguous",
                    "action": "Identify those axes first.",
                },
                {
                    "source_ref": "E2@1",
                    "decision": "skip",
                    "reason": "Duplicate guidance.",
                },
            ]
        }

    generator = Generator(rewrite=rewrite)
    learner = ExperienceLearningV2(generator, ConstantEmbedder())
    refs = learner.retrieve(task, state)
    assert len(refs) == 1 and refs[0].experience_id == "E1"
    assert set(learner.last_retrieval_audit["retrieved_refs"]) == {"E1@1", "E2@1"}
    assert learner.last_retrieval_audit["injected_refs"] == ["E1@1"]
    for operation, payload, media in generator.calls:
        assert payload["task"]["reference_answer"] is None
        assert "PRIVATE_REFERENCE" not in str(payload)
        assert [part.uri for part in media if part.kind is ContentKind.IMAGE] == [
            "image.png"
        ]
        assert media[0].kind is ContentKind.TEXT
    assert state.experiences.get("E1@1").metadata == {}


def test_rewrite_must_decide_every_source_and_validate_length():
    state = knowledge(item("E1"), item("E2"))
    generator = Generator(rewrite=lambda payload: {"items": []})
    with pytest.raises(ValueError, match="every retrieved reference"):
        ExperienceLearningV2(generator, ConstantEmbedder()).retrieve(
            rows(state)[0].task, state
        )


def test_empty_bank_short_circuits_retrieval_without_generator_or_embedder():
    state = knowledge()
    generator = Generator()
    learner = ExperienceLearningV2(generator, ConstantEmbedder())
    assert learner.retrieve(rows(state)[0].task, state) == ()
    assert generator.calls == []
    assert learner.last_retrieval_audit["status"] == "skipped_empty_bank"


def test_too_long_knowledge_is_rejected_without_string_truncation():
    state = knowledge()
    result = ExperienceLearningV2(
        Generator([add(action="word " * 70)]), ConstantEmbedder()
    ).update(state, rows(state))
    assert result["status"] == "skipped_validation_failure"
    assert result["experience_bank"] == state.experiences.to_dict()
    assert "64 words" in result["failure_reason"]


def test_infrastructure_failure_propagates_and_original_bank_is_unchanged():
    state = knowledge(item())

    def unavailable(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        ExperienceLearningV2(unavailable, ConstantEmbedder()).update(state, rows(state))
    assert state.experiences.get("E1@1").metadata == {}


def test_summary_has_actions_injected_text_and_verified_offline_answer():
    state = knowledge(item())
    generator = Generator()
    ExperienceLearningV2(generator, ConstantEmbedder()).update(state, rows(state))
    summaries = [
        payload
        for operation, payload, media in generator.calls
        if operation == "experience.summary"
    ]
    assert len(summaries) == 4
    assert summaries[0]["steps"][0]["action"]["final_answer"] == "A"
    assert (
        summaries[0]["injected_experiences"][0]["original_text"]
        == state.experiences.get("E1@1").prompt_text
    )
    assert summaries[0]["task"]["reference_answer"] == "PRIVATE_REFERENCE"


def test_no_decomposition_and_no_rewrite_use_raw_retrieval_without_llm():
    state = knowledge(item())
    generator = Generator()
    learner = ExperienceLearningV2(
        generator, ConstantEmbedder(), {"no_decomposition": True, "no_rewrite": True}
    )
    refs = learner.retrieve(rows(state)[0].task, state)
    assert generator.calls == []
    assert (
        len(refs) == 1
        and refs[0].prompt_text == state.experiences.get("E1@1").prompt_text
    )
    assert learner.last_retrieval_audit["aspects"] == [
        {"type": "raw_question", "query": "Which object is left?"}
    ]


def test_no_local_merge_still_runs_manager_when_over_capacity():
    state = knowledge(item())
    generator = Generator(
        [add()],
        manage=lambda payload: {
            "operations": [
                {
                    "type": "delete",
                    "target_ref": "E1@1",
                    "reason": "Redundant guidance.",
                }
            ]
        },
    )
    result = ExperienceLearningV2(
        generator, ConstantEmbedder(), {"no_local_merge": True, "capacity": 1}
    ).update(state, rows(state))
    assert result["status"] == "completed"
    assert not any(
        operation == "experience.merge" for operation, _, _ in generator.calls
    )
    assert any(operation == "experience.manage" for operation, _, _ in generator.calls)


def test_no_manager_overflow_is_explicit_transaction_rejection():
    state = knowledge(item())
    generator = Generator([add()])
    learner = ExperienceLearningV2(
        generator, ConstantEmbedder(), {"no_manager": True, "capacity": 1}
    )
    result = learner.update(state, rows(state))
    assert result["failed_stage"] == "disabled_manager_capacity_rejection"
    assert result["experience_bank"] == state.experiences.to_dict()
    assert not any(
        operation == "experience.manage" for operation, _, _ in generator.calls
    )


def test_no_visual_summary_retains_text_and_removes_media():
    state = knowledge()
    generator = Generator()
    result = ExperienceLearningV2(
        generator, ConstantEmbedder(), {"no_visual_summary": True}
    ).update(state, rows(state))
    assert result["status"] == "no_content_update"
    calls = [
        (payload, media)
        for operation, payload, media in generator.calls
        if operation == "experience.summary"
    ]
    assert all(not media and payload["steps"] for payload, media in calls)


def test_no_critique_extracts_individual_lessons_in_summary():
    state = knowledge()
    generator = Generator()
    result = ExperienceLearningV2(
        generator, ConstantEmbedder(), {"no_critique": True, "no_local_merge": True}
    ).update(state, rows(state))
    assert result["status"] == "completed"
    assert len(ExperienceBank.from_dict(result["experience_bank"]).active()) == 4
    assert result["critique"]["extraction_strategy"] == "individual_summary_extraction"
    assert not any(
        operation == "experience.critique" for operation, _, _ in generator.calls
    )


def test_provider_value_error_is_not_a_knowledge_validation_skip():
    state = knowledge()

    def unavailable(*args, **kwargs):
        raise ValueError("unsupported provider configuration")

    with pytest.raises(ValueError, match="unsupported provider configuration"):
        ExperienceLearningV2(unavailable, ConstantEmbedder()).update(state, rows(state))
