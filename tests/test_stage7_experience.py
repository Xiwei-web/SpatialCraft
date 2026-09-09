from __future__ import annotations

from pathlib import Path

import pytest

from spatialcraft.agent import (
    ActionParser,
    ContextComposer,
    ExecutionLoop,
    SpatialAgent,
)
from spatialcraft.knowledge.experience import (
    ContextualExperienceRewriter,
    CrossRolloutCritic,
    ExperienceBank,
    ExperienceBankError,
    ExperienceConsolidator,
    ExperienceDeployment,
    ExperienceIndex,
    ExperienceRetriever,
    ExperienceSnapshotStore,
    HashingEmbedder,
    apply_updates,
    update_from_critique,
)
from spatialcraft.models import (
    Capability,
    LocalModelConfig,
    ModelCapabilities,
    ModelConfig,
    ModelProvider,
    ModelResponse,
    ProviderKind,
)
from spatialcraft.schemas import AnswerType, ExperienceItem, TaskSample
from spatialcraft.storage import SnapshotManifestStore, StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor, create_mock_tool_registry


class _FixedProvider(ModelProvider):
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts = []

    def generate(self, request):
        prompt = "\n".join(message.text_content for message in request.messages)
        assert "Reference answer:" not in prompt
        assert "secret-target" not in prompt
        self.prompts.append(prompt)
        return ModelResponse(provider="fixed", model="fixed", text=self.answer)


def _model() -> ModelConfig:
    return ModelConfig(
        alias="fixed",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        model_id="fixed",
        capabilities=ModelCapabilities(
            supported=frozenset(
                {
                    Capability.TEXT_INPUT,
                    Capability.TOOL_CALLING,
                }
            )
        ),
        local=LocalModelConfig(path="."),
    )


def _task() -> TaskSample:
    return TaskSample(
        dataset="test",
        question="State the learned target name.",
        answer_type=AnswerType.FREE_FORM,
        reference_answer="secret-target",
        metadata={"question_type": "relation"},
    )


def _agent(tmp_path: Path, provider: ModelProvider, run_id: str) -> SpatialAgent:
    tools = create_mock_tool_registry()
    store = ArtifactStore(StorageLayout(tmp_path / "storage"), run_id)
    return SpatialAgent(
        ExecutionLoop(
            provider=provider,
            composer=ContextComposer(_model(), tools),
            action_parser=ActionParser(tools),
            tool_executor=ToolExecutor(tools, store),
        )
    )


def test_deployment_retrieval_rewrite_and_provenance(tmp_path) -> None:
    items = (
        ExperienceItem(
            condition="When a relation task asks for a learned target",
            action="Recall the target only from observed task evidence.",
        ),
        ExperienceItem(
            condition="When comparing two objects in an image",
            action="Detect both objects before comparing their centers.",
        ),
    )
    bank = ExperienceBank(items)
    embedder = HashingEmbedder(64)
    index = ExperienceIndex.build(bank, embedder)
    deployment = ExperienceDeployment(
        ExperienceRetriever(bank.freeze(), index, embedder),
        rewriter=ContextualExperienceRewriter(),
    )
    task = _task()
    state = deployment.prepare_state(task, top_k=2)
    assert len(state.retrieved_experiences) == 2
    assert all(item.contextualized_text for item in state.retrieved_experiences)

    provider = _FixedProvider("secret-target")
    trajectory = _agent(tmp_path, provider, "with-experience").solve(
        task, initial_state=state
    )
    assert trajectory.reward == 1.0
    assert set(trajectory.used_experience_ids) == {
        item.experience_id for item in state.retrieved_experiences
    }
    assert "Retrieved experience" in provider.prompts[0]

    vanilla = _agent(
        tmp_path, _FixedProvider("secret-target"), "without-experience"
    ).solve(task)
    assert vanilla.reward == 1.0
    assert vanilla.used_experience_ids == ()


def test_post_rollout_update_consolidation_and_new_snapshot(tmp_path) -> None:
    task = _task()
    good = _agent(tmp_path, _FixedProvider("secret-target"), "good").solve(
        task, rollout_index=0
    )
    bad = _agent(tmp_path, _FixedProvider("wrong"), "bad").solve(task, rollout_index=1)
    critique = CrossRolloutCritic().critique((good, bad))
    assert critique.best_trajectory_ids == (good.trajectory_id,)
    assert bad.trajectory_id in critique.failed_trajectory_ids
    assert "Reference answer: secret-target" in critique.summaries[0]

    initial = ExperienceBank()
    update = update_from_critique(critique, dataset=task.dataset)
    updated = apply_updates(initial, (update,))
    assert len(updated.active()) == 1
    learned = updated.active()[0]
    assert learned.provenance.trajectory_ids
    duplicate = ExperienceItem(
        condition=learned.condition,
        action=learned.action,
    )
    duplicated = ExperienceBank((*updated.all(), duplicate))
    merges = ExperienceConsolidator(similarity_threshold=0.99).propose(duplicated)
    assert len(merges) == 1
    consolidated = duplicated.apply_many(merges)
    assert len(consolidated.active()) == 1

    layout = StorageLayout(tmp_path / "snapshots")
    snapshot_store = ExperienceSnapshotStore(SnapshotManifestStore(layout))
    embedder = HashingEmbedder(32)
    empty_index = ExperienceIndex.build(initial, embedder)
    first = snapshot_store.commit(
        run_id="experience-run", iteration=0, bank=initial, index=empty_index
    )
    second_index = ExperienceIndex.build(updated, embedder)
    second = snapshot_store.commit(
        run_id="experience-run",
        iteration=1,
        parent_snapshot_id=first.snapshot_id,
        bank=updated,
        index=second_index,
        source_trajectory_ids=(good.trajectory_id, bad.trajectory_id),
    )
    loaded_bank, loaded_index = snapshot_store.load(second)
    assert len(loaded_bank.active()) == 1
    assert loaded_index.references == (learned.reference,)
    with pytest.raises(ExperienceBankError):
        loaded_bank.apply(update)
