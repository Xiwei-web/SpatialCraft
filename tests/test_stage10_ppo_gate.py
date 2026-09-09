from __future__ import annotations

from dataclasses import replace
from math import exp, isclose

from spatialcraft.knowledge.skill import (
    NonParametricPPOGate,
    PPOEvaluationExample,
    SeedCatalog,
    SurrogateSkillGate,
)
from spatialcraft.models import (
    MessageRole,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    SequenceScore,
)
from spatialcraft.models.scoring import (
    ScoringMode,
    SurrogateScore,
    TargetLogprobScorer,
    serialize_action,
    with_skill_prompt,
)
from spatialcraft.schemas import (
    AgentAction,
    PPOGateRecord,
    SkillCandidate,
    SkillEvolutionType,
)


class _FixedTargetProvider(ModelProvider):
    def generate(self, request):
        return ModelResponse(provider="fake", model="fake", text="unused")

    def score(self, request, target_text):
        prompt = "\n".join(message.text_content for message in request.messages)
        mean = (
            -0.8
            if "Good refinement" in prompt
            else -1.2
            if "Bad refinement" in prompt
            else -1.0
        )
        return SequenceScore(
            model="qwen-test-tokenizer",
            target_text=target_text,
            token_ids=(101, 102),
            token_logprobs=(mean, mean),
            prompt_token_count=17,
        )


def _candidates():
    parent = SeedCatalog.pool().active()[0]
    good_skill = replace(
        parent,
        name="Good refinement",
        version=2,
        evolution_type=SkillEvolutionType.REFINE,
        parent_skill_ref=parent.reference,
    )
    bad_skill = replace(
        parent,
        name="Bad refinement",
        version=2,
        evolution_type=SkillEvolutionType.REFINE,
        parent_skill_ref=parent.reference,
    )
    return parent, (
        SkillCandidate(
            candidate_id="candidate-good",
            skill=good_skill,
            evolution_type=SkillEvolutionType.REFINE,
        ),
        SkillCandidate(
            candidate_id="candidate-bad",
            skill=bad_skill,
            evolution_type=SkillEvolutionType.REFINE,
        ),
    )


def test_strict_ppo_is_recomputable_best_of_n_and_audited() -> None:
    parent, candidates = _candidates()
    base = ModelRequest(
        model_alias="qwen-scorer",
        messages=(
            ModelMessage.text(MessageRole.USER, "Historical state and evidence"),
        ),
    )
    target = serialize_action(AgentAction.final("A"))
    examples = (
        PPOEvaluationExample(
            trajectory_id="trajectory-positive",
            base_request=base,
            target_text=target,
            advantage=1.0,
        ),
        PPOEvaluationExample(
            trajectory_id="trajectory-negative",
            base_request=base,
            target_text=target,
            advantage=-0.5,
        ),
    )
    scorer = TargetLogprobScorer(_FixedTargetProvider(), model_alias="qwen-scorer")
    gate = NonParametricPPOGate(scorer, epsilon=0.2, acceptance_margin=0.04)
    result = gate.select(
        candidates,
        parents={candidate.candidate_id: parent for candidate in candidates},
        examples=examples,
    )
    assert result.selected_candidate_id == "candidate-good"
    assert result.accepted_candidate_id == "candidate-good"
    assert len(result.records) == 2
    record = next(item for item in result.records if item.accepted)
    first, second = record.trajectory_scores
    assert isclose(first.importance_ratio, exp(0.2))
    assert isclose(first.clipped_objective, 1.2)
    assert isclose(second.clipped_objective, -0.5 * exp(0.2))
    assert isclose(record.candidate_objective, (1.2 - 0.5 * exp(0.2)) / 2)
    assert record.metadata["formal_np_ppo"] is True
    assert record.metadata["scoring_mode"] == ScoringMode.STRICT_TEACHER_FORCED.value
    traces = record.metadata["token_traces"]
    assert traces[0]["old"]["token_ids"] == [101, 102]
    assert traces[0]["new"]["token_ids"] == [101, 102]
    assert traces[0]["old"]["metadata"]["prompt_tokens_excluded"] is True

    old_request = with_skill_prompt(base, parent)
    new_request = with_skill_prompt(base, candidates[0].skill)
    assert (
        old_request.messages[-1].text_content == new_request.messages[-1].text_content
    )
    assert old_request.messages[0].metadata["ppo_skill_slot"] is True
    assert new_request.messages[0].metadata["ppo_skill_slot"] is True


def test_surrogate_gate_cannot_be_mixed_with_formal_ppo_records() -> None:
    _, candidates = _candidates()
    result = SurrogateSkillGate(
        lambda candidate: 1.0 if candidate.candidate_id == "candidate-good" else 0.0,
        model_alias="api-model",
        threshold=0.5,
    ).select(candidates)
    assert isinstance(result, SurrogateScore)
    assert result.mode is ScoringMode.SURROGATE
    assert not isinstance(result, PPOGateRecord)


def test_zero_gain_and_zero_advantage_never_pass_positive_gate():
    parent, original = _candidates()
    candidates = tuple(
        replace(c, skill=replace(c.skill, name="unchanged score")) for c in original
    )
    base = ModelRequest(
        model_alias="qwen-scorer",
        messages=(ModelMessage.text(MessageRole.USER, "question"),),
    )
    gate = NonParametricPPOGate(
        TargetLogprobScorer(_FixedTargetProvider(), model_alias="qwen-scorer")
    )
    for advantage in (0.0, 1.0, -1.0):
        result = gate.select(
            candidates,
            parents={c.candidate_id: parent for c in candidates},
            examples=(
                PPOEvaluationExample(
                    trajectory_id="same",
                    base_request=base,
                    target_text="A",
                    advantage=advantage,
                ),
            ),
        )
        assert result.accepted_candidate_id is None
        assert not any(r.accepted for r in result.records)
