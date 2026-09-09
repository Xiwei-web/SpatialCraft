"""Strict non-parametric PPO candidate gate and separate surrogate path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp, isfinite

from spatialcraft.models import ModelRequest
from spatialcraft.models.scoring import (
    ScoringMode,
    SurrogateScore,
    TargetLogprobScorer,
    with_skill_prompt,
)
from spatialcraft.schemas import (
    PPOGateRecord,
    SkillCandidate,
    SkillItem,
    TrajectoryPPOScore,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOEvaluationExample:
    trajectory_id: str
    base_request: ModelRequest
    target_text: str
    advantage: float

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip() or not self.target_text:
            raise ValueError("PPO examples require trajectory_id and target_text")
        if not isfinite(float(self.advantage)):
            raise ValueError("PPO advantage must be finite")


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOGateResult:
    selected_candidate_id: str
    accepted_candidate_id: str | None
    records: tuple[PPOGateRecord, ...]

    @property
    def accepted(self) -> bool:
        return self.accepted_candidate_id is not None


class NonParametricPPOGate:
    """Score the same fixed actions under old/new prompts and apply PPO clipping."""

    def __init__(
        self,
        scorer: TargetLogprobScorer,
        *,
        epsilon: float = 0.2,
        acceptance_margin: float = 0.0,
    ) -> None:
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        if not isfinite(float(acceptance_margin)) or acceptance_margin < 0:
            raise ValueError("acceptance_margin must be finite and nonnegative")
        if scorer.mode is not ScoringMode.STRICT_TEACHER_FORCED:
            raise TypeError("formal NP-PPO requires strict teacher-forced scoring")
        self.scorer = scorer
        self.epsilon = float(epsilon)
        self.acceptance_margin = float(acceptance_margin)

    def _evaluate(
        self,
        candidate: SkillCandidate,
        parent: SkillItem | None,
        examples: tuple[PPOEvaluationExample, ...],
    ) -> tuple[float, float, tuple[TrajectoryPPOScore, ...], list[dict]]:
        scores = []
        traces = []
        old_objectives = []
        for example in examples:
            old_request = with_skill_prompt(example.base_request, parent)
            new_request = with_skill_prompt(example.base_request, candidate.skill)
            old_trace = self.scorer.score(old_request, example.target_text)
            new_trace = self.scorer.score(new_request, example.target_text)
            if old_trace.token_ids != new_trace.token_ids:
                raise ValueError(
                    "old/new skill prompts produced different target tokenization"
                )
            delta = new_trace.mean_logprob - old_trace.mean_logprob
            ratio = exp(max(-80.0, min(80.0, delta)))
            advantage = float(example.advantage)
            unclipped = ratio * advantage
            clipped_ratio = max(1.0 - self.epsilon, min(1.0 + self.epsilon, ratio))
            clipped = min(unclipped, clipped_ratio * advantage)
            old_objectives.append(advantage)
            scores.append(
                TrajectoryPPOScore(
                    trajectory_id=example.trajectory_id,
                    advantage=advantage,
                    old_mean_logprob=old_trace.mean_logprob,
                    new_mean_logprob=new_trace.mean_logprob,
                    importance_ratio=ratio,
                    unclipped_objective=unclipped,
                    clipped_objective=clipped,
                    target_token_count=old_trace.target_token_count,
                )
            )
            traces.append(
                {
                    "trajectory_id": example.trajectory_id,
                    "target_text": example.target_text,
                    "old": old_trace.to_dict(),
                    "new": new_trace.to_dict(),
                }
            )
        objective = sum(item.clipped_objective for item in scores) / len(scores)
        old_objective = sum(old_objectives) / len(old_objectives)
        return objective, old_objective, tuple(scores), traces

    def select(
        self,
        candidates: tuple[SkillCandidate, ...],
        *,
        parents: dict[str, SkillItem | None],
        examples: tuple[PPOEvaluationExample, ...],
    ) -> PPOGateResult:
        if not candidates or not examples:
            raise ValueError("NP-PPO requires candidates and evaluation examples")
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate IDs")
        for candidate in candidates:
            if candidate.candidate_id not in parents:
                raise ValueError("Each candidate requires an explicit parent or NONE")
            parent = parents[candidate.candidate_id]
            if candidate.skill.parent_skill_ref != (
                parent.reference if parent else None
            ):
                raise ValueError("Candidate/parent lineage mismatch")
        evaluated = []
        for candidate in candidates:
            parent = parents.get(candidate.candidate_id)
            objective, old_objective, scores, traces = self._evaluate(
                candidate, parent, examples
            )
            evaluated.append(
                (candidate, parent, objective, old_objective, scores, traces)
            )
        best = max(evaluated, key=lambda row: (row[2], row[0].candidate_id))
        threshold = best[3] + self.acceptance_margin
        accepted_id = best[0].candidate_id if best[2] > threshold else None
        records = []
        for candidate, parent, objective, old_objective, scores, traces in evaluated:
            is_best = candidate.candidate_id == best[0].candidate_id
            accepted = is_best and accepted_id is not None
            records.append(
                PPOGateRecord(
                    candidate_id=candidate.candidate_id,
                    parent_skill_ref=parent.reference if parent else "NONE",
                    scorer_model=self.scorer.model_alias,
                    epsilon=self.epsilon,
                    acceptance_margin=self.acceptance_margin,
                    candidate_objective=objective,
                    accepted=accepted,
                    trajectory_scores=scores,
                    reason=(
                        "best_of_n_and_margin_passed"
                        if accepted
                        else "not_best_of_n"
                        if not is_best
                        else "acceptance_margin_not_met"
                    ),
                    metadata={
                        "scoring_mode": ScoringMode.STRICT_TEACHER_FORCED.value,
                        "formal_np_ppo": True,
                        "old_objective": old_objective,
                        "required_objective": old_objective + self.acceptance_margin,
                        "candidate_ranked_best": is_best,
                        "token_traces": traces,
                    },
                )
            )
        return PPOGateResult(
            selected_candidate_id=best[0].candidate_id,
            accepted_candidate_id=accepted_id,
            records=tuple(records),
        )


class SurrogateSkillGate:
    """Non-PPO fallback whose result type cannot be mistaken for PPOGateRecord."""

    def __init__(
        self,
        scorer: Callable[[SkillCandidate], float],
        *,
        model_alias: str,
        threshold: float = 0.0,
    ) -> None:
        self.scorer = scorer
        self.model_alias = model_alias
        self.threshold = float(threshold)

    def select(self, candidates: tuple[SkillCandidate, ...]) -> SurrogateScore:
        if not candidates:
            raise ValueError("surrogate gate requires candidates")
        values = [
            (float(self.scorer(candidate)), candidate) for candidate in candidates
        ]
        score, candidate = max(values, key=lambda pair: (pair[0], pair[1].candidate_id))
        return SurrogateScore(
            model_alias=self.model_alias,
            candidate_id=candidate.candidate_id,
            score=score,
            reason="surrogate_threshold_passed"
            if score >= self.threshold
            else "surrogate_threshold_failed",
        )


__all__ = [
    "NonParametricPPOGate",
    "PPOEvaluationExample",
    "PPOGateResult",
    "SurrogateSkillGate",
]
