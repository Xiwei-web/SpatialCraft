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


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceLikelihoodExample:
    """A recorded action span; every trajectory receives equal total weight."""

    trajectory_id: str
    transition_id: str
    base_request: ModelRequest
    target_text: str
    target_token_ids: tuple[int, ...]
    advantage: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceLikelihoodResult:
    accepted_candidate_id: str | None
    records: tuple[dict, ...]


class LikelihoodScoreRejected(ValueError):
    """A computed candidate score is invalid; provider failures still propagate."""


class SequenceLikelihoodGate:
    """PPO-style raw-model sequence-likelihood surrogate, not sampling PPO.

    J = mean_trajectories(mean_matching_actions(min(r*A, clip(r)*A))).
    A is the original task-group advantage, with no trajectory-length divisor.
    Numerical overflow rejects the candidate; it never silently clamps a ratio.
    """

    def __init__(self, scorer, *, epsilon=0.2, acceptance_margin=0.0):
        if (
            not 0 <= epsilon < 1
            or not isfinite(acceptance_margin)
            or acceptance_margin < 0
        ):
            raise ValueError("Invalid likelihood Gate configuration")
        if scorer.mode is not ScoringMode.STRICT_TEACHER_FORCED:
            raise TypeError("Sequence likelihood requires fixed-target teacher forcing")
        self.scorer, self.epsilon, self.margin = scorer, epsilon, acceptance_margin

    @staticmethod
    def _average(rows):
        groups = {}
        for trajectory_id, value in rows:
            groups.setdefault(trajectory_id, []).append(value)
        return sum(sum(values) / len(values) for values in groups.values()) / len(
            groups
        )

    def select(self, candidates, *, parent, examples):
        if not candidates or not examples:
            raise ValueError(
                "Likelihood Gate requires candidates and historical actions"
            )
        if len({item.transition_id for item in examples}) != len(examples):
            raise ValueError("Duplicate historical actions in Gate")
        expected_ref = parent.reference if parent else None
        advantages = {}
        for example in examples:
            if example.base_request.model_alias != self.scorer.model_alias:
                raise ValueError("Scorer model must match the recorded executor model")
            if (
                not example.target_text
                or not example.target_token_ids
                or not isfinite(example.advantage)
            ):
                raise ValueError(
                    "Missing original action span/token IDs or invalid advantage"
                )
            if (
                example.trajectory_id in advantages
                and advantages[example.trajectory_id] != example.advantage
            ):
                raise ValueError(
                    "A trajectory must retain one original group advantage"
                )
            advantages[example.trajectory_id] = example.advantage
            slots = [
                m
                for m in example.base_request.messages
                if m.metadata.get("ppo_skill_slot")
            ]
            if (
                len(slots) != 1
                or slots[0].metadata.get("skill_reference") != expected_ref
            ):
                raise ValueError(
                    "Behavior Skill slot does not match the recorded parent/NONE"
                )
            expected_text = "Active procedural skill:\n" + (
                parent.format_for_prompt() if parent else "NONE"
            )
            if slots[0].text_content != expected_text:
                raise ValueError(
                    "Behavior Skill text differs from the actual parent version"
                )
        parent_j = sum(advantages.values()) / len(advantages)
        records = []
        old_traces = {}
        for candidate in candidates:
            record = {
                "candidate_id": candidate.candidate_id,
                "parent_skill_ref": expected_ref or "NONE",
                "scorer_model": self.scorer.model_alias,
                "epsilon": self.epsilon,
                "acceptance_margin": self.margin,
                "candidate_objective": None,
                "accepted": False,
                "trajectory_scores": [],
                "metadata": {
                    "scoring_mode": "sequence_raw_model_likelihood_surrogate",
                    "formal_np_ppo": False,
                    "weighting": "mean_trajectory_mean_matching_action",
                    "old_objective": parent_j,
                    "token_traces": [],
                },
            }
            if candidate.skill.parent_skill_ref != expected_ref:
                raise ValueError("Candidate lineage does not match the original parent")
            if all(value == 0 for value in advantages.values()):
                record.update(
                    reason="no_relative_scoring_signal", candidate_objective=0.0
                )
                records.append(record)
                continue
            objectives = []
            try:
                for example in examples:
                    old = old_traces.get(example.transition_id)
                    if old is None:
                        old = self.scorer.score(
                            example.base_request, example.target_text
                        )
                        old_traces[example.transition_id] = old
                    new = self.scorer.score(
                        with_skill_prompt(example.base_request, candidate.skill),
                        example.target_text,
                    )
                    if tuple(old.token_ids) != tuple(example.target_token_ids) or tuple(
                        new.token_ids
                    ) != tuple(example.target_token_ids):
                        raise LikelihoodScoreRejected(
                            "Target tokenization differs from recorded action tokens"
                        )
                    log_ratio = sum(new.token_logprobs) - sum(old.token_logprobs)
                    if not isfinite(log_ratio):
                        raise LikelihoodScoreRejected("Nonfinite sequence log ratio")
                    try:
                        ratio = exp(log_ratio)
                    except OverflowError as exc:
                        raise LikelihoodScoreRejected(
                            "Sequence ratio overflow"
                        ) from exc
                    if not isfinite(ratio) or ratio == 0:
                        raise LikelihoodScoreRejected(
                            "Sequence ratio overflow or underflow"
                        )
                    clipped = max(1 - self.epsilon, min(1 + self.epsilon, ratio))
                    objective = min(
                        ratio * example.advantage, clipped * example.advantage
                    )
                    if not isfinite(objective):
                        raise LikelihoodScoreRejected("Nonfinite clipped objective")
                    objectives.append((example.trajectory_id, objective))
                    record["trajectory_scores"].append(
                        {
                            "trajectory_id": example.trajectory_id,
                            "transition_id": example.transition_id,
                            "advantage": example.advantage,
                            "old_sequence_logprob": sum(old.token_logprobs),
                            "new_sequence_logprob": sum(new.token_logprobs),
                            "old_mean_logprob": old.mean_logprob,
                            "new_mean_logprob": new.mean_logprob,
                            "importance_ratio": ratio,
                            "log_ratio": log_ratio,
                            "clipped_objective": objective,
                            "target_token_count": len(example.target_token_ids),
                        }
                    )
                    record["metadata"]["token_traces"].append(
                        {
                            "transition_id": example.transition_id,
                            "target_text": example.target_text,
                            "old": old.to_dict(),
                            "new": new.to_dict(),
                        }
                    )
                record["candidate_objective"] = self._average(objectives)
                record["reason"] = "acceptance_margin_not_met"
            except LikelihoodScoreRejected as exc:
                record["reason"] = "invalid_likelihood_score"
                record["metadata"]["error"] = str(exc)
            records.append(record)
        valid = [r for r in records if r["candidate_objective"] is not None]
        best = (
            max(valid, key=lambda r: (r["candidate_objective"], r["candidate_id"]))
            if valid
            else None
        )
        accepted_id = None
        if best is not None and best["candidate_objective"] - parent_j > self.margin:
            best["accepted"] = True
            best["reason"] = "best_of_n_and_positive_gain"
            accepted_id = best["candidate_id"]
        for record in valid:
            record["metadata"]["gain"] = record["candidate_objective"] - parent_j
            if record is not best and record["reason"] != "no_relative_scoring_signal":
                record["reason"] = "not_best_of_n"
        return SequenceLikelihoodResult(
            accepted_candidate_id=accepted_id, records=tuple(records)
        )
