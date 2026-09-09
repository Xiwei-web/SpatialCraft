"""Task-level procedural skills and non-parametric PPO audit schemas."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any

from ._base import SchemaMixin, new_id, normalize_datetime, require_non_empty, utc_now


class SkillStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class SkillEvolutionType(str, Enum):
    SEED = "seed"
    REFINE = "refine"
    NEW = "new"


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillStats(SchemaMixin):
    """Online statistics used for selection and pool maintenance."""

    frequency: int = 0
    total_gain: float = 0.0
    average_gain: float = 0.0
    maturity: int = 0
    success_count: int = 0
    last_evolved_iteration: int | None = None

    def __post_init__(self) -> None:
        if min(self.frequency, self.maturity, self.success_count) < 0:
            raise ValueError("skill counters cannot be negative")
        if self.success_count > self.frequency:
            raise ValueError("success_count cannot exceed frequency")
        if not isfinite(float(self.total_gain)) or not isfinite(
            float(self.average_gain)
        ):
            raise ValueError("skill gain values must be finite")
        expected = self.total_gain / max(1, self.frequency)
        if abs(expected - self.average_gain) > 1e-8:
            raise ValueError("average_gain must equal total_gain / max(1, frequency)")
        if self.last_evolved_iteration is not None and self.last_evolved_iteration < 0:
            raise ValueError("last_evolved_iteration cannot be negative")

    def record_usage(
        self, *, advantage: float, call_count: int = 1, successful: bool = False
    ) -> SkillStats:
        if call_count < 1:
            raise ValueError("call_count must be positive")
        if not isfinite(float(advantage)):
            raise ValueError("advantage must be finite")
        frequency = self.frequency + call_count
        total_gain = self.total_gain + float(advantage)
        return replace(
            self,
            frequency=frequency,
            total_gain=total_gain,
            average_gain=total_gain / frequency,
            success_count=self.success_count + int(successful),
        )

    def increment_maturity(self) -> SkillStats:
        return replace(self, maturity=self.maturity + 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillItem(SchemaMixin):
    """A temporally extended skill following initiation/policy/termination."""

    name: str
    initiation: str
    policy: tuple[str, ...]
    termination: str
    skill_id: str = field(default_factory=lambda: new_id("skill"))
    version: int = 1
    status: SkillStatus = SkillStatus.ACTIVE
    evolution_type: SkillEvolutionType = SkillEvolutionType.NEW
    parent_skill_ref: str | None = None
    stats: SkillStats = field(default_factory=SkillStats)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "initiation", "termination", "skill_id"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        policy = tuple(step.strip() for step in self.policy)
        if not policy or any(not step for step in policy):
            raise ValueError("skill policy requires one or more non-empty steps")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.version < 1:
            raise ValueError("skill version must be at least 1")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if (
            self.evolution_type is SkillEvolutionType.REFINE
            and not self.parent_skill_ref
        ):
            raise ValueError("refined skills require parent_skill_ref")

    @property
    def reference(self) -> str:
        return f"{self.skill_id}@{self.version}"

    def format_for_prompt(self) -> str:
        steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(self.policy, 1)
        )
        return (
            f"Skill: {self.name}\n"
            f"Initiation: {self.initiation}\n"
            f"Policy:\n{steps}\n"
            f"Termination: {self.termination}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticGradient(SchemaMixin):
    """Trajectory-specific diagnosis aligned to the three skill components."""

    trajectory_id: str
    diagnosis: str
    is_related: bool
    initiation: str = ""
    policy: str = ""
    termination: str = ""
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_id",
            require_non_empty(self.trajectory_id, "trajectory_id"),
        )
        object.__setattr__(
            self, "diagnosis", require_non_empty(self.diagnosis, "diagnosis")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.reward is not None and not isfinite(float(self.reward)):
            raise ValueError("semantic-gradient reward must be finite")
        if self.is_related and not any(
            text.strip() for text in (self.initiation, self.policy, self.termination)
        ):
            raise ValueError("related semantic gradients require a component update")


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillCandidate(SchemaMixin):
    """A proposed skill plus the evidence that generated it."""

    skill: SkillItem
    evolution_type: SkillEvolutionType
    source_gradient_ids: tuple[str, ...] = ()
    source_trajectory_ids: tuple[str, ...] = ()
    candidate_id: str = field(default_factory=lambda: new_id("skill_candidate"))
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", require_non_empty(self.candidate_id, "candidate_id")
        )
        object.__setattr__(self, "source_gradient_ids", tuple(self.source_gradient_ids))
        object.__setattr__(
            self, "source_trajectory_ids", tuple(self.source_trajectory_ids)
        )
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.evolution_type is SkillEvolutionType.SEED:
            raise ValueError("evolution candidates cannot have type SEED")
        if self.skill.evolution_type is not self.evolution_type:
            raise ValueError("candidate and skill evolution types must agree")


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryPPOScore(SchemaMixin):
    """Per-trajectory terms used by the non-parametric PPO objective."""

    trajectory_id: str
    advantage: float
    old_mean_logprob: float
    new_mean_logprob: float
    importance_ratio: float
    unclipped_objective: float
    clipped_objective: float
    target_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_id",
            require_non_empty(self.trajectory_id, "trajectory_id"),
        )
        numeric = (
            self.advantage,
            self.old_mean_logprob,
            self.new_mean_logprob,
            self.importance_ratio,
            self.unclipped_objective,
            self.clipped_objective,
        )
        if not all(isfinite(float(value)) for value in numeric):
            raise ValueError("PPO score values must be finite")
        if self.importance_ratio <= 0:
            raise ValueError("importance_ratio must be positive")
        if self.target_token_count < 1:
            raise ValueError("target_token_count must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOGateRecord(SchemaMixin):
    """Auditable accept/reject decision for one candidate skill."""

    candidate_id: str
    parent_skill_ref: str
    scorer_model: str
    epsilon: float
    acceptance_margin: float
    candidate_objective: float
    accepted: bool
    trajectory_scores: tuple[TrajectoryPPOScore, ...]
    gate_id: str = field(default_factory=lambda: new_id("ppo_gate"))
    reason: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "parent_skill_ref",
            "scorer_model",
            "gate_id",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "trajectory_scores", tuple(self.trajectory_scores))
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("PPO epsilon must be in [0, 1)")
        if not isfinite(float(self.acceptance_margin)) or not isfinite(
            float(self.candidate_objective)
        ):
            raise ValueError("PPO objective and margin must be finite")
        if not self.trajectory_scores:
            raise ValueError("PPO gate requires at least one trajectory score")
