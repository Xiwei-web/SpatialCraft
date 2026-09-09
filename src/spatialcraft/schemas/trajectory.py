"""Trajectory schemas connecting states, actions, tools, skills, and rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any

from ._base import SchemaMixin, new_id, normalize_datetime, require_non_empty, utc_now
from .action import ActionType, AgentAction
from .spatial_state import ActiveSkillRef, SpatialState
from .task import TaskSample
from .tool_result import ToolResult


class TrajectoryStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True, kw_only=True)
class LogProbTrace(SchemaMixin):
    """Token-level score for one fixed target action.

    The prompt tokens are deliberately excluded from ``token_logprobs`` so the
    same object can be used by the Skill-Pro-style PPO gate.
    """

    model_alias: str
    target_text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    prompt_token_count: int = 0
    tokenizer_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_alias", require_non_empty(self.model_alias, "model_alias")
        )
        object.__setattr__(
            self, "target_text", require_non_empty(self.target_text, "target_text")
        )
        object.__setattr__(self, "token_ids", tuple(int(x) for x in self.token_ids))
        object.__setattr__(
            self, "token_logprobs", tuple(float(x) for x in self.token_logprobs)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.token_ids:
            raise ValueError("a log-prob trace requires at least one target token")
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("token_ids and token_logprobs must have equal length")
        if not all(isfinite(value) for value in self.token_logprobs):
            raise ValueError("token_logprobs must be finite")
        if self.prompt_token_count < 0:
            raise ValueError("prompt_token_count cannot be negative")

    @property
    def target_token_count(self) -> int:
        return len(self.token_ids)

    @property
    def sum_logprob(self) -> float:
        return sum(self.token_logprobs)

    @property
    def mean_logprob(self) -> float:
        return self.sum_logprob / self.target_token_count


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifierOutcome(SchemaMixin):
    """Post-rollout evaluation, kept separate from the execution context."""

    verifier_name: str
    score: float
    is_correct: bool | None = None
    analysis: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verifier_name",
            require_non_empty(self.verifier_name, "verifier_name"),
        )
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "details", dict(self.details))
        if not isfinite(self.score):
            raise ValueError("verifier score must be finite")


@dataclass(frozen=True, slots=True, kw_only=True)
class Transition(SchemaMixin):
    """One environment transition with exact knowledge attribution."""

    step_index: int
    state_before: SpatialState
    action: AgentAction
    transition_id: str = field(default_factory=lambda: new_id("transition"))
    tool_results: tuple[ToolResult, ...] = ()
    state_after: SpatialState | None = None
    active_skill: ActiveSkillRef | None = None
    used_experience_ids: tuple[str, ...] = ()
    policy_logprobs: LogProbTrace | None = None
    reward: float | None = None
    done: bool = False
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transition_id",
            require_non_empty(self.transition_id, "transition_id"),
        )
        object.__setattr__(self, "tool_results", tuple(self.tool_results))
        object.__setattr__(self, "used_experience_ids", tuple(self.used_experience_ids))
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        if self.state_before.step_index != self.step_index:
            raise ValueError("state_before.step_index must match transition step_index")
        if self.state_after is not None:
            if self.state_after.task_id != self.state_before.task_id:
                raise ValueError("a transition cannot change task_id")
            if self.state_after.step_index != self.step_index + 1:
                raise ValueError("state_after must represent the next step")
        if self.reward is not None and not isfinite(float(self.reward)):
            raise ValueError("transition reward must be finite")
        if len(self.used_experience_ids) != len(set(self.used_experience_ids)):
            raise ValueError("used_experience_ids cannot contain duplicates")

        expected_calls = {call.call_id for call in self.action.tool_calls}
        actual_calls = {result.tool_call_id for result in self.tool_results}
        if (
            self.action.action_type is ActionType.TOOL
            and expected_calls != actual_calls
        ):
            raise ValueError(
                "tool_results must contain exactly one result for every tool call"
            )
        if self.action.action_type is not ActionType.TOOL and self.tool_results:
            raise ValueError("non-tool actions cannot contain tool results")
        if len(actual_calls) != len(self.tool_results):
            raise ValueError("a transition cannot contain duplicate tool-call results")
        if self.action.action_type is ActionType.FINAL and not self.done:
            raise ValueError("a final-answer transition must set done=True")


@dataclass(frozen=True, slots=True, kw_only=True)
class Trajectory(SchemaMixin):
    """A complete, replayable rollout for one task sample."""

    task: TaskSample
    rollout_index: int
    executor_model: str
    knowledge_snapshot_id: str
    trajectory_id: str = field(default_factory=lambda: new_id("trajectory"))
    status: TrajectoryStatus = TrajectoryStatus.CREATED
    transitions: tuple[Transition, ...] = ()
    final_answer: str | None = None
    verifier: VerifierOutcome | None = None
    reward: float | None = None
    random_seed: int | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("executor_model", "knowledge_snapshot_id", "trajectory_id"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "started_at", normalize_datetime(self.started_at))
        if self.finished_at is not None:
            object.__setattr__(
                self, "finished_at", normalize_datetime(self.finished_at)
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.rollout_index < 0:
            raise ValueError("rollout_index cannot be negative")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")
        if self.reward is not None and not isfinite(float(self.reward)):
            raise ValueError("trajectory reward must be finite")

        expected_indices = list(range(len(self.transitions)))
        actual_indices = [transition.step_index for transition in self.transitions]
        if actual_indices != expected_indices:
            raise ValueError(
                "trajectory transition indices must be contiguous from zero"
            )
        if any(
            transition.state_before.task_id != self.task.task_id
            for transition in self.transitions
        ):
            raise ValueError("all trajectory states must reference the trajectory task")

        terminal_actions = [
            transition
            for transition in self.transitions
            if transition.action.action_type is ActionType.FINAL
        ]
        if len(terminal_actions) > 1:
            raise ValueError("a trajectory can contain at most one final answer")
        if terminal_actions and terminal_actions[0] is not self.transitions[-1]:
            raise ValueError("the final answer must be the last transition")

        action_answer = (
            terminal_actions[0].action.final_answer if terminal_actions else None
        )
        if action_answer is not None and self.final_answer is None:
            object.__setattr__(self, "final_answer", action_answer)
        elif action_answer is not None and self.final_answer != action_answer:
            raise ValueError("trajectory final_answer disagrees with its final action")

        if self.status is TrajectoryStatus.COMPLETED:
            if not terminal_actions:
                raise ValueError("completed trajectories require a final-answer action")
            if self.finished_at is None:
                raise ValueError("completed trajectories require finished_at")
        if self.status is TrajectoryStatus.FAILED and not self.failure_reason:
            raise ValueError("failed trajectories require failure_reason")

    @property
    def used_skill_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                transition.active_skill.skill_id
                for transition in self.transitions
                if transition.active_skill is not None
            )
        )

    @property
    def used_experience_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                experience_id
                for transition in self.transitions
                for experience_id in transition.used_experience_ids
            )
        )

    @property
    def total_tool_calls(self) -> int:
        return sum(len(transition.action.tool_calls) for transition in self.transitions)
