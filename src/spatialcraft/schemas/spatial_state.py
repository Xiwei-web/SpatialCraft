"""State representation for multi-step spatial reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

from ._base import (
    SchemaMixin,
    new_id,
    normalize_datetime,
    require_non_empty,
    require_probability,
    utc_now,
)


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class EvidenceKind(str, Enum):
    VISUAL = "visual"
    GEOMETRIC = "geometric"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    POSE = "pose"
    SCALE = "scale"
    MOTION = "motion"
    OCR = "ocr"
    SCENE_GRAPH = "scene_graph"
    OTHER = "other"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationMessage(SchemaMixin):
    """Compact provider-neutral conversation message."""

    role: MessageRole
    content: str
    message_id: str = field(default_factory=lambda: new_id("message"))
    tool_call_id: str | None = None
    artifact_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", require_non_empty(self.content, "content"))
        object.__setattr__(
            self, "message_id", require_non_empty(self.message_id, "message_id")
        )
        object.__setattr__(self, "artifact_ids", tuple(self.artifact_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialEvidence(SchemaMixin):
    """A concise claim grounded in one tool result and optional artifacts."""

    kind: EvidenceKind
    summary: str
    source_tool_result_id: str
    evidence_id: str = field(default_factory=lambda: new_id("evidence"))
    artifact_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    frame_id: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("summary", "source_tool_result_id", "evidence_id"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(self, "artifact_ids", tuple(self.artifact_ids))
        object.__setattr__(self, "entity_ids", tuple(self.entity_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                require_probability(self.confidence, "confidence"),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveSkillRef(SchemaMixin):
    """The skill active for the current temporal abstraction window."""

    skill_id: str
    version: int
    activated_at_step: int
    age_steps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "skill_id", require_non_empty(self.skill_id, "skill_id")
        )
        if self.version < 1:
            raise ValueError("skill version must be at least 1")
        if self.activated_at_step < 0 or self.age_steps < 0:
            raise ValueError("skill step counters cannot be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedExperienceRef(SchemaMixin):
    """One retrieved experience and its task-specific rewritten form."""

    experience_id: str
    version: int
    retrieval_score: float
    original_text: str
    contextualized_text: str | None = None
    matched_subtasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experience_id",
            require_non_empty(self.experience_id, "experience_id"),
        )
        object.__setattr__(
            self,
            "original_text",
            require_non_empty(self.original_text, "original_text"),
        )
        object.__setattr__(self, "matched_subtasks", tuple(self.matched_subtasks))
        if self.version < 1:
            raise ValueError("experience version must be at least 1")
        if not -1.0 <= self.retrieval_score <= 1.0:
            raise ValueError("retrieval_score must be a cosine score in [-1, 1]")

    @property
    def prompt_text(self) -> str:
        return self.contextualized_text or self.original_text


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialState(SchemaMixin):
    """Immutable state snapshot immediately before or after an agent action."""

    task_id: str
    step_index: int
    state_id: str = field(default_factory=lambda: new_id("state"))
    messages: tuple[ConversationMessage, ...] = ()
    tool_result_ids: tuple[str, ...] = ()
    evidence: tuple[SpatialEvidence, ...] = ()
    active_skill: ActiveSkillRef | None = None
    retrieved_experiences: tuple[RetrievedExperienceRef, ...] = ()
    token_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", require_non_empty(self.task_id, "task_id"))
        object.__setattr__(
            self, "state_id", require_non_empty(self.state_id, "state_id")
        )
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tool_result_ids", tuple(self.tool_result_ids))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(
            self, "retrieved_experiences", tuple(self.retrieved_experiences)
        )
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.step_index < 0 or self.token_count < 0:
            raise ValueError("step_index and token_count cannot be negative")
        if self.active_skill and self.active_skill.age_steps > self.step_index:
            raise ValueError("active skill age cannot exceed the current step index")

        for label, values in (
            ("message_id", [message.message_id for message in self.messages]),
            ("tool_result_id", list(self.tool_result_ids)),
            ("evidence_id", [item.evidence_id for item in self.evidence]),
            (
                "experience_id",
                [item.experience_id for item in self.retrieved_experiences],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"state contains duplicate {label} values")

    def next_step(self, **changes: Any) -> SpatialState:
        """Create the following state while preserving immutability."""

        if "step_index" in changes:
            raise ValueError("next_step manages step_index automatically")
        return replace(
            self,
            state_id=new_id("state"),
            step_index=self.step_index + 1,
            created_at=utc_now(),
            **changes,
        )
