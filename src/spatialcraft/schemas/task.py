"""Canonical task inputs shared by all SpatialCraft datasets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from typing import Any

from ._base import SchemaMixin, new_id, require_non_empty


class TaskSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNSPECIFIED = "unspecified"


class AnswerType(str, Enum):
    FREE_FORM = "free_form"
    MULTIPLE_CHOICE = "multiple_choice"
    SPATIAL_RELATION = "spatial_relation"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    POINTING = "pointing"
    STRUCTURED = "structured"


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInput(SchemaMixin):
    """A task image referenced by URI rather than embedded bytes."""

    uri: str
    image_id: str = field(default_factory=lambda: new_id("img"))
    media_type: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", require_non_empty(self.uri, "uri"))
        object.__setattr__(
            self, "image_id", require_non_empty(self.image_id, "image_id")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSample(SchemaMixin):
    """Dataset-independent representation of one spatial reasoning task."""

    dataset: str
    question: str
    task_id: str = field(default_factory=lambda: new_id("task"))
    split: TaskSplit = TaskSplit.UNSPECIFIED
    images: tuple[ImageInput, ...] = ()
    answer_type: AnswerType = AnswerType.FREE_FORM
    reference_answer: Any | None = None
    choices: tuple[str, ...] = ()
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", require_non_empty(self.dataset, "dataset"))
        object.__setattr__(
            self, "question", require_non_empty(self.question, "question")
        )
        object.__setattr__(self, "task_id", require_non_empty(self.task_id, "task_id"))
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "choices", tuple(self.choices))
        object.__setattr__(self, "metadata", dict(self.metadata))

        image_ids = [image.image_id for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("images must have unique image_id values")
        if self.answer_type is AnswerType.MULTIPLE_CHOICE and len(self.choices) < 2:
            raise ValueError("multiple-choice tasks require at least two choices")
        if any(not choice.strip() for choice in self.choices):
            raise ValueError("choices cannot contain empty strings")

    @property
    def fingerprint(self) -> str:
        """Content fingerprint for reproducibility and duplicate detection."""

        payload = "\n".join(
            [
                self.dataset,
                self.source_id or "",
                self.question,
                *(image.sha256 or image.uri for image in self.images),
            ]
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def without_reference_answer(self) -> TaskSample:
        """Return the execution-safe task view used to prevent answer leakage."""

        return replace(
            self,
            reference_answer=None,
            images=tuple(replace(image, metadata={}) for image in self.images),
            metadata={
                key: value
                for key, value in self.metadata.items()
                if key
                in {
                    "question_type",
                    "source_answer_type",
                    "choice_labels",
                    "visual_indices",
                    "experiment_split",
                    "source_split",
                }
            },
        )
