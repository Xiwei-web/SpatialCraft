"""SAT multi-view spatial reasoning benchmark adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from spatialcraft.schemas import AnswerType, TaskSample

from ..base import (
    DatasetConfigurationError,
    DatasetFormatError,
    RecordDatasetAdapter,
    iter_parquet_records,
)
from ..normalizer import (
    ImageMaterializer,
    choice_label,
    make_task_metadata,
    normalize_text,
    stable_task_id,
)

_FILES = {
    "train": "SAT_train.parquet",
    "static": "SAT_static.parquet",
    "validation": "SAT_val.parquet",
}


class SATAdapter(RecordDatasetAdapter):
    """Normalize SAT and prefer the SMA-compatible 300-row circular test split."""

    dataset_name: ClassVar[str] = "sat"
    available_splits: ClassVar[tuple[str, ...]] = (
        "train",
        "static",
        "validation",
        "test",
    )
    default_split: ClassVar[str] = "test"

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.images = ImageMaterializer(self.root / ".spatialcraft" / "media")

    def _source_path(self, split: str) -> Path:
        if split == "test":
            preferred = self.root / "SAT_test_circular_300.parquet"
            fallback = self.root / "SAT_test.parquet"
            if preferred.is_file():
                return preferred
            if fallback.is_file():
                return fallback
            raise DatasetConfigurationError(
                f"SAT test source not found under {self.root}"
            )
        return self.root / _FILES[split]

    def iter_records(self, split: str) -> Iterator[Mapping[str, Any]]:
        yield from iter_parquet_records(self._source_path(split))

    def normalize_record(
        self, record: Mapping[str, Any], *, index: int, split: str
    ) -> TaskSample:
        source_row_index = int(record.get("source_row_index", index))
        circular_shift = int(record.get("circular_shift", 0))
        source_id = f"{source_row_index}:shift-{circular_shift}"
        raw_choices = record.get("answers")
        if not isinstance(raw_choices, list) or len(raw_choices) < 2:
            raise DatasetFormatError(f"SAT row {source_id} has invalid answers")
        choices = tuple(
            normalize_text(choice, field_name="answer choice") for choice in raw_choices
        )
        reference = normalize_text(
            record.get("correct_answer"), field_name="correct_answer"
        )
        correct_index = record.get("correct_answer_index")
        if correct_index is None:
            normalized = [choice.casefold() for choice in choices]
            try:
                correct_index = normalized.index(reference.casefold())
            except ValueError as exc:
                raise DatasetFormatError(
                    f"SAT row {source_id} reference is not one of its choices"
                ) from exc
        correct_index = int(correct_index)
        if not 0 <= correct_index < len(choices):
            raise DatasetFormatError(f"SAT row {source_id} has invalid answer index")
        raw_images = record.get("image_bytes")
        if not isinstance(raw_images, list) or not raw_images:
            raise DatasetFormatError(f"SAT row {source_id} has no images")
        images = tuple(
            self.images.from_bytes(
                value,
                dataset=self.dataset_name,
                source_id=source_id,
                index=image_index,
                suggested_path=f"frame-{image_index}.jpg",
                metadata={"frame_index": image_index},
            )
            for image_index, value in enumerate(raw_images)
        )
        question_type = normalize_text(
            record.get("question_type"), field_name="question_type"
        )
        labels = tuple(choice_label(i) for i in range(len(choices)))
        return TaskSample(
            dataset=self.dataset_name,
            task_id=stable_task_id(self.dataset_name, split, source_id),
            source_id=source_id,
            split=self.canonical_task_split(split),
            question=normalize_text(record.get("question"), field_name="question"),
            images=images,
            answer_type=AnswerType.MULTIPLE_CHOICE,
            reference_answer=choice_label(correct_index),
            choices=choices,
            metadata=make_task_metadata(
                source_split=split,
                source_fields={
                    "question_type": question_type,
                    "choice_labels": labels,
                    "reference_answer_text": reference,
                    "correct_answer_index": correct_index,
                    "source_row_index": source_row_index,
                    "circular_shift": circular_shift,
                    "is_circular_expansion": "circular"
                    in self._source_path(split).name,
                },
            ),
        )


__all__ = ["SATAdapter"]
