"""ERQA benchmark adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from spatialcraft.schemas import AnswerType, TaskSample

from ..base import DatasetFormatError, RecordDatasetAdapter, iter_parquet_records
from ..normalizer import (
    ImageMaterializer,
    make_task_metadata,
    normalize_text,
    parse_inline_choices,
    stable_task_id,
)


class ERQAAdapter(RecordDatasetAdapter):
    """Normalize the 400-row FlagEval ERQA release."""

    dataset_name: ClassVar[str] = "erqa"
    available_splits: ClassVar[tuple[str, ...]] = ("test",)
    default_split: ClassVar[str] = "test"

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.source_path = self.root / "data" / "test-00000-of-00001.parquet"
        self.images = ImageMaterializer(self.root / ".spatialcraft" / "media")

    def iter_records(self, split: str) -> Iterator[Mapping[str, Any]]:
        yield from iter_parquet_records(self.source_path)

    def normalize_record(
        self, record: Mapping[str, Any], *, index: int, split: str
    ) -> TaskSample:
        source_id = normalize_text(
            record.get("question_id") or f"row-{index}", field_name="question_id"
        )
        question, labels, choices = parse_inline_choices(record.get("question"))
        if not choices:
            raise DatasetFormatError(f"ERQA row {source_id} has no embedded choices")
        answer = normalize_text(record.get("answer"), field_name="answer").upper()
        if answer not in labels:
            raise DatasetFormatError(
                f"ERQA row {source_id} answer {answer!r} is not a choice label"
            )
        raw_images = record.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            raise DatasetFormatError(f"ERQA row {source_id} has no images")
        images = tuple(
            self.images.from_huggingface_image(
                value,
                dataset=self.dataset_name,
                source_id=source_id,
                index=image_index,
                metadata={"visual_index": image_index},
            )
            for image_index, value in enumerate(raw_images)
        )
        question_type = normalize_text(
            record.get("question_type"), field_name="question_type"
        )
        return TaskSample(
            dataset=self.dataset_name,
            task_id=stable_task_id(self.dataset_name, split, source_id),
            source_id=source_id,
            split=self.canonical_task_split(split),
            question=question,
            images=images,
            answer_type=AnswerType.MULTIPLE_CHOICE,
            reference_answer=answer,
            choices=choices,
            metadata=make_task_metadata(
                source_split=split,
                source_fields={
                    "question_type": question_type,
                    "choice_labels": labels,
                    "visual_indices": tuple(record.get("visual_indices") or ()),
                },
            ),
        )


__all__ = ["ERQAAdapter"]
