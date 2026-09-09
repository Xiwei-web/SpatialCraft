"""Omni3D-Bench adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from spatialcraft.schemas import TaskSample

from ..base import RecordDatasetAdapter, iter_parquet_records
from ..normalizer import (
    ImageMaterializer,
    infer_answer_type,
    make_task_metadata,
    normalize_text,
    parse_braced_choices,
    stable_task_id,
)


class Omni3DAdapter(RecordDatasetAdapter):
    """Normalize the 501-question public Omni3D benchmark release."""

    dataset_name: ClassVar[str] = "omni3d"
    available_splits: ClassVar[tuple[str, ...]] = ("test",)
    default_split: ClassVar[str] = "test"

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.source_path = self.root / "data" / "train-00000-of-00001.parquet"
        self.images = ImageMaterializer(self.root / ".spatialcraft" / "media")

    def iter_records(self, split: str) -> Iterator[Mapping[str, Any]]:
        yield from iter_parquet_records(self.source_path)

    def normalize_record(
        self, record: Mapping[str, Any], *, index: int, split: str
    ) -> TaskSample:
        image_index = normalize_text(
            record.get("image_index"), field_name="image_index"
        )
        question_index = int(record.get("q_index", index))
        source_id = f"{image_index}:{question_index}"
        question, choices = parse_braced_choices(record.get("question"))
        reference = normalize_text(record.get("answer"), field_name="answer")
        source_answer_type = normalize_text(
            record.get("answer_type") or "str", field_name="answer_type"
        ).lower()
        image = self.images.from_huggingface_image(
            record.get("image"),
            dataset=self.dataset_name,
            source_id=source_id,
            index=0,
            source_root=self.root,
        )
        return TaskSample(
            dataset=self.dataset_name,
            task_id=stable_task_id(self.dataset_name, split, source_id),
            source_id=source_id,
            split=self.canonical_task_split(split),
            question=question,
            images=(image,),
            answer_type=infer_answer_type(
                reference, choices=choices, source_type=source_answer_type
            ),
            reference_answer=reference,
            choices=choices,
            metadata=make_task_metadata(
                source_split=split,
                source_fields={
                    "image_index": image_index,
                    "question_index": question_index,
                    "source_answer_type": source_answer_type,
                    "numeric_absolute_tolerance": 1e-3,
                    "numeric_relative_tolerance": 1e-2,
                },
            ),
        )


__all__ = ["Omni3DAdapter"]
