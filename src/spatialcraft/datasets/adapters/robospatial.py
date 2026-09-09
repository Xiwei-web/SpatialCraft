"""RoboSpatial-Home benchmark adapter."""

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
    stable_task_id,
)

_FILES = {
    "context": "context-00000-of-00001.parquet",
    "configuration": "configuration-00000-of-00001.parquet",
    "compatibility": "compatibility-00000-of-00001.parquet",
}


class RoboSpatialAdapter(RecordDatasetAdapter):
    """Normalize context-pointing and yes/no RoboSpatial tasks."""

    dataset_name: ClassVar[str] = "robospatial"
    available_splits: ClassVar[tuple[str, ...]] = (
        "test",
        "context",
        "configuration",
        "compatibility",
    )
    default_split: ClassVar[str] = "test"

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.images = ImageMaterializer(self.root / ".spatialcraft" / "media")

    def iter_records(self, split: str) -> Iterator[Mapping[str, Any]]:
        categories = tuple(_FILES) if split == "test" else (split,)
        for category in categories:
            path = self.root / "data" / _FILES[category]
            for category_index, record in enumerate(iter_parquet_records(path)):
                enriched = dict(record)
                enriched["_category"] = category
                enriched["_category_index"] = category_index
                yield enriched

    def normalize_record(
        self, record: Mapping[str, Any], *, index: int, split: str
    ) -> TaskSample:
        category = str(record.get("_category") or record.get("category") or "").lower()
        if category not in _FILES:
            raise DatasetFormatError(f"unknown RoboSpatial category: {category!r}")
        category_index = int(record.get("_category_index", index))
        source_id = f"{category}:{category_index}"
        rgb = self.images.from_huggingface_image(
            record.get("img"),
            dataset=self.dataset_name,
            source_id=source_id,
            index=0,
            source_root=self.root,
            metadata={"role": "rgb"},
        )
        auxiliary: dict[str, Any] = {}
        depth_value = record.get("depth_image")
        if depth_value is not None:
            depth = self.images.from_huggingface_image(
                depth_value,
                dataset=self.dataset_name,
                source_id=f"{source_id}:depth",
                index=0,
                source_root=self.root,
                metadata={"role": "depth"},
            )
            auxiliary["depth_uri"] = depth.uri
            auxiliary["depth_sha256"] = depth.sha256
        mask_value = record.get("mask")
        if mask_value is not None:
            mask = self.images.from_huggingface_image(
                mask_value,
                dataset=self.dataset_name,
                source_id=f"{source_id}:mask",
                index=0,
                source_root=self.root,
                metadata={"role": "target_mask"},
            )
            auxiliary["mask_uri"] = mask.uri
            auxiliary["mask_sha256"] = mask.sha256
        answer_type = (
            AnswerType.POINTING if category == "context" else AnswerType.BOOLEAN
        )
        return TaskSample(
            dataset=self.dataset_name,
            task_id=stable_task_id(self.dataset_name, category, source_id),
            source_id=source_id,
            split=self.canonical_task_split(split),
            question=normalize_text(record.get("question"), field_name="question"),
            images=(rgb,),
            answer_type=answer_type,
            reference_answer=normalize_text(record.get("answer"), field_name="answer"),
            metadata=make_task_metadata(
                source_split=split,
                source_fields={
                    "question_type": category,
                    "source_category": category,
                    "source_category_index": category_index,
                    "pointing_pass_threshold": 0.5,
                    **auxiliary,
                },
            ),
        )


__all__ = ["RoboSpatialAdapter"]
