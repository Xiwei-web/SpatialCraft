"""ViewSpatial-Bench adapter with lazy archive-backed image materialization."""

from __future__ import annotations

import json
import threading
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from spatialcraft.schemas import AnswerType, TaskSample

from ..base import DatasetConfigurationError, DatasetFormatError, RecordDatasetAdapter
from ..normalizer import (
    ImageMaterializer,
    make_task_metadata,
    normalize_text,
    parse_choice_block,
    split_labeled_answer,
    stable_task_id,
)


class ViewSpatialAdapter(RecordDatasetAdapter):
    """Normalize ViewSpatial JSON rows and resolve images from release ZIP files."""

    dataset_name: ClassVar[str] = "viewspatial"
    available_splits: ClassVar[tuple[str, ...]] = ("test",)
    default_split: ClassVar[str] = "test"

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.source_path = self.root / "ViewSpatial-Bench.json"
        self.images = ImageMaterializer(self.root / ".spatialcraft" / "media")
        self._archives: dict[str, zipfile.ZipFile] = {}
        self._archive_lock = threading.Lock()

    def iter_records(self, split: str) -> Iterator[Mapping[str, Any]]:
        if not self.source_path.is_file():
            raise DatasetConfigurationError(
                f"ViewSpatial JSON not found: {self.source_path}"
            )
        try:
            records = json.loads(self.source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetFormatError(f"cannot read {self.source_path}: {exc}") from exc
        if not isinstance(records, list):
            raise DatasetFormatError("ViewSpatial JSON root must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise DatasetFormatError("ViewSpatial records must be JSON objects")
            yield record

    @staticmethod
    def _member_name(source_path: str) -> str:
        parts = PurePosixPath(source_path).parts
        if parts and parts[0] == "ViewSpatial-Bench":
            parts = parts[1:]
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise DatasetFormatError(f"unsafe ViewSpatial image path: {source_path!r}")
        return PurePosixPath(*parts).as_posix()

    def _read_image(self, source_path: str) -> bytes:
        member = self._member_name(source_path)
        archive_stem = member.split("/", maxsplit=1)[0]
        archive_path = self.root / f"{archive_stem}.zip"
        if not archive_path.is_file():
            raise DatasetConfigurationError(
                f"ViewSpatial archive not found: {archive_path}"
            )
        with self._archive_lock:
            archive = self._archives.get(archive_stem)
            if archive is None:
                archive = zipfile.ZipFile(archive_path)
                self._archives[archive_stem] = archive
            try:
                return archive.read(member)
            except KeyError as exc:
                raise DatasetFormatError(
                    f"image {member!r} is missing from {archive_path.name}"
                ) from exc

    def close(self) -> None:
        with self._archive_lock:
            for archive in self._archives.values():
                archive.close()
            self._archives.clear()

    def normalize_record(
        self, record: Mapping[str, Any], *, index: int, split: str
    ) -> TaskSample:
        source_id = f"row-{index}"
        labels, choices = parse_choice_block(record.get("choices"))
        answer_label, answer_text = split_labeled_answer(record.get("answer"))
        if answer_label is None:
            normalized_choices = [choice.casefold() for choice in choices]
            try:
                answer_label = labels[normalized_choices.index(answer_text.casefold())]
            except ValueError as exc:
                raise DatasetFormatError(
                    f"ViewSpatial row {index} answer is not one of its choices"
                ) from exc
        if answer_label not in labels:
            raise DatasetFormatError(
                f"ViewSpatial row {index} answer label {answer_label!r} is invalid"
            )
        raw_paths = record.get("image_path")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise DatasetFormatError(f"ViewSpatial row {index} has no image paths")
        source_paths = tuple(
            normalize_text(path, field_name="image_path") for path in raw_paths
        )
        images = tuple(
            self.images.from_bytes(
                self._read_image(path),
                dataset=self.dataset_name,
                source_id=source_id,
                index=image_index,
                suggested_path=path,
                metadata={"source_path": path, "view_index": image_index},
            )
            for image_index, path in enumerate(source_paths)
        )
        question_type = normalize_text(
            record.get("question_type"), field_name="question_type"
        )
        return TaskSample(
            dataset=self.dataset_name,
            task_id=stable_task_id(self.dataset_name, split, source_id),
            source_id=source_id,
            split=self.canonical_task_split(split),
            question=normalize_text(record.get("question"), field_name="question"),
            images=images,
            answer_type=AnswerType.MULTIPLE_CHOICE,
            reference_answer=answer_label,
            choices=choices,
            metadata=make_task_metadata(
                source_split=split,
                source_fields={
                    "question_type": question_type,
                    "choice_labels": labels,
                    "reference_answer_text": answer_text,
                    "source_image_paths": source_paths,
                },
            ),
        )


__all__ = ["ViewSpatialAdapter"]
