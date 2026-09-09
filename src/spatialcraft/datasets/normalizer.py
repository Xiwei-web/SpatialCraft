"""Shared text, answer, choice, identifier, and image normalization."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from spatialcraft.schemas import AnswerType, ImageInput
from spatialcraft.storage.atomic_io import atomic_write_bytes

from .base import DatasetFormatError

_CHOICE_MARKER = re.compile(r"(?<!\w)([A-Z])[.):]\s+")
_TRAILING_INSTRUCTION = re.compile(
    r"\s*(?:Please\s+)?(?:answer|respond)\b.*$", re.IGNORECASE | re.DOTALL
)
_LABEL_PREFIX = re.compile(r"^\s*([A-Z])[.)]\s*(.*)$", re.DOTALL)


def normalize_text(value: Any, *, field_name: str = "text") -> str:
    """Return Unicode-normalized text with stable whitespace."""

    if value is None:
        raise DatasetFormatError(f"{field_name} is missing")
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise DatasetFormatError(f"{field_name} cannot be empty")
    return text


def stable_task_id(dataset: str, split: str, source_id: str) -> str:
    payload = f"{dataset}\0{split}\0{source_id}".encode()
    return f"task_{dataset}_{sha256(payload).hexdigest()[:24]}"


def choice_label(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("choice index must be between 0 and 25")
    return chr(ord("A") + index)


def split_labeled_answer(value: Any) -> tuple[str | None, str]:
    """Split ``A. answer`` into a label and answer text."""

    text = normalize_text(value, field_name="answer")
    match = _LABEL_PREFIX.match(text)
    if not match:
        return None, text
    return match.group(1), match.group(2).strip()


def parse_choice_block(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse newline or inline ``A. ... B. ...`` choices."""

    text = normalize_text(value, field_name="choices")
    matches = list(_CHOICE_MARKER.finditer(text))
    if len(matches) < 2:
        raise DatasetFormatError("choice block contains fewer than two labeled choices")

    # Select the first monotonically increasing A/B/C/... marker sequence.
    # This avoids treating point-label answers such as ``A. A. B. B.`` as
    # eight separate options.
    selected = []
    expected = "A"
    for match in matches:
        if match.group(1) == expected:
            selected.append(match)
            expected = chr(ord(expected) + 1)
    if len(selected) < 2:
        raise DatasetFormatError("choice labels must form an A/B/... sequence")

    labels: list[str] = []
    choices: list[str] = []
    for index, match in enumerate(selected):
        start = match.end()
        end = selected[index + 1].start() if index + 1 < len(selected) else len(text)
        item = _TRAILING_INSTRUCTION.sub("", text[start:end]).strip()
        label = match.group(1)
        # Some visual point questions intentionally encode a label-only option.
        item = item or label
        labels.append(label)
        choices.append(item)
    return tuple(labels), tuple(choices)


def parse_inline_choices(
    prompt: Any,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Separate an ERQA-style question from its embedded choice block."""

    text = normalize_text(prompt, field_name="question")
    marker = re.search(r"\bChoices\s*:\s*", text, re.IGNORECASE)
    if marker is not None:
        question = text[: marker.start()].strip()
        choice_block = text[marker.end() :]
    else:
        first_choice = re.search(r"(?<!\w)A[.):]\s+", text)
        if first_choice is None:
            return text, (), ()
        question = text[: first_choice.start()].strip()
        choice_block = text[first_choice.start() :]
    labels, choices = parse_choice_block(choice_block)
    return normalize_text(question, field_name="question"), labels, choices


def parse_braced_choices(prompt: Any) -> tuple[str, tuple[str, ...]]:
    """Extract an Omni3D-style trailing ``Options: {x, y}`` block."""

    text = normalize_text(prompt, field_name="question")
    match = re.search(r"\s*Options\s*:\s*\{([^{}]+)\}\s*$", text, re.IGNORECASE)
    if match is None:
        return text, ()
    choices = tuple(item.strip() for item in match.group(1).split(","))
    if len(choices) < 2 or any(not item for item in choices):
        raise DatasetFormatError("invalid braced choice block")
    return text[: match.start()].strip(), choices


def infer_answer_type(
    answer: Any,
    *,
    choices: Sequence[str] = (),
    source_type: str | None = None,
) -> AnswerType:
    if choices:
        return AnswerType.MULTIPLE_CHOICE
    source = (source_type or "").strip().lower()
    value = str(answer).strip().lower()
    if source in {"int", "integer", "float", "number", "numeric"}:
        return AnswerType.NUMERIC
    if value in {"yes", "no", "true", "false"}:
        return AnswerType.BOOLEAN
    if source in {"point", "pointing", "context"}:
        return AnswerType.POINTING
    return AnswerType.FREE_FORM


def _media_type(data: bytes, suggested_path: str | None) -> tuple[str, str]:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", ".gif"
    suffix = Path(suggested_path or "").suffix.lower()
    known = {
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if suffix in known:
        return known[suffix], suffix
    raise DatasetFormatError("cannot determine embedded image media type")


class ImageMaterializer:
    """Persist embedded images once and return portable image references."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def from_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        dataset: str,
        source_id: str,
        index: int,
        suggested_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ImageInput:
        payload = bytes(data)
        if not payload:
            raise DatasetFormatError("embedded image bytes cannot be empty")
        digest = sha256(payload).hexdigest()
        media_type, suffix = _media_type(payload, suggested_path)
        target = self.root / digest[:2] / f"{digest}{suffix}"
        if not target.exists():
            try:
                atomic_write_bytes(target, payload, overwrite=False)
            except FileExistsError:
                pass
        image_metadata = dict(metadata or {})
        if suggested_path:
            image_metadata.setdefault("source_path", suggested_path)
        return ImageInput(
            uri=str(target),
            image_id=f"{dataset}_{source_id}_{index}_{digest[:12]}",
            media_type=media_type,
            sha256=digest,
            metadata=image_metadata,
        )

    def from_huggingface_image(
        self,
        value: Any,
        *,
        dataset: str,
        source_id: str,
        index: int,
        source_root: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ImageInput:
        if not isinstance(value, Mapping):
            raise DatasetFormatError("Hugging Face image must be a mapping")
        data = value.get("bytes")
        source_path = value.get("path")
        if data is not None:
            return self.from_bytes(
                data,
                dataset=dataset,
                source_id=source_id,
                index=index,
                suggested_path=source_path,
                metadata=metadata,
            )
        if not source_path:
            raise DatasetFormatError("image contains neither bytes nor path")
        path = Path(source_path)
        if not path.is_absolute() and source_root is not None:
            path = Path(source_root) / path
        if not path.is_file():
            raise DatasetFormatError(f"image path not found: {path}")
        return self.from_bytes(
            path.read_bytes(),
            dataset=dataset,
            source_id=source_id,
            index=index,
            suggested_path=str(path),
            metadata=metadata,
        )


def make_task_metadata(
    *, source_split: str, source_fields: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "source_split": source_split,
        **dict(source_fields or {}),
    }


__all__ = [
    "ImageMaterializer",
    "choice_label",
    "infer_answer_type",
    "make_task_metadata",
    "normalize_text",
    "parse_braced_choices",
    "parse_choice_block",
    "parse_inline_choices",
    "split_labeled_answer",
    "stable_task_id",
]
