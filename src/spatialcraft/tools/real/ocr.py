"""EasyOCR adapter and spatial text-recognition tool."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..builtin.draw import render_annotations
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import array_schema, object_schema
from .common import LazyResource, SpatialToolPaths, add_python_path

OCRPredictor = Callable[[str, Sequence[str]], list[dict[str, Any]]]


class EasyOCRAdapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        gpu: bool = True,
        predictor: OCRPredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.gpu = gpu
        self.predictor = predictor
        self._readers: dict[tuple[str, ...], LazyResource] = {}

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or any(
            item.loaded for item in self._readers.values()
        )

    def _reader(self, languages: tuple[str, ...]):
        if languages not in self._readers:
            self._readers[languages] = LazyResource(lambda: self._load(languages))
        return self._readers[languages].get()

    def _load(self, languages: tuple[str, ...]):
        add_python_path(self.paths.easyocr_repo)
        import easyocr

        return easyocr.Reader(
            list(languages),
            gpu=self.gpu,
            model_storage_directory=str(self.paths.easyocr_models),
            download_enabled=False,
        )

    def predict(self, image_uri: str, languages: Sequence[str]) -> list[dict[str, Any]]:
        if self.predictor is not None:
            return self.predictor(image_uri, languages)
        values = self._reader(tuple(languages)).readtext(image_uri)
        return [
            {
                "polygon": [[float(x), float(y)] for x, y in polygon],
                "text": str(text),
                "confidence": float(confidence),
            }
            for polygon, text, confidence in values
        ]


class OCRTool(SpatialTool):
    spec = ToolSpec(
        name="ocr",
        description="Recognize text and return pixel-space polygons using EasyOCR.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "languages": array_schema(
                    {"type": "string", "minLength": 2}, min_items=1, max_items=8
                ),
                "min_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            required=("image_uri",),
        ),
        metadata={"model": "EasyOCR"},
    )

    def __init__(self, adapter: EasyOCRAdapter | None = None) -> None:
        self.adapter = adapter or EasyOCRAdapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        from PIL import Image

        image_uri = str(arguments["image_uri"])
        with Image.open(image_uri) as image:
            width, height = image.size
        values = self.adapter.predict(
            image_uri, tuple(arguments.get("languages") or ("en",))
        )
        threshold = float(arguments.get("min_confidence", 0.0))
        values = [item for item in values if float(item["confidence"]) >= threshold]
        boxes = []
        for item in values:
            xs = [point[0] for point in item["polygon"]]
            ys = [point[1] for point in item["polygon"]]
            boxes.append(
                {
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "label": item["text"],
                    "color": "yellow",
                }
            )
        overlay, _, _ = render_annotations(image_uri, boxes=boxes)
        frame = image_pixel_frame(Path(image_uri).stem, width, height)
        output = {"regions": values, "frame_id": frame.frame_id}
        confidence = (
            sum(float(item["confidence"]) for item in values) / len(values)
            if values
            else 0.0
        )
        return ToolExecution(
            text="Recognized text: "
            + (" | ".join(item["text"] for item in values) or "none"),
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.TEXT,
                    text="\n".join(item["text"] for item in values),
                    suffix=".txt",
                    mime_type="text/plain",
                    frame_id=frame.frame_id,
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.JSON,
                    json_value=output,
                    suffix=".json",
                    mime_type="application/json",
                    frame_id=frame.frame_id,
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=overlay,
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width, 3),
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={"role": "ocr_overlay"},
                ),
            ),
            coordinate_frames=(frame,),
            confidence=confidence,
            unit="pixel",
            metadata={"backend_model": "EasyOCR"},
        )


__all__ = ["EasyOCRAdapter", "OCRTool"]
