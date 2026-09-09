"""SAM3 text/box-prompt segmentation adapter and tool."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import array_schema, object_schema
from .common import (
    LazyResource,
    SpatialToolPaths,
    add_python_path,
    png_bytes,
    require_file,
)

SegmentationPredictor = Callable[
    [str, str | None, Sequence[Sequence[float]], Sequence[bool], float],
    Mapping[str, Any],
]


class SAM3Adapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: SegmentationPredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        add_python_path(self.paths.sam3_repo)
        checkpoint = require_file(self.paths.sam3_checkpoint, "SAM3 checkpoint")
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.device,
        )
        return Sam3Processor(model, device=self.device)

    def predict(
        self,
        image_uri: str,
        prompt: str | None,
        boxes: Sequence[Sequence[float]],
        box_labels: Sequence[bool],
        threshold: float,
    ) -> Mapping[str, Any]:
        if self.predictor is not None:
            return self.predictor(image_uri, prompt, boxes, box_labels, threshold)
        import torch
        from PIL import Image

        processor = self._runtime.get()
        processor.confidence_threshold = threshold
        with Image.open(image_uri) as source:
            image = source.convert("RGB")
        width, height = image.size
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ),
        ):
            state = processor.set_image(image)
            if prompt:
                state = processor.set_text_prompt(prompt, state)
            for box, label in zip(boxes, box_labels, strict=True):
                x1, y1, x2, y2 = (float(value) for value in box)
                normalized = [
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
                state = processor.add_geometric_prompt(normalized, bool(label), state)
        masks = state["masks"].detach().cpu().numpy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        return {
            "masks": masks,
            "boxes": state["boxes"].detach().float().cpu().numpy(),
            "scores": state["scores"].detach().float().cpu().numpy(),
        }


class SAM3Tool(SpatialTool):
    spec = ToolSpec(
        name="segment",
        description="Segment objects with SAM3 using a text prompt and/or pixel-space boxes.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "prompt": {"type": "string", "minLength": 1},
                "boxes": array_schema(
                    array_schema({"type": "number"}, min_items=4, max_items=4)
                ),
                "box_labels": array_schema({"type": "boolean"}),
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            },
            required=("image_uri",),
        ),
        metadata={"model": "SAM3"},
    )

    def __init__(self, adapter: SAM3Adapter | None = None) -> None:
        self.adapter = adapter or SAM3Adapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import numpy as np
        from PIL import Image

        prompt = arguments.get("prompt")
        boxes = tuple(arguments.get("boxes") or ())
        labels = tuple(arguments.get("box_labels") or (True for _ in boxes))
        if not prompt and not boxes:
            raise ValueError("SAM3 requires a text prompt or at least one box")
        if len(labels) != len(boxes):
            raise ValueError("box_labels must align with boxes")
        image_uri = str(arguments["image_uri"])
        with Image.open(image_uri) as image:
            width, height = image.size
        prediction = self.adapter.predict(
            image_uri,
            str(prompt) if prompt else None,
            boxes,
            labels,
            float(arguments.get("threshold", 0.5)),
        )
        masks = np.asarray(prediction["masks"], dtype=bool)
        if masks.ndim == 2:
            masks = masks[None]
        scores = np.asarray(prediction.get("scores", np.ones(len(masks))), dtype=float)
        frame = image_pixel_frame(Path(image_uri).stem, width, height)
        artifacts = tuple(
            ArtifactPayload(
                artifact_type=ArtifactType.MASK,
                data=png_bytes(mask.astype(np.uint8) * 255, mode="L"),
                suffix=".png",
                mime_type="image/png",
                shape=(height, width),
                dtype="uint8",
                frame_id=frame.frame_id,
                metadata={"mask_index": index, "score": float(scores[index])},
            )
            for index, mask in enumerate(masks)
        )
        output = {
            "mask_count": len(masks),
            "scores": scores.tolist(),
            "boxes": np.asarray(prediction.get("boxes", ())).tolist(),
            "frame_id": frame.frame_id,
        }
        return ToolExecution(
            text=f"SAM3 produced {len(masks)} mask(s).",
            structured_output=output,
            artifacts=artifacts,
            coordinate_frames=(frame,),
            confidence=float(scores.mean()) if len(scores) else 0.0,
            unit="pixel",
            metadata={"backend_model": "SAM3"},
        )


__all__ = ["SAM3Adapter", "SAM3Tool"]
