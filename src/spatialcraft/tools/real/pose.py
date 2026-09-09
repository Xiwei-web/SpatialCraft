"""Orient Anything object-pose adapter and tool."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..schema_builder import array_schema, object_schema
from .common import LazyResource, SpatialToolPaths, add_python_path, require_file

PosePredictor = Callable[[str, Sequence[float] | None], Mapping[str, float]]


class OrientAnythingAdapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: PosePredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        import torch
        from transformers import AutoImageProcessor

        add_python_path(self.paths.orient_repo)
        checkpoint = require_file(
            self.paths.orient_checkpoint, "Orient Anything checkpoint"
        )
        if not self.paths.orient_backbone.is_dir():
            raise FileNotFoundError(
                f"Orient Anything backbone not found: {self.paths.orient_backbone}"
            )
        paths_module = importlib.import_module("paths")
        paths_module.DINO_SMALL = str(self.paths.orient_backbone)
        vision = importlib.import_module("vision_tower")
        model = vision.DINOv2_MLP(
            dino_mode="small",
            in_dim=384,
            out_dim=722,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = model.to(self.device).eval()
        processor = AutoImageProcessor.from_pretrained(str(self.paths.orient_backbone))
        return model, processor

    def predict(
        self, image_uri: str, bbox: Sequence[float] | None
    ) -> Mapping[str, float]:
        if self.predictor is not None:
            return self.predictor(image_uri, bbox)
        from PIL import Image

        with Image.open(image_uri) as source:
            image = source.convert("RGB")
        if bbox is not None:
            image = image.crop(tuple(float(value) for value in bbox))
        import torch

        model, processor = self._runtime.get()
        # The downloaded cropsmallEx03 is the 722-output checkpoint. Match
        # upstream b62f5328505648be5fe9b24acb149a3df865365b/inference.py,
        # not the newer 902-output decoder (which uses a different checkpoint).
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            prediction = model(inputs)
        values = (
            prediction[:, :360].argmax(-1).item(),
            prediction[:, 360:540].argmax(-1).item() - 90,
            prediction[:, 540:720].argmax(-1).item() - 90,
            prediction[:, -2:].softmax(-1)[0, 0].item(),
        )
        return {
            "azimuth_deg": float(values[0]),
            "polar_deg": float(values[1]),
            "rotation_deg": float(values[2]),
            "confidence": float(values[3]),
        }


class PoseTool(SpatialTool):
    spec = ToolSpec(
        name="pose",
        description="Estimate object azimuth, polar angle, and in-plane rotation with Orient Anything.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "bbox": array_schema({"type": "number"}, min_items=4, max_items=4),
            },
            required=("image_uri",),
        ),
        metadata={"model": "Orient Anything"},
    )

    def __init__(self, adapter: OrientAnythingAdapter | None = None) -> None:
        self.adapter = adapter or OrientAnythingAdapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        output = dict(
            self.adapter.predict(str(arguments["image_uri"]), arguments.get("bbox"))
        )
        confidence = float(output.get("confidence", 0.0))
        frame = CoordinateFrame(
            frame_id=f"{Path(arguments['image_uri']).stem}:object-pose",
            unit="degree",
            convention="azimuth[0,360),polar[-90,90],rotation[-180,180]",
        )
        return ToolExecution(
            text=(
                f"Pose azimuth={output['azimuth_deg']:.1f}°, "
                f"polar={output['polar_deg']:.1f}°, "
                f"rotation={output['rotation_deg']:.1f}°."
            ),
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.POSE,
                    json_value=output,
                    suffix=".json",
                    mime_type="application/json",
                    frame_id=frame.frame_id,
                ),
            ),
            coordinate_frames=(frame,),
            confidence=confidence,
            unit="degree",
            metadata={"backend_model": "Orient Anything"},
        )


__all__ = ["OrientAnythingAdapter", "PoseTool"]
