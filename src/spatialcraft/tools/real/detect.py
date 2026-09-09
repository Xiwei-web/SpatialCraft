"""GroundingDINO open-vocabulary detection adapter and tool."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..builtin.draw import render_annotations
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import array_schema, object_schema
from .common import LazyResource, SpatialToolPaths, add_python_path, require_file

DetectionPredictor = Callable[[str, Sequence[str], float, float], list[dict[str, Any]]]


def sanitize_detections(detections, width: int, height: int):
    """Intersect predicted boxes with our inclusive pixel frame and audit changes.

    Normalize detector output, not arbitrary agent arguments. Never reorder
    inverted coordinates or invent a box entirely outside the image.
    """
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    valid, audit = [], []
    for index, item in enumerate(detections):
        raw = [float(value) for value in item["bbox"]]
        if len(raw) != 4:
            raise ValueError("detector bbox must contain four coordinates")
        if not all(math.isfinite(value) for value in raw):
            audit.append(
                {
                    "index": index,
                    "raw_bbox": [str(v) for v in raw],
                    "reason": "nonfinite",
                    "operation": "drop",
                }
            )
            continue
        x1, y1, x2, y2 = raw
        reason = None
        if x2 <= x1 or y2 <= y1:
            reason = "inverted_or_degenerate"
        elif x2 < 0 or y2 < 0 or x1 > width - 1 or y1 > height - 1:
            reason = "outside_image"
        clipped = [
            max(0.0, min(v, float(limit)))
            for v, limit in zip(
                raw, (width - 1, height - 1, width - 1, height - 1), strict=True
            )
        ]
        if reason is None and (clipped[2] <= clipped[0] or clipped[3] <= clipped[1]):
            reason = "empty_after_clipping"
        if reason:
            audit.append(
                {"index": index, "raw_bbox": raw, "reason": reason, "operation": "drop"}
            )
            continue
        if clipped != raw:
            audit.append(
                {"index": index, "raw_bbox": raw, "bbox": clipped, "operation": "clip"}
            )
        valid.append({**item, "bbox": clipped})
    return valid, audit


class GroundingDINOAdapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: DetectionPredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        add_python_path(self.paths.groundingdino_repo)
        require_file(self.paths.groundingdino_config, "GroundingDINO config")
        require_file(self.paths.groundingdino_checkpoint, "GroundingDINO checkpoint")
        import torch
        from groundingdino.models import build_model
        from groundingdino.models.GroundingDINO import groundingdino, ms_deform_attn
        from groundingdino.util.inference import load_image, predict
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(str(self.paths.groundingdino_config))
        args.device = self.device
        args.text_encoder_type = str(self.paths.groundingdino_bert)
        # Upstream's wrapper still requests a Transformers-4 head-mask helper.
        # Detection never masks attention heads; attach only to this BERT instance.
        original_wrapper = groundingdino.BertModelWarper

        def compatible_wrapper(bert_model):
            if not hasattr(bert_model, "get_head_mask"):

                def unmasked_heads(head_mask, num_hidden_layers):
                    if head_mask is not None:
                        raise ValueError("Only unmasked BERT inference is supported")
                    return [None] * num_hidden_layers

                bert_model.get_head_mask = unmasked_heads
            wrapper = original_wrapper(bert_model)
            import inspect

            original_mask = bert_model.get_extended_attention_mask
            if "device" not in inspect.signature(original_mask).parameters:
                wrapper.get_extended_attention_mask = lambda mask, shape, device=None: (
                    original_mask(mask, shape)
                )
            return wrapper

        groundingdino.BertModelWarper = compatible_wrapper
        try:
            model = build_model(args)
        finally:
            groundingdino.BertModelWarper = original_wrapper
        if not hasattr(ms_deform_attn, "_C"):

            class TorchDeformableAttention:
                @staticmethod
                def apply(value, shapes, starts, locations, weights, im2col_step):
                    return ms_deform_attn.multi_scale_deformable_attn_pytorch(
                        value, shapes, locations, weights
                    )

            # Use upstream's mathematically equivalent PyTorch implementation
            # on CUDA when its optional compiled extension is unavailable.
            ms_deform_attn.MultiScaleDeformableAttnFunction = TorchDeformableAttention
        checkpoint = torch.load(self.paths.groundingdino_checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model, load_image, predict

    def predict(
        self,
        image_uri: str,
        queries: Sequence[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        if self.predictor is not None:
            return self.predictor(image_uri, queries, box_threshold, text_threshold)
        model, load_image, predict = self._runtime.get()
        source, transformed = load_image(image_uri)
        height, width = source.shape[:2]
        boxes, logits, phrases = predict(
            model=model,
            image=transformed,
            caption=" . ".join(queries) + " .",
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        detections = []
        for box, score, phrase in zip(
            boxes.detach().cpu().tolist(),
            logits.detach().cpu().tolist(),
            phrases,
            strict=True,
        ):
            cx, cy, bw, bh = box
            detections.append(
                {
                    "label": str(phrase),
                    "bbox": [
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ],
                    "confidence": float(score),
                }
            )
        return detections


class GroundingDINOTool(SpatialTool):
    spec = ToolSpec(
        name="detect",
        description="Detect text-queried objects with GroundingDINO pixel-space boxes in [x_min,y_min,x_max,y_max] format, not xywh.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "queries": array_schema(
                    {"type": "string", "minLength": 1}, min_items=1, max_items=32
                ),
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "text_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "max_detections": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            required=("image_uri", "queries"),
        ),
        metadata={"model": "GroundingDINO"},
    )

    def __init__(self, adapter: GroundingDINOAdapter | None = None) -> None:
        self.adapter = adapter or GroundingDINOAdapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        from PIL import Image

        image_uri = str(arguments["image_uri"])
        with Image.open(image_uri) as image:
            width, height = image.size
        detections = self.adapter.predict(
            image_uri,
            tuple(arguments["queries"]),
            float(arguments.get("threshold", 0.35)),
            float(arguments.get("text_threshold", 0.25)),
        )
        detections, box_audit = sanitize_detections(detections, width, height)
        detections = detections[: int(arguments.get("max_detections", 100))]
        frame = image_pixel_frame(Path(image_uri).stem, width, height)
        output = {
            "detections": detections,
            "width": width,
            "height": height,
            "frame_id": frame.frame_id,
        }
        if box_audit:
            output["bbox_postprocessing"] = {
                "policy": "clip_to_inclusive_pixel_bounds_drop_invalid_v1",
                "changes": box_audit,
            }
        overlay, _, _ = render_annotations(image_uri, boxes=detections)
        confidence = (
            sum(float(item["confidence"]) for item in detections) / len(detections)
            if detections
            else 0.0
        )
        return ToolExecution(
            text=f"GroundingDINO detected {len(detections)} instance(s).",
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.BOUNDING_BOXES,
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
                    metadata={"role": "detection_overlay"},
                ),
            ),
            coordinate_frames=(frame,),
            confidence=confidence,
            unit="pixel",
            metadata={"backend_model": "GroundingDINO"},
        )


__all__ = ["GroundingDINOAdapter", "GroundingDINOTool"]
