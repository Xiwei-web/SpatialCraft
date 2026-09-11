"""Orient Anything object-pose adapter and tool."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..schema_builder import ToolSchemaError, array_schema, enum_schema, object_schema
from ..spatial_arrays import masked_world_points, se3
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
        version="2.0.0",
        description="Estimate Orient Anything angles, or operation=object_frame combines reconstruction_uri, mask_uri and frame_index into a model-estimated semantic frame (+Z front,+Y bottom,+X right-hand rule), visible-surface OBB and overlay. Low orientation confidence or insufficient points returns unavailable.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "operation": enum_schema("angles", "object_frame"),
                "reconstruction_uri": {"type": "string", "minLength": 1},
                "mask_uri": {"type": "string", "minLength": 1},
                "frame_index": {"type": "integer", "minimum": 0},
                "mask_space": enum_schema("source", "processed"),
                "min_points": {"type": "integer", "minimum": 1},
                "min_orientation_confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "front_world": array_schema(
                    {"type": "number"}, min_items=3, max_items=3
                ),
                "up_world": array_schema({"type": "number"}, min_items=3, max_items=3),
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
        if arguments.get("operation") == "object_frame":
            return self._object_frame(arguments, context)
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

    @staticmethod
    def _camera_axes(output: Mapping[str, Any]):
        """722-head angles, matching upstream utils.get_proj2D_XYZ.

        Upstream plots use x-right/y-up; we convert to x-right/y-down/z-forward.
        The third components make the projected front/right/top axes orthonormal.
        This is a declared model convention, not a claim of calibrated true pose.
        """
        import numpy as np

        phi, theta, gamma = np.radians(
            [output["azimuth_deg"], output["polar_deg"], -output["rotation_deg"]]
        )
        sp, cp, st, ct, sg, cg = (
            np.sin(phi),
            np.cos(phi),
            np.sin(theta),
            np.cos(theta),
            np.sin(gamma),
            np.cos(gamma),
        )
        front = np.array([-sp * cg - cp * st * sg, -sp * sg + cp * st * cg, cp * ct])
        up = np.array([ct * sg, -ct * cg, st])
        return front, up

    def _object_frame(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import io
        import itertools

        import numpy as np
        from PIL import Image, ImageDraw

        if "reconstruction_uri" not in arguments or "mask_uri" not in arguments:
            raise ToolSchemaError(
                "object_frame requires reconstruction_uri and mask_uri"
            )
        extraction = {**arguments, "source_image_uri": arguments["image_uri"]}
        points, view = masked_world_points(extraction)
        world = CoordinateFrame(
            frame_id=view["world_frame_id"],
            unit=view["length_unit"],
            convention="source reconstruction world",
        )
        reason = None
        output = {
            "available": False,
            "frame_index": view["frame_index"],
            "world_frame_id": world.frame_id,
            "point_count": len(points),
            "length_unit": view["length_unit"],
            "scale_status": view["scale_status"],
            "centroid_semantics": "coordinate-wise median of visible masked surface points",
            "obb_semantics": "bounds of observed valid surface points in estimated semantic axes; not full object extent",
        }
        if len(points) < int(arguments.get("min_points", 3)):
            reason = "insufficient_valid_masked_points"
        elif ("front_world" in arguments) != ("up_world" in arguments):
            raise ToolSchemaError("provide both front_world and up_world or neither")
        elif "front_world" in arguments:
            front = np.asarray(arguments["front_world"], dtype=float)
            up = np.asarray(arguments["up_world"], dtype=float)
            output["orientation_source"] = "explicit_world_axes"
        else:
            bbox = arguments.get("bbox")
            if bbox is None:
                with Image.open(arguments["mask_uri"]) as image:
                    ys, xs = np.nonzero(np.asarray(image.convert("L")) > 0)
                bbox = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ]
                if arguments.get("mask_space", "source") == "processed":
                    corners = (
                        np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
                        @ np.linalg.inv(view["source_to_processed"]).T
                    )
                    corners = corners[:, :2] / corners[:, 2:3]
                    bbox = [*corners[0], *corners[1]]
            angles = dict(self.adapter.predict(str(arguments["image_uri"]), bbox))
            confidence = float(angles.get("confidence", 0.0))
            output.update(
                orientation_angles=angles,
                orientation_confidence=confidence,
                orientation_source="orient_anything_v1_722_projection_v1",
                orientation_status="model_estimated",
            )
            if not np.isfinite(confidence) or confidence < float(
                arguments.get("min_orientation_confidence", 0.5)
            ):
                reason = "orientation_confidence_below_threshold"
            else:
                front_camera, up_camera = self._camera_axes(angles)
                rotation = se3(view["camera_to_world"])[:3, :3]
                front, up = rotation @ front_camera, rotation @ up_camera
        if reason is None:
            if not np.isfinite([front, up]).all() or np.linalg.norm(front) <= 1e-8:
                reason = "invalid_orientation_axes"
            else:
                front = front / np.linalg.norm(front)
                bottom = -up + np.dot(up, front) * front
                if np.linalg.norm(bottom) <= 1e-8:
                    reason = "degenerate_orientation_axes"
                else:
                    bottom /= np.linalg.norm(bottom)
                    x_axis = np.cross(bottom, front)
                    rotation = np.column_stack((x_axis, bottom, front))
        if reason is not None:
            output["reason"] = reason
            return ToolExecution(
                text=f"Object frame unavailable: {reason}",
                structured_output=output,
                coordinate_frames=(world,),
                unit=view["length_unit"],
            )
        centroid = np.median(points, axis=0)
        local = (points - centroid) @ rotation
        low, high = local.min(axis=0), local.max(axis=0)
        corners_local = np.array(list(itertools.product(*zip(low, high))))
        corners = corners_local @ rotation.T + centroid
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, centroid
        object_id = f"object:{context.invocation_id}"
        output.update(
            available=True,
            centroid=centroid.tolist(),
            front=front.tolist(),
            rotation=rotation.tolist(),
            object_to_world=transform.tolist(),
            object_frame_id=object_id,
            obb_center=(((low + high) / 2) @ rotation.T + centroid).tolist(),
            obb_extent=(high - low).tolist(),
            obb_corners=corners.tolist(),
            axis_convention="right-handed: +Z=semantic front,+Y=semantic bottom,+X=+Y cross +Z",
        )
        object_frame = CoordinateFrame(
            frame_id=object_id,
            parent_frame_id=world.frame_id,
            transform_to_parent=tuple(float(value) for value in transform.reshape(-1)),
            unit=view["length_unit"],
            convention=output["axis_convention"],
        )
        artifacts = [
            ArtifactPayload(
                artifact_type=ArtifactType.POSE,
                json_value=output,
                suffix=".json",
                mime_type="application/json",
                frame_id=object_id,
            )
        ]
        # Project numeric geometry onto the original source image for inspection.
        c2w, k = se3(view["camera_to_world"]), view["intrinsics"]
        axis_length = max(float(np.max(high - low)) * 0.4, 1e-4)
        landmarks = np.concatenate(
            (centroid[None], centroid[None] + rotation.T * axis_length, corners)
        )
        camera = (landmarks - c2w[:3, 3]) @ c2w[:3, :3]
        good = camera[:, 2] > 1e-8
        projected = camera @ k.T
        safe_z = np.where(good, projected[:, 2], 1)
        pixels = projected[:, :2] / safe_z[:, None]
        homogeneous = (
            np.column_stack((pixels, np.ones(len(pixels))))
            @ np.linalg.inv(view["source_to_processed"]).T
        )
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        with Image.open(arguments["image_uri"]) as source:
            overlay = source.convert("RGB")
        painter = ImageDraw.Draw(overlay)
        for i, color in enumerate(("red", "green", "blue"), 1):
            if good[0] and good[i]:
                painter.line([tuple(pixels[0]), tuple(pixels[i])], fill=color, width=2)
        for i, first in enumerate(corners_local):
            for j, second in enumerate(corners_local[i + 1 :], i + 1):
                if (
                    np.count_nonzero(first != second) == 1
                    and good[i + 4]
                    and good[j + 4]
                ):
                    painter.line(
                        [tuple(pixels[i + 4]), tuple(pixels[j + 4])],
                        fill="yellow",
                        width=1,
                    )
        data = io.BytesIO()
        overlay.save(data, format="PNG")
        artifacts.append(
            ArtifactPayload(
                artifact_type=ArtifactType.IMAGE,
                data=data.getvalue(),
                suffix=".png",
                mime_type="image/png",
                shape=(overlay.height, overlay.width, 3),
                dtype="uint8",
                frame_id=world.frame_id,
                metadata={
                    "role": "object_frame_overlay",
                    "source_image_uri": arguments["image_uri"],
                    "axis_colors": {"X": "red", "Y": "green", "Z_front": "blue"},
                    "pixel_space": "source image",
                },
            )
        )
        return ToolExecution(
            text=f"Object frame centroid={output['centroid']}, front={output['front']}; axes are estimated and OBB covers visible points.",
            structured_output=output,
            artifacts=tuple(artifacts),
            coordinate_frames=(world, object_frame),
            unit=view["length_unit"],
        )


__all__ = ["OrientAnythingAdapter", "PoseTool"]
