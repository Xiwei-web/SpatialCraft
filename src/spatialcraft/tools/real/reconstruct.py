"""Depth Anything 3 multi-view reconstruction adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..schema_builder import array_schema, object_schema
from ..spatial_arrays import array_bundle, backproject, camera_intrinsics, se3
from .common import (
    LazyResource,
    SpatialToolPaths,
    add_python_path,
    normalized_preview,
)

ReconstructionPredictor = Callable[[Sequence[str]], Mapping[str, Any]]


class DepthAnything3Adapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: ReconstructionPredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        add_python_path(self.paths.da3_repo / "src")
        if not self.paths.da3_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Depth Anything 3 checkpoint not found: {self.paths.da3_checkpoint}"
            )
        from depth_anything_3.api import DepthAnything3

        return DepthAnything3.from_pretrained(str(self.paths.da3_checkpoint)).to(
            device=self.device
        )

    def predict(self, image_uris: Sequence[str]) -> Mapping[str, Any]:
        if self.predictor is not None:
            return self.predictor(image_uris)
        prediction = self._runtime.get().inference(
            list(image_uris), process_res_method="upper_bound_resize"
        )
        return {
            name: getattr(prediction, name, None)
            for name in (
                "processed_images",
                "depth",
                "conf",
                "extrinsics",
                "intrinsics",
            )
        }


class ReconstructionTool(SpatialTool):
    spec = ToolSpec(
        name="reconstruct",
        version="2.0.0",
        description="Reconstruct depth and world point maps with DA3-BASE. Scale is unverified, not meters; equal-sized source views required. Pass the NPZ artifact as reconstruction_uri to mask, pose, or scale.",
        input_schema=object_schema(
            {
                "image_uris": array_schema(
                    {"type": "string", "minLength": 1}, min_items=1, max_items=64
                ),
                "frame_indices": array_schema(
                    {"type": "integer", "minimum": 0}, min_items=1, max_items=64
                ),
            },
            required=("image_uris",),
        ),
        metadata={"model": "Depth Anything 3"},
        default_timeout_s=300,
    )

    def __init__(self, adapter: DepthAnything3Adapter | None = None) -> None:
        self.adapter = adapter or DepthAnything3Adapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import numpy as np
        from PIL import Image

        image_uris = tuple(str(item) for item in arguments["image_uris"])
        source_shapes = []
        for uri in image_uris:
            with Image.open(uri) as image:
                source_shapes.append((image.height, image.width))
        # DA3 center-crops mixed-sized batches without returning that mapping.
        # Reject the ambiguous path; an injected backend can supply the mapping.
        if len(set(source_shapes)) != 1 and self.adapter.predictor is None:
            raise ValueError(
                "DA3 v2 requires equal-sized views to avoid undocumented batch center crops"
            )
        frame_indices = tuple(arguments.get("frame_indices", range(len(image_uris))))
        if len(frame_indices) != len(image_uris) or len(set(frame_indices)) != len(
            frame_indices
        ):
            raise ValueError("frame_indices must uniquely identify each source view")
        prediction = self.adapter.predict(image_uris)
        depth = np.asarray(prediction["depth"], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        if depth.ndim != 3 or len(depth) != len(image_uris):
            raise ValueError("backend depth must have shape (view_count,H,W)")
        confidence_available = prediction.get("conf") is not None
        confidence = np.asarray(
            prediction["conf"]
            if prediction.get("conf") is not None
            else np.ones_like(depth),
            dtype=np.float32,
        )
        if confidence.shape != depth.shape:
            raise ValueError("confidence and depth shapes must agree")
        intrinsics = np.asarray(prediction.get("intrinsics"), dtype=float)
        extrinsics = np.asarray(prediction.get("extrinsics"), dtype=float)
        if intrinsics.shape != (len(depth), 3, 3) or extrinsics.shape not in (
            (len(depth), 3, 4),
            (len(depth), 4, 4),
        ):
            raise ValueError(
                "backend must supply per-view intrinsics and world-to-camera extrinsics"
            )
        intrinsics = np.stack([camera_intrinsics(item) for item in intrinsics])
        extrinsics = np.stack([se3(item) for item in extrinsics])
        camera_to_worlds = np.linalg.inv(extrinsics)
        height, width = depth.shape[1:]
        transforms = prediction.get("source_to_processed")
        if transforms is None:
            if len(set(source_shapes)) != 1:
                raise ValueError(
                    "mixed source dimensions require explicit source_to_processed transforms"
                )
            transforms = [
                np.array(
                    [
                        [width / w, 0, (width / w - 1) / 2],
                        [0, height / h, (height / h - 1) / 2],
                        [0, 0, 1],
                    ]
                )
                for h, w in source_shapes
            ]
        transforms = np.asarray(transforms, dtype=float)
        if transforms.shape != (len(depth), 3, 3) or not np.isfinite(transforms).all():
            raise ValueError(
                "source_to_processed must contain one finite 3x3 mapping per view"
            )
        # DA3-BASE is an any-view model, not the metric or nested checkpoint.
        unit, scale_status = "reconstruction_unit", "unverified"
        world_id = f"reconstruction:{context.invocation_id}:world"
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
        points = np.stack(
            [
                backproject(d, k, c)
                for d, k, c in zip(depth, intrinsics, camera_to_worlds, strict=True)
            ]
        )
        points[~valid] = np.nan
        frames = tuple(
            CoordinateFrame(
                frame_id=f"reconstruction:{context.invocation_id}:camera:{frame_indices[i]}",
                parent_frame_id=world_id,
                transform_to_parent=tuple(
                    float(v) for v in camera_to_worlds[i].reshape(-1)
                ),
                unit=unit,
                convention="right-handed OpenCV: x=right,y=down,z=forward",
                metadata={
                    "source_extrinsics": "world_to_camera",
                    "transform_to_parent": "camera_to_world",
                    "source_image_uri": image_uris[i],
                },
            )
            for i in range(len(depth))
        )
        world = CoordinateFrame(
            frame_id=world_id,
            unit=unit,
            convention="DA3 reconstruction gauge; not gravity aligned",
            metadata={"scale_status": scale_status, "gravity_aligned": False},
        )
        bundle = array_bundle(
            schema_version="spatial_arrays_v2",
            array=depth,
            depth=depth,
            points=points.astype(np.float32),
            valid=valid,
            **({"confidence": confidence} if confidence_available else {}),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            world_to_camera=extrinsics,
            camera_to_world=camera_to_worlds,
            source_image_uris=image_uris,
            source_shapes=source_shapes,
            source_to_processed=transforms,
            frame_indices=frame_indices,
            world_frame_id=world_id,
            length_unit=unit,
            scale_status=scale_status,
        )
        artifacts = [
            ArtifactPayload(
                artifact_type=ArtifactType.POINT_CLOUD,
                data=bundle,
                suffix=".npz",
                mime_type="application/x-npz",
                shape=points.shape,
                dtype="float32",
                frame_id=world_id,
                metadata={
                    "content": "spatial_arrays_v2",
                    "fields": [
                        "points",
                        "depth",
                        "valid",
                        "intrinsics",
                        "camera_to_world",
                        "source_to_processed",
                    ],
                    "downstream": [
                        "mask.centroid_3d",
                        "mask.masked_points",
                        "pose.object_frame",
                        "scale",
                    ],
                    "confidence_available": confidence_available,
                    "length_unit": unit,
                    "scale_status": scale_status,
                },
            )
        ]
        for i, item in enumerate(depth):
            artifacts.append(
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=normalized_preview(item),
                    suffix=".png",
                    mime_type="image/png",
                    shape=item.shape,
                    dtype="uint8",
                    frame_id=frames[i].frame_id,
                    metadata={
                        "role": "depth_preview",
                        "view_index": i,
                        "frame_index": frame_indices[i],
                        "source_image_uri": image_uris[i],
                        "source_shape": list(source_shapes[i]),
                        "source_to_processed": transforms[i].tolist(),
                    },
                )
            )
        finite_confidence = (
            confidence[np.isfinite(confidence)]
            if confidence_available
            else np.array([])
        )
        output = {
            "view_count": len(depth),
            "depth_shape": list(depth.shape),
            "mean_confidence": float(finite_confidence.mean())
            if finite_confidence.size
            else None,
            "confidence_available": confidence_available,
            "confidence_semantics": "raw backend score; not calibrated probability"
            if confidence_available
            else "unavailable",
            "camera_frame_ids": [frame.frame_id for frame in frames],
            "world_frame_id": world_id,
            "length_unit": unit,
            "scale_status": scale_status,
            "gravity_aligned": False,
            "frame_indices": list(frame_indices),
            "valid_point_count": int(valid.sum()),
            "views": [
                {
                    "frame_index": frame_indices[i],
                    "source_image_uri": uri,
                    "source_shape": list(source_shapes[i]),
                    "processed_shape": [height, width],
                    "source_to_processed": transforms[i].tolist(),
                    "intrinsics_pixels": intrinsics[i].tolist(),
                    "camera_to_world": camera_to_worlds[i].tolist(),
                }
                for i, uri in enumerate(image_uris)
            ],
            "artifact_usage": "Pass NPZ URI as reconstruction_uri to mask, pose, or scale; masks default to source-image pixels.",
        }
        return ToolExecution(
            text=f"DA3 reconstructed {len(depth)} view(s); scale unverified.",
            structured_output=output,
            artifacts=tuple(artifacts),
            coordinate_frames=frames + (world,),
            unit=unit,
            metadata={
                "backend_model": "Depth Anything 3",
                "checkpoint": str(self.adapter.paths.da3_checkpoint),
                "scale_status": scale_status,
            },
        )


__all__ = ["DepthAnything3Adapter", "ReconstructionTool"]
