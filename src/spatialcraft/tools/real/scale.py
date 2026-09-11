"""MoGe-2 metric geometry and scale adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import object_schema
from ..spatial_arrays import (
    align_source_array,
    array_bundle,
    camera_intrinsics,
    read_bundle,
    reconstruction_view,
)
from .common import (
    LazyResource,
    SpatialToolPaths,
    add_python_path,
    normalized_preview,
    numpy_bytes,
    require_file,
)

ScalePredictor = Callable[[str, int], Mapping[str, Any]]


class MoGe2Adapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: ScalePredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        add_python_path(self.paths.moge_repo)
        checkpoint = require_file(self.paths.moge_checkpoint, "MoGe-2 checkpoint")
        from moge.model.v2 import MoGeModel

        return MoGeModel.from_pretrained(str(checkpoint)).to(self.device).eval()

    def predict(self, image_uri: str, resolution_level: int) -> Mapping[str, Any]:
        if self.predictor is not None:
            return self.predictor(image_uri, resolution_level)
        import cv2
        import torch

        image = cv2.imread(image_uri)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {image_uri}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(
            image / 255.0, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)
        output = self._runtime.get().infer(tensor, resolution_level=resolution_level)
        return {
            key: value.detach().float().cpu().numpy()
            if hasattr(value, "detach")
            else value
            for key, value in output.items()
        }


class ScaleTool(SpatialTool):
    spec = ToolSpec(
        name="scale",
        version="2.0.0",
        description="Estimate metric depth with MoGe-2. Optional reconstruction_uri/frame_index estimates a robust scale correction from matched valid pixels. High disagreement returns no correction; a reliable correction produces a new metric-estimated reconstruction artifact without mutating the source.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "resolution_level": {"type": "integer", "minimum": 0, "maximum": 9},
                "reconstruction_uri": {"type": "string", "minLength": 1},
                "frame_index": {"type": "integer", "minimum": 0},
                "min_valid_pixels": {"type": "integer", "minimum": 1},
                "max_relative_disagreement": {"type": "number", "minimum": 0},
            },
            required=("image_uri",),
        ),
        metadata={"model": "MoGe-2"},
    )

    def __init__(self, adapter: MoGe2Adapter | None = None) -> None:
        self.adapter = adapter or MoGe2Adapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import numpy as np
        from PIL import Image

        image_uri = str(arguments["image_uri"])
        prediction = self.adapter.predict(
            image_uri, int(arguments.get("resolution_level", 6))
        )
        points = np.asarray(prediction["points"], dtype=np.float32)
        depth = np.asarray(prediction.get("depth", points[..., 2]), dtype=np.float32)
        if depth.ndim != 2 or points.shape != (*depth.shape, 3):
            raise ValueError("MoGe must return HxW depth and HxWx3 camera points")
        with Image.open(image_uri) as source:
            if depth.shape != (source.height, source.width):
                raise ValueError("MoGe output must match source-image resolution")
        height, width = depth.shape
        intrinsics_normalized = camera_intrinsics(prediction.get("intrinsics"))
        # MoGe v2 publishes normalized intrinsics, not pixel-space intrinsics.
        intrinsics = (
            np.array([[width, 0, -0.5], [0, height, -0.5], [0, 0, 1]])
            @ intrinsics_normalized
        )
        mask = np.asarray(
            prediction.get("mask", np.ones_like(depth, dtype=bool)), dtype=bool
        )
        if mask.shape != depth.shape:
            raise ValueError("MoGe validity mask must align with depth")
        valid = (
            mask & np.isfinite(depth) & (depth > 0) & np.isfinite(points).all(axis=-1)
        )
        finite = depth[valid]
        pixel_frame = image_pixel_frame(Path(image_uri).stem, width, height)
        camera_frame = CoordinateFrame(
            frame_id=f"moge:{context.invocation_id}:camera",
            unit="meter",
            convention="right-handed OpenCV: x=right,y=down,z=forward",
            metadata={
                "scale_status": "estimated_metric",
                "source_image_uri": image_uri,
            },
        )
        output = {
            "available": bool(finite.size),
            "min_depth_m": float(finite.min()) if finite.size else None,
            "max_depth_m": float(finite.max()) if finite.size else None,
            "median_depth_m": float(np.median(finite)) if finite.size else None,
            "intrinsics": intrinsics.tolist(),
            "intrinsics_convention": "pixel",
            "intrinsics_normalized": intrinsics_normalized.tolist(),
            "width": width,
            "height": height,
            "pixel_frame_id": pixel_frame.frame_id,
            "camera_frame_id": camera_frame.frame_id,
            "source_image_uri": image_uri,
            "valid_pixel_count": int(valid.sum()),
            "scale_status": "estimated_metric",
        }
        artifacts = [
            ArtifactPayload(
                artifact_type=ArtifactType.DEPTH,
                data=numpy_bytes(depth),
                suffix=".npy",
                mime_type="application/x-npy",
                shape=depth.shape,
                dtype="float32",
                frame_id=pixel_frame.frame_id,
                metadata={
                    "value_unit": "meter",
                    "source_image_uri": image_uri,
                    "scale_status": "estimated_metric",
                },
            ),
            ArtifactPayload(
                artifact_type=ArtifactType.IMAGE,
                data=normalized_preview(np.where(valid, depth, np.nan)),
                suffix=".png",
                mime_type="image/png",
                shape=depth.shape,
                dtype="uint8",
                frame_id=pixel_frame.frame_id,
                metadata={"role": "depth_preview", "source_image_uri": image_uri},
            ),
            ArtifactPayload(
                artifact_type=ArtifactType.POINT_CLOUD,
                data=array_bundle(
                    schema_version="spatial_arrays_v2",
                    array=depth[None],
                    points=points[None],
                    depth=depth[None],
                    valid=valid[None],
                    intrinsics=intrinsics[None],
                    camera_to_world=np.eye(4)[None],
                    world_to_camera=np.eye(4)[None],
                    extrinsics=np.eye(4)[None],
                    frame_indices=[0],
                    source_image_uris=[image_uri],
                    source_shapes=[[height, width]],
                    source_to_processed=np.eye(3)[None],
                    world_frame_id=camera_frame.frame_id,
                    length_unit="meter",
                    scale_status="estimated_metric",
                ),
                suffix=".npz",
                mime_type="application/x-npz",
                shape=(1, *points.shape),
                dtype="float32",
                frame_id=camera_frame.frame_id,
                metadata={
                    "content": "spatial_arrays_v2",
                    "role": "metric_camera_reconstruction",
                    "length_unit": "meter",
                    "scale_status": "estimated_metric",
                    "downstream": [
                        "mask.centroid_3d",
                        "mask.masked_points",
                        "pose.object_frame",
                    ],
                },
            ),
        ]
        frames = [pixel_frame, camera_frame]
        if "reconstruction_uri" in arguments:
            view = reconstruction_view(
                str(arguments["reconstruction_uri"]),
                int(arguments.get("frame_index", 0)),
            )
            if (
                Path(image_uri).resolve()
                != Path(str(view["source_image_uris"])).resolve()
            ):
                raise ValueError(
                    "scale image must be the source image of the selected reconstruction view"
                )
            # Nearest-neighbor paired samples avoid invalid-depth interpolation and edge mixing.
            matched_depth = align_source_array(
                np.where(valid, depth, 0).astype(np.float32), view
            )
            matched_valid = align_source_array(valid.astype(np.uint8), view).astype(
                bool
            )
            common = (
                matched_valid
                & np.asarray(view["valid"], dtype=bool)
                & (view["depth"] > 0)
            )
            ratios = matched_depth[common] / view["depth"][common]
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            enough = len(ratios) >= int(arguments.get("min_valid_pixels", 32))
            factor = float(np.median(ratios)) if len(ratios) else None
            # Median absolute relative residual is robust but does not certify accuracy.
            disagreement = (
                float(np.median(np.abs(ratios / factor - 1))) if factor else None
            )
            reliable = enough and disagreement <= float(
                arguments.get("max_relative_disagreement", 0.25)
            )
            alignment = {
                "available": reliable,
                "correction_factor": factor if reliable else None,
                "estimated_factor": factor,
                "relative_disagreement": disagreement,
                "disagreement_definition": "median(abs(depth_m / (factor * depth_recon) - 1))",
                "num_valid_pixels": len(ratios),
                "selected_frames": [view["frame_index"]],
                "per_frame_factors": {str(view["frame_index"]): factor},
                "source_world_frame_id": view["world_frame_id"],
                "reason": None
                if reliable
                else "insufficient_overlap"
                if not enough
                else "high_disagreement",
                "scale_status": "estimated_metric" if reliable else "unverified",
            }
            output["alignment"] = alignment
            if reliable:
                bundle = read_bundle(str(arguments["reconstruction_uri"]))
                bundle["depth"] = bundle["depth"] * factor
                bundle["array"] = bundle["depth"]
                bundle["points"] = bundle["points"] * factor
                bundle["camera_to_world"] = bundle["camera_to_world"].copy()
                bundle["camera_to_world"][:, :3, 3] *= factor
                bundle["world_to_camera"] = np.linalg.inv(bundle["camera_to_world"])
                bundle["extrinsics"] = bundle["world_to_camera"]
                calibrated_id = (
                    f"{view['world_frame_id']}:metric:{context.invocation_id}"
                )
                bundle.update(
                    world_frame_id=calibrated_id,
                    length_unit="meter",
                    scale_status="estimated_metric",
                    scale_correction_factor=factor,
                    source_reconstruction_uri=str(arguments["reconstruction_uri"]),
                )
                frames.append(
                    CoordinateFrame(
                        frame_id=calibrated_id,
                        unit="meter",
                        convention="scaled reconstruction gauge; not gravity aligned",
                        metadata={
                            "scale_status": "estimated_metric",
                            "correction_factor": factor,
                        },
                    )
                )
                artifacts.append(
                    ArtifactPayload(
                        artifact_type=ArtifactType.POINT_CLOUD,
                        data=array_bundle(**bundle),
                        suffix=".npz",
                        mime_type="application/x-npz",
                        shape=bundle["points"].shape,
                        dtype=str(bundle["points"].dtype),
                        frame_id=calibrated_id,
                        metadata={
                            "content": "spatial_arrays_v2",
                            "role": "calibrated_reconstruction",
                            "length_unit": "meter",
                            "scale_status": "estimated_metric",
                            "downstream": [
                                "mask.centroid_3d",
                                "mask.masked_points",
                                "pose.object_frame",
                            ],
                        },
                    )
                )
                alignment["calibrated_world_frame_id"] = calibrated_id
        return ToolExecution(
            text=f"MoGe-2 metric estimate: median_depth_m={output['median_depth_m']}; alignment={output.get('alignment')}",
            structured_output=output,
            artifacts=tuple(artifacts),
            coordinate_frames=tuple(frames),
            unit="meter",
            metadata={"backend_model": "MoGe-2", "scale_status": "estimated_metric"},
        )


__all__ = ["MoGe2Adapter", "ScaleTool"]
