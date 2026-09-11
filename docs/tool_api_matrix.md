# Spatial tool contract v2

This document describes implemented interfaces and their limits. CPU analytic tests
and injected-backend integration tests do not establish perception accuracy. Real
backend verification must be reported separately. The registered tool names remain
`detect`, `segment`, `reconstruct`, `scale`, `mask`, `geometry`, `pose`, `graph`,
`draw`, `ocr`, and `motion`.

## Coordinate and artifact decisions

- The configured reconstruction checkpoint is **DA3-BASE**. The local upstream
  `Depth-Anything-3/README.md` distinguishes its any-view series from metric and
  nested models. Its outputs are therefore labeled `reconstruction_unit` and
  `scale_status=unverified`. Neither meter units nor gravity alignment is inferred
  from a model name.
- Camera coordinates are right handed: x right, y down, z forward. DA3 backend
  extrinsics are world-to-camera; exported `camera_to_world` is their inverse.
  Every reconstruction invocation has a distinct world-frame ID. Coordinates from
  different calls must be aligned explicitly before comparison.
- Integer image coordinates denote pixel centers. `source_to_processed` records
  the 3×3 source-to-depth-image mapping, including the half-pixel resize offset.
  Intrinsics are expressed in the processed image's pixel coordinates. Equal-sized
  DA3 source views use explicit `upper_bound_resize`, with no center crop. Mixed
  image sizes are rejected by the real adapter because upstream batch center
  cropping does not return its mapping. An injected/custom backend may provide
  an explicit mapping. This restriction is visible and is not a claim that all
  multiview image sets are supported.
- MoGe v2 intrinsics are normalized. Conversion to integer-centered pixels is
  `[[W,0,-0.5],[0,H,-0.5],[0,0,1]] @ K_normalized`, consistent with the local
  MoGe `moge/utils/data_augmentation.py` conversion. Its metric depth is an
  **estimate**, not independently calibrated ground truth.
- The NPZ schema `spatial_arrays_v2` contains `points`, `depth`, `valid`, optional
  raw `confidence`, `intrinsics`, `camera_to_world`, `world_to_camera`,
  `source_image_uris`, `source_shapes`, `source_to_processed`, `frame_indices`,
  `world_frame_id`, `length_unit`, and `scale_status`. No pickle is used. Both
  `reconstruct` and the point-map artifact from `scale` produce this contract.
- `frame_index` is the absolute ID in `frame_indices`, not the row position.
  Mask extraction defaults to source-image pixels; processed masks require
  `mask_space=processed`. Source dimensions and optional `source_image_uri` are
  checked before extraction. Segmentation artifacts expose the corresponding
  source image and dimensions. A plain PNG alone cannot establish its source
  image identity, so callers should preserve and pass that provenance.
- DA3 confidence is retained as a **raw backend score**. It is not clipped into
  [0,1] and advertised as a calibrated probability. Invalid or nonpositive depths
  are excluded. Empty/insufficient masks return `available=false` and no centroid.
- A 3D centroid is the coordinate-wise median of **visible valid surface points**.
  It is not the center of a complete solid object. Object bounds have the same
  visibility limitation.

## Implemented API and draft correspondence

| Draft API / capability | Actual invocation and output | Status / limit |
|---|---|---|
| B.1 `Reconstruct(frames, frame_indices)` | `reconstruct(image_uris, frame_indices?)`; self-contained NPZ, depth previews, per-view intrinsics/poses and source mapping | Implemented with the explicit equal-size real-backend restriction above; output depth resolution may differ from source |
| B.1 world point maps and camera-to-world poses | NPZ `points[N,H,W,3]`, `camera_to_world[N,4,4]`; raw depth, validity and intrinsics retained | Implemented, nonmetric until calibrated |
| B.1 gravity alignment / ground at Y≈0 | None | Unsupported: first-camera pose alone does not establish gravity or ground height; a visible plane is not automatically the floor |
| B.1 `render_bev`, moving-object trajectories | None | Unsupported; draft must remove or explicitly label proposed, and cannot describe these as used in current experiments |
| B.2 text and box SAM3 segmentation | `segment(image_uri, prompt?, boxes?, box_labels?, threshold?)`; persisted masks, boxes, scores | Implemented; boxes are source pixel xyxy; source-size masks are checked |
| B.2 point-prompt/video segmentation, object-existence batch API | None | Not exposed by the current adapter; repeated image segmentation is not video tracking or an existence-calibrated detector |
| B.2 masked world points / 3D median centroid | `mask(operation="masked_points" or "centroid_3d", mask_uris=[...], reconstruction_uri, frame_index, source_image_uri?)` | Implemented as an artifact-based wrapper instead of an in-process `PerFrameMask` method |
| B.2 segmentation visualization | Mask PNG artifacts | Binary masks are visible; colored source-image overlays and `VisualFeedback.show()` wrapper are not exposed |
| B.3 3D Euclidean distance | `geometry(operation="point_distance", first=[x,y,z], second=[x,y,z], length_unit, frame_id)` | Implemented; different declared operand frames are rejected; caller must not mix undeclared frames/units |
| B.3 angle between vectors | `geometry(operation="angle_between_vectors", first, second)` | Implemented in degrees; zero vectors rejected |
| B.3 vector rotation | `geometry(operation="rotation_matrix_from_vectors", first, second)` | Implemented, including antiparallel vectors; opposite-vector roll is chosen deterministically, not inferred physically |
| B.3 SE(3) transformation | `geometry(operation="transform_points", points or points_uri, matrix, target_frame_id)` | Implemented; validates proper rotation; full transformed points persisted and bounded preview returned |
| B.3 projection / inverse projection | `geometry(operation="project_points" or "backproject_points", points, intrinsics, camera_to_world, depth_values?)` | Implemented; inverse uses camera-z depth, not ray length; points behind camera return null and false validity |
| B.3 ground-plane RANSAC | None | Not implemented; robust plane fitting and semantic identification as floor are separate operations |
| B.3 normalized coordinates | Existing `geometry(operation="convert_points", ..., source_unit, target_unit)` | Existing normalized convention is 0–1, not the draft's 0–1000; callers must explicitly convert and manuscript must match |
| B.4 mask centroid/area/bbox | `mask(operation="statistics", mask_uris=[...])`; foreground count/fraction, `centroid_xy`, `bbox_xyxy` | Implemented median centroid and area; bbox_xyxy uses exact bounds with exclusive maxima, robust_bbox_xyxy exposes 1st/99th percentiles when foreground >100 pixels |
| B.4 batch centroids / IoU | statistics returns per-mask area/median; operation="iou" accepts exactly two masks | Implemented; empty/empty IoU is explicitly 1 |
| B.4 intersection / mask algebra | `mask(operation="intersection"/"union"/"difference"/"invert"/"dilate"/"erode", mask_uris)` | Implemented; inputs must have equal dimensions |
| B.5 numerical `Graph.plot` | `graph(operation="plot", values, validity?, x_label?, y_label?, title?)` | Implemented using matplotlib Agg; returns PNG and valid-sample min/max/mean/trend; invalid entries create gaps |
| Scene relation graph (additional API) | `graph(entities, max_output_edges?)`; direct relation edges plus full JSON artifact; `graph(graph_uri, entity_id?, relation?, offset?, max_output_edges?)` | Implemented. Image-plane left/right/above/below and pixel-near; depth front/behind only when both depths supplied. This is not object-frame or metric 3D adjacency |
| B.6 drawing boxes / points | `draw(image_uri, boxes?, points?)` | Implemented; annotations do not mutate source image |
| B.6 lines / thickness | `draw(image_uri, lines=[{line:[x1,y1,x2,y2],color:...}], thickness?, point_radius?)` | Implemented; thickness/radius apply to the call, colors may vary by primitive |
| B.7 OCR | `ocr(image_uri, languages?, min_confidence?)`; text, pixel polygons, confidence and annotation | Implemented, with structured `regions` rather than the draft's parallel arrays |
| B.8 detection | `detect(image_uri, queries, box_threshold?, text_threshold?)`; pixel xyxy boxes, labels, confidence and annotation | Implemented; outputs compose with SAM3 boxes |
| B.9 orientation angles | `pose(image_uri, bbox?)` or `operation="angles"` | Existing compatible mode retained |
| B.9 semantic object frame | `pose(operation="object_frame", image_uri, mask_uri, reconstruction_uri, frame_index, ...)` | Implemented with the named Orient Anything convention below, confidence gating and degenerate-axis rejection |
| B.9 object bounds / visualization | Output `object_to_world`, `front`, `centroid`, rotation, object-aligned bounds/corners; pose JSON and overlay PNG | Implemented for observed surface points; axis/bounds accuracy depends on perception and visibility |
| B.10 dense optical flow | `motion(first_image_uri, second_image_uri, ...)`; flow NPY, mean displacement/magnitude and color visualization | Implemented for equally sized images; flow is pixels, not metric ego-motion |
| B.10 region mask, validity/median summaries | `motion(..., mask_uri?)`; explicit validity PNG and median_flow | Implemented; empty valid regions return unavailable with null statistics; sparse arrow overlay is not exposed |
| B.11 metric scale verification | `scale(image_uri, reconstruction_uri?, frame_index?, min_valid_pixels?, max_relative_disagreement?)` | Implemented per selected view. It checks source identity, aligns valid pixels, and reports factor/disagreement |
| B.11 multiframe robust aggregation | One selected frame per call | Not implemented as one pooled estimate; independent estimates cannot be silently treated as multiframe consensus |

The table intentionally distinguishes interfaces already available from draft-only
capabilities. Missing physical information is not repaired by assigning convenient
axis labels or by inventing a floor plane. Unimplemented software APIs may be
implemented later; they must not appear as completed experiment capabilities.

## Pose convention and confidence

`orient_anything_v1_722_projection_v1` follows the local 722-output checkpoint's
azimuth/polar/in-plane angles and `Orient-Anything/utils.py:get_proj2D_XYZ`:
the upstream plot has x right and y up, while the camera frame has y down.
The third components complete its orthogonal front/right/top axes. The named
mapping has analytic cardinal-angle and nontrivial-angle projection tests. The
axes are transformed into the reconstruction world with its camera-to-world
rotation. This verifies convention consistency; it does not verify real-object
pose accuracy.

The output origin is the masked point median. +Z is the model's semantic-front
estimate, +Y is the orthogonalized semantic-bottom estimate, and +X is +Y×+Z.
No semantic front is inferred from a PCA box. Inputs with low model confidence,
insufficient points or degenerate axes return `available=false`. The confidence
threshold defaults to 0.5 and is an explicit tool argument, not a validated optimal
threshold. Explicit `front_world`/`up_world` vectors are also supported and labeled
as supplied axes, not model measurements. Overrides require both vectors.

## Scale correction

On mutually valid aligned pixels, `factor=median(depth_m/depth_recon)`. The
reported disagreement is `median(abs(depth_m/(factor*depth_recon)-1))`; this is a
robust agreement statistic, not an absolute error estimate. Defaults are at least
32 valid pixels and disagreement at most 0.25; both are explicit arguments.
Insufficient overlap or large disagreement returns `correction_factor=null`.

A reliable comparison creates a **new** `spatial_arrays_v2` artifact. Points,
depth and camera translations are scaled together, intrinsics remain unchanged,
and a distinct frame ID records `length_unit=meter`,
`scale_status=estimated_metric`. The original artifact is unchanged. This applies
one view's factor to the shared reconstruction gauge and must not be described as
multiview-calibrated ground-truth scale. Repeated calibration is another explicit
estimate, not an automatic in-place correction.

## Minimal execution chain

1. Call `reconstruct(image_uris=[source], frame_indices=[17])`.
2. Obtain a mask from `segment(image_uri=source, prompt="object")`.
3. Pass the saved NPZ and mask URIs to `mask(..., operation="centroid_3d",
   reconstruction_uri=..., frame_index=17, source_image_uri=source)`.
4. Compare centers in the same frame with `geometry(point_distance, ...)`, or
   obtain an object frame with `pose(operation="object_frame", frame_index=17, ...)` and use
   the inverse `object_to_world` for object-coordinate reasoning.
5. If metric scale is required, call `scale` with the same source image and
   reconstruction, explicitly passing `frame_index=17`. Use its new calibrated NPZ for downstream extraction only
   when `alignment.available` is true.

Large arrays remain in artifacts. Geometry returns a bounded point preview,
graph returns bounded relation pages or visible numerical charts, and each has an explicit consuming API.
Source image identity, frame ID, scale status and validity must accompany any
claimed geometric result. Real-backend acceptance should use scene-appropriate
tolerances; exact multiview agreement is only expected in analytic fixtures.

## Validation commands

```bash
PYTHONPATH=src /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python -m pytest -q \
  tests/test_spatial_tools_v2.py tests/test_stage4_tools.py \
  tests/test_stage6_real_tools.py tests/test_tool_artifact_chaining.py
```

These tests exercise analytic projection/transform cases, absolute frame IDs,
source mismatches, empty masks, array persistence/chaining, correction rejection,
non-mutating metric conversion, pose degeneracy, graph paging and old interfaces.
No model weights or GPU inference are used by this command.


## Recorded validation

- CPU checks: 22 tests in `test_spatial_tools_v2.py`, plus 8 existing tool tests,
  passed. Static `ruff check` passed for the tools tree, new tests and diagnostic
  script. The only CPU warnings were matplotlib/pyparsing deprecations.
- Real backends: Slurm job **229208**, report
  [`tools_229208/report.json`](../artifacts/v2_validation/tools_229208/report.json),
  completed all 11 checks in 89.90 seconds of diagnostic-script wall time (including
  child-process overhead; tool latency sum was 67.53 seconds), using the first RoboSpatial environment
  image. Detection, two segmentations, reconstruction, two centroids, 3D distance,
  scene graph, numerical plot, scale alignment and object-frame estimation all
  returned `passed` (the two centroid checks count separately).
- The measured center distance was 0.7887828 reconstruction units. MoGe estimated
  correction factor 1.5802128 from 190,512 matched valid pixels, with relative
  median absolute disagreement 0.0459359. These are model-derived consistency
  observations, not known true distance/scale accuracy.
- Pose returned model-estimated axes with confidence 0.7688 and a persisted OBB
  overlay. The smoke did not evaluate a final LLM answer or prove that the model
  consumed the resulting media. Such checks belong to the main Runtime validation.

The 229208 report predates the diagnostic script’s source-hash binding field. It proves the recorded component checks but is not a fully source-bound formal experiment; the updated script records source binding for future diagnostics. The historical report has not been rewritten.
