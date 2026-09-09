"""Real-weight, training-image tool diagnostics; no test labels or LLM/API calls."""

import argparse
import gc
import json
import traceback
from pathlib import Path

import torch

from spatialcraft.storage.atomic_io import atomic_write_json
from spatialcraft.tools.real.detect import GroundingDINOAdapter
from spatialcraft.tools.real.ocr import EasyOCRAdapter
from spatialcraft.tools.real.pose import OrientAnythingAdapter
from spatialcraft.tools.real.reconstruct import DepthAnything3Adapter
from spatialcraft.tools.real.scale import MoGe2Adapter
from spatialcraft.tools.real.segment import SAM3Adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--executor",
        action="store_true",
        help="Also validate serialized ToolResults and persisted artifacts",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Run in a GPU allocation")
    source = Path(
        "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1/robospatial/splits/environment.jsonl"
    )
    row = json.loads(source.read_text().splitlines()[0])
    image = row["images"][0]["uri"]
    checks = (
        (
            "detect",
            GroundingDINOAdapter,
            lambda a: a.predict(image, ("toilet", "frame"), 0.35, 0.25),
        ),
        ("segment", SAM3Adapter, lambda a: a.predict(image, "toilet", (), (), 0.5)),
        ("scale", MoGe2Adapter, lambda a: a.predict(image, 1)),
        ("reconstruct", DepthAnything3Adapter, lambda a: a.predict((image,))),
        ("pose", OrientAnythingAdapter, lambda a: a.predict(image, None)),
        ("ocr", EasyOCRAdapter, lambda a: a.predict(image, ("en",))),
    )
    report = {
        "kind": "standalone_tool_smoke_not_combined_qwen_validation",
        "image": image,
        "tools": {},
    }
    if args.executor:
        from spatialcraft.experiments.runtime import SerialToolExecutor
        from spatialcraft.schemas import ToolCall, ToolStatus
        from spatialcraft.storage import StorageLayout
        from spatialcraft.tools import ArtifactStore
        from spatialcraft.tools.real import create_real_tool_registry

        registry = create_real_tool_registry()
        executor = SerialToolExecutor(
            registry,
            ArtifactStore(
                StorageLayout(args.output.parent / args.output.stem), "tool_smoke"
            ),
        )
        arguments = {
            "detect": {"image_uri": image, "queries": ["toilet", "frame"]},
            "segment": {"image_uri": image, "prompt": "toilet"},
            "scale": {"image_uri": image, "resolution_level": 1},
            "reconstruct": {"image_uris": [image]},
            "pose": {"image_uri": image},
            "ocr": {"image_uri": image},
        }
        report["kind"] = "full_tool_executor_smoke_not_combined_qwen_validation"
        for name, values in arguments.items():
            print("TOOL_START", name, flush=True)
            result = executor.execute(ToolCall(tool_name=name, arguments=values))
            report["tools"][name] = {
                "status": "passed"
                if result.status is ToolStatus.SUCCEEDED
                else "failed",
                "result": result.to_dict(),
            }
            atomic_write_json(args.output, report)
            print("TOOL_DONE", name, report["tools"][name]["status"], flush=True)
        if any(r["status"] != "passed" for r in report["tools"].values()):
            raise SystemExit(1)
        return
    for name, factory, operation in checks:
        print("TOOL_START", name, flush=True)
        adapter = factory()
        try:
            result = operation(adapter)
            report["tools"][name] = {"status": "passed", "output_count": len(result)}
            del result
        except Exception as exc:  # noqa: BLE001 - collect independent tool diagnostics
            report["tools"][name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            traceback.print_exc()
        finally:
            resource = getattr(adapter, "_runtime", None)
            if resource is not None:
                resource.release()
            for resource in getattr(adapter, "_readers", {}).values():
                resource.release()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
        atomic_write_json(args.output, report)
        print("TOOL_DONE", name, report["tools"][name]["status"], flush=True)
    if any(r["status"] != "passed" for r in report["tools"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
