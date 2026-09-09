"""Fail before model/API work if Slurm has not exposed the requested usable GPUs."""

import argparse
import json
import os
import socket

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-gpus", type=int, default=1)
    args = parser.parse_args()
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    try:
        if (
            args.expected_gpus < 1
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != args.expected_gpus
        ):
            raise RuntimeError(
                f"Expected exactly {args.expected_gpus} Slurm-bound usable CUDA GPUs"
            )
        devices = []
        for index in range(args.expected_gpus):
            value = torch.ones((8, 8), device=f"cuda:{index}")
            assert (value @ value).sum().item() == 512
            devices.append(
                {
                    "index": index,
                    "gpu_name": torch.cuda.get_device_name(index),
                    "memory_bytes": list(torch.cuda.mem_get_info(index)),
                }
            )
        report.update(
            status="passed",
            devices=devices,
            gpu_name=torch.cuda.get_device_name(0),
            memory_bytes=list(torch.cuda.mem_get_info(0)),
        )
    except Exception as exc:  # noqa: BLE001 - persist GPU diagnostics then stop
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    with open(args.report, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
