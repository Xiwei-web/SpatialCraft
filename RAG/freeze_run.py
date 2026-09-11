"""Freeze and verify RAG source/configuration without importing experiment code."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def manifest(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("RAG", "src", "configs", "prompts")
        for path in sorted((root / name).rglob("*"))
        if path.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(root).parts
        )
        and path.suffix != ".pyc"
    }


def freeze(project, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ".rag_snapshot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            expected = json.loads((output / "source_manifest.json").read_text())
            if manifest(output) != expected:
                raise ValueError("Frozen RAG source changed")
            return output
        before = manifest(project)
        with tempfile.TemporaryDirectory(
            prefix=".rag-source-", dir=output.parent
        ) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            for name in before:
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project / name, destination)
            if manifest(staging) != before or manifest(project) != before:
                raise ValueError("Source changed during RAG snapshot creation")
            (staging / "source_manifest.json").write_text(
                json.dumps(before, sort_keys=True)
            )
            os.rename(staging, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.project.resolve(), args.output.resolve()))
