"""Preserve the exact executable source/configuration for future run resumption."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def files(root):
    return sorted(
        path
        for name in ("src", "configs", "prompts")
        for path in (root / name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def manifest(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files(root)
    }


def freeze(project, output):
    import fcntl

    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ".source_snapshot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            expected = json.loads((output / "source_manifest.json").read_text())
            if manifest(output) != expected:
                raise ValueError("Frozen source changed; refusing incompatible resume")
            return output
        with tempfile.TemporaryDirectory(
            prefix=".source-stage-", dir=output.parent
        ) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            before = manifest(project)
            for name in before:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project / name, target)
            if manifest(staging) != before or manifest(project) != before:
                raise ValueError("Source changed during snapshot creation")
            (staging / "source_manifest.json").write_text(
                json.dumps(before, sort_keys=True)
            )
            os.rename(staging, output)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.project.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
