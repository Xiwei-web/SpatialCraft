"""Record exact files in the isolated kernel overlay; never touch shared packages."""

from spatialcraft.experiments.fast_qwen import OVERLAY
from spatialcraft.storage.atomic_io import atomic_write_json, sha256_file

if __name__ == "__main__":
    files = {
        str(p.relative_to(OVERLAY)): sha256_file(p)
        for p in sorted(OVERLAY.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "installation_manifest.json"
        and p.suffix != ".pyc"
    }
    if not files:
        raise RuntimeError("Empty kernel overlay")
    atomic_write_json(
        OVERLAY / "installation_manifest.json",
        {
            "files": files,
            "source": "PyPI FLA 0.5.2 and official Dao-AILab causal-conv1d v1.7.0 wheel",
        },
        overwrite=False,
    )
    print("Pinned files:", len(files))
