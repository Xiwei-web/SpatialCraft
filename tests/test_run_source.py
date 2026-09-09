import runpy
from pathlib import Path

import pytest


def test_frozen_source_reuse_and_tamper_rejection(tmp_path):
    freeze = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts/inference/freeze_run_source.py"
        )
    )["freeze"]
    project = tmp_path / "project"
    for folder in ("src", "configs", "prompts"):
        (project / folder).mkdir(parents=True)
    source = project / "src/example.py"
    source.write_text("value = 1\n")
    output = tmp_path / "run/code_snapshot"
    assert freeze(project, output) == output
    source.write_text("value = 2\n")
    assert freeze(project, output) == output
    assert (output / "src/example.py").read_text() == "value = 1\n"
    (output / "src/example.py").write_text("value = 3\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        freeze(project, output)
