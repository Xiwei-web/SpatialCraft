"""Synthetic native-tool/API round trip; separate from all benchmark scoring."""

import argparse
from dataclasses import replace
from pathlib import Path

from PIL import Image

from spatialcraft.agent import (
    ActionParser,
    ExecutionConfig,
    ExecutionLoop,
    StateBuilder,
)
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.react_api import (
    ReActResponsesProvider,
    ScopedToolExecutor,
    TaskMedia,
    ToolOnlyComposer,
)
from spatialcraft.experiments.rollout import JournaledRollout
from spatialcraft.experiments.run import load_api_key_file
from spatialcraft.experiments.run_react_baseline import PrivateReward, trajectory_row
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import atomic_write_json, sha256_file
from spatialcraft.tools import ArtifactStore
from spatialcraft.tools.real import create_real_tool_registry


class ValidationComposer(ToolOnlyComposer):
    def compose(self, task, state):
        request = super().compose(task, state)
        return replace(
            request,
            tool_choice={0: "geometry", 1: "draw"}.get(state.step_index, "none"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = args.output / "red.png"
    if not image.exists():
        Image.new("RGB", (64, 64), "red").save(image)
    reference = TaskSample(
        task_id="synthetic_native_roundtrip",
        dataset="synthetic_validation",
        question=(
            "First call geometry with operation point_distance, first [0,0], second [3,4]. "
            "Next call draw on the provided image with one point [10,10]. "
            "After both tools, answer the distance calculated by geometry as one number."
        ),
        answer_type=AnswerType.NUMERIC,
        reference_answer=5,
        images=(ImageInput(uri=str(image), media_type="image/png"),),
    )
    task_file = args.output / "synthetic_task.json"
    if task_file.exists():
        import json

        reference = TaskSample.from_dict(json.loads(task_file.read_text()))
    else:
        atomic_write_json(task_file, reference.to_dict())
    task = TaskSample.from_dict(public_task(reference))
    config = ModelConfig.from_dict(
        load_yaml(args.project / "configs/models/gpt-5.4-mini-react.yaml")
    )
    layout = StorageLayout(args.output / "tool_store")
    media = TaskMedia(task, {"media": {str(image): sha256_file(image)}}, layout)
    registry = create_real_tool_registry()
    journal = RunJournal(
        args.output / "journal",
        {"kind": "synthetic_tool_api_roundtrip", "model": config.model_id},
    )
    load_api_key_file(Path("/home/xiwei.liu/.config/spatialcraft/openai_api_key"))
    provider = ReActResponsesProvider(config, media)
    loop = ExecutionLoop(
        provider=provider,
        composer=ValidationComposer(config, registry, media=media),
        action_parser=ActionParser(registry),
        tool_executor=ScopedToolExecutor(
            registry, ArtifactStore(layout, "synthetic"), media
        ),
        reward=PrivateReward(reference),
        skill_controller=None,
        config=ExecutionConfig(
            max_steps=4, knowledge_snapshot_id="tool_only_no_knowledge"
        ),
    )
    trajectory = JournaledRollout(loop, journal).run(
        task,
        prefix="check",
        initial_state=StateBuilder().initial(task),
        rollout_index=0,
        random_seed=None,
    )
    report = trajectory_row(task, trajectory, journal, "check")
    assert report["tool_names"] == ["geometry", "draw"], report["tool_names"]
    assert report["reward"] == 1 and report["usage"]["reasoning_tokens"] == 0
    assert report["call_counts"]["normal_llm_calls"] == 3
    third = journal.read_committed("check/steps/0002/decision")["request"]
    image_count = sum(
        p["kind"] == "image" for m in third["messages"] for p in m["content"]
    )
    assert image_count == 2, "Original and tool-generated image must reach next call"
    atomic_write_json(
        args.output / "report.json",
        {
            "validation_status": "passed",
            "synthetic_only": True,
            "image_count_at_final": image_count,
            **report,
        },
    )
    print(
        "PASS: real gpt-5.4-mini -> geometry -> draw -> visual observation -> final answer"
    )


if __name__ == "__main__":
    main()
