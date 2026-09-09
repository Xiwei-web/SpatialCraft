from __future__ import annotations

from pathlib import Path

from PIL import Image

from spatialcraft.agent import (
    ActionParser,
    ContextComposer,
    ExecutionConfig,
    ExecutionLoop,
    SpatialAgent,
)
from spatialcraft.models import (
    Capability,
    LocalModelConfig,
    MessageRole,
    ModelCapabilities,
    ModelConfig,
    ModelProvider,
    ModelResponse,
    ProviderKind,
    ResponseToolCall,
)
from spatialcraft.rollout import BatchScheduler, RolloutRunner, TrajectoryRecorder
from spatialcraft.schemas import (
    AnswerType,
    ImageInput,
    TaskSample,
    TrajectoryStatus,
)
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor, create_mock_tool_registry


class _ToolThenAnswerProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        text = "\n".join(message.text_content for message in request.messages)
        assert "Reference answer:" not in text
        tool_messages = [
            message for message in request.messages if message.role is MessageRole.TOOL
        ]
        image_uri = next(
            part.uri
            for message in request.messages
            for part in message.content
            if part.uri is not None
        )
        if not tool_messages:
            return ModelResponse(
                provider="scripted",
                model="mock-mllm",
                tool_calls=(
                    ResponseToolCall(
                        name="detect",
                        arguments={"image_uri": image_uri, "queries": ["square"]},
                    ),
                ),
            )
        if len(tool_messages) == 1:
            return ModelResponse(
                provider="scripted",
                model="mock-mllm",
                tool_calls=(
                    ResponseToolCall(name="depth", arguments={"image_uri": image_uri}),
                ),
            )
        return ModelResponse(provider="scripted", model="mock-mllm", text="A")


def _task(path: Path, suffix: str = "one") -> TaskSample:
    return TaskSample(
        dataset="test",
        task_id=f"task-{suffix}",
        question="Which option identifies the target?",
        images=(ImageInput(uri=str(path), media_type="image/png"),),
        answer_type=AnswerType.MULTIPLE_CHOICE,
        reference_answer="A",
        choices=("square", "circle"),
    )


def _agent(
    tmp_path: Path, run_id: str = "run"
) -> tuple[SpatialAgent, _ToolThenAnswerProvider]:
    registry = create_mock_tool_registry()
    model = ModelConfig(
        alias="mock-mllm",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        model_id="mock-mllm",
        capabilities=ModelCapabilities(
            supported=frozenset(
                {
                    Capability.TEXT_INPUT,
                    Capability.IMAGE_INPUT,
                    Capability.TOOL_CALLING,
                }
            )
        ),
        local=LocalModelConfig(path="."),
    )
    provider = _ToolThenAnswerProvider()
    store = ArtifactStore(StorageLayout(tmp_path / "storage"), run_id)
    loop = ExecutionLoop(
        provider=provider,
        composer=ContextComposer(model, registry),
        action_parser=ActionParser(registry),
        tool_executor=ToolExecutor(registry, store),
        config=ExecutionConfig(max_steps=5),
    )
    return SpatialAgent(loop), provider


def test_multistep_agent_loop_and_jsonl_replay(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (48, 32), "navy").save(image)
    task = _task(image)
    agent, provider = _agent(tmp_path)
    recorder = TrajectoryRecorder(StorageLayout(tmp_path / "storage"), "run")
    trajectories = RolloutRunner(agent, recorder=recorder).run(
        task, num_rollouts=2, seed=41
    )
    assert len(trajectories) == 2
    for index, trajectory in enumerate(trajectories):
        assert trajectory.status is TrajectoryStatus.COMPLETED
        assert trajectory.rollout_index == index
        assert trajectory.random_seed == 41 + index
        assert len(trajectory.transitions) == 3
        assert trajectory.total_tool_calls == 2
        assert trajectory.reward == 1.0
        assert trajectory.final_answer == "A"
        assert all(
            transition.state_before.metadata["reference_answer_exposed"] is False
            for transition in trajectory.transitions
        )
    assert tuple(recorder.replay()) == trajectories
    assert len(provider.requests) == 6


def test_batch_scheduler_preserves_task_order(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "white").save(image)
    tasks = (_task(image, "one"), _task(image, "two"))
    counter = iter(range(10))

    def factory() -> RolloutRunner:
        index = next(counter)
        agent, _ = _agent(tmp_path, f"worker-{index}")
        return RolloutRunner(agent)

    batches = BatchScheduler(factory, max_workers=2).run(tasks, seed=7)
    assert tuple(batch[0].task.task_id for batch in batches) == ("task-one", "task-two")
    assert all(batch[0].status is TrajectoryStatus.COMPLETED for batch in batches)
