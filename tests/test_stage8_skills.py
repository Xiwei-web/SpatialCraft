from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from spatialcraft.agent import (
    ActionParser,
    ContextComposer,
    ExecutionConfig,
    ExecutionLoop,
    SkillController,
    StateBuilder,
    TerminationController,
)
from spatialcraft.knowledge.skill import (
    SeedCatalog,
    SkillLifecyclePolicy,
    SkillSelector,
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
from spatialcraft.schemas import (
    AgentAction,
    AnswerType,
    ImageInput,
    TaskSample,
    ToolCall,
    TrajectoryStatus,
)
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor, create_mock_tool_registry


class _ScriptedProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        tool_messages = [m for m in request.messages if m.role is MessageRole.TOOL]
        image_uri = next(
            part.uri
            for message in request.messages
            for part in message.content
            if part.uri is not None
        )
        if len(tool_messages) < 2:
            name = "detect" if not tool_messages else "depth"
            arguments = {"image_uri": image_uri}
            if name == "detect":
                arguments["queries"] = ["target"]
            return ModelResponse(
                provider="scripted",
                model="mock",
                tool_calls=(ResponseToolCall(name=name, arguments=arguments),),
            )
        return ModelResponse(provider="scripted", model="mock", text="A")


def _task(image: Path) -> TaskSample:
    return TaskSample(
        dataset="test",
        task_id="skill-task",
        question="Which option describes the spatial relation of the target?",
        images=(ImageInput(uri=str(image), media_type="image/png"),),
        answer_type=AnswerType.MULTIPLE_CHOICE,
        choices=("left", "right"),
        reference_answer="A",
    )


def _run(tmp_path: Path, *, controller: SkillController | None):
    registry = create_mock_tool_registry()
    model = ModelConfig(
        alias="mock",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        model_id="mock",
        capabilities=ModelCapabilities(
            supported=frozenset(
                {Capability.TEXT_INPUT, Capability.IMAGE_INPUT, Capability.TOOL_CALLING}
            )
        ),
        local=LocalModelConfig(path="."),
    )
    provider = _ScriptedProvider()
    loop = ExecutionLoop(
        provider=provider,
        composer=ContextComposer(model, registry),
        action_parser=ActionParser(registry),
        tool_executor=ToolExecutor(
            registry, ArtifactStore(StorageLayout(tmp_path / "storage"), "skills")
        ),
        config=ExecutionConfig(max_steps=5),
        skill_controller=controller,
    )
    return loop.run(_task(tmp_path / "input.png")), provider


def test_termination_uses_new_observation_and_allows_same_skill_reactivation(tmp_path):
    pool = SeedCatalog.pool()
    parent = pool.active()[0]
    seen = []

    def terminate(skill, state, action):
        seen.append(state.metadata.get("observed_done", False))
        return seen[-1]

    controller = SkillController(
        pool,
        selector=SkillSelector(
            scorer=lambda skill, task, state: float(skill.reference == parent.reference)
        ),
        termination=TerminationController(
            SkillLifecyclePolicy(condition_evaluator=terminate)
        ),
    )
    task = _task(tmp_path / "input.png")
    state = controller.before_step(task, StateBuilder().initial(task))
    action = AgentAction.tool(ToolCall(tool_name="detect"))
    continued = controller.after_step(task, state, action, state.next_step())
    assert continued.active_skill is not None and continued.active_skill.age_steps == 1
    after = replace(
        continued.next_step(), metadata={**continued.metadata, "observed_done": True}
    )
    terminated = controller.after_step(task, continued, action, after)
    assert seen == [False, True] and terminated.active_skill is None
    reactivated = controller.before_step(task, terminated)
    assert reactivated.active_skill.skill_id == parent.skill_id
    assert reactivated.active_skill.activated_at_step == 2


def test_seed_pool_selection_persistence_and_termination(tmp_path) -> None:
    Image.new("RGB", (24, 24), "gray").save(tmp_path / "input.png")
    pool = SeedCatalog.pool()
    assert {item.name for item in pool.active()} == {
        "StructuredCoT",
        "ReActDecision",
        "HypothesisElimination",
        "SelfConsistencyCheck",
        "ExploreExploitArbitration",
        "StrategicPlanning",
    }
    controller = SkillController(
        pool,
        selector=SkillSelector(
            scorer=lambda skill, task, state: (
                1.0 if skill.name == "StructuredCoT" else 0.0
            )
        ),
        termination=TerminationController(SkillLifecyclePolicy(max_lifetime_steps=5)),
    )
    trajectory, provider = _run(tmp_path, controller=controller)
    assert trajectory.status is TrajectoryStatus.COMPLETED
    refs = [transition.active_skill for transition in trajectory.transitions]
    assert all(ref is not None for ref in refs)
    assert {(ref.skill_id, ref.version) for ref in refs if ref} == {
        ("skill_seed_structured_cot", 1)
    }
    assert [ref.age_steps for ref in refs if ref] == [0, 1, 2]
    assert trajectory.transitions[-1].state_after.active_skill is None
    assert (
        trajectory.transitions[-1].state_after.metadata["last_skill_termination_reason"]
        == "agent_final_answer"
    )
    prompts = [
        "\n".join(message.text_content for message in request.messages)
        for request in provider.requests
    ]
    assert all("Active procedural skill:" in prompt for prompt in prompts)


def test_max_lifetime_and_disabled_vanilla_equivalence(tmp_path) -> None:
    Image.new("RGB", (24, 24), "gray").save(tmp_path / "input.png")
    pool = SeedCatalog.pool()
    state = StateBuilder().initial(_task(tmp_path / "input.png"))
    controller = SkillController(
        pool,
        termination=TerminationController(SkillLifecyclePolicy(max_lifetime_steps=1)),
    )
    selected = controller.before_step(
        _task(tmp_path / "input.png").without_reference_answer(), state
    )
    retired = controller.after_step(
        _task(tmp_path / "input.png").without_reference_answer(),
        selected,
        AgentAction.noop(),
        selected.next_step(),
    )
    assert retired.active_skill is None
    assert retired.metadata["last_skill_termination_reason"] == "maximum_lifetime"

    (tmp_path / "vanilla").mkdir()
    Image.new("RGB", (24, 24), "gray").save(tmp_path / "vanilla" / "input.png")
    vanilla, _ = _run(tmp_path / "vanilla", controller=None)
    (tmp_path / "disabled").mkdir()
    Image.new("RGB", (24, 24), "gray").save(tmp_path / "disabled" / "input.png")
    disabled, disabled_provider = _run(
        tmp_path / "disabled", controller=SkillController(pool, enabled=False)
    )
    assert [item.action.action_type for item in disabled.transitions] == [
        item.action.action_type for item in vanilla.transitions
    ]
    assert [item.action.final_answer for item in disabled.transitions] == [
        item.action.final_answer for item in vanilla.transitions
    ]
    assert all(item.active_skill is None for item in disabled.transitions)
    assert all(
        "Active procedural skill:"
        not in "\n".join(m.text_content for m in request.messages)
        for request in disabled_provider.requests
    )
