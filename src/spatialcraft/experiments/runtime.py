"""Explicit Qwen/embedding/tool wiring; never a fallback to mock knowledge."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from spatialcraft.agent import (
    ActionParser,
    ContextComposer,
    ExecutionConfig,
    ExecutionLoop,
    SkillController,
    StateBuilder,
    TerminationController,
)
from spatialcraft.agent.decision import (
    BUDGET_POLICY,
    FORCED_FINAL_TOKENS,
    RECOVERY_TOKENS,
)
from spatialcraft.knowledge.experience.index import ModelEmbedder
from spatialcraft.knowledge.skill import SkillLifecyclePolicy, SkillSelector
from spatialcraft.models import ModelProvider, SequenceScore
from spatialcraft.models.providers.openai_embeddings import OpenAIEmbeddingsProvider
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.models.serialization import (
    request_to_dict,
    response_from_dict,
    response_to_dict,
)
from spatialcraft.rollout.reward import RewardComputer
from spatialcraft.schemas import Trajectory
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import sha256_file
from spatialcraft.tools import ArtifactStore, ToolExecutor
from spatialcraft.tools.real import create_real_tool_registry

from .accumulation import ProtocolPipeline
from .journal import RunJournal, digest
from .learning import LearningBuilders
from .rollout import JournaledRollout


class AuditedProvider(ModelProvider):
    def __init__(self, provider, journal):
        self.provider, self.journal = provider, journal

    def generate(self, request):
        inputs = {"request": request_to_dict(request, identity=False)}
        result, _ = self.journal.execute(
            "model_calls/" + digest(inputs),
            inputs,
            lambda: response_to_dict(self.provider.generate(request)),
        )
        return response_from_dict(result)

    def score(self, request, target_text):
        inputs = {
            "request": request_to_dict(request, identity=False),
            "target": target_text,
        }
        result, _ = self.journal.execute(
            "ppo_calls/" + digest(inputs),
            inputs,
            lambda: asdict(self.provider.score(request, target_text)),
        )
        return SequenceScore(**result)


class SerialToolExecutor(ToolExecutor):
    """Avoid keeping every spatial model resident beside the Qwen backbone."""

    def execute(self, call, *, context=None, backend="local"):
        try:
            return super().execute(call, context=context, backend=backend)
        finally:
            adapter = getattr(
                self.registry.get(call.tool_name, backend=backend), "adapter", None
            )
            resources = [getattr(adapter, "_runtime", None)]
            resources.extend(getattr(adapter, "_readers", {}).values())
            released = False
            for resource in resources:
                if resource is not None and resource.loaded:
                    resource.release()
                    released = True
            if released:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class ConfiguredLocalProvider(TransformersLocalProvider):
    def __init__(self, config, image_max_pixels):
        super().__init__(config)
        self.image_max_pixels = image_max_pixels

    def _load(self):
        model, processor = super()._load()
        processor.image_processor.size = {
            "shortest_edge": 65536,
            "longest_edge": self.image_max_pixels,
        }
        return model, processor


class ExperimentRuntime:
    def __init__(self, project: Path, output: Path, settings, binding: dict):
        self.project, self.output, self.settings = project, output, settings
        config = ModelConfig.from_dict(
            load_yaml(project / f"configs/models/{settings.backbone}.yaml")
        )
        self.model = replace(
            config,
            local=replace(
                config.local,
                device_map=config.local.device_map
                if config.metadata.get("experiment_auto_placement", False)
                else "cuda:0",
                trust_remote_code=False,
                extra_load_kwargs={
                    **config.local.extra_load_kwargs,
                    "local_files_only": True,
                    "attn_implementation": "sdpa",
                },
            ),
        )
        self.local = ConfiguredLocalProvider(self.model, settings.image_max_pixels)
        emb_config = ModelConfig.from_dict(
            load_yaml(project / "configs/models/text-embedding-3-small.yaml")
        )
        self.embedding = OpenAIEmbeddingsProvider(
            emb_config, cache_dir=output / "embedding_cache"
        )
        self.embedder = ModelEmbedder(self.embedding)
        self.binding = binding
        self.tools = create_real_tool_registry()

    def dataset(self, name: str) -> ProtocolPipeline:
        journal = RunJournal(
            self.output / name,
            {
                **self.binding,
                "dataset": name,
                "settings": self.settings.to_dict(),
                "model_local": asdict(self.model.local),
                "embedding_identity": self.embedder.identity,
                "trajectory_control": {
                    "policy": BUDGET_POLICY,
                    "max_tool_steps": self.settings.max_steps,
                    "actions_per_step": 1,
                    "max_recovery_calls_per_step": 1,
                    "recovery_max_tokens": RECOVERY_TOKENS,
                    "forced_final_max_tokens": FORCED_FINAL_TOKENS,
                },
            },
        )
        provider = AuditedProvider(self.local, journal)
        layout = StorageLayout(journal.root / "tool_store")
        executor = SerialToolExecutor(self.tools, ArtifactStore(layout, name))
        builders = LearningBuilders(
            self.settings,
            self.model,
            provider,
            self.embedder,
            layout.resolve_uri,
            trajectory_loader=lambda key: Trajectory.from_dict(
                journal.read_committed(key)
            ),
        )

        def rollout(task, knowledge, refs, index, seed, prefix, deployment):
            model = replace(
                self.model,
                generation=replace(
                    self.model.generation,
                    max_output_tokens=self.settings.max_output_tokens,
                    temperature=self.settings.deployment_temperature
                    if deployment
                    else self.settings.training_temperature,
                    top_p=1.0 if deployment else self.settings.training_top_p,
                ),
            )

            enable_thinking = self.settings.enable_thinking

            class Composer(ContextComposer):
                def compose(self, task, state):
                    request = super().compose(task, state)
                    return replace(
                        request,
                        metadata={
                            **request.metadata,
                            "chat_template_kwargs": {
                                "enable_thinking": enable_thinking
                            },
                        },
                    )

            system = (
                "Solve the spatial question using observed images and available tools. "
                + (
                    ""
                    if enable_thinking
                    else "Use instruct mode: follow the active Skill directly, emit the next tool action or final answer without a thinking block or extended reasoning preamble. "
                )
                + "Use exact tool schemas. Tool image_uri arguments must use the following input paths: "
                + ", ".join(im.uri for im in task.images)
                + ". Give your final answer on a final line 'Final Answer: ...'. "
                "For boolean tasks answer yes/no; multiple choice use the 1-based choice number shown; "
                "numeric tasks give a number; pointing tasks give normalized [x,y] coordinates in [0,1]."
            )
            loop = ExecutionLoop(
                provider=provider,
                composer=Composer(
                    model,
                    self.tools,
                    system_prompt=system,
                    artifact_resolver=layout.resolve_uri,
                    include_empty_skill_slot=True,
                ),
                action_parser=ActionParser(self.tools),
                tool_executor=executor,
                reward=RewardComputer(binary=True),
                config=ExecutionConfig(
                    max_steps=self.settings.max_steps,
                    knowledge_snapshot_id=knowledge.snapshot_id,
                ),
                skill_controller=SkillController(
                    knowledge.skills,
                    selector=SkillSelector(embedder=self.embedder),
                    termination=TerminationController(
                        SkillLifecyclePolicy(
                            max_lifetime_steps=self.settings.skill_max_lifetime,
                            condition_evaluator=builders.terminate,
                        )
                    ),
                ),
            )
            initial = StateBuilder().initial(
                task.without_reference_answer(), retrieved_experiences=refs
            )
            return JournaledRollout(loop, journal).run(
                task,
                prefix=prefix,
                initial_state=initial,
                rollout_index=index,
                random_seed=seed,
            )

        def validate_artifacts(row):
            for image in row.task.images:
                if image.sha256 and sha256_file(image.uri) != image.sha256:
                    raise ValueError("Task image content changed")
            for step in row.transitions:
                for result in step.tool_results:
                    for artifact in result.artifacts:
                        if (
                            artifact.sha256
                            and sha256_file(layout.resolve_uri(artifact.uri))
                            != artifact.sha256
                        ):
                            raise ValueError("Tool artifact content changed")

        return ProtocolPipeline(
            self.settings,
            journal,
            prepare_experiences=builders.retrieve,
            rollout=rollout,
            update_experiences=builders.update_experiences,
            evolve_skills=builders.evolve,
            artifact_validator=validate_artifacts,
        )
