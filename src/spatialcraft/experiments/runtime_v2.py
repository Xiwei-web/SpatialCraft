"""Production wiring for the explicitly versioned SpatialCraft v2 protocol."""

from dataclasses import asdict, replace
from functools import lru_cache
from time import perf_counter

from spatialcraft.agent import (
    ActionParser,
    ContextComposer,
    ExecutionConfig,
    ExecutionLoop,
    SkillController,
    StateBuilder,
    TerminationController,
)
from spatialcraft.agent.decision import BUDGET_POLICY
from spatialcraft.knowledge.skill import SkillLifecyclePolicy, SkillSelector
from spatialcraft.models.capabilities import Capability
from spatialcraft.models.registry import (
    ModelConfig,
    ModelRegistry,
    ProviderKind,
    load_yaml,
)
from spatialcraft.rollout.reward import RewardComputer
from spatialcraft.schemas import Trajectory
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import atomic_write_json, sha256_file
from spatialcraft.tools import ArtifactStore

from .accumulation import ProtocolPipeline
from .journal import RunJournal
from .knowledge_generator import KnowledgeGenerator
from .learning_v2 import LearningBuildersV2
from .operation_profiles import operation_profile, resolved_configuration
from .reasoning_modes_v2 import validate_operation_model
from .rollout import JournaledRollout
from .usage import AuditedProviderV2, ScopedEmbedder, UsageLedger, _elapsed, cost_report


def build_dataset(runtime, name):
    from .runtime import ConfiguredLocalProvider, SerialToolExecutor

    settings = runtime.settings
    resolved = resolved_configuration(settings)
    if resolved["roles"]["scorer"] != runtime.model.alias:
        raise ValueError(
            "The likelihood denominator must use the exact executor; a different scorer changes the method"
        )
    if not settings.ablations.get("no_ppo_gate"):
        runtime.model.capabilities.require(Capability.FIXED_TARGET_SCORING)
    kb_alias = resolved["roles"]["knowledge_builder"]
    models = {"executor": runtime.model, "scorer": runtime.model}
    raw_providers = {"executor": runtime.local, "scorer": runtime.local}
    if kb_alias == runtime.model.alias:
        models["knowledge_builder"], raw_providers["knowledge_builder"] = (
            runtime.model,
            runtime.local,
        )
    else:
        config = ModelConfig.from_dict(
            load_yaml(runtime.project / f"configs/models/{kb_alias}.yaml")
        )
        if config.provider is ProviderKind.TRANSFORMERS_LOCAL:
            if not config.metadata.get("experiment_auto_placement", False):
                raise ValueError(
                    "A separate local knowledge model needs an explicit safe placement profile"
                )
            provider = ConfiguredLocalProvider(config, settings.image_max_pixels)
        else:
            if config.api is not None:
                config = replace(config, api=replace(config.api, max_retries=0))
            provider = ModelRegistry((config,)).create_provider(config.alias)
        models["knowledge_builder"], raw_providers["knowledge_builder"] = (
            config,
            provider,
        )
    active_operations = {
        op for op in resolved["operations"] if not op.startswith("baseline.")
    }
    if settings.ablations.get("no_experience"):
        active_operations = {
            op
            for op in active_operations
            if not op.startswith(("experience.", "retrieval."))
        }
    if settings.ablations.get("no_skill"):
        active_operations = {
            op for op in active_operations if not op.startswith("skill.")
        }
    elif settings.ablations.get("static_seed_skills"):
        active_operations -= {
            "skill.semantic_gradient",
            "skill.gradient_aggregation",
            "skill.candidate_generation",
            "skill.deduplication",
        }
    for flag, operation in {
        "no_semantic_gradient": "skill.semantic_gradient",
        "no_gradient_aggregation": "skill.gradient_aggregation",
        "no_decomposition": "retrieval.decomposition",
        "no_critique": "experience.critique",
        "no_rewrite": "retrieval.rewrite",
        "no_local_merge": "experience.merge",
        "no_manager": "experience.manage",
    }.items():
        if settings.ablations.get(flag):
            active_operations.discard(operation)
    if settings.skill_options.get("deduplication_mode", "llm") == "exact":
        active_operations.discard("skill.deduplication")
    for operation in sorted(active_operations):
        profile = operation_profile(settings, operation)
        validate_operation_model(models[profile.role], profile)
    resolved["active_operations"] = sorted(active_operations)
    from .model_resources import validate_runtime_model_resources

    resource_audit = validate_runtime_model_resources(models, runtime.binding)
    journal = RunJournal(
        runtime.output / name,
        {
            **runtime.binding,
            "dataset": name,
            "settings": resolved,
            "model_local": asdict(runtime.model.local),
            "embedding_identity": runtime.embedder.identity,
            "model_resource_verification": resource_audit,
            "role_models": {
                role: {
                    "alias": config.alias,
                    "model_id": config.model_id,
                    "provider": config.provider.value,
                    "capabilities": config.capabilities.to_dict(),
                    "generation": asdict(config.generation),
                    "metadata": config.metadata,
                    "local": asdict(config.local) if config.local else None,
                    "api": asdict(config.api) if config.api else None,
                }
                for role, config in models.items()
            },
            "trajectory_control": {
                "policy": BUDGET_POLICY,
                "max_tool_steps": settings.max_steps,
                "actions_per_step": 1,
                "max_recovery_calls_per_step": 1,
            },
        },
    )
    providers = {
        role: AuditedProviderV2(provider, journal, role)
        for role, provider in raw_providers.items()
    }
    generator = KnowledgeGenerator(
        settings, models, providers, runtime.project, journal
    )
    embedder = ScopedEmbedder(runtime.embedder, journal, lambda: generator.scope)
    layout = StorageLayout(journal.root / "tool_store")
    ledger = UsageLedger(journal.root)

    class AuditedTools(SerialToolExecutor):
        def execute(self, call, **kwargs):
            started = perf_counter()
            try:
                result = super().execute(call, **kwargs)
            except Exception as exc:
                ledger.record(
                    {
                        **generator.scope,
                        "kind": "tool",
                        "operation": call.tool_name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "actual_call": True,
                        **_elapsed(started),
                    }
                )
                raise
            ledger.record(
                {
                    **generator.scope,
                    "kind": "tool",
                    "operation": call.tool_name,
                    "status": result.status.value,
                    "actual_call": True,
                    **_elapsed(started),
                }
            )
            return result

    executor = AuditedTools(runtime.tools, ArtifactStore(layout, name))

    @lru_cache(maxsize=1)
    def tokenizer():
        from transformers import AutoTokenizer

        local = runtime.model.local
        return AutoTokenizer.from_pretrained(
            local.tokenizer_path or local.path,
            local_files_only=True,
            trust_remote_code=False,
            revision=local.revision,
        )

    builders = LearningBuildersV2(
        settings,
        runtime.model,
        providers["scorer"],
        embedder,
        layout.resolve_uri,
        generator,
        lambda text: len(tokenizer().encode(text, add_special_tokens=False)),
        trajectory_loader=lambda key: Trajectory.from_dict(journal.read_committed(key)),
        tools=runtime.tools.definitions(),
    )

    def rollout(task, knowledge, refs, index, seed, prefix, deployment):
        phase = "deployment" if deployment else "accumulation_execution"
        operation = "execution.deployment" if deployment else "execution.accumulation"
        builders.set_scope(
            phase=phase,
            task_id=task.task_id,
            snapshot_id=knowledge.snapshot_id,
            rollout_index=index,
            rollout_prefix=prefix,
            operation=operation,
        )
        profile = operation_profile(settings, operation)
        model = replace(
            runtime.model,
            generation=replace(
                runtime.model.generation,
                max_output_tokens=profile.max_output_tokens,
                temperature=profile.temperature,
                top_p=profile.top_p,
            ),
        )

        class Composer(ContextComposer):
            def compose(self, task, state):
                request = super().compose(task, state)
                effective_operation = (
                    "execution.fallback" if state.active_skill is None else operation
                )
                effective_profile = operation_profile(
                    settings, effective_operation, deployment=deployment
                )
                return replace(
                    request,
                    settings=replace(
                        request.settings,
                        max_output_tokens=effective_profile.max_output_tokens,
                        temperature=effective_profile.temperature,
                        top_p=effective_profile.top_p,
                    ),
                    metadata={
                        **request.metadata,
                        **generator.scope,
                        "protocol_version": settings.protocol_version,
                        "operation": effective_operation,
                        "operation_profile": asdict(effective_profile),
                        "effective_reasoning_mode": "instruct",
                        "chat_template_kwargs": {"enable_thinking": False},
                        "recovery_profiles": {
                            k: asdict(operation_profile(settings, "execution." + k))
                            for k in ("recovery", "forced_final")
                        },
                    },
                )

        system = (
            "Solve the spatial question using observed images and available tools. "
            "Follow the active Skill when its conditions apply, using current evidence to resolve conflicts. "
            "Emit one next tool action or a final answer without a thinking block. "
            "Use exact tool schemas. Input image paths: "
            + ", ".join(im.uri for im in task.images)
            + ". Give the final answer on a line 'Final Answer: ...'. For boolean tasks answer yes/no; "
            "multiple choice use the displayed 1-based choice number; numeric give a number; "
            "pointing use normalized [x,y] coordinates. Artifact URIs returned by tools may be used as downstream inputs. "
            "Respect coordinate frames, units and scale status; reconstruction units are not meters unless metric scale is established."
        )
        controller = (
            None
            if settings.ablations.get("no_skill")
            else SkillController(
                knowledge.skills,
                selector=SkillSelector(
                    embedder=embedder, applicability_judge=builders.applicable
                ),
                termination=TerminationController(
                    SkillLifecyclePolicy(
                        max_lifetime_steps=settings.skill_max_lifetime,
                        task_condition_evaluator=builders.terminate,
                    )
                ),
            )
        )
        loop = ExecutionLoop(
            provider=providers["executor"],
            composer=Composer(
                model,
                runtime.tools,
                system_prompt=system,
                artifact_resolver=layout.resolve_uri,
                include_empty_skill_slot=True,
            ),
            action_parser=ActionParser(runtime.tools),
            tool_executor=executor,
            reward=RewardComputer(binary=True),
            config=ExecutionConfig(
                max_steps=settings.max_steps,
                knowledge_snapshot_id=knowledge.snapshot_id,
            ),
            skill_controller=controller,
        )
        initial = StateBuilder().initial(
            task.without_reference_answer(), retrieved_experiences=refs
        )
        result = JournaledRollout(loop, journal).run(
            task,
            prefix=prefix,
            initial_state=initial,
            rollout_index=index,
            random_seed=seed,
        )
        atomic_write_json(
            journal.root / "results/usage.json", cost_report(journal.root)
        )
        return result

    def validate_artifacts(row):
        for image in row.task.images:
            if image.sha256 and sha256_file(image.uri) != image.sha256:
                raise ValueError("Task image content changed")
        for transition in row.transitions:
            for result in transition.tool_results:
                for artifact in result.artifacts:
                    if (
                        artifact.sha256
                        and sha256_file(layout.resolve_uri(artifact.uri))
                        != artifact.sha256
                    ):
                        raise ValueError("Tool artifact content changed")

    pipeline = ProtocolPipeline(
        settings,
        journal,
        prepare_experiences=builders.retrieve,
        rollout=rollout,
        update_experiences=builders.update_experiences,
        evolve_skills=builders.evolve,
        artifact_validator=validate_artifacts,
    )
    pipeline.learning_builders = builders
    pipeline.builders = builders
    pipeline.generate_knowledge = generator
    pipeline.embedder = embedder
    pipeline.artifact_resolver = layout.resolve_uri
    pipeline.metric_token_counter = lambda text: len(
        tokenizer().encode(text, add_special_tokens=False)
    )
    pipeline.metric_tokenizer_id = {
        "model_alias": runtime.model.alias,
        "path": str(runtime.model.local.tokenizer_path or runtime.model.local.path),
        "revision": runtime.model.local.revision,
        "add_special_tokens": False,
    }
    return pipeline
