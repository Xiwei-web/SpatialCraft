"""Versioned operation-level configuration for the SpatialCraft v2 protocol."""

from dataclasses import asdict, dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class OperationProfile:
    role: str
    reasoning_mode: str
    max_output_tokens: int
    temperature: float
    top_p: float = 1.0
    repair_attempts: int = 1
    repair_max_tokens: int = 4096
    max_input_tokens: int | None = None


DEFAULT_PROFILES = {
    "execution.accumulation": OperationProfile(
        "executor", "instruct", 4096, 0.7, 0.9, 0
    ),
    "execution.deployment": OperationProfile("executor", "instruct", 4096, 0, 1, 0),
    "execution.fallback": OperationProfile("executor", "instruct", 4096, 0, 1, 0),
    "execution.recovery": OperationProfile("executor", "instruct", 1024, 0, 1, 0),
    "execution.forced_final": OperationProfile("executor", "instruct", 512, 0, 1, 0),
    "experience.summary": OperationProfile("knowledge_builder", "thinking", 8192, 0.6),
    "experience.critique": OperationProfile(
        "knowledge_builder", "thinking", 16384, 0.6
    ),
    "experience.merge": OperationProfile(
        "knowledge_builder", "instruct", 1024, 0, repair_max_tokens=1024
    ),
    "experience.manage": OperationProfile("knowledge_builder", "thinking", 8192, 0.6),
    "retrieval.decomposition": OperationProfile(
        "knowledge_builder", "instruct", 1024, 0, repair_max_tokens=1024
    ),
    "retrieval.rewrite": OperationProfile(
        "knowledge_builder", "instruct", 2048, 0, repair_max_tokens=2048
    ),
    "skill.semantic_gradient": OperationProfile(
        "knowledge_builder", "thinking", 8192, 0.6
    ),
    "skill.gradient_aggregation": OperationProfile(
        "knowledge_builder", "thinking", 16384, 0.6
    ),
    "skill.candidate_generation": OperationProfile(
        "knowledge_builder", "thinking", 16384, 0.7
    ),
    "skill.termination": OperationProfile(
        "knowledge_builder", "instruct", 256, 0, repair_max_tokens=256
    ),
    "skill.selection_judge": OperationProfile(
        "knowledge_builder", "instruct", 256, 0, repair_max_tokens=256
    ),
    "skill.deduplication": OperationProfile(
        "knowledge_builder", "instruct", 1024, 0, repair_max_tokens=1024
    ),
    "baseline.reflection": OperationProfile("knowledge_builder", "thinking", 8192, 0.6),
    "baseline.workflow": OperationProfile("knowledge_builder", "thinking", 8192, 0.6),
}

ALLOWED_ABLATIONS = {
    "no_experience",
    "no_skill",
    "no_decomposition",
    "no_rewrite",
    "no_visual_summary",
    "no_critique",
    "no_local_merge",
    "no_manager",
    "no_semantic_gradient",
    "no_gradient_aggregation",
    "no_ppo_gate",
    "no_score_pruning",
    "static_seed_skills",
}


def operation_profile(settings, operation, *, deployment=False):
    if operation not in DEFAULT_PROFILES:
        raise ValueError(f"Unknown operation: {operation}")
    values = asdict(DEFAULT_PROFILES[operation])
    if operation == "execution.accumulation":
        values.update(
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.training_temperature,
            top_p=settings.training_top_p,
        )
    elif operation == "execution.deployment":
        values.update(
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.deployment_temperature,
        )
    elif operation == "skill.candidate_generation":
        values["max_output_tokens"] = settings.skill_generation_max_tokens
    elif operation == "execution.fallback":
        inherited = operation_profile(
            settings, "execution.deployment" if deployment else "execution.accumulation"
        )
        values.update(
            max_output_tokens=inherited.max_output_tokens,
            temperature=inherited.temperature,
            top_p=inherited.top_p,
        )
    values.update(settings.operations.get(operation, {}))
    return OperationProfile(**values)


def resolved_configuration(settings):
    return {
        **settings.to_dict(),
        "roles": {
            "executor": settings.backbone,
            "knowledge_builder": settings.backbone,
            "scorer": settings.backbone,
            "embedding": settings.embedding_model,
            **settings.roles,
        },
        "operations": {
            k: asdict(operation_profile(settings, k)) for k in DEFAULT_PROFILES
        },
        "effective_fallback_profiles": {
            phase: asdict(
                operation_profile(
                    settings, "execution.fallback", deployment=phase == "deployment"
                )
            )
            for phase in ("accumulation", "deployment")
        },
        "skill_generation_max_tokens": operation_profile(
            settings, "skill.candidate_generation"
        ).max_output_tokens,
        "method_decisions": {
            "ratio": settings.skill_options.get("ratio_mode", "sequence"),
            "distribution": "raw_model_likelihood_surrogate",
            "target": "recorded_generated_action_token_span",
            "baseline": "original_task_group_mean",
            "objective_weighting": "equal_trajectory_then_equal_attributed_action",
            "acceptance": "best_candidate_J_minus_parent_J_strictly_positive",
            "skill_selection": "full_pool_llm_applicability_then_embedding_top1",
            "neutral_skill_pruning": False,
            "retrieval_threshold": settings.experience_options.get(
                "retrieval_minimum_score"
            ),
            "rewrite": "one_batch_per_task",
        },
    }


def validate_v2_settings(s):
    for name in (
        "roles",
        "operations",
        "experience_options",
        "skill_options",
        "ablations",
    ):
        if not isinstance(getattr(s, name), dict):
            raise ValueError(f"{name} must be an object")  # noqa: TRY004 - configuration validation
    if not isinstance(s.experiment_name, str) or not s.experiment_name.strip():
        raise ValueError("experiment_name must be nonempty")
    if type(s.enable_thinking) is not bool or type(s.accumulation_passes) is not int:
        raise ValueError("Invalid enable_thinking/accumulation_passes type")
    if any(
        not isinstance(v, str) or not v.strip()
        for v in (s.backbone, s.embedding_model, *s.roles.values())
    ):
        raise ValueError("Model role bindings must be nonempty aliases")
    experience_keys = {
        "max_words",
        "critique_max_ops",
        "merge_cosine_threshold",
        "decomposition_min_aspects",
        "decomposition_max_aspects",
        "retrieval_minimum_score",
    }
    skill_keys = {
        "ratio_mode",
        "stored_skill_max_tokens",
        "minimum_quality_trajectories",
        "deduplication_mode",
        "deduplication_cosine_threshold",
        "preferred_low_reward",
        "preferred_high_reward",
        "maximum_targets_per_round",
        "max_discovery_per_round",
    }
    aliases = {
        "experience_options": {
            "capacity": "experience_capacity",
            "top_k_per_aspect": "experience_top_k_per_subtask",
            "rollouts_per_task": "rollouts_per_task",
        },
        "skill_options": {
            "capacity": "skill_capacity",
            "batch_trajectories": "evolution_batch_trajectories",
            "candidates_per_target": "skill_candidates",
            "max_parent_skills_per_round": "max_parent_skills_per_round",
            "rollouts_per_task": "rollouts_per_task",
            "ppo_clip_epsilon": "ppo_epsilon",
            "acceptance_gain_margin": "ppo_positive_margin",
            "seed": "seed",
        },
    }
    for name, allowed in (
        ("experience_options", experience_keys),
        ("skill_options", skill_keys),
    ):
        value = getattr(s, name)
        duplicate = set(value) & aliases[name].keys()
        if duplicate:
            raise ValueError(
                f"Use canonical top-level settings instead of {name} aliases: "
                + ", ".join(f"{k} -> {aliases[name][k]}" for k in sorted(duplicate))
            )
        if set(value) - allowed:
            raise ValueError(f"Unknown {name}: {sorted(set(value) - allowed)}")
    if set(s.operations) - DEFAULT_PROFILES.keys():
        raise ValueError("Unknown operation profile override")
    profile_keys = {f.name for f in fields(OperationProfile)}
    for name, value in s.operations.items():
        if not isinstance(value, dict) or set(value) - profile_keys:
            raise ValueError(f"Unknown or invalid fields for operation {name}")
    if {"temperature", "top_p"} & s.operations.get("execution.fallback", {}).keys():
        raise ValueError(
            "execution.fallback inherits phase sampling; configure execution.accumulation/deployment temperature and top_p"
        )
    if s.auxiliary_temperature != 0:
        raise ValueError(
            "v2 auxiliary temperature is operation-specific; set operations instead of auxiliary_temperature"
        )
    integer_fields = (
        "rollouts_per_task",
        "evolution_batch_trajectories",
        "max_parent_skills_per_round",
        "experience_top_k_per_subtask",
        "experience_capacity",
        "skill_capacity",
        "skill_candidates",
        "max_steps",
        "max_output_tokens",
        "skill_max_lifetime",
        "image_max_pixels",
        "skill_generation_max_tokens",
    )
    if any(type(getattr(s, k)) is not int or getattr(s, k) < 1 for k in integer_fields):
        raise ValueError("v2 counts and budgets must be positive integers")
    if type(s.seed) is not int or s.seed < 0 or s.accumulation_passes != 1:
        raise ValueError("v2 requires nonnegative seed and one accumulation pass")
    if any(
        not isfinite(float(getattr(s, name)))
        for name in (
            "training_temperature",
            "training_top_p",
            "deployment_temperature",
            "ppo_epsilon",
            "ppo_positive_margin",
        )
    ):
        raise ValueError("Sampling and PPO values must be finite")
    if s.image_max_pixels < 65536:
        raise ValueError("image_max_pixels must be at least 65536")
    if s.max_steps > 50 or s.skill_max_lifetime > 8:
        raise ValueError(
            "Execution limits are 50 tool steps and an 8-step Skill horizon"
        )
    if s.experiment_name == "full" and s.rollouts_per_task != 4:
        raise ValueError("Name a rollout-count ablation explicitly before changing N=4")
    if s.experiment_name == "full" and s.ablations:
        raise ValueError("Ablations need an explicit experiment_name")
    if set(s.ablations) - ALLOWED_ABLATIONS or any(
        type(v) is not bool for v in s.ablations.values()
    ):
        raise ValueError("Unknown ablation or non-boolean switch")
    if s.enable_thinking or s.ppo_thinking_mode != "action_only":
        raise ValueError(
            "v2 main executor uses instruct; knowledge operations have independent modes"
        )
    if (
        not 0 < s.training_temperature <= 2
        or not 0 < s.training_top_p <= 1
        or s.deployment_temperature != 0
    ):
        raise ValueError("Invalid accumulation/deployment sampling")
    if (
        not 0 <= s.ppo_epsilon < 1
        or not isfinite(s.ppo_positive_margin)
        or s.ppo_positive_margin < 0
    ):
        raise ValueError("Invalid PPO gate settings")
    if s.embedding_model not in {"text-embedding-3-small", "text-embedding-3-large"}:
        raise ValueError("Provide a supported embedding model")
    if set(s.roles) - {"executor", "knowledge_builder", "scorer", "embedding"}:
        raise ValueError("Unknown model role")
    if s.roles.get("executor", s.backbone) != s.backbone:
        raise ValueError("backbone must match the executor role")
    if s.roles.get("scorer", s.backbone) != s.backbone:
        raise ValueError("Scorer must use the exact executor alias and weights")
    if s.roles.get("embedding", s.embedding_model) != s.embedding_model:
        raise ValueError("embedding_model must match the embedding role")
    if s.skill_options.get("ratio_mode", "sequence") != "sequence":
        raise ValueError(
            "v2 currently implements sequence likelihood; a normalized variant needs its own gate"
        )
    if set(s.operations) - DEFAULT_PROFILES.keys():
        raise ValueError("Unknown operation profile override")
    for name in DEFAULT_PROFILES:
        p = operation_profile(s, name)
        if p.reasoning_mode not in {"instruct", "thinking"} or p.role not in {
            "executor",
            "knowledge_builder",
        }:
            raise ValueError(f"Invalid operation mode/role: {name}")
        if type(p.max_output_tokens) is not int or p.max_output_tokens < 1:
            raise ValueError(f"Invalid output budget: {name}")
        if (
            type(p.temperature) not in (int, float)
            or type(p.top_p) not in (int, float)
            or not isfinite(p.temperature)
            or not isfinite(p.top_p)
            or not 0 <= p.temperature <= 2
            or not 0 < p.top_p <= 1
        ):
            raise ValueError(f"Invalid operation sampling: {name}")
        if (
            type(p.repair_attempts) is not int
            or p.repair_attempts not in (0, 1)
            or type(p.repair_max_tokens) is not int
            or p.repair_max_tokens < 1
        ):
            raise ValueError(
                "Knowledge repair is bounded to at most one additional call"
            )
        if name in {"execution.recovery", "execution.forced_final"}:
            cap = 1024 if name.endswith("recovery") else 512
            if p.max_output_tokens > cap or p.temperature != 0 or p.top_p != 1:
                raise ValueError(
                    "Recovery/final profiles require bounded deterministic decoding"
                )
        if name.startswith("execution.") and (
            p.reasoning_mode != "instruct"
            or p.role != "executor"
            or p.repair_attempts != 0
        ):
            raise ValueError(
                "Execution requires executor/Instruct and no knowledge JSON repair"
            )
        if name == "execution.accumulation" and not (0 < p.temperature <= 2):
            raise ValueError("Accumulation profile must sample independently")
        if name == "execution.deployment" and p.temperature != 0:
            raise ValueError("Deployment profile must remain deterministic")
        if p.max_input_tokens is not None and (
            type(p.max_input_tokens) is not int or p.max_input_tokens < 1
        ):
            raise ValueError("Invalid input context budget")

    def integer(value, name, minimum=1):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    e, k = s.experience_options, s.skill_options
    for name, default in (
        ("max_words", 64),
        ("critique_max_ops", 4),
        ("decomposition_min_aspects", 1),
        ("decomposition_max_aspects", 3),
    ):
        integer(e.get(name, default), name)
    if e.get("decomposition_min_aspects", 1) > e.get("decomposition_max_aspects", 3):
        raise ValueError("Decomposition minimum exceeds maximum")
    for options, name, default in (
        (e, "merge_cosine_threshold", 0.7),
        (e, "retrieval_minimum_score", None),
        (k, "deduplication_cosine_threshold", 0.9),
    ):
        value = options.get(name, default)
        if value is not None and (
            type(value) not in (int, float)
            or not isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError(f"Invalid cosine threshold: {name}")
    for name, default in (
        ("stored_skill_max_tokens", 1024),
        ("minimum_quality_trajectories", 3),
        ("maximum_targets_per_round", s.max_parent_skills_per_round),
    ):
        integer(k.get(name, default), name)
    for name, default in (
        ("preferred_low_reward", 3),
        ("preferred_high_reward", 3),
        ("max_discovery_per_round", 1),
    ):
        integer(k.get(name, default), name, minimum=0)
    if (
        k.get("preferred_low_reward", 3) + k.get("preferred_high_reward", 3)
        != s.evolution_batch_trajectories
    ):
        raise ValueError(
            "Preferred reward allocation must sum to evolution_batch_trajectories"
        )
    if k.get("max_discovery_per_round", 1) > k.get(
        "maximum_targets_per_round", s.max_parent_skills_per_round
    ):
        raise ValueError("Discovery budget exceeds total target budget")
    if k.get("deduplication_mode", "llm") not in {"llm", "exact"}:
        raise ValueError("deduplication_mode must be llm or exact")
