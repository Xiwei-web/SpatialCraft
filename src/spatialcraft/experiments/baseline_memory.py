"""Independent, explicitly adapted memory baselines using real rollout callbacks.

These are executable paper-inspired comparisons, not claims of faithful paper
reproduction. Every branch declares its storage, retrieval and update mechanism.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

from spatialcraft.agent.decision import is_controlled_failure
from spatialcraft.knowledge.evidence import (
    render_public_task,
    render_transition,
)
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.experience.index import ExperienceIndex
from spatialcraft.knowledge.experience.learning_v2 import (
    ExperienceLearningV2,
    word_count,
)
from spatialcraft.knowledge.skill import SeedCatalog, SkillPool
from spatialcraft.models import ContentPart
from spatialcraft.schemas import (
    ExperienceItem,
    ExperienceOperationType,
    ExperienceProvenance,
    ExperienceUpdate,
    RetrievedExperienceRef,
    SkillEvolutionType,
    SkillItem,
    SkillStats,
    TaskSplit,
    Trajectory,
    TrajectoryStatus,
)
from spatialcraft.schemas._base import utc_now
from spatialcraft.storage.atomic_io import atomic_write_json, file_lock

from .accumulation import KnowledgeState
from .journal import digest
from .knowledge_generator import KnowledgeValidationError

METHODS = {
    "rag_demonstrations": {
        "label": "RAG with verified spatial demonstrations",
        "available": True,
        "adaptation": "Build successful tool-trajectory demonstrations without memory during corpus construction; frozen semantic Top-k retrieval at deployment.",
    },
    "memp_reflection": {
        "label": "MemP-inspired procedural reflection",
        "available": True,
        "adaptation": "Independently reflect on each verified success or failure into a general procedure; semantic retrieval, without cross-rollout critique or PPO.",
    },
    "memrl_reward": {
        "label": "MemRL-inspired empirical-return reranking",
        "available": True,
        "adaptation": "Semantic candidate retrieval followed by cosine/empirical-return reranking; injected-memory outcomes are associations, not causal gains or a reproduction of a published Q-learning update.",
    },
    "sma_procedure": {
        "label": "SMA-inspired verified procedure memory",
        "available": True,
        "adaptation": "Write procedures only from successful training rollouts, calibrate reliability from later training injections, rank cosine times Laplace-smoothed success rate; one pass and final checkpoint.",
    },
    "xskill_dual_memory": {
        "label": "XSkill-inspired Experience plus workflow memory",
        "available": True,
        "adaptation": "Visual Experience learning plus separate LLM workflow add/modify operations; no semantic-gradient or PPO gate. Workflows use SpatialCraft initiation/policy/termination execution slots.",
    },
    "skill_pro_sequence": {
        "label": "Skill-Pro-inspired spatial sequence-gate adapter",
        "available": True,
        "adaptation": "Six seed skills, semantic diagnostics and independent Skill evolution with fixed-action sequence likelihood surrogate; no Experience bank or retrieval. Uses the declared SpatialCraft spatial/tool adaptation.",
    },
    "memrl_gt": {
        "label": "MemRL-GT (protocol unresolved)",
        "available": False,
        "adaptation": "GT variant semantics and an explicit oracle/non-oracle evaluation contract have not been established. It is unavailable rather than silently exposing deployment labels to retrieval.",
    },
}


def method_description(name):
    if name not in METHODS:
        raise ValueError(f"Unknown memory baseline: {name}")
    return {
        "method": name,
        **METHODS[name],
        "fidelity": "paper_inspired_spatial_adapter",
        "faithful_reproduction": False,
        "oracle": False,
    }


@dataclass(frozen=True)
class MemoryBaselineConfig:
    method: str
    memory_capacity: int = 100
    skill_capacity: int = 20
    retrieval_top_k: int = 3
    semantic_candidates: int = 10
    utility_weight: float = 0.5
    memory_max_words: int = 256
    demonstration_max_tokens: int = 1024
    workflow_max_words: int = 512
    workflow_max_operations: int = 2

    def __post_init__(self):
        description = method_description(self.method)
        if not description["available"]:
            raise ValueError(description["adaptation"])
        if (
            min(
                self.memory_capacity,
                self.skill_capacity,
                self.retrieval_top_k,
                self.semantic_candidates,
                self.memory_max_words,
                self.demonstration_max_tokens,
                self.workflow_max_words,
                self.workflow_max_operations,
            )
            < 1
        ):
            raise ValueError("Memory baseline limits must be positive")
        if self.method == "skill_pro_sequence" and self.skill_capacity < len(
            SeedCatalog.pool().active()
        ):
            raise ValueError("Skill-Pro capacity must fit all six initial seed skills")
        if (
            self.semantic_candidates < self.retrieval_top_k
            or not 0 <= self.utility_weight <= 1
        ):
            raise ValueError("Invalid candidate count or utility weight")


def render_demonstration(row):
    """Render public task/actions/observations/final, without incidental run IDs.

    Artifact and frame references are stable local aliases so necessary dependency
    links survive without corpus differences caused by random artifact identities.
    Observations are not silently truncated; admission budgets this rendered text.
    """
    aliases = {
        image.uri: f"task_image_{i + 1}" for i, image in enumerate(row.task.images)
    }
    frame_index, artifact_index = 0, 0
    for step in row.transitions:
        for result in step.tool_results:
            for frame in result.coordinate_frames:
                if frame.frame_id not in aliases:
                    frame_index += 1
                    aliases[frame.frame_id] = f"frame_{frame_index}"
            for artifact in result.artifacts:
                if artifact.uri not in aliases:
                    artifact_index += 1
                    aliases[artifact.uri] = f"tool_artifact_{artifact_index}"
                aliases[artifact.artifact_id] = aliases[artifact.uri]

    def local_refs(value):
        if isinstance(value, str):
            if value in aliases:
                return aliases[value]
            # Tool summaries sometimes interpolate an artifact URI inside prose.
            for old in sorted(aliases, key=len, reverse=True):
                if len(old) > 12:
                    value = value.replace(old, aliases[old])
            return value
        if isinstance(value, dict):
            return {key: local_refs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [local_refs(item) for item in value]
        return value

    payload = {
        "task": render_public_task(row.task),
        "steps": [render_transition(step) for step in row.transitions],
        "final_answer": row.final_answer,
    }
    return json.dumps(local_refs(payload), ensure_ascii=False, sort_keys=True)


class MemoryBaselineLearner:
    def __init__(
        self,
        config,
        *,
        generate,
        embedder,
        artifact_resolver=None,
        evolve_skills=None,
        experience_config=None,
        skill_ratio_mode="sequence",
        token_counter=None,
        tokenizer_id=None,
    ):
        self.config, self.generate, self.embedder = config, generate, embedder
        self.resolve = artifact_resolver or (lambda uri: uri)
        self.evolve_skills = evolve_skills
        self.token_counter, self.tokenizer_id = token_counter, tokenizer_id
        if config.method == "skill_pro_sequence" and (
            evolve_skills is None or skill_ratio_mode != "sequence"
        ):
            raise ValueError(
                "Skill-Pro adapter requires the wired fixed-action sequence-gate evolver"
            )
        self.experience = ExperienceLearningV2(
            generate, embedder, experience_config, self.resolve
        )

    def memory_budget_definition(self):
        token_based = (
            self.config.method == "rag_demonstrations"
            and self.token_counter is not None
        )
        return {
            "unit": "token" if token_based else "word",
            "limit": self.config.demonstration_max_tokens
            if token_based
            else self.config.memory_max_words,
            "measured_text": "exact ExperienceItem.prompt_text injected by retrieval",
            "tokenizer": self.tokenizer_id if token_based else None,
            "fallback": None
            if token_based
            else "explicit_whitespace_words_no_tokenizer",
            "overflow": "reject_whole_demonstration_or_memory_without_truncation",
        }

    def _memory_budget(self, proposal):
        definition = self.memory_budget_definition()
        measured = (
            int(self.token_counter(proposal.prompt_text))
            if definition["unit"] == "token"
            else word_count(proposal.prompt_text)
        )
        return {**definition, "measured": measured}

    def initial(self):
        skills = (
            SeedCatalog.pool()
            if self.config.method == "skill_pro_sequence"
            else SkillPool()
        )
        return KnowledgeState(ExperienceBank().freeze(), skills.freeze())

    def prepare(self, task, knowledge, *, deployment):
        task = task.without_reference_answer()
        method = self.config.method
        if method == "skill_pro_sequence" or (
            method == "rag_demonstrations" and not deployment
        ):
            return {
                "experiences": [],
                "retrieval_audit": {"status": "disabled_by_method", "method": method},
            }
        if method == "xskill_dual_memory":
            try:
                refs = self.experience.retrieve(task, knowledge)
            except KnowledgeValidationError as error:
                return {
                    "experiences": [],
                    "retrieval_audit": {
                        **self.experience.last_retrieval_audit,
                        "injected_refs": [],
                        "status": "rejected_model_output",
                        "error": str(error),
                    },
                }
            return {
                "experiences": [item.to_dict() for item in refs],
                "retrieval_audit": self.experience.last_retrieval_audit,
            }
        bank = knowledge.experiences
        if not bank.active():
            return {
                "experiences": [],
                "retrieval_audit": {"status": "skipped_empty_bank"},
            }
        index = ExperienceIndex.build(bank, self.embedder)
        rerank = method in {"memrl_reward", "sma_procedure"}
        candidates = index.search(
            task.question,
            self.embedder,
            top_k=self.config.semantic_candidates
            if rerank
            else self.config.retrieval_top_k,
        )
        ranked = []
        for reference, cosine in candidates:
            item = bank.get(reference)
            outcomes = item.metadata.get("baseline_outcomes", {})
            count, total = len(outcomes), sum(outcomes.values())
            reliability = (total + 1.0) / (count + 2.0)
            empirical = total / count if count else 0.5
            if method == "memrl_reward":
                score = (
                    1.0 - self.config.utility_weight
                ) * cosine + self.config.utility_weight * empirical
            elif method == "sma_procedure":
                score = max(0.0, cosine) * reliability
            else:
                score = cosine
            ranked.append((reference, cosine, score, count, reliability))
        ranked.sort(key=lambda value: (-value[2], -value[1], value[0]))
        chosen = ranked[: self.config.retrieval_top_k]
        refs = [
            RetrievedExperienceRef(
                experience_id=(item := bank.get(ref)).experience_id,
                version=item.version,
                retrieval_score=max(-1.0, min(1.0, cosine)),
                original_text=item.prompt_text,
                matched_subtasks=(task.question,),
            )
            for ref, cosine, _, _, _ in chosen
        ]
        return {
            "experiences": [item.to_dict() for item in refs],
            "retrieval_audit": {
                "status": "completed",
                "task_id": task.task_id,
                "snapshot_id": knowledge.snapshot_id,
                "method": method,
                "candidates": [
                    {
                        "reference": ref,
                        "cosine": cosine,
                        "ranking_score": score,
                        "associated_outcome_count": count,
                        "smoothed_success_rate": reliability,
                    }
                    for ref, cosine, score, count, reliability in ranked
                ],
                "injected_refs": [ref for ref, *_ in chosen],
                "uses_reference_answer": False,
            },
        }

    @staticmethod
    def _observed_usage(bank, rows):
        items = []
        for item in bank.all():
            outcomes = dict(item.metadata.get("baseline_outcomes", {}))
            for row in rows:
                refs = {
                    f"{ref.experience_id}@{ref.version}"
                    for transition in row.transitions
                    for ref in transition.state_before.retrieved_experiences
                }
                if item.reference in refs:
                    outcomes.setdefault(row.trajectory_id, float(row.reward))
            items.append(
                replace(item, metadata={**item.metadata, "baseline_outcomes": outcomes})
            )
        return ExperienceBank(tuple(items))

    def _reflection(self, row):
        allowed = {
            row.trajectory_id,
            *(
                f"{row.trajectory_id}:step-{step.step_index}"
                for step in row.transitions
            ),
        }

        def validate(value):
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("memories"), list)
                or len(value["memories"]) > 1
            ):
                raise ValueError("Reflection requires zero or one procedural memory")
            for memory in value["memories"]:
                if (
                    not isinstance(memory, dict)
                    or not isinstance(memory.get("condition"), str)
                    or not memory["condition"].strip()
                ):
                    raise ValueError("Memory condition must be nonempty")
                procedure = memory.get("procedure")
                if (
                    not isinstance(procedure, list)
                    or not procedure
                    or any(
                        not isinstance(step, str) or not step.strip()
                        for step in procedure
                    )
                ):
                    raise ValueError("Memory needs nonempty procedural steps")
                refs = memory.get("evidence_refs")
                if (
                    not isinstance(refs, list)
                    or not refs
                    or any(
                        not isinstance(ref, str) or ref not in allowed for ref in refs
                    )
                ):
                    raise ValueError(
                        "Memory evidence must reference this completed trajectory"
                    )
                if (
                    word_count(memory["condition"])
                    + sum(word_count(step) for step in procedure)
                    > self.config.memory_max_words
                ):
                    raise ValueError("Procedural memory exceeds word budget")
            return value

        media, manifest = self.experience._media(row.task, row.transitions)
        value = self.generate(
            "baseline.reflection",
            {
                "method": method_description(self.config.method),
                "task": {
                    **render_public_task(row.task),
                    "reference_answer": row.task.reference_answer,
                },
                "trajectory_id": row.trajectory_id,
                "reward": row.reward,
                "verifier": row.verifier.to_dict() if row.verifier else None,
                "image_manifest": manifest,
                "steps": [render_transition(step) for step in row.transitions],
                "max_words": self.config.memory_max_words,
                "expected_output": {
                    "memories": [
                        {
                            "condition": "When ...",
                            "procedure": ["Check ..."],
                            "evidence_refs": [row.trajectory_id],
                        }
                    ]
                },
            },
            media=media,
            validator=validate,
        )
        validate(value)
        return value["memories"]

    def _memory(self, row, memory, index):
        action = "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(memory["procedure"])
        )
        identity = digest([self.config.method, row.trajectory_id, index, memory])[:24]
        return ExperienceItem(
            experience_id="baseline_" + identity,
            condition=memory["condition"],
            action=action,
            provenance=ExperienceProvenance(
                trajectory_ids=(row.trajectory_id,),
                task_ids=(row.task.task_id,),
                datasets=(row.task.dataset,),
                rollout_indices=(row.rollout_index,),
            ),
            metadata={
                "baseline_method": self.config.method,
                "source_reward": row.reward,
                "evidence_refs": memory.get("evidence_refs", []),
                "baseline_outcomes": {},
            },
        )

    def _workflows(self, knowledge, rows, round_index):
        pool = SkillPool(knowledge.skills.all())
        active = {skill.reference: skill for skill in pool.active()}

        def validate(value):
            operations = value.get("operations") if isinstance(value, dict) else None
            if (
                not isinstance(operations, list)
                or len(operations) > self.config.workflow_max_operations
            ):
                raise ValueError("Invalid number of workflow operations")
            modified = set()
            for operation in operations:
                if not isinstance(operation, dict) or operation.get("type") not in {
                    "add",
                    "modify",
                }:
                    raise ValueError("Workflow supports add/modify only")
                if operation["type"] == "modify":
                    ref = operation.get("target_ref")
                    if ref not in active or ref in modified:
                        raise ValueError(
                            "Invalid or repeated workflow modify reference"
                        )
                    modified.add(ref)
                for key in ("name", "initiation", "termination"):
                    if (
                        not isinstance(operation.get(key), str)
                        or not operation[key].strip()
                    ):
                        raise ValueError("Workflow fields must be nonempty")
                policy = operation.get("policy")
                if (
                    not isinstance(policy, list)
                    or not policy
                    or any(
                        not isinstance(step, str) or not step.strip() for step in policy
                    )
                ):
                    raise ValueError("Workflow policy requires steps")
                if (
                    sum(
                        word_count(operation[key])
                        for key in ("name", "initiation", "termination")
                    )
                    + sum(word_count(step) for step in policy)
                    > self.config.workflow_max_words
                ):
                    raise ValueError("Workflow exceeds word budget")
            return value

        media = tuple(
            ContentPart.image_uri(image.uri, mime_type=image.media_type)
            for image in rows[0].task.images
        )
        value = self.generate(
            "baseline.workflow",
            {
                "method": method_description(self.config.method),
                "task": {
                    **render_public_task(rows[0].task),
                    "reference_answer": rows[0].task.reference_answer,
                },
                "rollouts": [
                    {
                        "trajectory_id": row.trajectory_id,
                        "reward": row.reward,
                        "steps": [render_transition(step) for step in row.transitions],
                    }
                    for row in rows
                ],
                "existing_workflows": [
                    {"reference": ref, "definition": skill.format_for_prompt()}
                    for ref, skill in active.items()
                ],
                "max_words": self.config.workflow_max_words,
                "max_operations": self.config.workflow_max_operations,
                "expected_output": {
                    "operations": [
                        {
                            "type": "add",
                            "name": "Resolve frame",
                            "initiation": "When ...",
                            "policy": ["Check ..."],
                            "termination": "Once ...",
                        }
                    ]
                },
            },
            media=media,
            validator=validate,
        )
        validate(value)
        for index, operation in enumerate(value["operations"]):
            fields = {
                key: operation[key] for key in ("name", "initiation", "termination")
            }
            fields["policy"] = tuple(operation["policy"])
            if operation["type"] == "modify":
                source = active[operation["target_ref"]]
                proposal = replace(
                    source,
                    **fields,
                    version=source.version + 1,
                    parent_skill_ref=source.reference,
                    evolution_type=SkillEvolutionType.REFINE,
                    stats=SkillStats(),
                    updated_at=utc_now(),
                )
                pool = pool.refine(proposal)
            else:
                proposal = SkillItem(
                    **fields,
                    skill_id="workflow_" + digest([round_index, index, operation])[:24],
                    metadata={
                        "baseline_method": self.config.method,
                        "source_trajectory_ids": [row.trajectory_id for row in rows],
                    },
                )
                pool = pool.add(proposal)
        archives = []
        for skill in sorted(
            pool.active(), key=lambda value: (value.created_at, value.reference)
        )[: max(0, len(pool.active()) - self.config.skill_capacity)]:
            pool = pool.archive(skill.reference)
            archives.append(skill.reference)
        return pool, {
            "llm_decision": value,
            "capacity_policy": "oldest_created_first",
            "archived_refs": archives,
        }

    def update(
        self, knowledge, rows, round_index, evolution_state=None, retrieval_audit=None
    ):
        method = self.config.method
        if method == "skill_pro_sequence":
            value = self.evolve_skills(knowledge, rows, round_index, evolution_state)
            pool = SkillPool.from_dict(value["skill_pool"], frozen=True)
            if len(pool.active()) > self.config.skill_capacity:
                raise ValueError("Skill-Pro evolver exceeded its declared capacity")
            return {
                "knowledge": KnowledgeState(knowledge.experiences, pool).to_dict(),
                "evolution_state": value["evolution_state"],
                "status": "completed",
                "skill_audit": value,
            }
        if method == "xskill_dual_memory":
            self.experience.last_retrieval_audit = retrieval_audit or {}
            experience_result = self.experience.update(knowledge, rows)
            try:
                pool, workflows = self._workflows(knowledge, rows, round_index)
            except KnowledgeValidationError as error:
                pool, workflows = (
                    knowledge.skills,
                    {"status": "skipped_validation_failure", "reason": str(error)},
                )
            return {
                "knowledge": KnowledgeState(
                    ExperienceBank.from_dict(
                        experience_result["experience_bank"], frozen=True
                    ),
                    pool.freeze(),
                ).to_dict(),
                "evolution_state": None,
                "status": "completed",
                "experience_audit": experience_result,
                "workflow_audit": workflows,
            }
        bank = self._observed_usage(ExperienceBank(knowledge.experiences.all()), rows)
        reflection_audit = []
        for row in rows:
            if method in {"rag_demonstrations", "sma_procedure"} and row.reward != 1.0:
                reflection_audit.append(
                    {
                        "trajectory_id": row.trajectory_id,
                        "status": "skipped_unsuccessful",
                    }
                )
                continue
            if method == "rag_demonstrations":
                procedure = [render_demonstration(row)]
                memories = [
                    {
                        "condition": "Verified demonstration for a similar spatial problem",
                        "procedure": procedure,
                        "evidence_refs": [row.trajectory_id],
                    }
                ]
            else:
                try:
                    memories = self._reflection(row)
                except KnowledgeValidationError as error:
                    reflection_audit.append(
                        {
                            "trajectory_id": row.trajectory_id,
                            "status": "skipped_validation_failure",
                            "reason": str(error),
                        }
                    )
                    continue
            for index, memory in enumerate(memories):
                proposal = self._memory(row, memory, index)
                # Long demonstrations need explicit bounded storage. Reject rather
                # than truncating a tool call or silently changing the method.
                measured = self._memory_budget(proposal)
                if measured["measured"] > measured["limit"]:
                    reflection_audit.append(
                        {
                            "trajectory_id": row.trajectory_id,
                            "status": "skipped_memory_" + measured["unit"] + "_budget",
                            "budget": measured,
                        }
                    )
                    continue
                bank = bank.apply(
                    ExperienceUpdate(
                        operation=ExperienceOperationType.ADD,
                        rationale="Declared baseline memory construction.",
                        proposed_experience=proposal,
                    )
                )
                reflection_audit.append(
                    {
                        "trajectory_id": row.trajectory_id,
                        "status": "added",
                        "reference": proposal.reference,
                        "budget": measured,
                    }
                )
        archives = sorted(
            bank.active(), key=lambda value: (value.created_at, value.reference)
        )[: max(0, len(bank.active()) - self.config.memory_capacity)]
        if archives:
            bank = bank.apply(
                ExperienceUpdate(
                    operation=ExperienceOperationType.ARCHIVE,
                    rationale="Baseline's declared oldest-created-first capacity policy.",
                    target_experience_refs=tuple(item.reference for item in archives),
                )
            )
        return {
            "knowledge": KnowledgeState(bank.freeze(), knowledge.skills).to_dict(),
            "evolution_state": None,
            "status": "completed",
            "reflection_audit": reflection_audit,
            "memory_construction": {
                "eligible_successful_trajectories": sum(
                    row.reward == 1.0 for row in rows
                ),
                "saved_memories": sum(
                    item["status"] == "added" for item in reflection_audit
                ),
                "budget_rejected_memories": sum(
                    item["status"]
                    in {"skipped_memory_word_budget", "skipped_memory_token_budget"}
                    for item in reflection_audit
                ),
                "validation_failed_trajectories": sum(
                    item["status"] == "skipped_validation_failure"
                    for item in reflection_audit
                ),
                "active_memories": len(bank.active()),
                "budget_definition": self.memory_budget_definition(),
            },
            "capacity_policy": "oldest_created_first",
            "archived_refs": [item.reference for item in archives],
        }


class MemoryBaselinePipeline:
    """Own accumulation/deployment orchestration, sharing only real execution."""

    def __init__(
        self,
        *,
        learner,
        journal,
        rollout,
        seed=42,
        rollouts_per_task=4,
        artifact_validator=None,
    ):
        self.learner, self.journal, self.rollout = learner, journal, rollout
        self.seed, self.rollouts_per_task = seed, rollouts_per_task
        self.artifact_validator = artifact_validator
        if rollouts_per_task < 1:
            raise ValueError("rollouts_per_task must be positive")
        descriptor = {
            **method_description(learner.config.method),
            "config": asdict(learner.config),
            "seed": seed,
            "rollouts_per_task": rollouts_per_task,
            "memory_budget": learner.memory_budget_definition(),
        }
        self.description, _ = self.journal.execute(
            "memory_baseline/descriptor", descriptor, lambda: descriptor
        )

    def _scope(self, *, phase, task_id, snapshot_id):
        generator = self.learner.generate
        if hasattr(generator, "scope"):
            generator.scope = {
                "phase": phase,
                "task_id": task_id,
                "snapshot_id": snapshot_id,
                "baseline_method": self.learner.config.method,
            }

    def _rows(self, task, state, prepared, prefix, *, deployment, task_index):
        refs = tuple(
            RetrievedExperienceRef.from_dict(item) for item in prepared["experiences"]
        )
        count = 1 if deployment else self.rollouts_per_task
        rows = []
        for index in range(count):
            seed = self.seed + task_index * self.rollouts_per_task + index
            isolated = state.copy()
            data, _ = self.journal.execute(
                prefix + f"/rollouts/{index:02d}/complete",
                {
                    "task": task.to_dict(),
                    "snapshot": state.snapshot_id,
                    "prepared": prepared,
                    "seed": seed,
                    "deployment": deployment,
                },
                lambda isolated=isolated, index=index, seed=seed: self.rollout(
                    task,
                    isolated,
                    refs,
                    index,
                    seed,
                    prefix + f"/rollouts/{index:02d}/execution",
                    deployment,
                ).to_dict(),
            )
            if isolated.snapshot_id != state.snapshot_id:
                raise RuntimeError("Baseline rollout mutated frozen memory")
            row = Trajectory.from_dict(data)
            if self.learner.config.method == "skill_pro_sequence" and not deployment:
                # The shared evolution queue resolves this standard source key.
                # Alias already committed data; never execute the rollout twice.
                self.journal.execute(
                    f"tasks/{task_index:05d}/rollouts/{index:02d}/complete",
                    {
                        "baseline_source_key": prefix
                        + f"/rollouts/{index:02d}/complete",
                        "trajectory_sha256": digest(data),
                    },
                    lambda data=data: data,
                )
            if (
                row.task.to_dict() != task.to_dict()
                or row.knowledge_snapshot_id != state.snapshot_id
                or row.rollout_index != index
                or row.random_seed != seed
                or row.reward not in (0.0, 1.0)
            ):
                raise ValueError(
                    "Baseline rollout failed task/snapshot/seed/reward binding"
                )
            if row.status not in {
                TrajectoryStatus.COMPLETED,
                TrajectoryStatus.TRUNCATED,
            } and not (
                row.status is TrajectoryStatus.FAILED and is_controlled_failure(row)
            ):
                raise ValueError(
                    "Baseline rollout has an unverified infrastructure failure/status"
                )
            if self.artifact_validator is not None:
                self.artifact_validator(row)
            rows.append(row)
        return tuple(rows)

    @staticmethod
    def _validate_tasks(tasks, split):
        if (
            not tasks
            or len({task.dataset for task in tasks}) != 1
            or len({task.task_id for task in tasks}) != len(tasks)
        ):
            raise ValueError("Expected unique tasks within one benchmark")
        if any(task.split is not split for task in tasks):
            raise ValueError("Wrong baseline data split")

    def accumulate(self, tasks):
        tasks = tuple(tasks)
        self._validate_tasks(tasks, TaskSplit.TRAIN)
        with file_lock(self.journal.root / ".memory_baseline.lock", timeout=0):
            initial, _ = self.journal.execute(
                "memory_baseline/initial",
                {
                    "description": self.description,
                    "tasks": [task.to_dict() for task in tasks],
                },
                lambda: {
                    "knowledge": self.learner.initial().to_dict(),
                    "dataset": tasks[0].dataset,
                    "training_task_ids": [task.task_id for task in tasks],
                },
            )
            state, evolution = KnowledgeState.from_dict(initial["knowledge"]), None
            construction = {
                "eligible_successful_trajectories": 0,
                "saved_memories": 0,
                "budget_rejected_memories": 0,
                "validation_failed_trajectories": 0,
            }
            for task_index, task in enumerate(tasks):
                prefix = f"memory_baseline/training/{task_index:05d}"
                self._scope(
                    phase="accumulation_retrieval",
                    task_id=task.task_id,
                    snapshot_id=state.snapshot_id,
                )
                prepared, _ = self.journal.execute(
                    prefix + "/prepare",
                    {
                        "task": task.without_reference_answer().to_dict(),
                        "snapshot": state.snapshot_id,
                    },
                    lambda task=task, state=state: self.learner.prepare(
                        task, state, deployment=False
                    ),
                )
                rows = self._rows(
                    task,
                    state,
                    prepared,
                    prefix,
                    deployment=False,
                    task_index=task_index,
                )
                self._scope(
                    phase="accumulation_learning",
                    task_id=task.task_id,
                    snapshot_id=state.snapshot_id,
                )
                value, _ = self.journal.execute(
                    prefix + "/update",
                    {
                        "knowledge": state.to_dict(),
                        "rows": [row.to_dict() for row in rows],
                        "evolution_state": evolution,
                        "retrieval_audit": prepared["retrieval_audit"],
                    },
                    lambda state=state, rows=rows, task_index=task_index, evolution=evolution, prepared=prepared: (
                        self.learner.update(
                            state,
                            rows,
                            task_index,
                            evolution,
                            prepared["retrieval_audit"],
                        )
                    ),
                )
                state, evolution = (
                    KnowledgeState.from_dict(value["knowledge"]),
                    value["evolution_state"],
                )
                construction["eligible_successful_trajectories"] += sum(
                    row.reward == 1.0 for row in rows
                )
                for key in construction:
                    if key != "eligible_successful_trajectories":
                        construction[key] += value.get("memory_construction", {}).get(
                            key, 0
                        )
            construction.update(
                active_memories=len(state.experiences.active()),
                memory_budget=self.learner.memory_budget_definition(),
            )
            if self.learner.config.method in {
                "xskill_dual_memory",
                "skill_pro_sequence",
            }:
                for key in (
                    "saved_memories",
                    "budget_rejected_memories",
                    "validation_failed_trajectories",
                ):
                    construction[key] = None
                construction["counter_coverage"] = (
                    "LLM Experience/Skill operations use their own learning audits; simple-memory admission counts not applicable"
                )
            else:
                construction["counter_coverage"] = (
                    "all simple-memory admission attempts"
                )
            construction, _ = self.journal.execute(
                "memory_baseline/construction",
                {"snapshot": state.snapshot_id, "statistics": construction},
                lambda: construction,
            )
            atomic_write_json(
                self.journal.root / "results/memory_construction.json", construction
            )
            final, _ = self.journal.execute(
                "memory_baseline/frozen",
                {"snapshot": state.snapshot_id, "evolution_state": evolution},
                lambda: {"knowledge": state.to_dict(), "evolution_state": evolution},
            )
            return KnowledgeState.from_dict(final["knowledge"])

    def deploy(self, tasks, knowledge):
        tasks = tuple(tasks)
        self._validate_tasks(tasks, TaskSplit.TEST)
        with file_lock(self.journal.root / ".memory_baseline.lock", timeout=0):
            initial = self.journal.read_committed("memory_baseline/initial")
            final = KnowledgeState.from_dict(
                self.journal.read_committed("memory_baseline/frozen")["knowledge"]
            )
            if final.snapshot_id != knowledge.snapshot_id:
                raise ValueError("Deploy only the committed frozen baseline snapshot")
            if any(
                task.dataset != initial["dataset"]
                or task.task_id in initial["training_task_ids"]
                for task in tasks
            ):
                raise ValueError(
                    "Baseline deployment dataset differs or overlaps training IDs"
                )
            before, rows = knowledge.snapshot_id, []
            retrieval_requests, retrieval_hits = 0, 0
            for task_index, task in enumerate(tasks):
                prefix = f"memory_baseline/deployment/{task_index:05d}"
                self._scope(
                    phase="deployment", task_id=task.task_id, snapshot_id=before
                )
                prepared, _ = self.journal.execute(
                    prefix + "/prepare",
                    {
                        "task": task.without_reference_answer().to_dict(),
                        "snapshot": before,
                    },
                    lambda task=task: self.learner.prepare(
                        task, knowledge, deployment=True
                    ),
                )
                retrieval_requests += 1
                retrieval_hits += bool(prepared["experiences"])
                rows.extend(
                    self._rows(
                        task,
                        knowledge,
                        prepared,
                        prefix,
                        deployment=True,
                        task_index=task_index,
                    )
                )
                if knowledge.snapshot_id != before:
                    raise RuntimeError("Deployment changed baseline knowledge")
            result = {
                "baseline": self.description,
                "snapshot_id": before,
                "count": len(rows),
                "correct": sum(row.reward for row in rows),
                "accuracy": sum(row.reward for row in rows) / len(rows),
                "trajectory_ids": [row.trajectory_id for row in rows],
                "result_kind": "paper_inspired_adapter_evaluation",
                "memory_construction": self.journal.read_committed(
                    "memory_baseline/construction"
                ),
                "retrieval": {
                    "requests": retrieval_requests,
                    "nonempty_injections": retrieval_hits,
                    "hit_rate": retrieval_hits / retrieval_requests
                    if retrieval_requests
                    else None,
                    "definition": "deployment tasks with at least one injected memory / deployment tasks; not semantic relevance accuracy",
                },
            }
            saved, _ = self.journal.execute(
                "memory_baseline/results",
                {"snapshot": before, "rows": [row.to_dict() for row in rows]},
                lambda: result,
            )
            from spatialcraft.evaluation.protocol_metrics_v2 import protocol_metrics

            saved["protocol_metrics_path"] = "results/protocol_metrics.json"
            atomic_write_json(
                self.journal.root / saved["protocol_metrics_path"],
                protocol_metrics(
                    rows,
                    knowledge,
                    token_counter=getattr(self, "metric_token_counter", None),
                    tokenizer_id=getattr(self, "metric_tokenizer_id", None),
                ),
            )
            atomic_write_json(
                self.journal.root / "results" / "memory_baseline.json", saved
            )
            return saved
