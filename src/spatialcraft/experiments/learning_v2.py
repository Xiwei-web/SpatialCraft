"""Runtime bridge for operation-scoped Experience and Skill learning."""

from functools import wraps

from spatialcraft.knowledge.evidence import render_action
from spatialcraft.knowledge.experience.learning_v2 import ExperienceLearningV2
from spatialcraft.knowledge.skill.learning_v2 import SkillLearningV2
from spatialcraft.models import ContentPart
from spatialcraft.models.scoring.target_logprob import TargetLogprobScorer

from .knowledge_generator import KnowledgeValidationError


def _scoped_embedding_operation(operation):
    """A cached stage must not change metadata of later model requests."""

    def decorate(method):
        @wraps(method)
        def scoped(self, *args, **kwargs):
            previous = self.generator.scope
            self.generator.scope = {**previous, "embedding_operation": operation}
            try:
                return method(self, *args, **kwargs)
            finally:
                self.generator.scope = previous

        return scoped

    return decorate


class LearningBuildersV2:
    def __init__(
        self,
        settings,
        model,
        provider,
        embedder,
        resolver,
        generator,
        token_counter,
        trajectory_loader=None,
        tools=(),
    ):
        self.settings, self.generator, self.resolve = settings, generator, resolver
        self.experience = ExperienceLearningV2(
            generator,
            embedder,
            {
                "capacity": settings.experience_capacity,
                "rollouts_per_task": settings.rollouts_per_task,
                "top_k_per_aspect": settings.experience_top_k_per_subtask,
                **settings.experience_options,
                **settings.ablations,
            },
            artifact_resolver=resolver,
        )
        self.skill = SkillLearningV2(
            generator,
            embedder,
            TargetLogprobScorer(provider, model_alias=model.alias),
            token_counter,
            trajectory_loader=trajectory_loader,
            resolver=resolver,
            config={
                "capacity": settings.skill_capacity,
                "batch_trajectories": settings.evolution_batch_trajectories,
                "candidates_per_target": settings.skill_candidates,
                "max_parent_skills_per_round": settings.max_parent_skills_per_round,
                "maximum_targets_per_round": settings.max_parent_skills_per_round,
                "max_discovery_per_round": 1,
                "rollouts_per_task": settings.rollouts_per_task,
                "ppo_clip_epsilon": settings.ppo_epsilon,
                "acceptance_gain_margin": settings.ppo_positive_margin,
                "seed": settings.seed,
                "tokenizer_id": model.local.tokenizer_path or model.local.path,
                "available_tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                    for t in tools
                ],
                **settings.skill_options,
                **settings.ablations,
            },
        )

    def set_scope(self, **scope):
        self.generator.scope = scope

    @_scoped_embedding_operation("retrieval.index_and_query")
    def retrieve(self, task, knowledge):
        if self.settings.ablations.get("no_experience"):
            return {
                "experiences": [],
                "retrieval_audit": {
                    "task_id": task.task_id,
                    "snapshot_id": knowledge.snapshot_id,
                    "retrieved_refs": [],
                    "injected_refs": [],
                    "status": "disabled_no_experience",
                },
            }
        try:
            refs = self.experience.retrieve(task, knowledge)
        except KnowledgeValidationError as exc:
            return {
                "experiences": [],
                "retrieval_audit": {
                    **self.experience.last_retrieval_audit,
                    "task_id": task.task_id,
                    "snapshot_id": knowledge.snapshot_id,
                    "injected_refs": [],
                    "status": "rejected_model_output",
                    "error": str(exc),
                },
            }
        return {
            "experiences": [r.to_dict() for r in refs],
            "retrieval_audit": self.experience.last_retrieval_audit,
        }

    def update_experiences(self, knowledge, rows, retrieval_audit=None):
        self.set_scope(
            phase="accumulation_learning",
            task_id=rows[0].task.task_id,
            snapshot_id=knowledge.snapshot_id,
            embedding_operation="experience.merge_candidates",
        )
        if self.settings.ablations.get("no_experience"):
            return {
                "experience_bank": knowledge.experiences.to_dict(),
                "status": "disabled_no_experience",
            }
        self.experience.last_retrieval_audit = dict(retrieval_audit or {})
        return self.experience.update(knowledge, rows)

    def evolve(self, knowledge, rows, batch_index, evolution_state=None):
        self.set_scope(
            phase="accumulation_learning",
            task_id=rows[0].task.task_id,
            snapshot_id=knowledge.snapshot_id,
            batch_index=batch_index,
            embedding_operation="skill.evolution_and_deduplication",
        )
        if self.settings.ablations.get("no_skill") or self.settings.ablations.get(
            "static_seed_skills"
        ):
            return {
                "skill_pool": knowledge.skills.to_dict(),
                "evolution_state": evolution_state,
                "status": "disabled_skill_evolution",
            }
        return self.skill.evolve(knowledge, rows, batch_index, evolution_state)

    def _public(self, task, state):
        media = [
            ContentPart.image_uri(im.uri, mime_type=im.media_type) for im in task.images
        ]
        for message in state.messages:
            if message.role.value != "tool":
                continue
            import json

            for artifact in json.loads(message.content).get("artifacts", ()):
                if str(artifact.get("mime_type", "")).startswith("image/"):
                    media.append(
                        ContentPart.text_part("Tool image evidence: " + artifact["uri"])
                    )
                    media.append(
                        ContentPart.image_uri(
                            str(self.resolve(artifact["uri"])),
                            mime_type=artifact["mime_type"],
                        )
                    )
        return {
            "task": task.without_reference_answer().to_dict(),
            "state": state.to_dict(),
        }, media

    @_scoped_embedding_operation("skill.selection")
    def applicable(self, candidates, task, state):
        payload, media = self._public(task, state)
        available = {s.reference for s in candidates}
        payload.update(
            skills=[
                {"reference": s.reference, "initiation": s.initiation}
                for s in candidates
            ],
            expected_output={"applicable_refs": ["zero or more exact references"]},
        )

        def validate(value):
            refs = value.get("applicable_refs")
            if (
                not isinstance(refs, list)
                or any(not isinstance(r, str) for r in refs)
                or len(set(refs)) != len(refs)
                or not set(refs) <= available
            ):
                raise ValueError(
                    "Applicability requires unique known references; empty means NONE"
                )

        try:
            return tuple(
                self.generator(
                    "skill.selection_judge", payload, media=media, validator=validate
                )["applicable_refs"]
            )
        except KnowledgeValidationError:
            return ()

    def terminate(self, task, skill, state, action):
        payload, media = self._public(task, state)
        payload.update(
            skill=skill.to_dict(),
            action=render_action(action),
            expected_output={"terminate": "boolean"},
        )

        def validate(value):
            if type(value.get("terminate")) is not bool:
                raise ValueError("Termination requires an actual boolean")

        try:
            return self.generator(
                "skill.termination", payload, media=media, validator=validate
            )["terminate"]
        except KnowledgeValidationError:
            return False
