"""Validated user-confirmed algorithm protocol plus explicit engineering defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from spatialcraft.models.registry import load_yaml


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    seed: int = 42
    accumulation_passes: int = 1
    rollouts_per_task: int = 4
    evolution_batch_trajectories: int = 6
    max_parent_skills_per_round: int = 2
    skill_generation_max_tokens: int = 4096
    experience_top_k_per_subtask: int = 3
    experience_capacity: int = 100
    skill_capacity: int = 20
    skill_candidates: int = 3
    embedding_model: str = "text-embedding-3-small"
    backbone: str = "qwen3.5-9b"
    max_steps: int = 50
    max_output_tokens: int = 4096
    training_temperature: float = 0.7
    training_top_p: float = 0.9
    deployment_temperature: float = 0.0
    auxiliary_temperature: float = 0.0
    ppo_epsilon: float = 0.2
    ppo_positive_margin: float = 0.0
    skill_max_lifetime: int = 8
    image_max_pixels: int = 1048576
    enable_thinking: bool = False
    ppo_thinking_mode: str = "action_only"

    def __post_init__(self):
        if type(self.enable_thinking) is not bool:
            raise TypeError("enable_thinking must be boolean")
        expected_mode = (
            "fixed_sampled_prefix" if self.enable_thinking else "action_only"
        )
        if self.ppo_thinking_mode != expected_mode:
            raise ValueError(
                f"PPO scoring mode must be {expected_mode} for enable_thinking={self.enable_thinking}"
            )
        if any(
            not isfinite(float(getattr(self, field)))
            for field in (
                "training_temperature",
                "training_top_p",
                "deployment_temperature",
                "auxiliary_temperature",
                "ppo_epsilon",
                "ppo_positive_margin",
            )
        ):
            raise ValueError("Sampling/PPO values must be finite")
        for name in (
            "seed",
            "evolution_batch_trajectories",
            "max_parent_skills_per_round",
            "skill_generation_max_tokens",
            "max_steps",
            "max_output_tokens",
            "skill_max_lifetime",
            "image_max_pixels",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0 or self.image_max_pixels < 65536:
            raise ValueError("Require nonnegative seed and image budget >=65536 pixels")
        for name, value in {
            "accumulation_passes": 1,
            "rollouts_per_task": 4,
            "experience_top_k_per_subtask": 3,
            "experience_capacity": 100,
            "skill_capacity": 20,
            "skill_candidates": 3,
        }.items():
            if getattr(self, name) != value:
                raise ValueError(f"Confirmed protocol requires {name}={value}")
        if self.embedding_model != "text-embedding-3-small":
            raise ValueError("Confirmed embedding model is text-embedding-3-small")
        if self.backbone not in {"qwen3.5-9b", "qwen3.6-27b-local"}:
            raise ValueError("Unsupported local experiment backbone")
        if any(
            getattr(self, name) < 1
            for name in (
                "evolution_batch_trajectories",
                "max_parent_skills_per_round",
                "skill_generation_max_tokens",
                "max_steps",
                "max_output_tokens",
                "skill_max_lifetime",
                "image_max_pixels",
            )
        ):
            raise ValueError("Execution budgets must be positive")
        if self.skill_generation_max_tokens > self.max_output_tokens:
            raise ValueError(
                "Skill generation cannot exceed the single-call output cap"
            )
        if (
            self.evolution_batch_trajectories != 6
            or self.max_parent_skills_per_round > 2
        ):
            raise ValueError(
                "Evolution uses 6 distinct trajectories and at most 2 parents per round"
            )
        if (
            self.max_steps > 50
            or self.skill_max_lifetime > 8
            or self.max_output_tokens > 4096
        ):
            raise ValueError(
                "Confirmed caps: trajectory 50 steps, Skill 8 steps, output 4096 tokens"
            )
        if not 0 < self.training_temperature <= 2 or not 0 < self.training_top_p <= 1:
            raise ValueError(
                "Independent accumulation rollouts require stochastic sampling"
            )
        if (
            self.deployment_temperature != 0
            or self.auxiliary_temperature != 0
            or not 0 <= self.ppo_epsilon < 1
            or not self.ppo_positive_margin >= 0
        ):
            raise ValueError("Invalid deployment/PPO setting")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        return cls(**load_yaml(path)["settings"])
