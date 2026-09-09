"""Central reward computation for completed agent rollouts."""

from __future__ import annotations

from dataclasses import replace

from spatialcraft.schemas import AgentAction, TaskSample, VerifierOutcome
from spatialcraft.verification import VerifierRouter


class RewardComputer:
    def __init__(
        self, verifier: VerifierRouter | None = None, *, binary: bool = False
    ) -> None:
        self.verifier = verifier or VerifierRouter()
        self.binary = binary

    def compute(self, task: TaskSample, action: AgentAction) -> VerifierOutcome:
        result = self.verifier.verify(task, action)
        if not self.binary:
            return result
        if result.is_correct is None:
            raise ValueError("Binary reward requires a definitive correctness verdict")
        return replace(
            result,
            score=float(result.is_correct),
            details={
                **result.details,
                "raw_verifier_score": result.score,
                "reward_mode": "binary",
            },
        )


__all__ = ["RewardComputer"]
