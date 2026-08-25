"""`RewardBackend`: scores a `Trajectory`'s response, filling in
`.reward`. `MathVerifierReward` (math_verifier.py) is the first, real
implementation - deterministic answer verification, no LLM judge (see
its own docstring). `MockRewardBackend` (mock.py) is a GPU-free stand-in."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from quic_rl.trajectory import Trajectory


@runtime_checkable
class RewardBackend(Protocol):
    def score(self, trajectory: Trajectory) -> float:
        """Returns the reward for one trajectory - does NOT mutate
        `trajectory.reward` itself (the caller, e.g.
        `workers/reward_worker.py`, is responsible for assigning it),
        so `RewardBackend` implementations stay pure functions of
        (prompt, response), trivially testable in isolation."""
        ...

    def batch_score(self, trajectories: list[Trajectory]) -> list[float]:
        """Same contract as `score`, batched - the default/expected
        implementation is just `[self.score(t) for t in trajectories]`;
        override only if a real batching speedup exists (e.g. a reward
        MODEL forward pass, not this project's rule-based verifiers)."""
        ...
