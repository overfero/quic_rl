"""GPU-free, deterministic stand-in for `RewardBackend` - reward is a
pure function of the response TEXT (its length), so the same trajectory
always scores the same way across test runs, and so tests can assert on
an exact expected value rather than a range."""
from __future__ import annotations

from quic_rl.trajectory import Trajectory


class MockRewardBackend:
    def __init__(self, target_length: int = 40) -> None:
        self.target_length = target_length

    def score(self, trajectory: Trajectory) -> float:
        length = len(trajectory.response.split())
        return -abs(length - self.target_length) / self.target_length

    def batch_score(self, trajectories: list[Trajectory]) -> list[float]:
        return [self.score(t) for t in trajectories]
