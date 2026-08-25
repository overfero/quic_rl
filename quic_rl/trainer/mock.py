"""GPU-free stand-in for `TrainerBackend` - "trains" by computing real
statistics over the given batch's rewards (so `TrainResult` is
meaningful, not fabricated) and incrementing a fake policy version -
enough to exercise the orchestrator's policy-version bookkeeping and
weight-sync flow without any real model."""
from __future__ import annotations

import statistics

from quic_rl.trainer.base import TrainResult
from quic_rl.trajectory import Trajectory


class MockTrainerBackend:
    def __init__(self) -> None:
        self._policy_version: int | None = None
        self._policy_path: str | None = None

    def initialize_policy(self, policy_path: str) -> int:
        self._policy_path = policy_path
        self._policy_version = 0
        return self._policy_version

    def train(self, batch: list[Trajectory]) -> TrainResult:
        if self._policy_version is None:
            raise RuntimeError("MockTrainerBackend.train() called before initialize_policy()")
        if not batch:
            raise ValueError("MockTrainerBackend.train() called with an empty batch")

        rewards = [t.reward for t in batch]
        if any(r is None for r in rewards):
            raise ValueError("MockTrainerBackend.train(): every trajectory must have .reward set before training")

        self._policy_version += 1
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        # A fake but deterministic "loss" - decreases as rewards approach
        # 0 (this mock's reward is always <= 0, see MockRewardBackend),
        # just enough signal for a test to assert the loop is coherent.
        train_loss = -reward_mean
        completion_rate = sum(1 for t in batch if t.response.strip()) / len(batch)

        return TrainResult(
            policy_version=self._policy_version,
            train_loss=train_loss,
            reward_mean=reward_mean,
            reward_std=reward_std,
            completion_rate=completion_rate,
            tokens_per_sec=0.0,
            step_time_s=0.0,
            kl=0.0,
        )

    def get_policy_version(self) -> int:
        if self._policy_version is None:
            raise RuntimeError("MockTrainerBackend: policy not initialized yet")
        return self._policy_version

    def export_policy(self, output_dir: str) -> str:
        import os

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"mock_policy_v{self._policy_version}.txt")
        with open(path, "w") as f:
            f.write(f"mock policy weights, version={self._policy_version}\n")
        return path

    def checkpoint(self, output_dir: str) -> str:
        import os

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"mock_checkpoint_v{self._policy_version}.txt")
        with open(path, "w") as f:
            f.write(f"mock training state, version={self._policy_version}\n")
        return path

    def health_check(self) -> bool:
        return self._policy_version is not None
