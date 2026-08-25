"""`TrainerBackend`: the orchestrator's only view of "run one GRPO update
somewhere" - it must not implement GRPO itself (that logic lives in
quic-train). `QuicTrainBackend` (quic_train.py) is the real
implementation; `MockTrainerBackend` (mock.py) is a GPU-free stand-in."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quic_rl.trajectory import Trajectory


@dataclass
class TrainResult:
    policy_version: int  # the NEW version this update produced
    train_loss: float
    reward_mean: float
    reward_std: float
    completion_rate: float  # fraction of trajectories that had a usable (non-truncated/non-empty) response
    tokens_per_sec: float
    step_time_s: float
    kl: float | None = None  # None when the backend doesn't compute/report it


@runtime_checkable
class TrainerBackend(Protocol):
    def initialize_policy(self, policy_path: str) -> int:
        """Loads the starting checkpoint, returns its policy_version
        (0 for a fresh run, or whatever a resumed checkpoint's version
        was)."""
        ...

    def train(self, batch: list[Trajectory]) -> TrainResult:
        """Runs exactly one GRPO update on `batch` (every trajectory
        must already have `.reward` set and share one
        `policy_version`, enforced by the caller - see
        `orchestrator/state.py`). Does not generate anything itself."""
        ...

    def get_policy_version(self) -> int:
        ...

    def export_policy(self, output_dir: str) -> str:
        """Writes the CURRENT policy's weights (trainable-params-only,
        e.g. a LoRA adapter - matching quic-train's own
        `training_utils.save_checkpoint` convention) to `output_dir`.
        Returns the path actually written. This is weight export for
        ROLLOUT sync, distinct from `checkpoint()` below (training-state
        resume) - `WeightSynchronizer` is responsible for anything
        rollout-engine-specific (merging into a full checkpoint,
        sharding), not this method."""
        ...

    def checkpoint(self, output_dir: str) -> str:
        """Writes full resumable TRAINING state (model, optimizer, RNG,
        step) to `output_dir`, for resuming an interrupted training run
        - not for rollout sync. Returns the path actually written."""
        ...

    def health_check(self) -> bool:
        ...
