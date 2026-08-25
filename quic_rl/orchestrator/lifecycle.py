"""Startup/shutdown sequencing - kept separate from `controller.py`'s
per-iteration loop so "how does a run begin/end" and "what happens each
iteration" can be read (and tested) independently."""
from __future__ import annotations

from quic_rl.rollout.base import RolloutBackend
from quic_rl.trainer.base import TrainerBackend


def initialize(rollout: RolloutBackend, trainer: TrainerBackend, initial_policy_path: str) -> int:
    """Loads the starting policy into the trainer, then propagates that
    SAME version to the rollout backend before any generation happens -
    returns the resulting policy_version both sides now agree on."""
    policy_version = trainer.initialize_policy(initial_policy_path)
    rollout.load_policy(initial_policy_path, policy_version)
    return policy_version


def shutdown(rollout: RolloutBackend, trainer: TrainerBackend) -> None:
    """Best-effort clean teardown - a backend that has nothing to release
    for one of these should implement it as a no-op, not raise."""
    rollout.unload_policy()
