"""Policy-version bookkeeping and orchestrator run state.

The prompt this repo's architecture was designed against is explicit and
non-negotiable on this point: "Never silently mix trajectories from
incompatible policy versions" and "the orchestrator should explicitly
track: current_policy_version, rollout_policy_version,
training_policy_version." These are three genuinely different things,
not one counter read from three places:

- `current_policy_version`: what the orchestrator believes is canonical
  RIGHT NOW (what a freshly-synced rollout worker would serve).
- `rollout_policy_version`: what version the trajectories just collected
  were ACTUALLY generated under (stamped by `RolloutBackend.generate()`,
  read back off the `Trajectory` objects themselves - never assumed to
  equal `current_policy_version` without checking).
- `training_policy_version`: what version `TrainerBackend.train()` just
  trained FROM (i.e. the batch's version), producing `version + 1`.

`PolicyVersionMismatchError` is the fail-loud path this file exists to
enforce - never caught and silently ignored anywhere in this repo."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from quic_rl.trajectory import Trajectory


class PolicyVersionMismatchError(RuntimeError):
    """Raised the instant trajectories from more than one policy version
    (or a version other than the currently-expected one) would otherwise
    get silently combined into one training batch."""


@dataclass
class PolicyVersionState:
    current_policy_version: int = 0
    rollout_policy_version: int | None = None
    training_policy_version: int | None = None

    def record_rollout(self, trajectories: list[Trajectory]) -> None:
        """Call once per batch of freshly-generated trajectories.
        Verifies every trajectory shares ONE version, and that it
        matches what the orchestrator currently believes is canonical -
        raises `PolicyVersionMismatchError` otherwise (e.g. a rollout
        worker that hadn't finished loading the latest synced weights
        yet, or a stale cached response)."""
        if not trajectories:
            raise ValueError("record_rollout() called with an empty trajectory list")
        versions = {t.policy_version for t in trajectories}
        if len(versions) != 1:
            raise PolicyVersionMismatchError(
                f"trajectories in one batch came from {len(versions)} different policy versions: "
                f"{sorted(versions)} - refusing to mix them into one training batch"
            )
        (version,) = versions
        if version != self.current_policy_version:
            raise PolicyVersionMismatchError(
                f"rollout trajectories are stamped policy_version={version}, but the orchestrator's "
                f"current_policy_version is {self.current_policy_version} - a rollout worker is serving "
                f"stale (or, less likely, unexpectedly-future) weights"
            )
        self.rollout_policy_version = version

    def record_training(self, result_policy_version: int) -> None:
        """Call once per `TrainerBackend.train()` call. Verifies the
        trainer trained FROM the version the just-recorded rollout
        actually used, and that the new version is a real increment."""
        if self.rollout_policy_version is None:
            raise RuntimeError("record_training() called before record_rollout() this iteration")
        if result_policy_version <= self.rollout_policy_version:
            raise PolicyVersionMismatchError(
                f"trainer returned policy_version={result_policy_version}, which is not newer than "
                f"the version it should have trained from ({self.rollout_policy_version})"
            )
        self.training_policy_version = self.rollout_policy_version
        self.current_policy_version = result_policy_version

    def record_sync(self, synced_version: int) -> None:
        """Call once weight sync to rollout workers has actually
        completed - verifies rollout workers are now serving exactly
        what training just produced, not some other version."""
        if synced_version != self.current_policy_version:
            raise PolicyVersionMismatchError(
                f"weight sync reported syncing policy_version={synced_version}, but the orchestrator's "
                f"current_policy_version is {self.current_policy_version} - sync target drifted"
            )


@dataclass
class OrchestratorState:
    """Full resumable orchestrator run state - "checkpoint orchestrator
    state" in the training loop below is THIS, saved to a JSON file. This
    is the orchestrator's OWN crash-resume, a different and much smaller
    thing than detecting/replacing a dead rollout/reward worker (that
    worker-level fault tolerance is explicitly out of scope for this
    repo - see ARCHITECTURE.md)."""

    iteration: int = 0
    policy_version: PolicyVersionState = field(default_factory=PolicyVersionState)
    total_trajectories_collected: int = 0
    total_training_steps: int = 0

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OrchestratorState | None":
        """Returns None (a clean "nothing to resume" signal, not an
        exception) when no checkpoint exists yet - matches quic-train's
        own `training_utils.load_checkpoint` contract deliberately, same
        ecosystem convention."""
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            data = json.load(f)
        pv = data.pop("policy_version")
        return cls(policy_version=PolicyVersionState(**pv), **data)
