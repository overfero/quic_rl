"""`WeightSynchronizer`: Trainer -> Rollout weight propagation, kept as
its own abstraction deliberately (the prompt's own instruction: "Do NOT
assume that the only possible synchronization mechanism is a full model
reload"). The real implementation for this ecosystem
(`QuicVLLMWeightSynchronizer`, added in Phase C) performs merge -> shard
-> transfer -> restart - see `docs/ARCHITECTURE.md` for why vLLM's own
built-in NCCL/IPC live weight-transfer engines do NOT fit this NAT'd,
multi-machine topology and restart-based sync is the correct choice, not
a stopgap.

Every real implementation must report the 5 cost fields the prompt
explicitly asks not to hide from benchmarks - `SyncResult` makes that
structural, not optional per-implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SyncResult:
    policy_version: int
    weight_size_bytes: int
    transfer_time_s: float
    sync_time_s: float  # end-to-end wall clock for the whole sync() call
    reload_time_s: float
    total_overhead_s: float  # sync_time_s, restated explicitly so it's never accidentally omitted from a report


@runtime_checkable
class WeightSynchronizer(Protocol):
    def sync(self, policy_dir: str, policy_version: int) -> SyncResult:
        """Propagates the weights at `policy_dir` (as written by
        `TrainerBackend.export_policy()`) to every rollout worker, and
        blocks until they're actually serving `policy_version` - the
        orchestrator's `PolicyVersionState.record_sync()` depends on
        that being true the instant this returns."""
        ...


class MockWeightSynchronizer:
    """GPU/network-free stand-in - simulates the sync steps' ORDER and
    timing structure without any real transfer, so orchestrator tests
    exercise the same call shape a real sync would."""

    def __init__(self, simulated_latency_s: float = 0.0) -> None:
        self.simulated_latency_s = simulated_latency_s
        self.sync_calls: list[tuple[str, int]] = []

    def sync(self, policy_dir: str, policy_version: int) -> SyncResult:
        t0 = time.monotonic()
        self.sync_calls.append((policy_dir, policy_version))
        if self.simulated_latency_s:
            time.sleep(self.simulated_latency_s)
        elapsed = time.monotonic() - t0
        return SyncResult(
            policy_version=policy_version,
            weight_size_bytes=0,
            transfer_time_s=elapsed / 2,
            sync_time_s=elapsed,
            reload_time_s=elapsed / 2,
            total_overhead_s=elapsed,
        )
