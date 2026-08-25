"""`RolloutBackend`: the orchestrator's only view of "generate completions
somewhere" - it must not depend on vLLM internals (or any other engine's
internals) directly. `QuicVLLMRollout` (quic_vllm.py) is the real
implementation; `MockRolloutBackend` (mock.py) is a GPU-free stand-in used
by every orchestrator test and `examples/mock_loop.py`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quic_rl.trajectory import Trajectory


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    top_p: float = 1.0


@dataclass
class GenerationRequest:
    """One prompt, asking for `num_samples` independent completions
    (GRPO's "group" - `GRPOConfig.group_size` on the quic-train side)."""

    prompt_id: str
    prompt: str
    num_samples: int = 1


@runtime_checkable
class RolloutBackend(Protocol):
    def load_policy(self, policy_path: str, policy_version: int) -> None:
        """Makes `policy_version` (weights at `policy_path`) the one
        `generate()` samples from. Blocking - returns only once rollout
        workers would actually serve this version, so the orchestrator's
        `rollout_policy_version` bookkeeping is trustworthy the instant
        this returns."""
        ...

    def generate(self, requests: list[GenerationRequest], sampling: SamplingParams) -> list[Trajectory]:
        """Returns exactly `sum(r.num_samples for r in requests)`
        `Trajectory` objects, `policy_version` stamped to whatever
        `load_policy()` last set. `token_ids`/`logprobs` populated when
        the underlying engine's response includes them - callers must
        not assume they're always present (see `Trajectory`'s own
        docstring)."""
        ...

    def health_check(self) -> bool:
        """Cheap, fast liveness check - not a full generation round trip."""
        ...

    def get_status(self) -> dict:
        """Free-form status snapshot (current policy version, worker
        count, queue depth, ...) - for logging/metrics, not control flow."""
        ...

    def unload_policy(self) -> None:
        """Optional - releases whatever `load_policy()` holds. A backend
        for which this is a no-op (e.g. always-one-model-resident) may
        implement it as `pass`."""
        ...
