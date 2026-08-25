"""`quic_rl`: the orchestration/integration layer between `quic-train`
(distributed LLM training over QUIC) and `quic-vllm` (distributed
inference over QUIC) - NOT a place that reimplements either. This
package owns rollout scheduling, trajectory collection, reward dispatch,
policy-version management, weight synchronization, and training-iteration
coordination; GRPO's actual loss math stays in quic-train, and vLLM's
actual inference stays in quic-vllm.

See `docs/ARCHITECTURE.md` for the full integration architecture and why
each of the two other repos' real, verified APIs shaped the design
choices here."""
from __future__ import annotations

from quic_rl.trajectory import Trajectory

__all__ = ["Trajectory"]
