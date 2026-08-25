"""GPU-free stand-in for `RolloutBackend` - lets the whole orchestration
loop (and every test in `tests/`) run without vLLM, without a model,
without a GPU. Generates deterministic, fake completions - real enough
in SHAPE (token_ids, logprobs, per-prompt grouping) to exercise every
downstream consumer honestly."""
from __future__ import annotations

import random

from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.trajectory import Trajectory


class MockRolloutBackend:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._policy_version: int | None = None
        self._policy_path: str | None = None
        self._loaded = False

    def load_policy(self, policy_path: str, policy_version: int) -> None:
        self._policy_path = policy_path
        self._policy_version = policy_version
        self._loaded = True

    def generate(self, requests: list[GenerationRequest], sampling: SamplingParams) -> list[Trajectory]:
        if not self._loaded:
            raise RuntimeError("MockRolloutBackend.generate() called before load_policy()")
        out: list[Trajectory] = []
        for req in requests:
            for i in range(req.num_samples):
                n_tokens = self._rng.randint(3, sampling.max_tokens // 8 or 3)
                token_ids = [self._rng.randint(0, 50000) for _ in range(n_tokens)]
                response = f"mock-response-{req.prompt_id}-{i} (" + " ".join(str(t) for t in token_ids) + ")"
                out.append(
                    Trajectory(
                        prompt_id=req.prompt_id,
                        policy_version=self._policy_version,
                        prompt=req.prompt,
                        response=response,
                        token_ids=token_ids,
                        logprobs=[-abs(self._rng.gauss(1.0, 0.5)) for _ in token_ids],
                        metadata={"sample_index": i, "backend": "mock"},
                    )
                )
        return out

    def health_check(self) -> bool:
        return self._loaded

    def get_status(self) -> dict:
        return {"loaded": self._loaded, "policy_version": self._policy_version, "policy_path": self._policy_path}

    def unload_policy(self) -> None:
        self._loaded = False
