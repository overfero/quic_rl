"""`Trajectory`: the one shared data type every backend in this repo
passes around - a rollout worker produces it, a reward worker fills in
`reward`, `orchestrator/controller.py` groups them into a training batch
for `TrainerBackend.train()`. Schema is fixed by the integration
architecture in `docs/ARCHITECTURE.md`, not redesigned here.

Optional fields stay `None`-able throughout the pipeline on purpose -
not every `RolloutBackend` can report token ids/logprobs (e.g. a plain
text-only completion API), and `reward` is genuinely unset until the
reward stage runs. Code that consumes a `Trajectory` must not assume a
field is populated just because it COULD be."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    prompt_id: str
    policy_version: int
    prompt: str
    response: str
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    reward: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "policy_version": self.policy_version,
            "prompt": self.prompt,
            "response": self.response,
            "token_ids": self.token_ids,
            "logprobs": self.logprobs,
            "reward": self.reward,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        return cls(
            prompt_id=data["prompt_id"],
            policy_version=data["policy_version"],
            prompt=data["prompt"],
            response=data["response"],
            token_ids=data.get("token_ids"),
            logprobs=data.get("logprobs"),
            reward=data.get("reward"),
            metadata=data.get("metadata", {}),
        )
