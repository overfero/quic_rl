"""Typed, validated experiment configuration - mirrors the prompt's own
example YAML schema field-for-field (top-level `model`/`algorithm`/
`rollout`/`reward`/`trainer`/`synchronization`/`dataset`/`evaluation`
keys), not a redesigned schema. Uses pydantic (already a dependency of
quic-vllm's own OpenAI-compatible request/response schemas - same
convention, not a new one introduced here)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name: str


class AlgorithmConfig(BaseModel):
    name: str = "grpo"

    @field_validator("name")
    @classmethod
    def _known_algorithm(cls, v: str) -> str:
        # Only grpo has a real TrainerBackend implementation right now -
        # see ARCHITECTURE.md's "future algorithms" section for why the
        # Protocol abstractions stay algorithm-general even though this
        # repo intentionally only wires up one of them today.
        allowed = {"grpo"}
        if v not in allowed:
            raise ValueError(f"algorithm.name={v!r} not implemented yet (only {sorted(allowed)} for now)")
        return v


class RolloutConfig(BaseModel):
    backend: str = "quic-vllm"
    workers: int = Field(default=1, ge=1)
    samples_per_prompt: int = Field(default=1, ge=1)  # GRPO's group_size


class RewardConfig(BaseModel):
    backend: str = "math"


class TrainerConfig(BaseModel):
    backend: str = "quic-train"


class SynchronizationConfig(BaseModel):
    strategy: str = "checkpoint"  # export -> transfer -> reload, see ARCHITECTURE.md


class DatasetConfig(BaseModel):
    name: str
    split: str = "train"
    num_examples: int | None = None


class EvaluationConfig(BaseModel):
    benchmarks: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    model: ModelConfig
    algorithm: AlgorithmConfig = AlgorithmConfig()
    rollout: RolloutConfig = RolloutConfig()
    reward: RewardConfig = RewardConfig()
    trainer: TrainerConfig = TrainerConfig()
    synchronization: SynchronizationConfig = SynchronizationConfig()
    dataset: DatasetConfig
    evaluation: EvaluationConfig = EvaluationConfig()

    # Orchestration knobs that aren't part of the prompt's example YAML
    # but the training loop genuinely needs - kept here rather than
    # invented as a second config file.
    max_iterations: int = Field(default=1, ge=1)
    state_dir: str = "./quic_rl_state"

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
