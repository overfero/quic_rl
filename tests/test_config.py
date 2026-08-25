"""Config validation - typed, type-safe per the prompt's requirement,
schema mirrors the prompt's own example YAML field-for-field."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from quic_rl.config.schema import ExperimentConfig


def _minimal_config_dict():
    return {
        "model": {"name": "Qwen/Qwen3-1.7B-Base"},
        "dataset": {"name": "DAPO-Math-17k"},
    }


def test_minimal_config_applies_documented_defaults():
    config = ExperimentConfig(**_minimal_config_dict())
    assert config.algorithm.name == "grpo"
    assert config.rollout.backend == "quic-vllm"
    assert config.rollout.workers == 1
    assert config.reward.backend == "math"
    assert config.trainer.backend == "quic-train"
    assert config.synchronization.strategy == "checkpoint"


def test_full_config_matches_prompt_example_shape():
    data = {
        "model": {"name": "Qwen/Qwen3-1.7B-Base"},
        "algorithm": {"name": "grpo"},
        "rollout": {"backend": "quic-vllm", "workers": 4, "samples_per_prompt": 8},
        "reward": {"backend": "math"},
        "trainer": {"backend": "quic-train"},
        "synchronization": {"strategy": "checkpoint"},
        "dataset": {"name": "DAPO-Math-17k"},
        "evaluation": {"benchmarks": ["math500", "aime2024", "aime2025", "amc2023"]},
    }
    config = ExperimentConfig(**data)
    assert config.rollout.workers == 4
    assert config.rollout.samples_per_prompt == 8
    assert config.evaluation.benchmarks == ["math500", "aime2024", "aime2025", "amc2023"]


def test_unknown_algorithm_rejected():
    data = _minimal_config_dict()
    data["algorithm"] = {"name": "some_future_algorithm"}
    with pytest.raises(ValidationError, match="not implemented"):
        ExperimentConfig(**data)


def test_missing_required_model_name_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(dataset={"name": "x"})


def test_zero_rollout_workers_rejected():
    data = _minimal_config_dict()
    data["rollout"] = {"workers": 0}
    with pytest.raises(ValidationError):
        ExperimentConfig(**data)


def test_from_file_yaml_roundtrip(tmp_path):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_minimal_config_dict()))
    config = ExperimentConfig.from_file(str(path))
    assert config.model.name == "Qwen/Qwen3-1.7B-Base"
    assert config.dataset.name == "DAPO-Math-17k"
