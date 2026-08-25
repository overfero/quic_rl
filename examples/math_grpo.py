"""The real first experiment: Qwen3-1.7B-Base + DAPO-Math-17k + GRPO +
MathVerifierReward, over real quic-train/quic-vllm backends.

STATUS: not runnable yet. `quic_rl.rollout.quic_vllm.QuicVLLMRollout` and
`quic_rl.trainer.quic_train.QuicTrainBackend` (the real, non-mock
implementations) don't exist yet - see docs/ARCHITECTURE.md's phasing
(Phase B/C). This script is written now, in its real intended shape, so
the gap between "the mock loop" (examples/mock_loop.py, works today) and
"the real experiment" is exactly those two imports - nothing structural
changes once they exist. Run examples/mock_loop.py today; come back to
this once Phase B/C land.

Run (once real, Phase E): python3 examples/math_grpo.py configs/qwen3_1.7b_grpo.yaml
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quic_rl.config.schema import ExperimentConfig
from quic_rl.orchestrator import lifecycle
from quic_rl.orchestrator.controller import Controller
from quic_rl.reward.math_verifier import MathVerifierReward
from quic_rl.rollout.base import GenerationRequest, SamplingParams


def build_prompt_source(config: ExperimentConfig):
    """Loads `config.dataset.name` (DAPO-Math-17k / GSM8K), yields
    `GenerationRequest`s with `metadata={"ground_truth": ...}` -
    `MathVerifierReward` requires that field (see its own docstring)."""
    from datasets import load_dataset

    split = config.dataset.split if config.dataset.num_examples is None else f"{config.dataset.split}[:{config.dataset.num_examples}]"
    ds = load_dataset(config.dataset.name, split=split)
    counter = itertools.count()

    def source() -> list[GenerationRequest]:
        batch = []
        for _ in range(8):  # prompts per iteration - a real tunable, not fixed here
            try:
                ex = next(iter(ds))
            except StopIteration:
                break
            batch.append(
                GenerationRequest(
                    prompt_id=f"prompt-{next(counter)}", prompt=ex["prompt"],
                    num_samples=config.rollout.samples_per_prompt,
                )
            )
        return batch

    return source


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen3_1.7b_grpo.yaml"
    config = ExperimentConfig.from_file(config_path)

    # The two imports below are the ONLY thing separating this from
    # examples/mock_loop.py - see this file's own module docstring.
    from quic_rl.rollout.quic_vllm import QuicVLLMRollout  # noqa: PLC0415 - imported lazily so mock_loop.py-only usage never needs these installed
    from quic_rl.trainer.quic_train import QuicTrainBackend

    rollout = QuicVLLMRollout(driver_url="http://localhost:8080")  # real launch details: docs/ARCHITECTURE.md
    trainer = QuicTrainBackend(config=config)
    reward = MathVerifierReward()

    from quic_rl.synchronization.weights import MockWeightSynchronizer  # TODO(Phase C): QuicVLLMWeightSynchronizer

    weight_synchronizer = MockWeightSynchronizer()

    initial_version = lifecycle.initialize(rollout, trainer, initial_policy_path=config.model.name)
    controller = Controller(
        rollout=rollout, trainer=trainer, reward=reward, weight_synchronizer=weight_synchronizer,
        prompt_source=build_prompt_source(config), sampling=SamplingParams(temperature=1.0, max_tokens=1024),
        state_dir=config.state_dir, rollout_workers=config.rollout.workers,
        metrics_path=f"{config.state_dir}/metrics.jsonl",
    )
    controller.resume_or_start(initial_version)
    results = controller.run(max_iterations=config.max_iterations)

    for r in results:
        print(f"iter={r.iteration} policy_version={r.policy_version} train_loss={r.train_loss:.4f} "
              f"reward_mean={r.reward_mean:.4f}", flush=True)

    lifecycle.shutdown(rollout, trainer)


if __name__ == "__main__":
    main()
