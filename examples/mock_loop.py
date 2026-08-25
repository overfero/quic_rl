"""Runs the full orchestration loop with zero GPUs, zero real models,
zero network calls - `MockRolloutBackend`/`MockTrainerBackend`/
`MockRewardBackend`/`MockWeightSynchronizer` stand in for the real
quic-vllm/quic-train/MathVerifierReward/real weight sync. This is the
"test the whole loop before attempting real multi-node training" step
the prompt explicitly asks for - if this script doesn't produce a
coherent multi-iteration run, nothing downstream will either.

Run: python3 examples/mock_loop.py
"""
from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quic_rl.orchestrator import lifecycle
from quic_rl.orchestrator.controller import Controller
from quic_rl.reward.mock import MockRewardBackend
from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.rollout.mock import MockRolloutBackend
from quic_rl.synchronization.weights import MockWeightSynchronizer
from quic_rl.trainer.mock import MockTrainerBackend

STATE_DIR = "/tmp/quic_rl_mock_loop_state"
NUM_ITERATIONS = 5
PROMPTS_PER_ITERATION = 4
GROUP_SIZE = 4  # GRPO's samples-per-prompt


def make_prompt_source():
    counter = itertools.count()

    def source() -> list[GenerationRequest]:
        return [
            GenerationRequest(prompt_id=f"prompt-{next(counter)}", prompt="What is 2 + 2?", num_samples=GROUP_SIZE)
            for _ in range(PROMPTS_PER_ITERATION)
        ]

    return source


def main() -> None:
    shutil.rmtree(STATE_DIR, ignore_errors=True)  # fresh run each time this example is invoked directly

    rollout = MockRolloutBackend(seed=0)
    trainer = MockTrainerBackend()
    reward = MockRewardBackend(target_length=6)
    weight_synchronizer = MockWeightSynchronizer(simulated_latency_s=0.05)

    initial_version = lifecycle.initialize(rollout, trainer, initial_policy_path="mock://qwen3-1.7b-base")
    print(f"initialized: policy_version={initial_version}", flush=True)

    controller = Controller(
        rollout=rollout,
        trainer=trainer,
        reward=reward,
        weight_synchronizer=weight_synchronizer,
        prompt_source=make_prompt_source(),
        sampling=SamplingParams(temperature=1.0, max_tokens=64),
        state_dir=STATE_DIR,
        rollout_workers=2,
        reward_workers=2,
        metrics_path=f"{STATE_DIR}/metrics.jsonl",
    )
    controller.resume_or_start(initial_version)

    results = controller.run(max_iterations=NUM_ITERATIONS)

    print("\n=== mock loop results ===", flush=True)
    for r in results:
        print(
            f"iter={r.iteration} policy_version={r.policy_version} "
            f"train_loss={r.train_loss:.4f} reward_mean={r.reward_mean:.4f} "
            f"sync_overhead_s={r.sync_overhead_s:.4f}",
            flush=True,
        )

    lifecycle.shutdown(rollout, trainer)
    print(f"\nmetrics written to {STATE_DIR}/metrics.jsonl", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
