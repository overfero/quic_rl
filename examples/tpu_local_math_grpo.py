"""Real, runnable full orchestrator loop on a single shared 8-chip TPU
host: LocalVLLMRollout (vLLM-TPU, a RESIDENT worker process pinned to
chips [0, vllm_tensor_parallel_size)) + QuicTrainBackend (quic-train's
real GRPO rank process(es), pinned to the disjoint remainder via
`tpu_chip_offset=vllm_tensor_parallel_size`, LoRA + tensor_parallel_size
sharding) + MathVerifierReward (real GSM8K/MATH answer verification) +
LocalWeightSynchronizer (same-host, no network transfer) +
build_prompt_source (real gsm8k_train+math_train, matching the
reference reproduction's own training_config.yaml).

Both backends run CONCURRENTLY, not sequentially - see
`quic_rl/rollout/local_vllm.py`'s own module docstring for why this
split (chips 0..vllm_tensor_parallel_size-1 for vLLM, the rest for
quic-train) is now known to be physically valid on this host's real
2x4x1 topology, and why a resident vLLM worker (booted once) replaces
this file's earlier one-shot-subprocess-per-call design.

Unlike examples/math_grpo.py (written speculatively before Phase B/C
landed, against QuicVLLMRollout's cross-machine HTTP-driver model and
an outdated QuicTrainBackend constructor shape - not runnable as-is),
this script targets the ACTUAL current backend interfaces directly.

Run:
  python3 examples/tpu_local_math_grpo.py \\
    --quic-dist-repo-dir /kaggle/working/quic_dist \\
    --vllm-venv-python /kaggle/working/vllm_deploy/venv/bin/python3 \\
    --state-dir /kaggle/working/quic_rl_state \\
    --hf-home /hf_cache \\
    --vllm-tensor-parallel-size 2 \\
    --quic-train-tensor-parallel-size 6 \\
    --max-iterations 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quic_rl.dataset.math_dataset import build_prompt_source
from quic_rl.orchestrator import lifecycle
from quic_rl.orchestrator.controller import Controller
from quic_rl.reward.math_verifier import MathVerifierReward
from quic_rl.rollout.base import SamplingParams
from quic_rl.rollout.local_vllm import LocalVLLMRollout
from quic_rl.synchronization.weights import LocalWeightSynchronizer
from quic_rl.trainer.quic_train import QuicTrainBackend


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3-1.7B")
    p.add_argument("--quic-dist-repo-dir", required=True)
    p.add_argument("--vllm-venv-python", required=True)
    p.add_argument("--signaling-url", default="http://localhost:8000")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--hf-home", default=None)
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=2,
                    help="chips [0, N) - pinned for the resident vLLM rollout worker")
    p.add_argument("--quic-train-tensor-parallel-size", type=int, default=6,
                    help="chips [vllm_tensor_parallel_size, vllm_tensor_parallel_size+N) for quic-train's rank process(es)")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-iteration", type=int, default=1)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--num-examples", type=int, default=8, help="real dataset examples to cycle through")
    p.add_argument("--max-iterations", type=int, default=1)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    rollout = LocalVLLMRollout(
        model_path=args.model_path, vllm_venv_python=args.vllm_venv_python,
        work_dir=os.path.join(args.state_dir, "vllm_worker"),
        tensor_parallel_size=args.vllm_tensor_parallel_size, max_model_len=args.max_prompt_len + args.max_new_tokens,
        hf_home=args.hf_home,
    )
    trainer = QuicTrainBackend(
        quic_dist_repo_dir=args.quic_dist_repo_dir, signaling_url=args.signaling_url,
        world_size=1, num_layers=args.num_layers, state_dir=args.state_dir,
        quantization="none", compute_dtype="bfloat16", tensor_parallel_size=args.quic_train_tensor_parallel_size,
        tpu_chip_offset=args.vllm_tensor_parallel_size,
        max_prompt_len=args.max_prompt_len, kl_coef=0.0, lr=1e-6,
    )
    reward = MathVerifierReward()
    weight_synchronizer = LocalWeightSynchronizer()

    initial_version = lifecycle.initialize(rollout, trainer, initial_policy_path=args.model_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt_source = build_prompt_source(
        tokenizer, samples_per_prompt=args.group_size, prompts_per_iteration=args.prompts_per_iteration,
        num_examples=args.num_examples,
    )

    controller = Controller(
        rollout=rollout, trainer=trainer, reward=reward, weight_synchronizer=weight_synchronizer,
        prompt_source=prompt_source,
        sampling=SamplingParams(temperature=1.0, max_tokens=args.max_new_tokens),
        state_dir=args.state_dir, metrics_path=f"{args.state_dir}/metrics.jsonl",
        wandb_project=args.wandb_project, wandb_run_name=args.wandb_run_name,
    )
    controller.resume_or_start(initial_version)
    results = controller.run(max_iterations=args.max_iterations)

    for r in results:
        print(f"iter={r.iteration} policy_version={r.policy_version} train_loss={r.train_loss:.4f} "
              f"reward_mean={r.reward_mean:.4f} reward_std={r.reward_std:.4f} "
              f"sync_overhead_s={r.sync_overhead_s:.2f}", flush=True)

    lifecycle.shutdown(rollout, trainer)
    trainer.shutdown()


if __name__ == "__main__":
    main()
