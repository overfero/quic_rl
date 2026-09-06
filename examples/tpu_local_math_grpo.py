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

QuicTrainBackend's own hyperparameters here (kl_coef=0.0, lr=1e-6,
grad_clip=0.3, weight_decay=0.01) match
jaygala24/Qwen3-1.7B-GRPO-math-reasoning's model card exactly - see
quic_dist/examples/configs/tpu_qwen3_1.7b_grpo_math_real.yaml's own
comments for the same reproduction via quic_dist's OWN generation path
(run_grpo_math_training) instead of this file's vLLM-rollout path.
`--group-size` defaults to 8, not the reference's 16: confirmed
directly that group_size=16 OOMs `_grpo_update_from_rollout`'s
GSM8K-forced fp32 logits cast even at tensor_parallel_size=8 (see
rlhf.py's own history) - splitting 2 chips off for vLLM leaves
quic-train only 6, making that OOM MORE likely, not less, so 8 stays
the real, working ceiling until that cast gets a genuine memory fix.

Run:
  python3 examples/tpu_local_math_grpo.py \\
    --quic-dist-repo-dir /kaggle/working/quic_dist \\
    --vllm-venv-python /kaggle/working/vllm_deploy/venv/bin/python3 \\
    --state-dir /kaggle/working/quic_rl_state \\
    --hf-home /hf_cache \\
    --vllm-tensor-parallel-size 2 \\
    --quic-train-tensor-parallel-size 6 \\
    --max-iterations 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Deliberately does NOT set LIBTPU_INIT_ARGS here - tried that (both
# tpu_inference's single flag alone, and a hand-merged string containing
# every flag both sides' own init code would otherwise add on their own)
# and made things WORSE: injecting `--xla_tpu_use_dynamic_smem_negotiation
# =true` into the ORCHESTRATOR's env means BOTH children inherit it via
# their own plain `dict(os.environ)` copies, including quic-train's rank
# subprocess - and confirmed directly that torch_xla crashes on THIS
# flag specifically whenever it isn't the process that ends up "owning"
# this host's shared libtpu runtime coordination service (the
# "SliceBuilder" service named in that service's own warning) - i.e. the
# flag itself isn't safe to force onto torch_xla at all, matching or not.
# Every test that actually WORKED (see `_local_vllm_worker.py`'s own
# docstring and this repo's own validation history) left LIBTPU_INIT_ARGS
# completely untouched and let each side's own library compute its own
# natural defaults (torch_xla's `_setup_libtpu_flags()`; tpu_inference's
# `env_override.py`) - the ACTUAL fix is `--tpu-init-stagger-s` below,
# not this flag.
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
    p.add_argument("--tpu-init-stagger-s", type=float, default=20.0,
                    help="delay between starting quic-train's rank process and vLLM's worker - "
                         "see this file's own LIBTPU_INIT_ARGS comment for why: even with IDENTICAL "
                         "flags, confirmed directly that two processes touching this host's TPU at "
                         "the same wall-clock instant race on a shared local coordination service "
                         "and one of them gets rejected outright; staggering by a real margin (not a "
                         "flag fix) is what actually avoids it")
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
        # Matches jaygala24/Qwen3-1.7B-GRPO-math-reasoning's model card exactly.
        grad_clip=0.3, weight_decay=0.01,
    )
    reward = MathVerifierReward()
    weight_synchronizer = LocalWeightSynchronizer()

    initial_version = lifecycle.initialize(rollout, trainer, initial_policy_path=args.model_path)

    # quic-train's rank subprocess just launched (inside initialize_policy())
    # and is already touching the TPU; give it a real head start before
    # LocalVLLMRollout's OWN first generate() call starts its worker and
    # touches the TPU too - see --tpu-init-stagger-s's own help text.
    time.sleep(args.tpu_init_stagger_s)

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
