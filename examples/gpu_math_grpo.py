"""Real, runnable full orchestrator loop across TWO SEPARATE machines:
quic-train's own TPU host (full 8 chips, no chip-sharing needed at all -
see SshMultiMachineTrainBackend's own `tpu`/`tensor_parallel_size`
fields) for training, and a genuinely separate CUDA GPU box (2xT4)
running this project's OWN custom quic-vllm fork (github.com/overfero/vllm
- real UDP hole-punch pipeline-parallel transport, NOT vanilla PyPI
vllm) for rollout generation - see this topology replaced the earlier
same-host 2+6 TPU chip-split attempt (a hard, currently-unresolvable
dependency ceiling on that single TPU host).

`full_finetune=True`: confirmed directly against
jaygala24/Qwen3-1.7B-GRPO-math-reasoning's own model card (the real
recreation target this whole backend exists for) that it trains with
genuine full-parameter fine-tuning (PipelineRL + Transformers +
DeepSpeed ZeRO Stage 3 - no LoRA/PEFT anywhere) - matching that means a
real full checkpoint gets moved per policy update, not a small LoRA
adapter.

Weight transfer uses REAL P2P QUIC (`QuicWeightSynchronizer` +
`quic_transfer.py`'s already-validated hole-punch transport, the SAME
one quic-train's own pipeline communication runs on) directly between
the TPU and GPU machines - confirmed working end-to-end with a real
file transfer through a PUBLIC zrok signaling URL (`zrok share public
localhost:8000` on the TPU machine, since the two Kaggle VMs cannot
reach each other's private SSH-tunnel-only endpoints directly). This
avoids relaying multi-GB checkpoints through this orchestrator's own
(comparatively low-bandwidth) machine - see quic_transfer.py's own
docstring for the deeper rationale.

This orchestrator process itself runs on NEITHER Kaggle machine - it
runs wherever it has SSH access (via ~/.ssh/config aliases) to BOTH:
`SshMultiMachineTrainBackend` launches quic-train's rank subprocess on
the TPU machine over SSH; `QuicWeightSynchronizer` (weight bytes, real
QUIC P2P) + `ssh_launcher.SshMultiMachineStageLauncher` (server restart,
the custom vllm fork's OWN `scripts/launch_pp_stage.py --transport quic`
- a SINGLE stage here, n==1, this model fits on one T4) together make
the GPU machine serve the new policy; `QuicVLLMRollout`'s own HTTP
generation calls go through the SshMultiMachineStageLauncher's own local
SSH port-forward (small, frequent request/response bodies - unlike the
checkpoint, cheap enough over the existing SSH tunnel).

Run:
  python3 examples/gpu_math_grpo.py \\
    --tpu-ssh-alias Kaggle_Zrok2 \\
    --tpu-quic-dist-repo-dir /kaggle/working/quic_dist \\
    --tpu-state-dir /kaggle/working/quic_rl_state \\
    --gpu-ssh-alias Kaggle_Zrok3 \\
    --gpu-vllm-repo-dir /kaggle/working/vllm \\
    --gpu-work-dir /kaggle/working/gpu_vllm_state \\
    --public-signaling-url https://gj5t9o1guwr2.share.zrok.io \\
    --local-state-dir /tmp/quic_rl_gpu_math_state \\
    --max-iterations 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quic_rl.dataset.math_dataset import build_prompt_source
from quic_rl.orchestrator import lifecycle
from quic_rl.orchestrator.controller import Controller
from quic_rl.reward.math_verifier import MathVerifierReward
from quic_rl.rollout.base import SamplingParams
from quic_rl.rollout.quic_vllm import QuicVLLMRollout
from quic_rl.rollout.ssh_launcher import RemoteMachine, SshMultiMachineStageLauncher
from quic_rl.rollout.vllm_serve_launcher import NoOpStageLauncher
from quic_rl.synchronization.weights import QuicWeightSynchronizer
from quic_rl.trainer.quic_train_multi import SshMultiMachineTrainBackend, TrainerMachine


def _cleanup_remote_exports(tpu_alias: str, exports_root: str, keep_latest: int = 1) -> None:
    """Real disk-usage cleanup ON THE TPU MACHINE - `Controller`'s own
    `PolicyRegistry.register()` (called from `_sync_policy()`) only
    prunes a LOCAL path, and this orchestrator deliberately never pulls
    the actual checkpoint files to its own local disk (see this file's
    own module docstring - QuicWeightSynchronizer sends them P2P
    straight from the TPU machine instead), so PolicyRegistry's own
    pruning is silently a no-op here - without this, every iteration's
    full checkpoint (multi-GB, full_finetune=True) would accumulate on
    the TPU machine's disk forever."""
    out = subprocess.run(
        ["ssh", tpu_alias, f"ls -1 {exports_root} 2>/dev/null"],
        capture_output=True, text=True, timeout=20,
    )
    versions = sorted(
        (d for d in out.stdout.split() if d.startswith("v") and d[1:].isdigit()),
        key=lambda d: int(d[1:]),
    )
    for stale in versions[:-keep_latest] if keep_latest > 0 else versions:
        subprocess.run(["ssh", tpu_alias, f"rm -rf {exports_root}/{stale}"], timeout=60, check=False)


def _upload_checkpoint_to_drive(tpu_alias: str, local_export_dir: str, gdrive_folder: str, label: str) -> None:
    """Real rclone copy ON THE TPU MACHINE (rclone runs there, already
    configured - no relay through this orchestrator's own link) straight
    to Drive, THEN deletes the previous `label` version - never both
    versions present at once for long, and never more than one `label`
    checkpoint accumulating on Drive (this project's own explicit
    instruction: "hanya ada best dan last... hapus model lamanya biar ga
    menuhi drive"). Uploads to a staging name first so a failed/partial
    upload never destroys the still-good previous version."""
    staging = f"{gdrive_folder}/{label}_staging"
    final = f"{gdrive_folder}/{label}"
    subprocess.run(
        ["ssh", tpu_alias, f"rclone purge gdrive:{staging} 2>/dev/null; "
                            f"rclone copy {local_export_dir} gdrive:{staging} --transfers 8 --checkers 8"],
        timeout=1800, check=True,
    )
    subprocess.run(
        ["ssh", tpu_alias, f"rclone purge gdrive:{final} 2>/dev/null; "
                            f"rclone moveto gdrive:{staging} gdrive:{final}"],
        timeout=300, check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3-1.7B")
    p.add_argument("--tpu-ssh-alias", required=True)
    p.add_argument("--tpu-quic-dist-repo-dir", required=True)
    p.add_argument("--tpu-state-dir", required=True, help="path ON THE TPU MACHINE")
    p.add_argument("--tpu-tensor-parallel-size", type=int, default=8, help="full host - no chip split needed anymore")
    p.add_argument("--local-signaling-url", default="http://localhost:8000",
                    help="reachable FROM the TPU machine only - used for quic-train's own single-rank init")
    p.add_argument("--public-signaling-url", required=True,
                    help="a real, internet-reachable URL for the SAME signaling server (e.g. `zrok share public "
                         "localhost:8000` run on the TPU machine) - required for the cross-machine QUIC weight "
                         "transfer, since the two Kaggle VMs can't reach each other's private tunnels directly")
    p.add_argument("--gpu-ssh-alias", required=True)
    p.add_argument("--gpu-vllm-repo-dir", required=True, help="this project's own custom quic-vllm fork, path ON THE GPU MACHINE")
    p.add_argument("--gpu-vllm-venv", default="/vllm_build_venv",
                    help="setup_inference_machine.sh's own default build venv path ON THE GPU MACHINE")
    p.add_argument("--gpu-work-dir", required=True, help="path ON THE GPU MACHINE")
    p.add_argument("--gpu-driver-port", type=int, default=8080)
    p.add_argument("--gpu-tensor-parallel-size", type=int, default=2,
                    help="the rollout machine has 2xT4 - default uses both via real tensor parallelism instead "
                         "of leaving the second GPU fully idle. Set to 1 to pin to a single GPU (cuda_device 0).")
    p.add_argument("--gpu-max-num-seqs", type=int, default=64,
                    help="SshMultiMachineStageLauncher's own default (4) is tuned for quic-vllm's real "
                         "multi-machine pipeline deployment, not one GRPO iteration's real concurrent batch "
                         "(group_size x prompts_per_iteration completions all in flight at once) - confirmed "
                         "directly this bottlenecks real throughput badly on an otherwise-idle T4")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                    help="same reasoning as --gpu-max-num-seqs - the launcher's own 0.5 default leaves real "
                         "KV-cache headroom on the table for a single-model, single-GPU workload like this one. "
                         "0.95 (not 1.0) still leaves a small margin for CUDA context/allocator overhead")
    p.add_argument("--local-state-dir", required=True, help="path on THIS orchestrator's own machine")
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--group-size", type=int, default=16, help="matches jaygala24's own GRPO group size")
    p.add_argument("--prompts-per-iteration", type=int, default=4,
                    help="real dataset (gsm8k_train+math_train, ~20k examples) needs a real batch size per "
                         "iteration to finish in a tractable step count - see --max-iterations' own help")
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=8192,
                    help="real output budget - 8192 max generated tokens per completion. Total context "
                         "(vLLM's --max-model-len) is this + --max-prompt-len = 8704, so the model's own "
                         "position budget covers prompt + full 8k output with room to spare, never truncating "
                         "the output itself. Qwen3-1.7B's real max_position_embeddings is 40960, so this still "
                         "leaves plenty of headroom.")
    p.add_argument("--num-examples", type=int, default=None,
                    help="None (default) = the FULL combined gsm8k_train+math_train dataset, not a slice")
    p.add_argument("--max-iterations", type=int, default=None,
                    help="None (default) = computed from the real dataset size so every example gets seen at "
                         "least once (ceil(total_examples / prompts_per_iteration)) - can exceed 1500 (the "
                         "reference's own step count) since that run never covered the whole dataset either")
    p.add_argument("--checkpoint-interval", type=int, default=25,
                    help="export+upload the running 'last' checkpoint to Drive every this many iterations - "
                         "every iteration would be a genuine multi-GB export+upload each time, real overhead "
                         "this project explicitly asked to keep low (\"efisiensikan pipeline\")")
    p.add_argument("--gdrive-folder", default="Models/Qwen3-1.7B-GRPO-math-recreation",
                    help="path under the TPU machine's own rclone 'gdrive:' remote - holds only best/ and last/, "
                         "each ever ONE checkpoint (the old one is deleted right after a new upload succeeds)")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    if args.num_examples is None or args.max_iterations is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from quic_rl.dataset.math_dataset import load_combined_math_examples

        total_examples = len(load_combined_math_examples(args.num_examples))
        if args.max_iterations is None:
            import math

            args.max_iterations = math.ceil(total_examples / args.prompts_per_iteration)
        print(f"real dataset size: {total_examples} examples -> max_iterations={args.max_iterations} "
              f"(prompts_per_iteration={args.prompts_per_iteration})", flush=True)

    tpu_machine = TrainerMachine(
        name="tpu", ssh_alias=args.tpu_ssh_alias, cuda_devices=["0"],
        quic_dist_repo_dir=args.tpu_quic_dist_repo_dir, state_dir=args.tpu_state_dir,
    )
    gpu_cuda_device = ",".join(str(i) for i in range(args.gpu_tensor_parallel_size))
    gpu_machine = RemoteMachine(name="gpu", ssh_alias=args.gpu_ssh_alias, cuda_device=gpu_cuda_device)

    trainer = SshMultiMachineTrainBackend(
        machines=[tpu_machine],
        signaling_url=args.local_signaling_url, num_layers=args.num_layers,
        full_finetune=True, quantization="none", compute_dtype="bfloat16",
        tpu=True, tensor_parallel_size=args.tpu_tensor_parallel_size,
        max_prompt_len=args.max_prompt_len, kl_coef=0.0, lr=1e-6,
        # Matches jaygala24/Qwen3-1.7B-GRPO-math-reasoning's model card exactly.
        grad_clip=0.3, weight_decay=0.01,
        # This orchestrator's own machine needs a LOCAL quic_dist checkout
        # (rust extension built, no TPU/GPU use - purely the RolloutBatch
        # dataclass import) - see SshMultiMachineTrainBackend's own field
        # docstring for why the remote-path default is wrong here.
        local_quic_dist_parent_dir=str(Path(__file__).resolve().parents[2]),
    )
    # The rollout server's own serving cap must stay strictly ABOVE what
    # training will ever actually request (prompt + max_new_tokens) - a
    # margin exactly equal to that sum leaves zero room for chat-template/
    # special-token overhead added on top of the raw prompt, which would
    # silently truncate real generations right at the boundary instead of
    # ever hitting max_new_tokens cleanly. ROLLOUT_CONTEXT_MARGIN keeps
    # vLLM's real ceiling comfortably above training's real requirement.
    ROLLOUT_CONTEXT_MARGIN = 512
    real_stage_launcher = SshMultiMachineStageLauncher(
        vllm_repo_dir=args.gpu_vllm_repo_dir, machines=[gpu_machine], signaling_url=args.public_signaling_url,
        max_model_len=args.max_prompt_len + args.max_new_tokens + ROLLOUT_CONTEXT_MARGIN,
        driver_port=args.gpu_driver_port,
        remote_log_dir=args.gpu_work_dir,
        # See --gpu-max-num-seqs/--gpu-memory-utilization's own help text -
        # the launcher's own defaults badly bottleneck a real GRPO batch
        # (confirmed directly: max_num_seqs=4 forces group_size x
        # prompts_per_iteration completions through in small serial
        # batches instead of all at once on an otherwise-idle T4).
        max_num_seqs=args.gpu_max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=None, tensor_parallel_size=args.gpu_tensor_parallel_size,
    )
    weight_synchronizer = QuicWeightSynchronizer(
        receiver=gpu_machine,
        remote_policy_root=f"{args.gpu_work_dir}/policy_versions",
        remote_quic_transfer_script=f"{args.gpu_work_dir}/../quic_rl/quic_rl/synchronization/quic_transfer.py",
        remote_quic_dist_repo_dir=f"{args.gpu_work_dir}/../quic_dist",
        local_quic_dist_repo_dir=args.tpu_quic_dist_repo_dir,  # a path ON THE SENDER (tpu_machine) - see class docstring
        signaling_url=args.public_signaling_url,
        stage_launcher=real_stage_launcher,
        sender=tpu_machine,
        sender_quic_transfer_script=f"{args.tpu_quic_dist_repo_dir}/../quic_rl/quic_rl/synchronization/quic_transfer.py",
    )
    reward = MathVerifierReward()

    rollout = QuicVLLMRollout(
        driver_url=real_stage_launcher.driver_url(),
        stage_launcher=NoOpStageLauncher(),  # the REAL restart already happens inside QuicWeightSynchronizer.sync()
        model_name=args.model_path,
        # QuicVLLMRollout's own 120s default is nowhere near enough for a
        # real request here: group_size x max_new_tokens completions in
        # ONE /v1/completions call (e.g. 16 x 2048 tokens), PLUS the
        # custom fork's own first-request compile/warmup cost on a
        # single T4 - confirmed directly: the first real generate() call
        # timed out at 120s with no result at all.
        request_timeout=1800.0,
    )

    try:
        # Bootstrap: nothing ever calls real_stage_launcher.restart() for
        # the INITIAL base model otherwise - lifecycle.initialize() only
        # calls rollout.load_policy() (a no-op here, see NoOpStageLauncher's
        # own docstring), and QuicWeightSynchronizer.sync() (which drives
        # the real launcher) only runs from INSIDE the training loop,
        # after the first real training step. Downloads the base model
        # into the SAME `{remote_policy_root}/v0/stage0` layout later
        # checkpoints will use, directly on the GPU machine (its own
        # transformers/huggingface_hub, exactly like quic-train's own
        # build_stage_model() resolves an HF repo id - no cross-machine
        # transfer needed for a public HF model).
        bootstrap_dir = f"{args.gpu_work_dir}/policy_versions/v0"
        subprocess.run(
            ["ssh", args.gpu_ssh_alias,
             f"mkdir -p {bootstrap_dir}/stage0 && {args.gpu_vllm_venv}/bin/python3 -c "
             f"\"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{args.model_path}', local_dir='{bootstrap_dir}/stage0')\""],
            timeout=1800, check=True,
        )
        real_stage_launcher.restart(bootstrap_dir)

        # restart() only launches the process and returns (see
        # StageLauncher's own Protocol docstring: "NOT necessarily once
        # the driver's HTTP API is healthy yet") - real model load +
        # compilation on the GPU machine takes real time (minutes, not
        # the ~2s restart() itself sleeps for the port-forward to
        # establish). QuicVLLMRollout.load_policy() polls this for every
        # LATER policy update; this bootstrap call bypassed that (calling
        # the launcher directly, not through load_policy()), so it needs
        # its own explicit wait here - confirmed directly: without this,
        # the very first generate() call hits a real Connection Refused
        # / timeout because nothing is listening on driver_port yet.
        bootstrap_deadline = time.monotonic() + real_stage_launcher.transport_connect_timeout + 300.0
        while not rollout.health_check():
            if time.monotonic() > bootstrap_deadline:
                raise RuntimeError(
                    "gpu_math_grpo.py: the bootstrap vLLM fork server never became healthy - "
                    f"check {args.gpu_work_dir}/quic_rl_stage_gpu.log on {args.gpu_ssh_alias}"
                )
            time.sleep(5.0)

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
            state_dir=args.local_state_dir, metrics_path=f"{args.local_state_dir}/metrics.jsonl",
            wandb_project=args.wandb_project, wandb_run_name=args.wandb_run_name,
            # Real bug found reading this path: Controller's own default
            # (rollout_workers=1) sends QuicVLLMRollout.generate() ONE
            # prompt at a time - group_size (n) samples of the SAME prompt
            # are genuinely parallel (vLLM's own `n` param, one request),
            # but different prompts within one iteration were serialized
            # through the same blocking HTTP call, leaving most of the
            # rollout machine's real verified concurrency (24x at the
            # current 9216-token cap) idle. One worker per prompt fans all
            # of them out concurrently via collect_rollouts()'s own
            # ThreadPoolExecutor path instead.
            rollout_workers=args.prompts_per_iteration,
        )
        controller.resume_or_start(initial_version)

        exports_root = f"{args.local_state_dir}/exports"  # same string is valid on both this process's own
        # disk AND the TPU machine's - see Controller._sync_policy()'s own export_dir construction and this
        # file's own module docstring for why that's true here (both are Kaggle-VM-style /tmp roots)
        best_reward_mean = float("-inf")

        t_start = time.monotonic()
        results = []
        # Manual per-iteration loop (not controller.run(max_iterations=...))
        # so real remote-disk cleanup and best/last Drive upload can run
        # right after EACH iteration's own export - see
        # _cleanup_remote_exports's/_upload_checkpoint_to_drive's own
        # docstrings for why both are real, load-bearing steps here, not
        # just logging.
        for i in range(args.max_iterations):
            result = controller._run_one_iteration()
            results.append(result)
            print(f"iter={result.iteration} policy_version={result.policy_version} train_loss={result.train_loss:.4f} "
                  f"reward_mean={result.reward_mean:.4f} reward_std={result.reward_std:.4f} "
                  f"sync_overhead_s={result.sync_overhead_s:.2f}", flush=True)

            latest_export_dir = f"{exports_root}/v{result.policy_version}"
            is_last_checkpoint = (i + 1) % args.checkpoint_interval == 0 or i == args.max_iterations - 1
            is_best_checkpoint = result.reward_mean > best_reward_mean
            if is_best_checkpoint:
                best_reward_mean = result.reward_mean

            if is_last_checkpoint:
                _upload_checkpoint_to_drive(args.tpu_ssh_alias, latest_export_dir, args.gdrive_folder, "last")
            if is_best_checkpoint:
                _upload_checkpoint_to_drive(args.tpu_ssh_alias, latest_export_dir, args.gdrive_folder, "best")
            _cleanup_remote_exports(args.tpu_ssh_alias, exports_root, keep_latest=1)

        print(f"total wall time for {len(results)} iterations: {time.monotonic() - t_start:.1f}s", flush=True)
        print(f"best reward_mean seen: {best_reward_mean:.4f}", flush=True)

        lifecycle.shutdown(rollout, trainer)
        trainer.shutdown()
    finally:
        real_stage_launcher.shutdown()


if __name__ == "__main__":
    main()
