"""`QuicTrainBackend`: the real `TrainerBackend` - launches quic-train's
own GRPO rank processes (`quic_dist/examples/grpo_external_rollout_rank.py`,
wrapping `quic_dist.rlhf.run_grpo_training_from_rollouts`) and feeds them
real, already-scored rollout batches from the orchestrator via quic-rl's
own file-based handoff (`RolloutBatch` files - see that launcher script's
docstring for the exact write/poll protocol). Reuses quic-train's EXACT
GRPO update math unmodified - this backend's only job is turning
`list[Trajectory]` into the `(prompt_ids, generated, rewards)` shape that
math already expects, and getting the real per-step result back.

Tokenization: uses the SAME tokenizer the trainer's own model loads
(`AutoTokenizer.from_pretrained(policy_path)`, matching every other RLHF
mode in `quic_dist/rlhf.py`). Response tokens come from
`Trajectory.token_ids` when the rollout backend populated them (the
sampler's own actual token ids - no re-tokenization ambiguity), falling
back to tokenizing `Trajectory.response` when it didn't.

Group handling: `_grpo_update_from_rollout` requires every sample in a
group to be the SAME length N - there is no padding mask in that code
path (`generated: (G, N)`, used as labels directly). Real rollouts can
finish at different lengths (different `finish_reason`). This backend
TRUNCATES every sample in a group to the group's shortest real
generation rather than padding, so no fabricated pad token ever enters
the loss - the only choice that keeps that (correct, unmodified) math
correct without touching it.

Only a LOCAL, single-machine launch is implemented (one subprocess per
rank on this machine's own GPUs) - matches `rollout/local_launcher.py`'s
role for quic-vllm: prove the real pipeline correctness here first. An
SSH multi-machine trainer launcher (mirroring `rollout/ssh_launcher.py`)
is real future work, not built yet.

`export_policy()`/`checkpoint()`: real. `checkpoint()` copies the raw
per-rank checkpoint files (full optimizer/RNG state, for training
resume) verbatim, backed by `run_grpo_training_from_rollouts`'s own
`training_utils.save_checkpoint` wiring. `export_policy()` does NOT load
a model in this (orchestrator) process at all - it sends a file-based
export REQUEST that rank 0's own `on_checkpoint` hook services directly
from its already-GPU-resident model (see
`grpo_external_rollout_rank.py`'s `_check_export_request`), then just
copies the (static, training never changes it) tokenizer files here.
Real bug this fixes: an earlier version loaded a SECOND full model copy
in this process for export - rank processes never exit between train()
calls, so with both still resident that second load pushed a real
machine over its actual RAM ceiling (not a GPU/VRAM problem - the GPUs
had headroom the whole time; this was host RAM exhaustion from 3
simultaneous full-model loads on one box).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

import torch
import yaml

from quic_rl.trainer.base import TrainResult
from quic_rl.trajectory import Trajectory


@dataclass
class QuicTrainBackend:
    quic_dist_repo_dir: str
    signaling_url: str
    world_size: int
    num_layers: int
    # Rollout batch/result files + the written GRPOConfig live under
    # here, AND (checkpoints/checkpoints/) - real disk usage found
    # running this for real: one full-parameter checkpoint is ~7.5GB
    # PER RANK, and with checkpoint_keep_last=2 plus one "best" copy
    # (see rlhf.py's own checkpoint-saving docstring) that's up to
    # ~22.5GB/rank, ~45GB total for world_size=2 - MUST be a real disk
    # path (e.g. under /data), never /kaggle/working, which is commonly
    # just a small (~20GB) loop device that filled up completely from
    # exactly this during development.
    state_dir: str

    cuda_devices: list[str] | None = None  # one entry per rank; defaults to "0".."world_size-1"
    # True: genuine full-parameter fine-tuning (no LoRA) - see
    # finetune.PipelineConfig.full_finetune's own docstring for the real
    # mechanics/constraints. Requires quantization="none" and kl_coef=0.0
    # (checked in initialize_policy() - fails loudly rather than
    # launching rank processes that would crash on the first real step).
    full_finetune: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    quantization: str = "4bit"
    compute_dtype: str = "bfloat16"
    max_prompt_len: int = 512
    kl_coef: float = 0.05
    lr: float = 1e-4
    # Micro-steps accumulated before a real optimizer.step() - see
    # quic_dist.rlhf.run_grpo_training_from_rollouts's own docstring.
    gradient_accumulation_steps: int = 1
    job_id: str = "quic_rl_grpo"
    step_result_timeout_s: float = 600.0
    step_result_poll_interval_s: float = 1.0
    rank_stagger_s: float = 0.5

    _procs: list[subprocess.Popen] = field(default_factory=list, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _RolloutBatch: object | None = field(default=None, init=False, repr=False)
    _policy_version: int | None = field(default=None, init=False, repr=False)
    _next_step: int = field(default=1, init=False, repr=False)
    _next_export_id: int = field(default=0, init=False, repr=False)
    _rollout_dir: str = field(default="", init=False, repr=False)
    _config_path: str = field(default="", init=False, repr=False)
    _checkpoint_dir: str = field(default="", init=False, repr=False)
    _policy_path: str = field(default="", init=False, repr=False)

    def _rollout_batch_cls(self):
        if self._RolloutBatch is None:
            parent = os.path.dirname(os.path.abspath(self.quic_dist_repo_dir))
            if parent not in sys.path:
                sys.path.insert(0, parent)
            from quic_dist.rlhf import RolloutBatch

            self._RolloutBatch = RolloutBatch
        return self._RolloutBatch

    def initialize_policy(self, policy_path: str) -> int:
        if self.full_finetune and self.quantization != "none":
            raise ValueError(
                f"QuicTrainBackend: full_finetune=True requires quantization='none' (got "
                f"{self.quantization!r}) - see finetune.PipelineConfig.full_finetune's docstring"
            )
        if self.full_finetune and self.kl_coef != 0.0:
            raise ValueError(
                "QuicTrainBackend: full_finetune=True requires kl_coef=0.0 - a full-parameter "
                "model has no LoRA adapter for the free-reference-model trick to disable, and "
                "quic-train's GRPO path only skips the reference forward pass when kl_coef==0 "
                "(see rlhf.py's _grpo_update_from_rollout)"
            )

        os.makedirs(self.state_dir, exist_ok=True)
        self._rollout_dir = os.path.join(self.state_dir, "rollouts")
        os.makedirs(self._rollout_dir, exist_ok=True)
        self._checkpoint_dir = os.path.join(self.state_dir, "checkpoints")
        self._policy_path = policy_path

        cfg = {
            "model_path": policy_path,
            "world_size": self.world_size,
            "num_layers": self.num_layers,
            "full_finetune": self.full_finetune,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_target_modules": self.lora_target_modules,
            "quantization": self.quantization,
            "compute_dtype": self.compute_dtype,
            "max_prompt_len": self.max_prompt_len,
            "kl_coef": self.kl_coef,
            "lr": self.lr,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "checkpoint_dir": self._checkpoint_dir,
            "checkpoint_every": 1,  # every real optimizer.step() (window), not every micro-batch
            "checkpoint_keep_last": 2,
        }
        self._config_path = os.path.join(self.state_dir, "grpo_config.yaml")
        with open(self._config_path, "w") as f:
            yaml.safe_dump(cfg, f)

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(policy_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        devices = self.cuda_devices or [str(i) for i in range(self.world_size)]
        script = os.path.join(self.quic_dist_repo_dir, "examples", "grpo_external_rollout_rank.py")
        for rank in range(self.world_size):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank])
            log_f = open(os.path.join(self.state_dir, f"quic_train_rank{rank}.log"), "w")
            proc = subprocess.Popen(
                ["python3", "-u", script, self._config_path, str(rank), self.signaling_url,
                 self._rollout_dir, self.job_id],
                cwd=self.quic_dist_repo_dir, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            )
            self._procs.append(proc)
            if rank < self.world_size - 1:
                time.sleep(self.rank_stagger_s)

        self._policy_version = 0
        self._next_step = 1
        return self._policy_version

    def _tokenize_group(self, group: list[Trajectory]) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt_ids_list = self._tokenizer(
            group[0].prompt, truncation=True, max_length=self.max_prompt_len, add_special_tokens=True,
        )["input_ids"]
        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long)  # (1, prompt_len)

        per_sample_ids: list[list[int]] = []
        for t in group:
            ids = list(t.token_ids) if t.token_ids else self._tokenizer(t.response, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"QuicTrainBackend: trajectory prompt_id={t.prompt_id!r} has no generated tokens")
            per_sample_ids.append(ids)

        n = min(len(ids) for ids in per_sample_ids)
        generated = torch.tensor([ids[:n] for ids in per_sample_ids], dtype=torch.long)  # (G, n)
        return prompt_ids, generated, prompt_ids.numel() + generated.numel()

    def _write_rollout_batch(self, step: int, prompt_ids: torch.Tensor, generated: torch.Tensor, rewards: torch.Tensor) -> None:
        batch = self._rollout_batch_cls()(prompt_ids=prompt_ids, generated=generated, rewards=rewards)
        pt_path = os.path.join(self._rollout_dir, f"step_{step:06d}.pt")
        tmp_path = pt_path + ".tmp"
        torch.save(batch, tmp_path)
        os.rename(tmp_path, pt_path)

    def _wait_for_result(self, step: int) -> dict:
        result_path = os.path.join(self._rollout_dir, f"step_{step:06d}.result.json")
        deadline = time.monotonic() + self.step_result_timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(result_path):
                with open(result_path) as f:
                    return json.load(f)
            time.sleep(self.step_result_poll_interval_s)
        raise RuntimeError(
            f"QuicTrainBackend.train(): step {step} never produced a result within "
            f"{self.step_result_timeout_s}s - check quic_train_rank*.log under {self.state_dir}"
        )

    def train(self, batch: list[Trajectory]) -> TrainResult:
        if self._policy_version is None:
            raise RuntimeError("QuicTrainBackend.train() called before initialize_policy()")
        if not batch:
            raise ValueError("QuicTrainBackend.train() called with an empty batch")
        rewards = [t.reward for t in batch]
        if any(r is None for r in rewards):
            raise ValueError("QuicTrainBackend.train(): every trajectory must have .reward set before training")

        t0 = time.monotonic()
        groups: dict[str, list[Trajectory]] = {}
        for t in batch:
            groups.setdefault(t.prompt_id, []).append(t)

        step_numbers: list[int] = []
        total_tokens = 0
        for prompt_id, group in groups.items():
            if len(group) < 2:
                raise ValueError(
                    f"QuicTrainBackend.train(): prompt_id={prompt_id!r} has only {len(group)} "
                    f"sample(s) - GRPO's group-relative advantage needs at least 2"
                )
            prompt_ids, generated, n_tokens = self._tokenize_group(group)
            rewards_t = torch.tensor([t.reward for t in group], dtype=torch.float32)

            step = self._next_step
            self._next_step += 1
            self._write_rollout_batch(step, prompt_ids, generated, rewards_t)
            step_numbers.append(step)
            total_tokens += n_tokens

        results = [self._wait_for_result(step) for step in step_numbers]
        elapsed = time.monotonic() - t0

        self._policy_version += 1
        loss_mean = statistics.fmean(r["loss"] for r in results)
        kl_values = [r["kl"] for r in results if r["kl"] is not None]
        kl_mean = statistics.fmean(kl_values) if kl_values else None
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        completion_rate = sum(1 for t in batch if t.response.strip()) / len(batch)

        return TrainResult(
            policy_version=self._policy_version,
            train_loss=loss_mean,
            reward_mean=reward_mean,
            reward_std=reward_std,
            completion_rate=completion_rate,
            tokens_per_sec=(total_tokens / elapsed) if elapsed > 0 else 0.0,
            step_time_s=elapsed,
            kl=kl_mean,
        )

    def get_policy_version(self) -> int:
        if self._policy_version is None:
            raise RuntimeError("QuicTrainBackend: policy not initialized yet")
        return self._policy_version

    def _quic_dist_training_utils(self):
        # training_utils.py is a bare top-level module inside quic_dist's
        # OWN repo root (not a quic_dist.training_utils submodule import),
        # matching a real, pre-existing pattern rlhf.py itself uses - it
        # only resolves when the repo root itself is on sys.path, not
        # just its parent (which is enough for `import quic_dist` alone).
        # Same real bug hit in grpo_external_rollout_rank.py, fixed the
        # same way there.
        repo_root = os.path.abspath(self.quic_dist_repo_dir)
        parent = os.path.dirname(repo_root)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import training_utils

        return training_utils

    def export_policy(self, output_dir: str) -> str:
        if not self._checkpoint_dir:
            raise RuntimeError("QuicTrainBackend.export_policy() called before initialize_policy()")

        # Real RAM-exhaustion bug found running this for real: rank
        # processes never exit between train() calls, so a SECOND full
        # model load here (a separate orchestrator process, on the SAME
        # machine) pushed a real box over its actual RAM ceiling with
        # both rank processes still resident. Fixed by asking rank 0
        # (the one that, in today's full_finetune non-sharded reality,
        # already holds the complete model on its own GPU - see
        # build_stage_model's own docstring) to serialize DIRECTLY from
        # its already-loaded model instead - no second copy anywhere.
        # See grpo_external_rollout_rank.py's _check_export_request()
        # for the rank-side half of this file-based protocol.
        self._next_export_id += 1
        request_id = self._next_export_id
        request_path = os.path.join(self._rollout_dir, "export_request.json")
        tmp_path = request_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"output_dir": output_dir, "request_id": request_id}, f)
        os.rename(tmp_path, request_path)

        done_path = os.path.join(self._rollout_dir, f"export_done_{request_id}.json")
        deadline = time.monotonic() + self.step_result_timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(done_path):
                break
            time.sleep(self.step_result_poll_interval_s)
        else:
            raise RuntimeError(
                f"QuicTrainBackend.export_policy(): rank 0 never serviced the export request within "
                f"{self.step_result_timeout_s}s - check quic_train_rank0.log under {self.state_dir}"
            )

        # The model weights are already written by rank 0 above - the
        # tokenizer is static (training never changes it), so copying it
        # here (cheap: no model load, just small JSON/text files) is all
        # that's left to make output_dir a complete, real HF checkpoint.
        self._tokenizer.save_pretrained(output_dir)
        return output_dir

    def checkpoint(self, output_dir: str) -> str:
        if not self._checkpoint_dir:
            raise RuntimeError("QuicTrainBackend.checkpoint() called before initialize_policy()")
        training_utils = self._quic_dist_training_utils()
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        for rank in range(self.world_size):
            ckpt_path = training_utils.find_latest_checkpoint(self._checkpoint_dir, rank=rank)
            if ckpt_path is None:
                raise RuntimeError(f"QuicTrainBackend.checkpoint(): no checkpoint found for rank {rank}")
            shutil.copy2(ckpt_path, os.path.join(output_dir, os.path.basename(ckpt_path)))
        return output_dir

    def health_check(self) -> bool:
        return self._policy_version is not None and all(p.poll() is None for p in self._procs)

    def shutdown(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 30.0
        for proc in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        self._procs = []
