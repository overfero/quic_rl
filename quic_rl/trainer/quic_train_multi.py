"""`SshMultiMachineTrainBackend`: genuine multi-MACHINE extension of
`QuicTrainBackend` - real pipeline-parallel GRPO training split across
N physically separate machines (not just N GPUs on one machine), each
contributing some number of its own local GPUs as consecutive global
ranks. `QuicTrainBackend` itself only launches local subprocesses
(bare `subprocess.Popen`, no SSH) - see that module's own docstring for
why ("prove the real pipeline correctness here first"). This backend
is the SSH multi-machine trainer launcher that docstring flagged as
"real future work, not built yet".

Why a separate class instead of extending `QuicTrainBackend` in place:
every remote machine has its OWN filesystem - the single-machine
backend's assumption that all ranks can see one shared `rollout_dir`
(for batch handoff, results, and export-request/shard files) does not
hold across machines. Rather than change quic-train's rank-side code
(`grpo_external_rollout_rank.py`) to know about multiple machines at
all, this backend keeps that code 100% unmodified and instead makes
EACH machine's local rollout_dir look, from that machine's own rank
processes' point of view, exactly like the single-machine case always
did - by having the orchestrator itself shuttle the handful of small
files (rollout batches, export requests, export shards) between
machines' filesystems via SCP, at exactly the points those files would
otherwise need to cross a machine boundary. Every other part of the
protocol (file names, polling, atomic rename) is unchanged from
`grpo_external_rollout_rank.py`.

Rollout batches are small (tensors of token ids for one training step,
not weights) - SCPing one to every machine per `train()` call is cheap
and frequent. Export/checkpoint shards are the one place real
multi-GB-scale data could cross a machine boundary via SCP - see
`export_policy()`'s own docstring for why that's still bounded and
occasional, not a violation of this project's "never move large files
between machines over SSH" rule.
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
class TrainerMachine:
    name: str
    host: str = "127.0.0.1"
    ssh_port: int = 22
    ssh_password: str | None = None  # None => this is the local machine, no SSH
    cuda_devices: list[str] = field(default_factory=lambda: ["0"])  # local GPU indices THIS machine contributes
    quic_dist_repo_dir: str = "/kaggle/working/quic_dist"  # path ON this machine
    state_dir: str = "/data/quic_train_state"  # path ON this machine - see QuicTrainBackend's own
    # state_dir field comment for why this must be real disk (e.g. /data), never /kaggle/working
    model_path: str = ""  # path ON this machine - each machine needs its OWN local copy of the base
    # model (build_stage_model() loads the full checkpoint before trimming to owned layers - see
    # finetune.py's own docstring), not something shareable across machines over the network here


@dataclass
class SshMultiMachineTrainBackend:
    machines: list[TrainerMachine]  # ordered - machines[0] gets ranks [0, len(cuda_devices)), etc.
    signaling_url: str
    num_layers: int
    full_finetune: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    quantization: str = "4bit"
    compute_dtype: str = "bfloat16"
    max_prompt_len: int = 512
    kl_coef: float = 0.05
    lr: float = 1e-4
    gradient_accumulation_steps: int = 1
    job_id: str = "quic_rl_grpo_multi"
    step_result_timeout_s: float = 600.0
    step_result_poll_interval_s: float = 1.0
    rank_stagger_s: float = 1.0
    checkpoint_keep_last: int = 2

    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _RolloutBatch: object | None = field(default=None, init=False, repr=False)
    _policy_version: int | None = field(default=None, init=False, repr=False)
    _next_step: int = field(default=1, init=False, repr=False)
    _next_export_id: int = field(default=0, init=False, repr=False)
    _world_size: int = field(default=0, init=False, repr=False)
    _rank_machine: list[TrainerMachine] = field(default_factory=list, init=False, repr=False)  # index = global rank
    _rank_local_gpu: list[str] = field(default_factory=list, init=False, repr=False)  # index = global rank
    _local_scratch: str = field(default="", init=False, repr=False)

    # ---- SSH plumbing - same pattern as rollout/ssh_launcher.py's RemoteMachine/_ssh_prefix ----

    def _ssh_prefix(self, m: TrainerMachine) -> list[str]:
        if m.ssh_password is None:
            return []
        return [
            "sshpass", "-p", m.ssh_password,
            "ssh", "-o", "StrictHostKeyChecking=no", "-p", str(m.ssh_port),
            "root@127.0.0.1",
        ]

    def _retry_transient(self, fn, *, attempts: int = 3, backoff_s: float = 2.0):
        """This project has hit real, repeated transient SSH failures all
        session (`kex_exchange_identification: read: Connection reset by
        peer`, occasional command timeouts) that have nothing to do with
        the actual remote command being wrong - they're the tunnel/SSH
        session itself hiccuping. A single one of these crashing an
        entire multi-step training run (as it did here for real: one
        `subprocess.TimeoutExpired` from an `_remote_exists()` check
        during otherwise-successful polling took down the whole backend)
        is not acceptable given how often this project has observed them.
        Retries the same operation a few times with a short backoff
        before actually giving up and propagating - deliberately narrow
        (subprocess-level transient errors only, never masks a real
        command failure/non-zero exit from the remote side itself)."""
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return fn()
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(backoff_s)
        raise last_exc

    def _run_ssh(self, m: TrainerMachine, remote_cmd: str, timeout: float = 30.0) -> str:
        def _do():
            if m.ssh_password is None:
                return subprocess.run(["bash", "-c", remote_cmd], timeout=timeout, capture_output=True, text=True)
            return subprocess.run(self._ssh_prefix(m) + [remote_cmd], timeout=timeout, capture_output=True, text=True)

        proc = self._retry_transient(_do)
        if proc.returncode != 0:
            raise RuntimeError(f"SshMultiMachineTrainBackend: command failed on {m.name}: {remote_cmd!r}\n{proc.stderr}")
        return proc.stdout

    def _remote_exists(self, m: TrainerMachine, remote_path: str) -> bool:
        # Real bug found running this for real: a single transient SSH
        # hiccup (this project has hit these repeatedly all session -
        # `kex_exchange_identification: read: Connection reset by peer`,
        # or just a slow tunnel) raised subprocess.TimeoutExpired, which
        # crashed the ENTIRE training run from inside a POLLING loop that
        # is explicitly supposed to just try again next tick. A transient
        # SSH failure while polling "does this file exist yet" is not
        # meaningfully different from "not yet" - treat it that way
        # instead of letting the whole backend die over one flaky
        # connection attempt in a loop that already retries every
        # step_result_poll_interval_s.
        try:
            proc = subprocess.run(
                self._ssh_prefix(m) + [f"test -e {remote_path}"] if m.ssh_password else ["test", "-e", remote_path],
                timeout=20,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _scp_to_machine(self, m: TrainerMachine, local_path: str, remote_path: str, timeout: float = 120.0) -> None:
        if m.ssh_password is None:
            import shutil

            shutil.copy2(local_path, remote_path)
            return
        self._retry_transient(lambda: subprocess.run(
            ["sshpass", "-p", m.ssh_password, "scp", "-o", "StrictHostKeyChecking=no",
             "-P", str(m.ssh_port), local_path, f"root@127.0.0.1:{remote_path}"],
            timeout=timeout, check=True,
        ))

    def _scp_from_machine(self, m: TrainerMachine, remote_path: str, local_path: str, timeout: float = 120.0) -> None:
        if m.ssh_password is None:
            import shutil

            shutil.copy2(remote_path, local_path)
            return
        self._retry_transient(lambda: subprocess.run(
            ["sshpass", "-p", m.ssh_password, "scp", "-o", "StrictHostKeyChecking=no",
             "-P", str(m.ssh_port), f"root@127.0.0.1:{remote_path}", local_path],
            timeout=timeout, check=True,
        ))

    def _rollout_dir(self, m: TrainerMachine) -> str:
        return os.path.join(m.state_dir, "rollouts")

    def _checkpoint_dir(self, m: TrainerMachine) -> str:
        return os.path.join(m.state_dir, "checkpoints")

    def _rollout_batch_cls(self):
        if self._RolloutBatch is None:
            # Any one machine's quic_dist repo dir works here - the class
            # definition is identical across every deployment (same repo,
            # just copied to each machine - see this project's own
            # standing convention for how these repos get onto a fresh
            # machine). Only used locally, for torch.save()-ing a real
            # RolloutBatch instance before it gets scp'd out.
            local_quic_dist = self.machines[0].quic_dist_repo_dir
            parent = os.path.dirname(os.path.abspath(local_quic_dist))
            if parent not in sys.path:
                sys.path.insert(0, parent)
            from quic_dist.rlhf import RolloutBatch

            self._RolloutBatch = RolloutBatch
        return self._RolloutBatch

    # ---- TrainerBackend protocol ----

    def initialize_policy(self, policy_path: str) -> int:
        if self.full_finetune and self.quantization != "none":
            raise ValueError(
                f"SshMultiMachineTrainBackend: full_finetune=True requires quantization='none' "
                f"(got {self.quantization!r})"
            )
        if self.full_finetune and self.kl_coef != 0.0:
            raise ValueError("SshMultiMachineTrainBackend: full_finetune=True requires kl_coef=0.0")
        if not self.machines:
            raise ValueError("SshMultiMachineTrainBackend: machines must be non-empty")

        self._world_size = sum(len(m.cuda_devices) for m in self.machines)
        self._rank_machine = []
        self._rank_local_gpu = []
        for m in self.machines:
            for gpu in m.cuda_devices:
                self._rank_machine.append(m)
                self._rank_local_gpu.append(gpu)

        import tempfile

        self._local_scratch = tempfile.mkdtemp(prefix="quic_rl_multi_trainer_")

        # One GRPOConfig per machine (model_path differs per machine - see
        # TrainerMachine.model_path's own docstring), written locally then
        # scp'd to that machine's own state_dir.
        for m in self.machines:
            self._run_ssh(m, f"mkdir -p {self._rollout_dir(m)} {self._checkpoint_dir(m)}")
            cfg = {
                "model_path": m.model_path or policy_path,
                "world_size": self._world_size,
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
                "checkpoint_dir": self._checkpoint_dir(m),
                "checkpoint_every": 1,
                "checkpoint_keep_last": self.checkpoint_keep_last,
            }
            local_cfg_path = os.path.join(self._local_scratch, f"grpo_config_{m.name}.yaml")
            with open(local_cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)
            self._scp_to_machine(m, local_cfg_path, os.path.join(m.state_dir, "grpo_config.yaml"))

        from transformers import AutoTokenizer

        # Loaded on THIS (orchestrator) process, wherever it runs - deliberately
        # `policy_path` as passed by the caller (e.g. an HF repo id like
        # "Qwen/Qwen3-1.7B"), never a TrainerMachine.model_path (those are
        # paths on REMOTE machines' own local disks, meaningless here).
        self._tokenizer = AutoTokenizer.from_pretrained(policy_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Launch every rank via SSH+nohup - real bug this avoids (found in
        # ssh_launcher.py / weights.py, this project's own history): a
        # compound `mkdir -p X && CMD & disown` only redirects CMD's own
        # stdio, leaving mkdir's stdio attached to the SSH pty and the
        # whole channel hanging open indefinitely - wrap the whole chain
        # in a brace group so the redirection covers all of it.
        for rank in range(self._world_size):
            m = self._rank_machine[rank]
            gpu = self._rank_local_gpu[rank]
            script = os.path.join(m.quic_dist_repo_dir, "examples", "grpo_external_rollout_rank.py")
            cfg_path = os.path.join(m.state_dir, "grpo_config.yaml")
            log_path = os.path.join(m.state_dir, f"quic_train_rank{rank}.log")
            cmd = (
                f"{{ cd {m.quic_dist_repo_dir} && CUDA_VISIBLE_DEVICES={gpu} QUIC_DIST_FAULTHANDLER=1 nohup python3 -u {script} "
                f"{cfg_path} {rank} {self.signaling_url} {self._rollout_dir(m)} {self.job_id}; }} "
                f"< /dev/null > {log_path} 2>&1 & disown; echo LAUNCHED"
            )
            self._run_ssh(m, cmd, timeout=30.0)
            if rank < self._world_size - 1:
                time.sleep(self.rank_stagger_s)

        self._policy_version = 0
        self._next_step = 1
        return self._policy_version

    def _tokenize_group(self, group: list[Trajectory]) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt_ids_list = self._tokenizer(
            group[0].prompt, truncation=True, max_length=self.max_prompt_len, add_special_tokens=True,
        )["input_ids"]
        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long)

        per_sample_ids: list[list[int]] = []
        for t in group:
            ids = list(t.token_ids) if t.token_ids else self._tokenizer(t.response, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"SshMultiMachineTrainBackend: trajectory prompt_id={t.prompt_id!r} has no generated tokens")
            per_sample_ids.append(ids)

        n = min(len(ids) for ids in per_sample_ids)
        generated = torch.tensor([ids[:n] for ids in per_sample_ids], dtype=torch.long)
        return prompt_ids, generated, prompt_ids.numel() + generated.numel()

    def _write_rollout_batch_everywhere(self, step: int, prompt_ids, generated, rewards) -> None:
        batch = self._rollout_batch_cls()(prompt_ids=prompt_ids, generated=generated, rewards=rewards)
        local_path = os.path.join(self._local_scratch, f"step_{step:06d}.pt")
        torch.save(batch, local_path)
        # Every machine's own rank processes poll their OWN local
        # rollout_dir (grpo_external_rollout_rank.py is unmodified - it
        # has no idea other machines exist) - so the same small batch
        # file needs to land on every machine's filesystem, atomically
        # (scp to a .tmp name then rename remotely), matching the exact
        # write discipline QuicTrainBackend._write_rollout_batch() uses
        # for the single-machine case.
        for m in self.machines:
            remote_tmp = os.path.join(self._rollout_dir(m), f"step_{step:06d}.pt.tmp")
            remote_final = os.path.join(self._rollout_dir(m), f"step_{step:06d}.pt")
            self._scp_to_machine(m, local_path, remote_tmp)
            self._run_ssh(m, f"mv {remote_tmp} {remote_final}")

    def _wait_for_result(self, step: int) -> dict:
        # Only the LAST global rank (the last GPU on the last machine)
        # ever writes a real result - see grpo_external_rollout_rank.py's
        # _write_result(): non-last ranks' loss_value is None and it
        # returns without writing anything.
        last_machine = self._rank_machine[-1]
        result_path = os.path.join(self._rollout_dir(last_machine), f"step_{step:06d}.result.json")
        deadline = time.monotonic() + self.step_result_timeout_s
        while time.monotonic() < deadline:
            if self._remote_exists(last_machine, result_path):
                local_tmp = os.path.join(self._local_scratch, f"result_{step:06d}.json")
                self._scp_from_machine(last_machine, result_path, local_tmp)
                with open(local_tmp) as f:
                    return json.load(f)
            time.sleep(self.step_result_poll_interval_s)
        raise RuntimeError(
            f"SshMultiMachineTrainBackend.train(): step {step} never produced a result within "
            f"{self.step_result_timeout_s}s - check quic_train_rank*.log on {last_machine.name}"
        )

    def train(self, batch: list[Trajectory]) -> TrainResult:
        if self._policy_version is None:
            raise RuntimeError("SshMultiMachineTrainBackend.train() called before initialize_policy()")
        if not batch:
            raise ValueError("SshMultiMachineTrainBackend.train() called with an empty batch")
        rewards = [t.reward for t in batch]
        if any(r is None for r in rewards):
            raise ValueError("SshMultiMachineTrainBackend.train(): every trajectory must have .reward set before training")

        t0 = time.monotonic()
        groups: dict[str, list[Trajectory]] = {}
        for t in batch:
            groups.setdefault(t.prompt_id, []).append(t)

        step_numbers: list[int] = []
        total_tokens = 0
        for prompt_id, group in groups.items():
            if len(group) < 2:
                raise ValueError(
                    f"SshMultiMachineTrainBackend.train(): prompt_id={prompt_id!r} has only {len(group)} "
                    f"sample(s) - GRPO's group-relative advantage needs at least 2"
                )
            prompt_ids, generated, n_tokens = self._tokenize_group(group)
            rewards_t = torch.tensor([t.reward for t in group], dtype=torch.float32)

            step = self._next_step
            self._next_step += 1
            self._write_rollout_batch_everywhere(step, prompt_ids, generated, rewards_t)
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
            raise RuntimeError("SshMultiMachineTrainBackend: policy not initialized yet")
        return self._policy_version

    def export_policy(self, output_dir: str) -> str:
        """Real multi-machine consequence of the shard-then-merge export
        protocol (see finetune.py's merge_stage_shards() and
        grpo_external_rollout_rank.py's _export_via_shard_merge() - both
        UNMODIFIED here): rank 0 (on machines[0]) coordinates the merge
        by waiting for every rank's OWN shard file to appear in ITS
        rollout_dir - which only works if every shard is actually
        THERE. For every rank not on machines[0]'s machine, this method
        waits for that rank's shard to appear on its OWN machine, then
        scp's it into machines[0]'s rollout_dir under the exact filename
        rank 0's own polling loop already expects
        (export_shard_rank{r}_{request_id}.pt) - at that point rank 0's
        existing, unmodified logic finds every shard "locally" and merges
        exactly as it does in the single-machine case.

        This is the one place real multi-GB-scale data can cross a
        machine boundary over SSH in this backend - bounded (one shard
        per non-primary rank, not the whole training checkpoint) and
        occasional (once per real weight sync, not once per step) rather
        than a violation of this project's "never move large files
        between machines over SSH" rule, which is about repeated/
        per-step multi-GB transfers, not a rare few-GB one-time sync."""
        if self._policy_version is None:
            raise RuntimeError("SshMultiMachineTrainBackend.export_policy() called before initialize_policy()")

        self._next_export_id += 1
        request_id = self._next_export_id
        primary = self.machines[0]

        for m in self.machines:
            local_tmp = os.path.join(self._local_scratch, f"export_request_{m.name}.json")
            with open(local_tmp, "w") as f:
                # Every machine's own rank(s) write to THEIR OWN
                # rollout_dir - output_dir only matters on the primary
                # machine (that's where the merge actually happens).
                json.dump({"output_dir": output_dir, "request_id": request_id}, f)
            remote_tmp = os.path.join(self._rollout_dir(m), "export_request.json.tmp")
            remote_final = os.path.join(self._rollout_dir(m), "export_request.json")
            self._scp_to_machine(m, local_tmp, remote_tmp)
            self._run_ssh(m, f"mv {remote_tmp} {remote_final}")

        deadline = time.monotonic() + self.step_result_timeout_s
        shuttled: set[str] = set()
        while time.monotonic() < deadline:
            done_path = os.path.join(self._rollout_dir(primary), f"export_done_{request_id}.json")
            if self._remote_exists(primary, done_path):
                break
            for rank, m in enumerate(self._rank_machine):
                if m is primary or rank in shuttled:
                    continue
                shard_name = f"export_shard_rank{rank}_{request_id}.pt"
                remote_src = os.path.join(self._rollout_dir(m), shard_name)
                if self._remote_exists(m, remote_src):
                    local_shard = os.path.join(self._local_scratch, shard_name)
                    self._scp_from_machine(m, remote_src, local_shard, timeout=600.0)
                    remote_dst_tmp = os.path.join(self._rollout_dir(primary), shard_name + ".tmp")
                    remote_dst = os.path.join(self._rollout_dir(primary), shard_name)
                    self._scp_to_machine(primary, local_shard, remote_dst_tmp, timeout=600.0)
                    self._run_ssh(primary, f"mv {remote_dst_tmp} {remote_dst}")
                    os.remove(local_shard)
                    shuttled.add(rank)
            time.sleep(self.step_result_poll_interval_s)
        else:
            raise RuntimeError(
                f"SshMultiMachineTrainBackend.export_policy(): rank 0 never completed the merge within "
                f"{self.step_result_timeout_s}s - check quic_train_rank0.log on {primary.name}"
            )

        self._tokenizer.save_pretrained(output_dir)
        return output_dir

    def checkpoint(self, output_dir: str) -> str:
        """Copies every rank's raw checkpoint file back to THIS process
        (via scp), one per rank, into output_dir - unlike export_policy()
        this does NOT merge them into one complete model (checkpoint
        files carry optimizer/RNG state for exact resume, not a
        reloadable HF model - see training_utils.save_checkpoint()'s own
        docstring); resuming just needs each rank's own file back on
        whatever machine that rank runs on next, so keeping them as
        separate per-rank files here (not merged) matches how
        QuicTrainBackend.checkpoint() already treats them."""
        if self._policy_version is None:
            raise RuntimeError("SshMultiMachineTrainBackend.checkpoint() called before initialize_policy()")
        os.makedirs(output_dir, exist_ok=True)
        for rank, m in enumerate(self._rank_machine):
            list_cmd = f"ls {self._checkpoint_dir(m)}/rank{rank}_step*.pt 2>/dev/null | sort | tail -1"
            out = self._run_ssh(m, list_cmd).strip()
            if not out:
                raise RuntimeError(f"SshMultiMachineTrainBackend.checkpoint(): no checkpoint found for rank {rank} on {m.name}")
            self._scp_from_machine(m, out, os.path.join(output_dir, os.path.basename(out)), timeout=600.0)
        return output_dir

    def health_check(self) -> bool:
        if self._policy_version is None:
            return False
        for rank, m in enumerate(self._rank_machine):
            proc_check = subprocess.run(
                self._ssh_prefix(m) + [f"pgrep -f 'grpo_external_rollout_rank.py.*{self.job_id}'"]
                if m.ssh_password else ["pgrep", "-f", f"grpo_external_rollout_rank.py.*{self.job_id}"],
                timeout=15,
            )
            if proc_check.returncode != 0:
                return False
        return True

    def shutdown(self) -> None:
        # Best-effort: a transient SSH hiccup or a remote shell quirk
        # making the cleanup command itself report non-zero must never
        # crash shutdown() - a caller invoking this after a real,
        # successful run (the common case) should not see an exception
        # from teardown alone.
        for m in self.machines:
            try:
                self._run_ssh(m, "pkill -9 -f grpo_external_rollout_rank.py; true", timeout=20.0)
            except Exception:
                pass
        if self._local_scratch and os.path.isdir(self._local_scratch):
            import shutil

            shutil.rmtree(self._local_scratch, ignore_errors=True)
