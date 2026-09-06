"""Real `RolloutBackend`: vLLM-TPU RESIDENT on the SAME host as the
orchestrator, in a genuine SEPARATE OS PROCESS (`_local_vllm_worker.py`,
running inside vllm-tpu's own venv - a different pinned jax/torch_xla
stack from the orchestrator's own) - not `QuicVLLMRollout`'s
HTTP-client-to-a-remote-driver-machine model (quic_vllm.py), because
this project's actual first topology (see docs/EXPERIMENT.md) is one
shared Kaggle TPU VM, not separate rollout/training machines.

Why a SEPARATE PROCESS at all, if not for per-call chip exclusivity
(see below - that's no longer the reason): vLLM's public `LLM` class
exposes no shutdown/close method (confirmed directly against this
installed version) and pulls in its own pinned jax/torch/vllm stack,
which is NOT the orchestrator's own quic-dist-side venv (transformers/
peft/torch_xla, a different pin set) - the two stacks are not proven
importable into one interpreter, so this stays a real subprocess
regardless of chip-sharing questions.

Why RESIDENT (boots once, stays up for this rollout's entire lifetime)
rather than one-shot-per-`generate()`-call (an earlier version of this
file worked that way): confirmed directly, via a standalone concurrent
test, that a torch_xla process and a JAX process CAN each claim their
own disjoint chip rectangle of this host's real physical 2x4x1 topology
AT THE SAME TIME - this worker is pinned to chips [0, tensor_parallel_size)
(see `_local_vllm_worker.py`'s own docstring), while `QuicTrainBackend`
is configured with `tpu_chip_offset=tensor_parallel_size` so its own
rank process(es) claim the disjoint remainder - so `Controller`'s
generate()-then-train() sequence no longer needs either side to fully
vacate the TPU for the other, and this worker's real first-boot cost
(~99s on Qwen3-1.7B: weight load + full JAX/XLA shape compilation) is
paid exactly ONCE for the whole run, not once per iteration.

Protocol: a plain polling directory (see `_local_vllm_worker.py`) -
`request_{N}.json` / `result_{N}.json` pairs in strictly increasing
order, `load_policy.json` / `load_policy_done_{version}.json` for
policy hot-swaps between calls, `shutdown.json` to stop the worker.
Chosen over a socket/RPC framework because both sides already share a
real filesystem (same host) and this project's own quic-dist rank
processes use the identical atomic write-then-rename convention
(`grpo_external_rollout_rank.py`) - one polling protocol, not two.

No LoRA hot-swap: tried `enable_lora=True` against this vllm-tpu
version and found its default model-impl path (`flax_nnx`, what Qwen3
resolves to) never actually builds a LoRA manager - confirmed directly
(`AssertionError: LoRA is not enabled` on the very first compile step,
`model.lora_manager` stays `None` regardless of `enable_lora=True`).
Forcing `MODEL_IMPL_TYPE=vllm` does give a real LoRA manager, but its
per-shape precompilation with LoRA active is drastically slower (many
individual shapes taking 40s+, some far longer) - a real, load-bearing
one-time cost not worth paying here: `load_policy()` records
`policy_path`/`policy_version` for bookkeeping/metadata only and is
intentionally NEVER pushed into the resident worker (see its own
docstring) - this rollout always generates from the SAME base-model
snapshot the worker booted with. quic-train still does its own real
LoRA training/checkpointing on the other chip block, completely
unaffected by this."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.trajectory import Trajectory


@dataclass
class LocalVLLMRollout:
    model_path: str
    vllm_venv_python: str  # e.g. "/kaggle/working/vllm_deploy/venv/bin/python3" - the
                            # vllm-tpu venv's own interpreter, NOT this process's (see module docstring)
    work_dir: str  # real, persistent directory for the request/result polling protocol -
                    # separate from `state_dir` so a stale rollout's leftover files never
                    # get mistaken for a fresh run's (see `_reset_work_dir`)
    tensor_parallel_size: int = 2  # chips [0, tensor_parallel_size) - see module docstring
    max_model_len: int = 4096
    dtype: str = "bfloat16"
    generate_timeout_s: float = 1800.0
    startup_timeout_s: float = 600.0  # real first-boot cost (~99s measured on Qwen3-1.7B) plus margin
    hf_home: str | None = None  # HF_HOME to export into the worker's env - keeps the model
                                 # cache on real disk (see this repo's own Kaggle disk-quota history)

    _policy_path: str | None = field(default=None, init=False, repr=False)
    _policy_version: int | None = field(default=None, init=False, repr=False)
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _next_n: int = field(default=1, init=False, repr=False)
    _loaded_version_on_worker: int | None = field(default=None, init=False, repr=False)

    def _reset_work_dir(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        for name in os.listdir(self.work_dir):
            os.remove(os.path.join(self.work_dir, name))

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._reset_work_dir()
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_local_vllm_worker.py")
        env = dict(os.environ)
        if self.hf_home:
            env["HF_HOME"] = self.hf_home
        self._proc = subprocess.Popen(
            [self.vllm_venv_python, "-u", worker, self.work_dir, self.model_path,
             str(self.tensor_parallel_size), str(self.max_model_len), self.dtype],
            env=env,
        )
        self._next_n = 1
        self._loaded_version_on_worker = None
        deadline = time.monotonic() + self.startup_timeout_s
        # Readiness is "the first request file we write gets picked up" -
        # there is no separate health-check file, so instead we just
        # confirm the process is still alive before handing it work;
        # the first real generate() call blocks on the boot cost directly.
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"LocalVLLMRollout worker exited during startup (returncode={self._proc.returncode})")
            return

    def _push_policy_if_changed(self) -> None:
        """Acks the current policy version to the worker (bookkeeping
        only - the worker never actually reloads weights for it, see
        this module's own docstring for why)."""
        if self._loaded_version_on_worker == self._policy_version:
            return
        policy_path = os.path.join(self.work_dir, "load_policy.json")
        done_path = os.path.join(self.work_dir, f"load_policy_done_{self._policy_version}.json")
        tmp_path = policy_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"policy_path": self._policy_path, "policy_version": self._policy_version}, f)
        os.rename(tmp_path, policy_path)

        deadline = time.monotonic() + self.generate_timeout_s
        while not os.path.exists(done_path):
            if time.monotonic() > deadline:
                raise TimeoutError(f"LocalVLLMRollout: worker did not ack policy version {self._policy_version} in time")
            if self._proc.poll() is not None:
                raise RuntimeError(f"LocalVLLMRollout worker died while loading policy (returncode={self._proc.returncode})")
            time.sleep(0.2)
        os.remove(done_path)
        self._loaded_version_on_worker = self._policy_version

    def load_policy(self, policy_path: str, policy_version: int) -> None:
        """Records `policy_path`/`policy_version` for `get_status()`/
        `Trajectory.policy_version` bookkeeping only - this rollout's
        resident worker NEVER actually reloads weights for it (see this
        module's own docstring for why: no cheap LoRA hot-swap path on
        this vllm-tpu version, and this backend generates from a single
        static base-model snapshot for the whole run). Real per-step
        policy updates come from quic-train's own training/checkpointing
        on its own disjoint chip block, independent of this rollout."""
        self._policy_path = policy_path
        self._policy_version = policy_version

    def generate(self, requests: list[GenerationRequest], sampling: SamplingParams) -> list[Trajectory]:
        if self._policy_version is None:
            raise RuntimeError("LocalVLLMRollout.generate() called before load_policy()")

        self._ensure_started()
        self._push_policy_if_changed()

        prompts: list[str] = []
        owners: list[GenerationRequest] = []
        for req in requests:
            for _ in range(req.num_samples):
                prompts.append(req.prompt)
                owners.append(req)

        n = self._next_n
        self._next_n += 1
        request_path = os.path.join(self.work_dir, f"request_{n:06d}.json")
        result_path = os.path.join(self.work_dir, f"result_{n:06d}.json")
        tmp_path = request_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({
                "prompts": prompts,
                "sampling": {"max_tokens": sampling.max_tokens, "temperature": sampling.temperature, "top_p": sampling.top_p},
            }, f)
        os.rename(tmp_path, request_path)

        deadline = time.monotonic() + self.generate_timeout_s
        while not os.path.exists(result_path):
            if time.monotonic() > deadline:
                raise TimeoutError(f"LocalVLLMRollout.generate(): worker did not respond to request {n} in time")
            if self._proc.poll() is not None:
                raise RuntimeError(f"LocalVLLMRollout worker died during generate() (returncode={self._proc.returncode})")
            time.sleep(0.2)
        with open(result_path) as f:
            result = json.load(f)

        out: list[Trajectory] = []
        counters: dict[str, int] = {}
        for owner, completion in zip(owners, result["completions"]):
            idx = counters.get(owner.prompt_id, 0)
            counters[owner.prompt_id] = idx + 1
            out.append(
                Trajectory(
                    prompt_id=owner.prompt_id,
                    policy_version=self._policy_version,
                    prompt=owner.prompt,
                    response=completion["text"],
                    token_ids=completion["token_ids"],
                    logprobs=completion["logprobs"],
                    metadata={**owner.metadata, "sample_index": idx, "backend": "local_vllm"},
                )
            )
        return out

    def health_check(self) -> bool:
        return os.path.exists(self.vllm_venv_python)

    def get_status(self) -> dict:
        return {
            "model_path": self.model_path,
            "policy_version": self._policy_version,
            "backend": "local_vllm",
            "worker_alive": self._proc is not None and self._proc.poll() is None,
        }

    def unload_policy(self) -> None:
        self._policy_path = None
        self._policy_version = None
        if self._proc is not None and self._proc.poll() is None:
            shutdown_path = os.path.join(self.work_dir, "shutdown.json")
            with open(shutdown_path, "w") as f:
                json.dump({}, f)
            try:
                self._proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None
