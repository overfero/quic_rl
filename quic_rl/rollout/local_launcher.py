"""`StageLauncher` for a single-machine, 2-GPU, TCP-loopback quic-vllm
deployment - real subprocess management (`stage_server.py` for the
non-driver rank, `launch_pp_stage.py --serve` for the driver rank),
validated end-to-end against a real Qwen3-1.7B-Base 2-stage split (see
this repo's own real-run notes). Intended for local development/testing
of the orchestrator against a real (if small-scale) quic-vllm deployment
before moving to `ssh_launcher.py`'s real multi-machine version.

`policy_path` passed to `restart()` must be a directory containing one
subdirectory per stage (`stage0/`, `stage1/`, ...), each a full HF-format
checkpoint shard - exactly what `scripts/extract_stage_checkpoint_qwen3.py`
(quic-vllm) produces. This launcher does not itself shard checkpoints;
that is `WeightSynchronizer`'s job upstream of this.

`VLLM_USE_V2_MODEL_RUNNER=0` is set on every launched process: the
currently-installed vLLM checkout's default "V2" model runner constructs a
`PPHandler` whose sampled-token broadcast optimization calls
`torch.distributed.new_group()` - a real cross-machine NCCL rendezvous the
transport-backed synthetic PP group cannot provide (by design - the whole
point of the transport layer is avoiding exactly that direct
reachability). Hits `assert self.pp_handler is not None` during warmup on
every non-last rank otherwise. The older "V1" model runner
(`vllm/v1/worker/gpu_model_runner.py`) has no such dependency and is what
this project's transport layer was actually built/validated against.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class LocalTcpStageLauncher:
    vllm_repo_dir: str
    max_model_len: int = 2048
    max_num_seqs: int = 4
    gpu_memory_utilization: float = 0.5
    num_gpu_blocks_override: int = 2000
    tcp_port_base_stage0: int = 31000
    tcp_port_base_stage1: int = 31001
    rpc_port: int = 40000
    driver_port: int = 8080
    cuda_device_stage0: str = "0"
    cuda_device_stage1: str = "1"
    startup_timeout: float = 300.0

    _stage0_proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _stage1_proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def _stop(self) -> None:
        for proc in (self._stage0_proc, self._stage1_proc):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 30.0
        for proc in (self._stage0_proc, self._stage1_proc):
            if proc is None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        self._stage0_proc = None
        self._stage1_proc = None

    def restart(self, policy_path: str) -> None:
        self._stop()

        stage0_model = os.path.join(policy_path, "stage0")
        stage1_model = os.path.join(policy_path, "stage1")
        for path in (stage0_model, stage1_model):
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"LocalTcpStageLauncher.restart: expected a stage checkpoint dir at {path!r}"
                )

        env0 = dict(os.environ, CUDA_VISIBLE_DEVICES=self.cuda_device_stage0, VLLM_USE_V2_MODEL_RUNNER="0")
        self._stage0_proc = subprocess.Popen(
            [
                "python3", "-u", "scripts/stage_server.py",
                "--pp-rank", "0", "--pp-world-size", "2",
                "--self-name", "Stage0", "--next-name", "Stage1", "--driver-name", "Stage1",
                "--transport", "tcp",
                "--tcp-port-base", str(self.tcp_port_base_stage0),
                "--tcp-connect-host-next", "127.0.0.1",
                "--rpc-port", str(self.rpc_port), "--rpc-listen-host", "0.0.0.0",
                "--model", stage0_model, "--tensor-parallel-size", "1", "--dtype", "float16",
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--max-model-len", str(self.max_model_len),
                "--max-num-seqs", str(self.max_num_seqs),
                "--num-gpu-blocks-override", str(self.num_gpu_blocks_override),
            ],
            cwd=self.vllm_repo_dir, env=env0,
        )

        time.sleep(3.0)  # let stage0 start listening before stage1 tries to connect

        env1 = dict(os.environ, CUDA_VISIBLE_DEVICES=self.cuda_device_stage1, VLLM_USE_V2_MODEL_RUNNER="0")
        self._stage1_proc = subprocess.Popen(
            [
                "python3", "-u", "scripts/launch_pp_stage.py",
                "--pp-rank", "1", "--pp-world-size", "2",
                "--self-name", "Stage1", "--prev-name", "Stage0",
                "--transport", "tcp",
                "--tcp-port-base", str(self.tcp_port_base_stage1),
                "--tcp-connect-host-prev", "127.0.0.1",
                "--model", stage1_model, "--tensor-parallel-size", "1", "--dtype", "float16",
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--max-model-len", str(self.max_model_len),
                "--max-num-seqs", str(self.max_num_seqs),
                "--serve", "--host", "0.0.0.0", "--port", str(self.driver_port),
                "--remote-stage-names", "Stage0", "--remote-stage-hosts", "127.0.0.1",
                "--rpc-port", str(self.rpc_port),
                "--num-gpu-blocks-override", str(self.num_gpu_blocks_override),
            ],
            cwd=self.vllm_repo_dir, env=env1,
        )

    def shutdown(self) -> None:
        self._stop()
