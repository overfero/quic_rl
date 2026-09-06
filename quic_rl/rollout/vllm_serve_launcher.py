"""`StageLauncher` implementations for plain vanilla `vllm serve` (CUDA) -
NOT `ssh_launcher.py`'s `SshMultiMachineStageLauncher`, which is built for
quic-vllm's own custom pipeline-parallel stage-server protocol
(`scripts/stage_server.py`/`launch_pp_stage.py`, QUIC RPC between stages).
This project's actual GPU rollout machine (a plain 2xT4 box) runs
ordinary, unmodified vLLM with a single stage - `vllm serve <model> ...`
- so it needs a much simpler launcher: kill whatever's running, start a
fresh server pointed at the given model path.

Paired with `synchronization.weights.QuicWeightSynchronizer` for real,
efficient weight transfer (see that module's own docstring for why P2P
QUIC, not SSH relay, moves the actual multi-GB checkpoint bytes) -
`QuicWeightSynchronizer.sync()` writes the received checkpoint to
`{remote_policy_root}/v{version}/stage0` (matching
`SshMultiMachineStageLauncher`'s own directory convention, which this
launcher also expects, for consistency - not because this backend has
real pipeline stages) and calls `restart({remote_policy_root}/v{version})`
- `VllmServeStageLauncher.restart()` appends the same `/stage0` suffix
internally to find the actual checkpoint.

`QuicVLLMRollout` (rollout/quic_vllm.py) is paired with a SEPARATE
`NoOpStageLauncher` instance, not this one - the real restart already
happens via `QuicWeightSynchronizer.sync()`'s own call above;
`QuicVLLMRollout.load_policy()` calling `restart()` a second time (with
the WRONG - local, not the remote received - path, since it receives
whatever `Controller._sync_policy()` passed as `exported_path`) would
both be redundant (a second, wasted full server reboot) and incorrect
(that local path means nothing on the GPU machine). `QuicVLLMRollout`'s
own post-restart `/health` poll and model-name re-resolution stay real
and useful regardless - `NoOpStageLauncher` only skips the restart call
itself."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


class NoOpStageLauncher:
    def restart(self, policy_path: str) -> None:
        pass


@dataclass
class VllmServeStageLauncher:
    ssh_alias: str
    port: int = 8000
    tensor_parallel_size: int = 1
    # T4 (compute capability 7.5) has no real bfloat16 support - confirmed
    # directly ("Bfloat16 is only supported on GPUs with compute
    # capability of at least 8.0"). float16 is the real, correct default
    # for a T4 box; override explicitly for anything newer.
    dtype: str = "float16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    remote_work_dir: str = "/kaggle/working/gpu_vllm_state"
    kill_settle_s: float = 5.0  # real GPU-memory-release delay - see quic_train_multi.py's identical comment

    def _ssh(self, remote_cmd: str, timeout: float = 30.0, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", self.ssh_alias, remote_cmd], timeout=timeout, capture_output=True, text=True, check=check,
        )

    def restart(self, policy_path: str) -> None:
        model_dir = f"{policy_path}/stage0"
        self._ssh("pkill -9 -f 'vllm serve'; true", check=False)
        time.sleep(self.kill_settle_s)

        self._ssh(f"mkdir -p {self.remote_work_dir}")
        log_path = f"{self.remote_work_dir}/vllm_server.log"
        cmd = (
            f"{{ cd {self.remote_work_dir} && nohup vllm serve {model_dir} "
            f"--port {self.port} --tensor-parallel-size {self.tensor_parallel_size} "
            f"--dtype {self.dtype} --max-model-len {self.max_model_len} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization}; }} "
            f"< /dev/null > {log_path} 2>&1 & disown; echo LAUNCHED"
        )
        self._ssh(cmd, timeout=30.0)
