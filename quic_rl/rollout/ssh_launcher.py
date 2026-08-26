"""`StageLauncher` for a REAL multi-machine quic-vllm deployment: one
local stage (rank 0) plus N remote stages launched over SSH, using the
real `--transport quic` + public signaling URL path - the exact
architecture validated end-to-end in quic-vllm's own real-run notes
(docs/history/ in that repo), after two real transport bugs were found
and fixed there (a QUIC idle-timeout config that silently never took
effect, and NAT keepalive traffic that never touched the QUIC protocol
layer itself - see that repo's commit "Fix real QUIC transport bugs
found in a genuine multi-machine deployment").

Unlike `local_launcher.py`'s TCP loopback (all stages on one machine),
this launches across genuinely separate, NAT'd machines via SSH, using
`--transport quic` and a real public signaling URL for hole-punching -
matching the real multi-Kaggle-session topology this whole project
targets. `policy_path` must contain one subdirectory per stage
(`stage0/`, `stage1/`, ...) - the last one is always the driver.

Each remote stage's checkpoint directory is assumed to already exist on
that machine (this launcher does not transfer checkpoints - see this
repo's own README for why: checkpoints are produced independently per
machine via quic-vllm's `scripts/extract_stage_checkpoint_qwen3.py`
against a locally-downloaded base model, never copied machine-to-machine
over SSH).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class RemoteMachine:
    name: str
    host: str = "127.0.0.1"
    ssh_port: int = 22
    ssh_password: str | None = None  # None => this is the LOCAL machine, no SSH
    cuda_device: str = "0"


@dataclass
class SshMultiMachineStageLauncher:
    vllm_repo_dir: str
    machines: list[RemoteMachine]  # ordered rank0..rankN-1; last is the driver
    signaling_url: str
    max_model_len: int = 2048
    max_num_seqs: int = 4
    gpu_memory_utilization: float = 0.5
    num_gpu_blocks_override: int = 2000
    rpc_port: int = 40100
    driver_port: int = 8080
    transport_connect_timeout: float = 300.0
    quic_idle_timeout_s: float = 1800.0
    remote_log_dir: str = "/kaggle/working"

    _procs: dict[str, subprocess.Popen] = field(default_factory=dict, init=False, repr=False)
    _port_forward: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def _ssh_prefix(self, m: RemoteMachine) -> list[str]:
        if m.ssh_password is None:
            return []
        return [
            "sshpass", "-p", m.ssh_password,
            "ssh", "-o", "StrictHostKeyChecking=no", "-p", str(m.ssh_port),
            "root@127.0.0.1",
        ]

    def _env_prefix(self, m: RemoteMachine) -> str:
        return (
            f"CUDA_VISIBLE_DEVICES={m.cuda_device} VLLM_USE_V2_MODEL_RUNNER=0 "
            f"VLLM_TRANSPORT_QUIC_IDLE_TIMEOUT_S={self.quic_idle_timeout_s}"
        )

    def _stop(self) -> None:
        if self._port_forward is not None:
            self._port_forward.terminate()
            try:
                self._port_forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._port_forward.kill()
            self._port_forward = None
        for m in self.machines:
            log_hint = f"{self.remote_log_dir}/quic_rl_stage_{m.name}.log"
            kill_cmd = (
                "pkill -9 -f 'scripts/stage_server.py' 2>/dev/null; "
                "pkill -9 -f 'vllm.entrypoints.cli.main' 2>/dev/null; "
                "true"
            )
            if m.ssh_password is None:
                subprocess.run(kill_cmd, shell=True)
            else:
                subprocess.run(
                    self._ssh_prefix(m) + [kill_cmd],
                    timeout=20,
                )
        # Real observation running this for real: a worker process mid-CUDA-call
        # (e.g. VLLM::Worker_PP2 under GPU load) can take a few seconds to
        # actually die from SIGKILL, not the instant a signal usually implies -
        # 2s left one alive holding ~2.8GB of GPU memory after shutdown()
        # returned. 5s cleared it reliably in that same real run.
        time.sleep(5)
        self._procs = {}

    def restart(self, policy_path: str) -> None:
        self._stop()

        # Real bug hit running this for real: reusing the same --self-name
        # across restarts left a stale registration on the signaling
        # server (this launcher's _stop() only pkill -9's the processes -
        # a hard kill never runs the real QUICTransport.close(), so
        # nothing ever unregisters). A later restart with the SAME name
        # then rendezvoused with that stale entry instead of the new
        # process, hanging the RPC-channel connect indefinitely. Suffixing
        # every on-wire name with a fresh id per restart() call sidesteps
        # this entirely - `RemoteMachine.name` stays the stable,
        # caller-facing identity; only the wire name changes.
        run_suffix = f"-r{int(time.time())}"
        wire_names = [m.name + run_suffix for m in self.machines]

        n = len(self.machines)
        for rank, m in enumerate(self.machines):
            stage_dir = os.path.join(policy_path, f"stage{rank}")
            is_driver = rank == n - 1
            self_name = wire_names[rank]
            prev_name = None if rank == 0 else wire_names[rank - 1]
            next_name = None if is_driver else wire_names[rank + 1]
            driver_name = wire_names[-1]

            if not is_driver:
                cmd = (
                    f"{self._env_prefix(m)} nohup python3 -u "
                    f"{self.vllm_repo_dir}/scripts/stage_server.py "
                    f"--pp-rank {rank} --pp-world-size {n} "
                    f"--self-name {self_name} "
                    + (f"--prev-name {prev_name} " if prev_name else "")
                    + f"--next-name {next_name} --driver-name {driver_name} "
                    f"--transport quic --signaling-url {self.signaling_url} "
                    f"--transport-connect-timeout {self.transport_connect_timeout} "
                    f"--rpc-port {self.rpc_port} --rpc-listen-host 0.0.0.0 "
                    f"--model {stage_dir} --tensor-parallel-size 1 --dtype float16 "
                    f"--gpu-memory-utilization {self.gpu_memory_utilization} "
                    f"--max-model-len {self.max_model_len} "
                    f"--max-num-seqs {self.max_num_seqs} "
                    f"--num-gpu-blocks-override {self.num_gpu_blocks_override}"
                )
            else:
                # n==1 (this machine is the ONLY stage, e.g. the 2-machine
                # train/infer topology's inference side): no prev stage, no
                # remote stages at all - launch_pp_stage.py itself requires
                # --prev-name be OMITTED (not passed empty/"None") whenever
                # pp_rank==0, and only turns on the real transport/RPC
                # machinery when --remote-stage-names is actually given
                # (falls back to vanilla vLLM's own "mp" executor otherwise -
                # see that script's own main()). Real bug hit building this:
                # an earlier version always included both flags, rendering
                # literal "--prev-name None" / an empty "--remote-stage-names"
                # value that broke argparse for exactly this n==1 case.
                remote_names = ",".join(wire_names[:-1])
                cmd = (
                    f"{self._env_prefix(m)} nohup python3 -u "
                    f"{self.vllm_repo_dir}/scripts/launch_pp_stage.py "
                    f"--pp-rank {rank} --pp-world-size {n} "
                    f"--self-name {self_name} "
                    + (f"--prev-name {prev_name} " if prev_name else "")
                    + f"--transport quic --signaling-url {self.signaling_url} "
                    f"--transport-connect-timeout {self.transport_connect_timeout} "
                    f"--model {stage_dir} --tensor-parallel-size 1 --dtype float16 "
                    f"--gpu-memory-utilization {self.gpu_memory_utilization} "
                    f"--max-model-len {self.max_model_len} "
                    f"--max-num-seqs {self.max_num_seqs} "
                    f"--serve --host 0.0.0.0 --port {self.driver_port} "
                    + (f"--remote-stage-names {remote_names} --rpc-port {self.rpc_port} " if remote_names else "")
                    + f"--num-gpu-blocks-override {self.num_gpu_blocks_override}"
                )

            log_path = f"{self.remote_log_dir}/quic_rl_stage_{m.name}.log"
            if m.ssh_password is None:
                full_cmd = f"cd {self.vllm_repo_dir} && {cmd} > {log_path} 2>&1"
                proc = subprocess.Popen(["bash", "-c", full_cmd])
                self._procs[m.name] = proc
            else:
                remote_full_cmd = f"{cmd} > {log_path} 2>&1 & disown; echo LAUNCHED_{m.name}"
                subprocess.run(
                    self._ssh_prefix(m) + [remote_full_cmd],
                    timeout=20, check=True,
                )

            if not is_driver:
                time.sleep(1.0)  # stagger listens-before-connects, matches validated manual runs

        driver = self.machines[-1]
        if driver.ssh_password is not None:
            # The driver's HTTP API is only reachable from this orchestrator
            # process through the same SSH tunnel used to manage it - open
            # a local port-forward so QuicVLLMRollout's plain HTTP client
            # (urllib against a local URL) can reach it without needing to
            # know anything about SSH. Forwarded to the SAME port number
            # locally as remotely (driver_port), so driver_url() is a fixed,
            # known-upfront value - callers (e.g. QuicVLLMRollout) construct
            # it once at __init__ time, before restart() has ever run.
            fwd_cmd = [
                "sshpass", "-p", driver.ssh_password,
                "ssh", "-o", "StrictHostKeyChecking=no", "-N",
                "-L", f"{self.driver_port}:127.0.0.1:{self.driver_port}",
                "-p", str(driver.ssh_port), "root@127.0.0.1",
            ]
            self._port_forward = subprocess.Popen(fwd_cmd)
            time.sleep(2.0)  # let the forward establish before the first request

    def driver_url(self) -> str:
        return f"http://127.0.0.1:{self.driver_port}"

    def shutdown(self) -> None:
        self._stop()
