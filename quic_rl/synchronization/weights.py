"""`WeightSynchronizer`: Trainer -> Rollout weight propagation, kept as
its own abstraction deliberately (the prompt's own instruction: "Do NOT
assume that the only possible synchronization mechanism is a full model
reload"). The real implementation for this ecosystem
(`QuicWeightSynchronizer`, below) transfers `export_policy()`'s output
straight over `quic_dist`'s QUIC transport (see `quic_transfer.py`) and
restarts the rollout deployment - see `docs/ARCHITECTURE.md` for why
vLLM's own built-in NCCL/IPC live weight-transfer engines do NOT fit this
NAT'd, multi-machine topology and restart-based sync is the correct
choice, not a stopgap. No merge/shard step: `full_finetune=True` means
`export_policy()` already writes one complete checkpoint, and this
repo's real topology is single-stage per machine (n==1 in
`ssh_launcher.py`'s terms) - sharding across pipeline-parallel stages
is real, precedented work for a future multi-stage topology, not
something this implementation silently pretends to do.

Every real implementation must report the 5 cost fields the prompt
explicitly asks not to hide from benchmarks - `SyncResult` makes that
structural, not optional per-implementation."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SyncResult:
    policy_version: int
    weight_size_bytes: int
    transfer_time_s: float
    sync_time_s: float  # end-to-end wall clock for the whole sync() call
    reload_time_s: float
    total_overhead_s: float  # sync_time_s, restated explicitly so it's never accidentally omitted from a report


@runtime_checkable
class WeightSynchronizer(Protocol):
    def sync(self, policy_dir: str, policy_version: int) -> SyncResult:
        """Propagates the weights at `policy_dir` (as written by
        `TrainerBackend.export_policy()`) to every rollout worker, and
        blocks until they're actually serving `policy_version` - the
        orchestrator's `PolicyVersionState.record_sync()` depends on
        that being true the instant this returns."""
        ...


@dataclass
class QuicWeightSynchronizer:
    """Pushes `export_policy()`'s output from the machine this
    orchestrator runs on (the training machine, e.g. akun3 - `sync()`
    calls `send_directory()` IN-PROCESS, not over SSH, since it's
    already local there) to a remote rollout machine (e.g. akun6) over
    `quic_dist`'s real QUIC ProcessGroup transport - see
    `quic_transfer.py`'s module docstring for why that transport (not
    `vllm/transport/quic_transport.py`, not scp/ssh) is the right choice
    here. SSH is used ONLY to start the remote receiver process
    (`quic_transfer.py`'s own `--` CLI, deployed once on the receiver
    machine - see that module's `_cli()`) - the actual weight bytes never
    touch the SSH tunnel, so this does not draw against this project's
    documented ~5GB/day SSH budget no matter how large the checkpoint is.

    `send_directory()` returning successfully already guarantees the
    remote side wrote and sha256-verified every file (see
    `quic_transfer.py`'s own ack-then-return protocol) - no separate
    "did the remote side finish" polling is layered on top of that here.

    `sender=None` (the default) assumes THIS process already runs on the
    machine holding `policy_dir` (e.g. the orchestrator was itself
    launched by SSHing into the training machine directly - the pattern
    every real validation run in this repo has used so far) and calls
    `send_directory()` in-process. Pass a `sender` `RemoteMachine` instead
    when the orchestrator runs somewhere with SSH reachability to BOTH
    machines but isn't itself either one (real gap found running this for
    real: `ssh_launcher.py`'s hardcoded `root@127.0.0.1` target only
    resolves against tunnels that exist on ONE specific machine - akun3
    has no route to akun6's tunnel port and vice versa, only whichever
    machine actually holds both port-forwards does) - in that case both
    sides are launched over SSH and this polls the sender's own log for
    completion instead of relying on an in-process return value."""

    receiver: object  # quic_rl.rollout.ssh_launcher.RemoteMachine - the rollout machine's SSH details
    remote_policy_root: str  # e.g. "/kaggle/working/quic_rl_policy_versions" ON the receiver machine
    remote_quic_transfer_script: str  # quic_transfer.py's path ON the receiver machine (deployed once)
    remote_quic_dist_repo_dir: str  # quic_dist's repo root ON the receiver machine
    local_quic_dist_repo_dir: str  # quic_dist's repo root on the SENDER machine (local or remote - see `sender`)
    signaling_url: str
    stage_launcher: object  # anything with .restart(policy_path) - e.g. SshMultiMachineStageLauncher
    sender: object = None  # RemoteMachine | None - see class docstring
    sender_quic_transfer_script: str | None = None  # quic_transfer.py's path ON the sender machine - required if `sender` is set
    transfer_timeout_s: float = 1800.0
    remote_log_dir: str = "/kaggle/working"
    poll_interval_s: float = 2.0

    def _ssh_prefix(self, machine) -> list[str]:
        if machine is None or machine.ssh_password is None:
            return []
        return [
            "sshpass", "-p", machine.ssh_password,
            "ssh", "-o", "StrictHostKeyChecking=no", "-p", str(machine.ssh_port),
            "root@127.0.0.1",
        ]

    def _run_ssh(self, machine, remote_cmd: str, timeout: float) -> str:
        proc = subprocess.run(
            self._ssh_prefix(machine) + [remote_cmd], timeout=timeout, check=True, capture_output=True, text=True,
        )
        return proc.stdout

    def sync(self, policy_dir: str, policy_version: int) -> SyncResult:
        t_sync0 = time.monotonic()
        # job_id must be unique per call - reusing one hits the exact
        # stale-signaling-server-registration hang ssh_launcher.py's
        # restart() already found for real once (see quic_transfer.py's
        # module docstring for the full cross-reference).
        job_id = f"weight_sync_v{policy_version}_{int(time.time())}"
        remote_dest = f"{self.remote_policy_root}/v{policy_version}/stage0"
        recv_log = f"{self.remote_log_dir}/quic_weight_recv_v{policy_version}.log"

        # Real bug found running this for real: `mkdir -p X && CMD < /dev/null
        # > log 2>&1 &` only redirects CMD's own stdio - `mkdir`'s stdio (and
        # the shell's own bookkeeping for the `&&` chain) stays attached to
        # the SSH session's pty, which keeps the whole SSH channel open
        # indefinitely even though the visible command (echo) already ran and
        # the intended background job is fully detached from job control.
        # Wrapping the whole chain in `{ ...; }` applies the redirection to
        # the ENTIRE group, not just the last simple command - confirmed via
        # a minimal repro (a plain `mkdir -p X && sleep 100 &` hung the same
        # way; wrapping it in braces did not).
        recv_cmd = (
            f"{{ mkdir -p {remote_dest} && nohup python3 -u {self.remote_quic_transfer_script} recv {remote_dest} "
            f"--signaling-url {self.signaling_url} --job-id {job_id} "
            f"--quic-dist-repo-dir {self.remote_quic_dist_repo_dir} --timeout-s {self.transfer_timeout_s}; }} "
            f"< /dev/null > {recv_log} 2>&1 & disown; echo RECV_LAUNCHED"
        )
        self._run_ssh(self.receiver, recv_cmd, timeout=30)
        # Real receiver registers with the signaling server itself right
        # after this returns (import + argument parsing + reaching
        # init_process_group takes well under this) - a short, fixed
        # wait here just avoids the sender racing ahead of that, not a
        # correctness dependency (send_directory()'s own connect_timeout
        # inside quic_dist.init_process_group already covers the rest of
        # any real delay).
        time.sleep(2.0)

        t_transfer0 = time.monotonic()
        weight_size_bytes = self._send(policy_dir, job_id, policy_version)
        transfer_time_s = time.monotonic() - t_transfer0

        t_reload0 = time.monotonic()
        self.stage_launcher.restart(f"{self.remote_policy_root}/v{policy_version}")
        reload_time_s = time.monotonic() - t_reload0

        sync_time_s = time.monotonic() - t_sync0
        return SyncResult(
            policy_version=policy_version,
            weight_size_bytes=weight_size_bytes,
            transfer_time_s=transfer_time_s,
            sync_time_s=sync_time_s,
            reload_time_s=reload_time_s,
            total_overhead_s=sync_time_s,
        )

    def _send(self, policy_dir: str, job_id: str, policy_version: int) -> int:
        """Returns the transferred total_bytes. In-process for
        `sender=None` (see class docstring); over SSH + log-polling
        otherwise - `send_directory()` itself already blocks until the
        receiver's completion ack, but a DETACHED remote process (`nohup
        ... & disown`) can't hand this process a Python return value, so
        the remote side's own log is the only channel left to learn the
        real transferred byte count and to detect a remote failure."""
        if self.sender is None:
            from quic_rl.synchronization.quic_transfer import send_directory

            result = send_directory(
                policy_dir, signaling_url=self.signaling_url, job_id=job_id,
                quic_dist_repo_dir=self.local_quic_dist_repo_dir, timeout_s=self.transfer_timeout_s,
            )
            return result.total_bytes

        if not self.sender_quic_transfer_script:
            raise ValueError("QuicWeightSynchronizer: sender_quic_transfer_script is required when sender is set")

        send_log = f"{self.remote_log_dir}/quic_weight_send_v{policy_version}.log"
        send_cmd = (
            f"nohup python3 -u {self.sender_quic_transfer_script} send {policy_dir} "
            f"--signaling-url {self.signaling_url} --job-id {job_id} "
            f"--quic-dist-repo-dir {self.local_quic_dist_repo_dir} --timeout-s {self.transfer_timeout_s} "
            f"< /dev/null > {send_log} 2>&1 & disown; echo SEND_LAUNCHED"
        )
        self._run_ssh(self.sender, send_cmd, timeout=30)

        import re

        deadline = time.monotonic() + self.transfer_timeout_s
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"QuicWeightSynchronizer: sender never finished within {self.transfer_timeout_s}s - "
                    f"check {send_log} on the sender machine"
                )
            log = self._run_ssh(self.sender, f"cat {send_log} 2>/dev/null || true", timeout=15)
            if "Traceback" in log:
                raise RuntimeError(f"QuicWeightSynchronizer: remote send_directory() failed:\n{log}")
            match = re.search(r"total_bytes=(\d+)", log)
            if "SEND_RESULT" in log and match:
                return int(match.group(1))
            time.sleep(self.poll_interval_s)


class MockWeightSynchronizer:
    """GPU/network-free stand-in - simulates the sync steps' ORDER and
    timing structure without any real transfer, so orchestrator tests
    exercise the same call shape a real sync would."""

    def __init__(self, simulated_latency_s: float = 0.0) -> None:
        self.simulated_latency_s = simulated_latency_s
        self.sync_calls: list[tuple[str, int]] = []

    def sync(self, policy_dir: str, policy_version: int) -> SyncResult:
        t0 = time.monotonic()
        self.sync_calls.append((policy_dir, policy_version))
        if self.simulated_latency_s:
            time.sleep(self.simulated_latency_s)
        elapsed = time.monotonic() - t0
        return SyncResult(
            policy_version=policy_version,
            weight_size_bytes=0,
            transfer_time_s=elapsed / 2,
            sync_time_s=elapsed,
            reload_time_s=elapsed / 2,
            total_overhead_s=elapsed,
        )
