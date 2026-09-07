"""Generic remote shell-command execution over `quic_dist.store
.QuicRendezvousStore`'s KV API - for machine pairs that can each reach the
shared signaling server but can't reach each other directly.

Why this exists instead of SSH or a new HTTP service: zrok's `access
private` only works within one zrok account/environment - a DIFFERENT
account's environment (e.g. a freshly re-provisioned training machine) gets
a real `401 accessUnauthorized` trying to reach another account's private
share, and there's no local relay allowed (the local machine is for editing
only, never for holding live traffic). `QuicRendezvousStore` already exists,
is already validated (backs `torch.distributed`'s own rendezvous/barrier),
and already only needs both sides to reach the same public signaling URL -
the exact reachability both sides already have for the real QUIC weight
transfer. See ARCHITECTURE.md's "Cross-machine coordination WITHOUT SSH"
section for the full rationale.

Protocol: the controller `set()`s `{channel}_cmd_v{id}` to a JSON blob
`{"cmd": "<shell command>"}`; the watcher (running persistently on the
target machine) blocks on `get()` for that key, runs the command via
`bash -c`, and `set()`s `{channel}_result_v{id}` to a JSON blob
`{"returncode": int, "stdout": str, "stderr": str}` (each truncated to keep
well under the KV store's real payload size). `id` must increment per call -
reusing one would let a stale watcher iteration's old result satisfy a NEW
`run_remote_command()` call before the real command even ran.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import timedelta

from quic_dist.store import QuicRendezvousStore

_MAX_OUTPUT_CHARS = 4000


def watch_and_execute(signaling_url: str, channel: str, start_id: int = 0) -> None:
    """Runs forever on the TARGET machine. Call this from a persistent
    process (`screen`, not bare `nohup`/`disown` - see CLAUDE.md)."""
    store = QuicRendezvousStore(signaling_url, timeout=timedelta(seconds=3600))
    # Real bug found running this for real: `subprocess.run` inherits this
    # WATCHER process's own environment by default - if the watcher itself
    # needed e.g. PYTHONPATH set to import quic_rl/quic_dist, that same
    # PYTHONPATH leaked into every command it executed, breaking an
    # unrelated vLLM launch (`ImportError: cannot import name
    # 'SamplingParams' from 'vllm' (unknown location)` - vLLM's own import
    # resolution got confused by this watcher's PYTHONPATH). Executed
    # commands get a clean environment (this watcher's PATH/HOME kept,
    # PYTHONPATH stripped) so they behave exactly as they would launched
    # directly, independent of whatever this watcher process needed to
    # import itself.
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd_id = start_id
    while True:
        key = f"{channel}_cmd_v{cmd_id}"
        print(f"[kv_remote_exec] waiting for {key}", file=sys.stderr, flush=True)
        raw = store.get(key)  # blocks until present (or raises after this store's own 3600s timeout)
        payload = json.loads(raw)
        cmd = payload["cmd"]
        print(f"[kv_remote_exec] executing: {cmd}", file=sys.stderr, flush=True)
        proc = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True,
            timeout=payload.get("timeout_s", 300.0), env=clean_env,
        )
        result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-_MAX_OUTPUT_CHARS:],
            "stderr": proc.stderr[-_MAX_OUTPUT_CHARS:],
        }
        store.set(f"{channel}_result_v{cmd_id}", json.dumps(result))
        print(f"[kv_remote_exec] done: {key} rc={proc.returncode}", file=sys.stderr, flush=True)
        cmd_id += 1


def run_remote_command(signaling_url: str, channel: str, cmd_id: int, cmd: str, timeout_s: float = 60.0) -> dict:
    """Runs on the CONTROLLER machine. Submits `cmd`, blocks until the
    watcher's result is posted (or `timeout_s` elapses), returns
    `{"returncode": int, "stdout": str, "stderr": str}`."""
    store = QuicRendezvousStore(signaling_url, timeout=timedelta(seconds=timeout_s))
    store.set(f"{channel}_cmd_v{cmd_id}", json.dumps({"cmd": cmd, "timeout_s": timeout_s}))
    raw = store.get(f"{channel}_result_v{cmd_id}")
    return json.loads(raw)


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    watch_p = sub.add_parser("watch")
    watch_p.add_argument("--signaling-url", required=True)
    watch_p.add_argument("--channel", required=True)
    watch_p.add_argument("--start-id", type=int, default=0)

    run_p = sub.add_parser("run")
    run_p.add_argument("--signaling-url", required=True)
    run_p.add_argument("--channel", required=True)
    run_p.add_argument("--cmd-id", type=int, required=True)
    run_p.add_argument("--cmd", required=True)
    run_p.add_argument("--timeout-s", type=float, default=60.0)

    args = p.parse_args()
    if args.mode == "watch":
        watch_and_execute(args.signaling_url, args.channel, args.start_id)
    else:
        result = run_remote_command(args.signaling_url, args.channel, args.cmd_id, args.cmd, args.timeout_s)
        print(json.dumps(result))


if __name__ == "__main__":
    _cli()
