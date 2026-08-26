"""Chunked file/directory transfer over `quic_dist`'s own real, already-
validated QUIC `ProcessGroup` (`quic_dist.init_process_group` + plain
`torch.distributed.send`/`recv`) - the same transport this repo's actual
GRPO training already runs on akun3, NOT `vllm/transport/quic_transport.py`.

Why quic_dist and not vllm's transport: `vllm.transport.quic_transport`
can only be imported through the full `vllm` package, which eagerly
resolves the CUDA platform at import time (`vllm/__init__.py` ->
`env_override.py` -> `vllm._C_stable_libtorch`) - a real, heavy
dependency chain that requires quic-vllm's actual build to be present and
working. akun3 (the training machine) has no such build and no reason to
carry one. `quic_dist`, by contrast, is deliberately standalone (see
`process_group.py`'s own module docstring: "NO dependency on vLLM...
being installed or even present on disk") and is already deployed and
working on akun3 - deploying it to akun6 too (just the repo plus its
vendored Rust `.so`, no vLLM changes at all) is far cheaper than the
reverse.

Every transfer opens a THROWAWAY 2-rank process group (rank 0 = sender,
rank 1 = receiver) scoped to exactly one call, then tears it down.
`job_id` MUST be unique per call (callers should derive it from e.g. the
policy_version being synced) - reusing one across calls hits the exact
stale-signaling-server-registration hang `ssh_launcher.py`'s `restart()`
already found for real once (a later call rendezvous-ing with a stale,
never-unregistered old registration instead of the new peer); `job_id`
here plays exactly the role `run_suffix`/wire names play there.

`dist.send`/`dist.recv` require the receiver's tensor to be pre-allocated
with the SAME shape/dtype as what's coming - confirmed directly from
`process_group.py`'s `_recv_one` and `tensor.py`'s `wire_size` (no
implicit variable-length recv exists at this layer, unlike
`vllm.transport.Transport.recv()`). So every file is sent as a sequence
of FIXED-size `CHUNK_SIZE_BYTES` uint8 tensors; the real length of each
file's last (possibly short) chunk is derived independently by both sides
from the manifest's own `size` field - no separate per-chunk length
message needed. The manifest itself has no fixed size, so it gets one
real length-prefix message (tiny, tag 0) followed by a zero-padded
`MAX_MANIFEST_BYTES` buffer (tag 1).

`CHUNK_SIZE_BYTES` (32MB) is not an arbitrary guess - it matches
`process_group.py`'s own documented real cross-machine stress-test size
("an 8-message mixed-size stress run up to 32MB per message... all
byte-exact, no stall"), so this reuses an already-validated message size
rather than picking a new untested one.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass

CHUNK_SIZE_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024  # generous for any realistic checkpoint file listing

_DONE_ACK = b"__QUIC_TRANSFER_DONE__"


def _import_quic_dist(quic_dist_repo_dir: str):
    """Matches quic_train.py's own cross-repo import convention (insert
    the repo's PARENT dir, then `import <package>`)."""
    parent = os.path.dirname(os.path.abspath(quic_dist_repo_dir))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import torch
    import torch.distributed as dist

    import quic_dist

    return quic_dist, torch, dist


@dataclass
class TransferResult:
    total_bytes: int
    num_files: int
    connect_time_s: float
    transfer_time_s: float


def _iter_files(local_dir: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            full = os.path.join(root, name)
            out.append(os.path.relpath(full, local_dir))
    return sorted(out)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def send_directory(
    local_dir: str,
    *,
    signaling_url: str,
    job_id: str,
    quic_dist_repo_dir: str,
    timeout_s: float = 600.0,
    progress_cb=None,
) -> TransferResult:
    """Sends every file under `local_dir` to a peer's matching
    `recv_directory()` call (rank 1, same `job_id`/`signaling_url`).
    Blocks until the peer has confirmed every file verified - a caller
    must not treat the transfer as done, let alone restart a deployment
    against it, before this returns."""
    from datetime import timedelta

    quic_dist, torch, dist = _import_quic_dist(quic_dist_repo_dir)

    rel_paths = _iter_files(local_dir)
    files_meta = []
    for rel in rel_paths:
        full = os.path.join(local_dir, rel)
        files_meta.append({"path": rel, "size": os.path.getsize(full), "sha256": _sha256_file(full)})
    total_bytes = sum(m["size"] for m in files_meta)

    manifest_bytes = json.dumps({"files": files_meta, "total_bytes": total_bytes}).encode("utf-8")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError(
            f"send_directory: manifest is {len(manifest_bytes)} bytes, exceeds MAX_MANIFEST_BYTES "
            f"{MAX_MANIFEST_BYTES} ({len(files_meta)} files under {local_dir}) - raise MAX_MANIFEST_BYTES"
        )

    t0 = time.monotonic()
    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=0, world_size=2, job_id=job_id, timeout=timedelta(seconds=timeout_s)
    )
    connect_time_s = time.monotonic() - t0
    try:
        len_tensor = torch.tensor([len(manifest_bytes)], dtype=torch.int64)
        dist.send(len_tensor, dst=1, tag=0)

        manifest_buf = torch.zeros(MAX_MANIFEST_BYTES, dtype=torch.uint8)
        manifest_buf[: len(manifest_bytes)] = torch.frombuffer(bytearray(manifest_bytes), dtype=torch.uint8)
        dist.send(manifest_buf, dst=1, tag=1)

        t1 = time.monotonic()
        sent_bytes = 0
        chunk_buf = torch.zeros(CHUNK_SIZE_BYTES, dtype=torch.uint8)
        for meta in files_meta:
            full = os.path.join(local_dir, meta["path"])
            remaining = meta["size"]
            with open(full, "rb") as f:
                while remaining > 0:
                    n = min(CHUNK_SIZE_BYTES, remaining)
                    data = f.read(n)
                    if len(data) != n:
                        raise RuntimeError(f"send_directory: short read on {meta['path']} ({len(data)} != {n})")
                    if n < CHUNK_SIZE_BYTES:
                        chunk_buf.zero_()
                    chunk_buf[:n] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    dist.send(chunk_buf, dst=1, tag=2)
                    remaining -= n
                    sent_bytes += n
            if progress_cb is not None:
                progress_cb(meta["path"], sent_bytes, total_bytes)

        ack_buf = torch.zeros(len(_DONE_ACK), dtype=torch.uint8)
        dist.recv(ack_buf, src=1, tag=3)
        if bytes(ack_buf.numpy()) != _DONE_ACK:
            raise RuntimeError(
                "send_directory: peer did not ack completion - do not treat this transfer as successful"
            )
        transfer_time_s = time.monotonic() - t1
    finally:
        dist.destroy_process_group()

    return TransferResult(
        total_bytes=total_bytes, num_files=len(rel_paths), connect_time_s=connect_time_s, transfer_time_s=transfer_time_s
    )


def recv_directory(
    dest_dir: str,
    *,
    signaling_url: str,
    job_id: str,
    quic_dist_repo_dir: str,
    timeout_s: float = 600.0,
    progress_cb=None,
) -> TransferResult:
    """Receives a directory sent by `send_directory()` (rank 0, same
    `job_id`/`signaling_url`) into `dest_dir` (created if needed). Every
    file is written to a `<name>.part` temp path and only
    `os.replace()`'d into its real name after its sha256 matches the
    manifest - the same discipline `training_utils.save_checkpoint()`'s
    `.tmp` + `os.replace()` already applies to local writes, needed at
    least as much for a network write. Only after every file verifies
    does this function ack the sender and return."""
    from datetime import timedelta

    quic_dist, torch, dist = _import_quic_dist(quic_dist_repo_dir)
    os.makedirs(dest_dir, exist_ok=True)

    t0 = time.monotonic()
    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=1, world_size=2, job_id=job_id, timeout=timedelta(seconds=timeout_s)
    )
    connect_time_s = time.monotonic() - t0
    try:
        len_tensor = torch.zeros(1, dtype=torch.int64)
        dist.recv(len_tensor, src=0, tag=0)
        manifest_len = int(len_tensor.item())
        if manifest_len > MAX_MANIFEST_BYTES:
            raise ValueError(
                f"recv_directory: sender's manifest length {manifest_len} exceeds MAX_MANIFEST_BYTES "
                f"{MAX_MANIFEST_BYTES} - refusing (this receiver's cap is out of sync with the sender's)"
            )

        manifest_buf = torch.zeros(MAX_MANIFEST_BYTES, dtype=torch.uint8)
        dist.recv(manifest_buf, src=0, tag=1)
        manifest = json.loads(bytes(manifest_buf[:manifest_len].numpy()).decode("utf-8"))
        files_meta = manifest["files"]
        total_bytes = manifest["total_bytes"]

        t1 = time.monotonic()
        received_bytes = 0
        chunk_buf = torch.zeros(CHUNK_SIZE_BYTES, dtype=torch.uint8)
        for meta in files_meta:
            rel, size, expected_sha = meta["path"], meta["size"], meta["sha256"]
            final_path = os.path.join(dest_dir, rel)
            tmp_path = final_path + ".part"
            os.makedirs(os.path.dirname(final_path) or dest_dir, exist_ok=True)

            h = hashlib.sha256()
            remaining = size
            with open(tmp_path, "wb") as f:
                while remaining > 0:
                    n = min(CHUNK_SIZE_BYTES, remaining)
                    dist.recv(chunk_buf, src=0, tag=2)
                    data = bytes(chunk_buf[:n].numpy())
                    f.write(data)
                    h.update(data)
                    remaining -= n

            if h.hexdigest() != expected_sha:
                raise RuntimeError(
                    f"recv_directory: {rel} - sha256 mismatch after {size} bytes "
                    "(corrupted in transit despite QUIC's own integrity checks - do not trust this file)"
                )
            os.replace(tmp_path, final_path)
            received_bytes += size
            if progress_cb is not None:
                progress_cb(rel, received_bytes, total_bytes)

        transfer_time_s = time.monotonic() - t1
        ack_tensor = torch.frombuffer(bytearray(_DONE_ACK), dtype=torch.uint8).clone()
        dist.send(ack_tensor, dst=0, tag=3)
    finally:
        dist.destroy_process_group()

    return TransferResult(
        total_bytes=total_bytes, num_files=len(files_meta), connect_time_s=connect_time_s, transfer_time_s=transfer_time_s
    )


def _cli() -> None:
    """`python3 quic_transfer.py send|recv <path> --signaling-url ... --job-id
    ... --quic-dist-repo-dir ...` - lets `QuicWeightSynchronizer` (weights.py)
    start the REMOTE side of a transfer over SSH (`nohup python3 -u
    quic_transfer.py recv ...`) without needing a separate driver script
    deployed alongside this module - one file, one deployment, on both
    the sender and receiver machine (matches this project's own repeated
    complaint about needing more than one setup step per repo)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["send", "recv"])
    parser.add_argument("path", help="local_dir for send, dest_dir for recv")
    parser.add_argument("--signaling-url", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--quic-dist-repo-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    def progress(path: str, done: int, total: int) -> None:
        print(f"[{args.mode}] {path}: {done}/{total} bytes", flush=True)

    fn = send_directory if args.mode == "send" else recv_directory
    result = fn(
        args.path, signaling_url=args.signaling_url, job_id=args.job_id,
        quic_dist_repo_dir=args.quic_dist_repo_dir, timeout_s=args.timeout_s, progress_cb=progress,
    )
    print(f"{args.mode.upper()}_RESULT", result, flush=True)


if __name__ == "__main__":
    _cli()
