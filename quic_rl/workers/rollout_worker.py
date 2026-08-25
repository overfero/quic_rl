"""Rollout coordinator: partitions prompts across `num_workers` concurrent
dispatchers, each calling the SAME `RolloutBackend.generate()` (today
that's one HTTP endpoint - quic-vllm's PP driver machine - concurrency
here is about not serializing a large prompt batch through one blocking
Python call, not about talking to N different backend instances; nothing
stops a future `RolloutBackend` from fanning out to real distinct workers
internally, this coordinator doesn't need to know either way).

Deliberately NOT heartbeat/health-check based (worker-level fault
tolerance is out of scope for this repo - see ARCHITECTURE.md) - a
dispatch thread that raises just propagates the exception, no retry-with-
a-different-worker logic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from quic_rl.rollout.base import GenerationRequest, RolloutBackend, SamplingParams
from quic_rl.trajectory import Trajectory


def partition(requests: list[GenerationRequest], num_workers: int) -> list[list[GenerationRequest]]:
    """Round-robin partition, not a naive contiguous slice - keeps each
    worker's share close in size even when `num_samples` varies a lot
    per request (a few very-high-group_size prompts landing on the same
    worker would otherwise skew load badly)."""
    num_workers = max(1, num_workers)
    parts: list[list[GenerationRequest]] = [[] for _ in range(num_workers)]
    for i, req in enumerate(requests):
        parts[i % num_workers].append(req)
    return [p for p in parts if p]


def collect_rollouts(
    backend: RolloutBackend,
    requests: list[GenerationRequest],
    sampling: SamplingParams,
    num_workers: int = 1,
) -> list[Trajectory]:
    """Dispatches `requests` across `num_workers` concurrent calls to
    `backend.generate()`, collects and flattens the results. Every
    returned trajectory's `prompt_id` is verified to belong to a request
    that was actually sent - a backend bug that fabricates or drops
    prompt_ids fails loudly here instead of silently corrupting the
    training batch downstream."""
    requested_ids = {r.prompt_id for r in requests}
    parts = partition(requests, num_workers)

    if len(parts) == 1:
        results = [backend.generate(parts[0], sampling)]
    else:
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            futures = [pool.submit(backend.generate, part, sampling) for part in parts]
            results = [f.result() for f in futures]

    trajectories = [t for batch in results for t in batch]

    unexpected = {t.prompt_id for t in trajectories} - requested_ids
    if unexpected:
        raise RuntimeError(f"rollout backend returned trajectories for unrequested prompt_ids: {sorted(unexpected)}")

    expected_counts = {r.prompt_id: r.num_samples for r in requests}
    got_counts: dict[str, int] = {}
    for t in trajectories:
        got_counts[t.prompt_id] = got_counts.get(t.prompt_id, 0) + 1
    missing = {pid: expected_counts[pid] - got_counts.get(pid, 0) for pid in expected_counts if got_counts.get(pid, 0) < expected_counts[pid]}
    if missing:
        raise RuntimeError(f"rollout backend returned fewer samples than requested for: {missing}")

    return trajectories
