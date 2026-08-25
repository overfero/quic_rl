"""Reward dispatch + aggregation - scores every trajectory (concurrently
across `num_workers`, same rationale as `rollout_worker.py`: this
project's reward backends are pure/CPU-bound, so concurrency here is
about overlapping many independent scoring calls, not talking to
distinct remote workers), assigns `.reward`, and computes the per-prompt
group statistics GRPO's advantage computation needs (group mean/std -
matching quic-train's own `rlhf.py::run_grpo_training`'s
`(rewards - rewards.mean()) / (rewards.std() + eps)`, computed here so
`TrainerBackend.train()` receives trajectories that already carry
everything needed, no group-membership bookkeeping duplicated on the
quic-train side)."""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from quic_rl.reward.base import RewardBackend
from quic_rl.trajectory import Trajectory


@dataclass
class GroupStats:
    prompt_id: str
    reward_mean: float
    reward_std: float
    group_size: int


def score_trajectories(
    backend: RewardBackend,
    trajectories: list[Trajectory],
    num_workers: int = 1,
) -> list[Trajectory]:
    """Returns NEW `Trajectory` objects (does not mutate the input) with
    `.reward` filled in."""
    num_workers = max(1, num_workers)
    if num_workers == 1 or len(trajectories) <= 1:
        rewards = backend.batch_score(trajectories)
    else:
        # Contiguous chunks, each remembering its own start offset - the
        # simplest possible scheme that can't scramble order on reassembly.
        chunk_size = -(-len(trajectories) // num_workers)  # ceil division
        chunks = [
            (i, trajectories[i : i + chunk_size])
            for i in range(0, len(trajectories), chunk_size)
        ]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {pool.submit(backend.batch_score, chunk): start for start, chunk in chunks}
            rewards: list[float | None] = [None] * len(trajectories)
            for future, start in futures.items():
                chunk_rewards = future.result()
                rewards[start : start + len(chunk_rewards)] = chunk_rewards

    if len(rewards) != len(trajectories):
        raise RuntimeError(
            f"reward backend returned {len(rewards)} scores for {len(trajectories)} trajectories - "
            "batch_score() must return exactly one score per input, in order"
        )

    return [
        Trajectory(
            prompt_id=t.prompt_id, policy_version=t.policy_version, prompt=t.prompt, response=t.response,
            token_ids=t.token_ids, logprobs=t.logprobs, reward=r, metadata=t.metadata,
        )
        for t, r in zip(trajectories, rewards)
    ]


def group_statistics(trajectories: list[Trajectory]) -> dict[str, GroupStats]:
    """One `GroupStats` per distinct `prompt_id` - the group GRPO's
    advantage computation normalizes against. Every trajectory must
    already have `.reward` set (call `score_trajectories` first)."""
    by_prompt: dict[str, list[float]] = {}
    for t in trajectories:
        if t.reward is None:
            raise ValueError(f"trajectory {t.prompt_id!r} has no reward yet - call score_trajectories() first")
        by_prompt.setdefault(t.prompt_id, []).append(t.reward)

    return {
        pid: GroupStats(
            prompt_id=pid,
            reward_mean=statistics.fmean(rewards),
            reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            group_size=len(rewards),
        )
        for pid, rewards in by_prompt.items()
    }
