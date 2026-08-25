"""Rollout worker registration/dispatch (prompt's testing item 5) and
reward dispatch/aggregation - the coordinator logic in
`workers/rollout_worker.py`/`reward_worker.py`."""
from __future__ import annotations

import pytest

from quic_rl.reward.mock import MockRewardBackend
from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.rollout.mock import MockRolloutBackend
from quic_rl.workers.reward_worker import group_statistics, score_trajectories
from quic_rl.workers.rollout_worker import collect_rollouts, partition


def test_partition_round_robins_across_workers():
    requests = [GenerationRequest(prompt_id=str(i), prompt="p") for i in range(5)]
    parts = partition(requests, num_workers=2)
    assert len(parts) == 2
    assert sum(len(p) for p in parts) == 5


def test_partition_with_more_workers_than_requests_drops_empty_partitions():
    requests = [GenerationRequest(prompt_id="0", prompt="p")]
    parts = partition(requests, num_workers=4)
    assert len(parts) == 1


def test_collect_rollouts_single_worker():
    backend = MockRolloutBackend(seed=0)
    backend.load_policy("dummy", policy_version=0)
    requests = [GenerationRequest(prompt_id="a", prompt="hi", num_samples=2)]
    trajectories = collect_rollouts(backend, requests, SamplingParams(), num_workers=1)
    assert len(trajectories) == 2
    assert all(t.prompt_id == "a" for t in trajectories)
    assert all(t.policy_version == 0 for t in trajectories)


def test_collect_rollouts_multi_worker_returns_all_samples():
    backend = MockRolloutBackend(seed=0)
    backend.load_policy("dummy", policy_version=1)
    requests = [GenerationRequest(prompt_id=str(i), prompt="hi", num_samples=3) for i in range(4)]
    trajectories = collect_rollouts(backend, requests, SamplingParams(), num_workers=3)
    assert len(trajectories) == 12
    counts = {}
    for t in trajectories:
        counts[t.prompt_id] = counts.get(t.prompt_id, 0) + 1
    assert counts == {"0": 3, "1": 3, "2": 3, "3": 3}


def test_collect_rollouts_raises_before_policy_loaded():
    backend = MockRolloutBackend(seed=0)
    with pytest.raises(RuntimeError, match="load_policy"):
        collect_rollouts(backend, [GenerationRequest(prompt_id="a", prompt="hi")], SamplingParams())


def test_score_trajectories_preserves_order_single_worker():
    backend = MockRolloutBackend(seed=0)
    backend.load_policy("dummy", policy_version=0)
    requests = [GenerationRequest(prompt_id=str(i), prompt="hi") for i in range(6)]
    trajectories = collect_rollouts(backend, requests, SamplingParams(), num_workers=1)

    reward = MockRewardBackend()
    scored = score_trajectories(reward, trajectories, num_workers=1)
    assert [t.prompt_id for t in scored] == [t.prompt_id for t in trajectories]
    assert all(t.reward is not None for t in scored)


def test_score_trajectories_preserves_order_multi_worker():
    backend = MockRolloutBackend(seed=0)
    backend.load_policy("dummy", policy_version=0)
    requests = [GenerationRequest(prompt_id=str(i), prompt="hi") for i in range(10)]
    trajectories = collect_rollouts(backend, requests, SamplingParams(), num_workers=1)

    reward = MockRewardBackend()
    single = score_trajectories(reward, trajectories, num_workers=1)
    multi = score_trajectories(reward, trajectories, num_workers=3)
    assert [t.reward for t in single] == [t.reward for t in multi]
    assert [t.prompt_id for t in multi] == [t.prompt_id for t in trajectories]


def test_group_statistics_groups_by_prompt_id():
    backend = MockRolloutBackend(seed=0)
    backend.load_policy("dummy", policy_version=0)
    requests = [GenerationRequest(prompt_id="a", prompt="hi", num_samples=4)]
    trajectories = collect_rollouts(backend, requests, SamplingParams(), num_workers=1)
    scored = score_trajectories(MockRewardBackend(), trajectories, num_workers=1)

    stats = group_statistics(scored)
    assert set(stats) == {"a"}
    assert stats["a"].group_size == 4


def test_group_statistics_requires_reward_set():
    from quic_rl.trajectory import Trajectory

    unscored = [Trajectory(prompt_id="a", policy_version=0, prompt="p", response="r")]
    with pytest.raises(ValueError, match="no reward"):
        group_statistics(unscored)
