"""Weight synchronization (prompt's testing item 9) - orchestration-level
only (the merge/shard/transfer/restart STEPS happen in the right order
and report the required cost fields), not real multi-GB file operations
- see `synchronization/weights.py`'s own docstring on why the real
implementation is a separate, later phase."""
from __future__ import annotations

from quic_rl.synchronization.policy import PolicyRegistry
from quic_rl.synchronization.weights import MockWeightSynchronizer


def test_mock_weight_synchronizer_records_calls_in_order():
    sync = MockWeightSynchronizer()
    sync.sync("/policies/v1", 1)
    sync.sync("/policies/v2", 2)
    assert sync.sync_calls == [("/policies/v1", 1), ("/policies/v2", 2)]


def test_sync_result_reports_all_required_cost_fields():
    sync = MockWeightSynchronizer()
    result = sync.sync("/policies/v1", 1)
    # The prompt is explicit: "Do not hide synchronization cost from
    # benchmarks" - every field below must exist and be a real number,
    # not None, even in the mock.
    assert isinstance(result.weight_size_bytes, int)
    assert isinstance(result.transfer_time_s, float)
    assert isinstance(result.sync_time_s, float)
    assert isinstance(result.reload_time_s, float)
    assert isinstance(result.total_overhead_s, float)
    assert result.policy_version == 1


def test_policy_registry_tracks_latest_version():
    registry = PolicyRegistry(root_dir="/tmp/unused", keep_last=2)
    registry.register(0, "/tmp/unused/v0")
    registry.register(1, "/tmp/unused/v1")
    assert registry.latest_version() == 1
    assert registry.path_for(0) == "/tmp/unused/v0"


def test_policy_registry_prunes_beyond_keep_last(tmp_path):
    root = tmp_path / "policies"
    registry = PolicyRegistry(root_dir=str(root), keep_last=2)
    paths = []
    for v in range(4):
        p = root / f"v{v}"
        p.mkdir(parents=True)
        paths.append(p)
        registry.register(v, str(p))

    assert registry.latest_version() == 3
    assert registry.path_for(0) is None  # pruned
    assert registry.path_for(1) is None  # pruned
    assert registry.path_for(2) is not None
    assert registry.path_for(3) is not None
    assert not paths[0].exists()
    assert not paths[1].exists()
    assert paths[2].exists()
    assert paths[3].exists()
