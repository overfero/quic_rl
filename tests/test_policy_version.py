"""Policy-version mismatch detection (prompt's testing item 3) - the
"never silently mix trajectories from incompatible policy versions"
requirement."""
from __future__ import annotations

import pytest

from quic_rl.orchestrator.state import OrchestratorState, PolicyVersionMismatchError, PolicyVersionState
from quic_rl.trajectory import Trajectory


def _traj(prompt_id: str, version: int) -> Trajectory:
    return Trajectory(prompt_id=prompt_id, policy_version=version, prompt="p", response="r")


def test_record_rollout_accepts_matching_single_version():
    state = PolicyVersionState(current_policy_version=5)
    state.record_rollout([_traj("a", 5), _traj("b", 5)])
    assert state.rollout_policy_version == 5


def test_record_rollout_rejects_mixed_versions_in_one_batch():
    state = PolicyVersionState(current_policy_version=5)
    with pytest.raises(PolicyVersionMismatchError, match="different policy versions"):
        state.record_rollout([_traj("a", 5), _traj("b", 6)])


def test_record_rollout_rejects_stale_version():
    state = PolicyVersionState(current_policy_version=5)
    with pytest.raises(PolicyVersionMismatchError, match="stale"):
        state.record_rollout([_traj("a", 4), _traj("b", 4)])


def test_record_rollout_rejects_empty_batch():
    state = PolicyVersionState(current_policy_version=0)
    with pytest.raises(ValueError):
        state.record_rollout([])


def test_record_training_requires_rollout_recorded_first():
    state = PolicyVersionState(current_policy_version=0)
    with pytest.raises(RuntimeError, match="record_rollout"):
        state.record_training(1)


def test_record_training_rejects_non_incrementing_version():
    state = PolicyVersionState(current_policy_version=5)
    state.record_rollout([_traj("a", 5)])
    with pytest.raises(PolicyVersionMismatchError, match="not newer"):
        state.record_training(5)  # trainer claims it produced the SAME version, not a new one


def test_record_training_advances_current_version():
    state = PolicyVersionState(current_policy_version=5)
    state.record_rollout([_traj("a", 5)])
    state.record_training(6)
    assert state.current_policy_version == 6
    assert state.training_policy_version == 5


def test_record_sync_rejects_wrong_target_version():
    state = PolicyVersionState(current_policy_version=5)
    state.record_rollout([_traj("a", 5)])
    state.record_training(6)
    with pytest.raises(PolicyVersionMismatchError, match="drifted"):
        state.record_sync(5)  # synced the OLD version, not the new one


def test_full_happy_path_cycle():
    state = PolicyVersionState(current_policy_version=0)
    state.record_rollout([_traj("a", 0), _traj("a", 0)])
    state.record_training(1)
    state.record_sync(1)
    assert state.current_policy_version == 1
    assert state.rollout_policy_version == 0
    assert state.training_policy_version == 0


def test_orchestrator_state_save_load_roundtrip(tmp_path):
    state = OrchestratorState(iteration=3, total_trajectories_collected=42, total_training_steps=3)
    state.policy_version.current_policy_version = 3
    path = str(tmp_path / "state.json")
    state.save(path)

    loaded = OrchestratorState.load(path)
    assert loaded.iteration == 3
    assert loaded.total_trajectories_collected == 42
    assert loaded.policy_version.current_policy_version == 3


def test_orchestrator_state_load_missing_file_returns_none(tmp_path):
    assert OrchestratorState.load(str(tmp_path / "does_not_exist.json")) is None
