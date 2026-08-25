"""Orchestrator state transitions + checkpoint/resume (prompt's testing
items 10-11) - the full mock loop, exercised end to end with zero GPUs,
per the prompt's own explicit testing philosophy."""
from __future__ import annotations

import itertools

import pytest

from quic_rl.orchestrator import lifecycle
from quic_rl.orchestrator.controller import Controller
from quic_rl.orchestrator.state import PolicyVersionMismatchError
from quic_rl.reward.mock import MockRewardBackend
from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.rollout.mock import MockRolloutBackend
from quic_rl.synchronization.weights import MockWeightSynchronizer
from quic_rl.trainer.mock import MockTrainerBackend


def _prompt_source(n_prompts: int = 3, group_size: int = 2):
    counter = itertools.count()

    def source() -> list[GenerationRequest]:
        return [GenerationRequest(prompt_id=f"p{next(counter)}", prompt="2+2=?", num_samples=group_size)
                for _ in range(n_prompts)]

    return source


def _build_controller(state_dir, **overrides):
    rollout = overrides.pop("rollout", MockRolloutBackend(seed=0))
    trainer = overrides.pop("trainer", MockTrainerBackend())
    reward = overrides.pop("reward", MockRewardBackend())
    weight_sync = overrides.pop("weight_sync", MockWeightSynchronizer())

    initial_version = lifecycle.initialize(rollout, trainer, initial_policy_path="mock://initial")

    controller = Controller(
        rollout=rollout, trainer=trainer, reward=reward, weight_synchronizer=weight_sync,
        prompt_source=_prompt_source(), sampling=SamplingParams(max_tokens=32),
        state_dir=str(state_dir), **overrides,
    )
    controller.resume_or_start(initial_version)
    return controller, rollout, trainer, reward, weight_sync


def test_single_iteration_advances_policy_version(tmp_path):
    controller, rollout, trainer, _, weight_sync = _build_controller(tmp_path / "state")
    results = controller.run(max_iterations=1)

    assert len(results) == 1
    assert results[0].iteration == 0
    assert results[0].policy_version == 1
    assert trainer.get_policy_version() == 1
    # the rollout backend must have been synced to the SAME new version
    assert rollout.get_status()["policy_version"] == 1
    assert weight_sync.sync_calls[-1][1] == 1


def test_multiple_iterations_run_consecutively(tmp_path):
    controller, rollout, trainer, _, _ = _build_controller(tmp_path / "state")
    results = controller.run(max_iterations=5)

    assert [r.iteration for r in results] == [0, 1, 2, 3, 4]
    assert [r.policy_version for r in results] == [1, 2, 3, 4, 5]
    assert trainer.get_policy_version() == 5
    assert rollout.get_status()["policy_version"] == 5


def test_run_is_idempotent_about_max_iterations(tmp_path):
    controller, *_ = _build_controller(tmp_path / "state")
    controller.run(max_iterations=3)
    # calling run() again with the SAME max_iterations does nothing more -
    # state.iteration is already >= max_iterations.
    more = controller.run(max_iterations=3)
    assert more == []


def test_orchestrator_state_checkpointed_every_iteration(tmp_path):
    state_dir = tmp_path / "state"
    controller, *_ = _build_controller(state_dir)
    controller.run(max_iterations=2)

    from quic_rl.orchestrator.state import OrchestratorState

    saved = OrchestratorState.load(str(state_dir / "orchestrator_state.json"))
    assert saved is not None
    assert saved.iteration == 2
    assert saved.total_training_steps == 2


def test_resume_after_simulated_crash_continues_from_saved_iteration(tmp_path):
    state_dir = tmp_path / "state"

    # "Run 1": crashes after 2 iterations (we just stop calling run()).
    controller1, rollout1, trainer1, _, _ = _build_controller(state_dir)
    controller1.run(max_iterations=2)
    assert trainer1.get_policy_version() == 2

    # "Run 2": a FRESH process (new backend instances - a mock trainer
    # restarted from scratch would normally reload its own real
    # checkpoint; here we only assert the ORCHESTRATOR's own state
    # resumed correctly, which is this repo's scope - see
    # orchestrator/state.py's OrchestratorState docstring).
    rollout2 = MockRolloutBackend(seed=1)
    trainer2 = MockTrainerBackend()
    reward2 = MockRewardBackend()
    weight_sync2 = MockWeightSynchronizer()
    initial_version = lifecycle.initialize(rollout2, trainer2, initial_policy_path="mock://resumed")
    controller2 = Controller(
        rollout=rollout2, trainer=trainer2, reward=reward2, weight_synchronizer=weight_sync2,
        prompt_source=_prompt_source(), sampling=SamplingParams(max_tokens=32), state_dir=str(state_dir),
    )
    assert controller2.state.iteration == 2  # resumed, not starting fresh at 0
    assert controller2.state.total_training_steps == 2


def test_policy_version_mismatch_fails_loudly_not_silently(tmp_path):
    """A rollout backend that ends up stamping trajectories with a
    DIFFERENT version than the orchestrator believes is current (e.g. a
    sync that silently failed to actually take effect) must blow up the
    iteration, not silently train on mismatched data."""
    controller, rollout, trainer, reward, weight_sync = _build_controller(tmp_path / "state")

    # Directly desync the rollout backend's internal version from what
    # the orchestrator's state believes is current (0) - simulating
    # exactly the failure mode this check exists to catch.
    rollout._policy_version = 99

    with pytest.raises(PolicyVersionMismatchError, match="stale"):
        controller.run(max_iterations=1)
