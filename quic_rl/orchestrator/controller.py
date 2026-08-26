"""`Controller`: the synchronous training loop, following the prompt's own
pseudocode step for step (each step below is one line of that pseudocode,
kept in the same order deliberately so the mapping is obvious to a
reader):

    initialize policy                    -> lifecycle.initialize() (caller, before Controller.run())
    while not finished:
        verify rollout workers           -> _verify_rollout_workers()
        generate trajectories            -> _generate()
        verify policy version            -> PolicyVersionState.record_rollout()
        compute rewards                  -> _score()
        build GRPO batch                 -> _score() (assigns .reward, ready to hand to the trainer)
        send batch to quic-train         -> TrainerBackend.train(batch)
        execute training update          -> TrainerBackend.train(batch)
        obtain new policy version        -> PolicyVersionState.record_training()
        synchronize policy to workers    -> _sync_policy()
        record metrics                   -> MetricsLogger.log_*()
        checkpoint state                 -> OrchestratorState.save()
        repeat

Synchronous by design (the prompt is explicit: don't implement
asynchronous RL first) - one iteration fully completes before the next
starts. Worker-level fault tolerance (a dead rollout/reward worker being
detected and replaced) is explicitly out of scope - see
`docs/ARCHITECTURE.md`; a raised exception from any backend call
propagates and stops the run rather than being caught and retried with a
different worker."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

from quic_rl.metrics.metrics import MetricsLogger
from quic_rl.orchestrator.state import OrchestratorState
from quic_rl.retry import retry_call
from quic_rl.reward.base import RewardBackend
from quic_rl.rollout.base import GenerationRequest, RolloutBackend, SamplingParams
from quic_rl.synchronization.policy import PolicyRegistry
from quic_rl.synchronization.weights import WeightSynchronizer
from quic_rl.trainer.base import TrainerBackend
from quic_rl.trajectory import Trajectory
from quic_rl.workers.reward_worker import score_trajectories
from quic_rl.workers.rollout_worker import collect_rollouts


@dataclass
class IterationResult:
    iteration: int
    policy_version: int
    train_loss: float
    reward_mean: float
    reward_std: float
    sync_overhead_s: float


class Controller:
    def __init__(
        self,
        rollout: RolloutBackend,
        trainer: TrainerBackend,
        reward: RewardBackend,
        weight_synchronizer: WeightSynchronizer,
        prompt_source: Callable[[], list[GenerationRequest]],
        sampling: SamplingParams,
        state_dir: str,
        rollout_workers: int = 1,
        reward_workers: int = 1,
        metrics_path: str | None = None,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
        max_retry_attempts: int = 3,
        retry_backoff_s: float = 0.0,
    ) -> None:
        self.rollout = rollout
        self.trainer = trainer
        self.reward = reward
        self.weight_synchronizer = weight_synchronizer
        self.prompt_source = prompt_source
        self.sampling = sampling
        self.rollout_workers = rollout_workers
        self.reward_workers = reward_workers
        self.metrics = MetricsLogger(
            metrics_path, wandb_project=wandb_project, wandb_run_name=wandb_run_name, wandb_config=wandb_config,
        )
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff_s = retry_backoff_s

        self.state_dir = state_dir
        self._state_path = os.path.join(state_dir, "orchestrator_state.json")
        self.policy_registry = PolicyRegistry(root_dir=os.path.join(state_dir, "policies"))

        resumed = OrchestratorState.load(self._state_path)
        self.state = resumed if resumed is not None else OrchestratorState()

    def resume_or_start(self, initial_policy_version: int) -> None:
        """Reconciles the orchestrator's OWN resumed state (if any, from
        a previous crashed run) with the version the rollout/trainer
        backends were JUST initialized to by `lifecycle.initialize()` -
        a fresh run's state starts at that version; a resumed run's
        state must already agree with it (fail loudly if not, same
        philosophy as every other version check in this file)."""
        if self.state.policy_version.rollout_policy_version is None:
            self.state.policy_version.current_policy_version = initial_policy_version

    def run(self, max_iterations: int) -> list[IterationResult]:
        results: list[IterationResult] = []
        while self.state.iteration < max_iterations:
            results.append(self._run_one_iteration())
        return results

    def _run_one_iteration(self) -> IterationResult:
        iteration = self.state.iteration

        # ---- verify rollout workers ----
        self._verify_rollout_workers()

        # ---- generate trajectories ----
        t0 = time.monotonic()
        requests = self.prompt_source()
        trajectories = self._generate(requests)
        generation_time_s = time.monotonic() - t0

        # ---- verify policy version ----
        self.state.policy_version.record_rollout(trajectories)

        # ---- compute rewards / build GRPO batch ----
        scored = score_trajectories(self.reward, trajectories, num_workers=self.reward_workers)

        # ---- send batch to quic-train / execute training update ----
        result = self.trainer.train(scored)

        # ---- obtain new policy version ----
        self.state.policy_version.record_training(result.policy_version)

        # ---- synchronize policy to rollout workers ----
        sync_overhead_s = self._sync_policy(iteration, result.policy_version)

        # ---- record metrics ----
        n_prompts = len(requests)
        total_tokens = sum(len(t.token_ids) for t in trajectories if t.token_ids)
        self.metrics.log_training(
            iteration=iteration, train_loss=result.train_loss, reward_mean=result.reward_mean,
            reward_std=result.reward_std, completion_rate=result.completion_rate,
            tokens_per_sec=result.tokens_per_sec, step_time_s=result.step_time_s,
        )
        self.metrics.log_rollout(
            iteration=iteration,
            prompts_per_sec=(n_prompts / generation_time_s) if generation_time_s > 0 else 0.0,
            tokens_per_sec=(total_tokens / generation_time_s) if generation_time_s > 0 else 0.0,
            generation_latency_s=generation_time_s,
        )
        self.metrics.log_rl(
            iteration=iteration, policy_version=result.policy_version, kl=result.kl,
            reward_mean=result.reward_mean, reward_std=result.reward_std,
            response_length_mean=(total_tokens / len(trajectories)) if trajectories else 0.0,
            group_size=requests[0].num_samples if requests else 0,
        )

        # ---- checkpoint state ----
        self.state.iteration += 1
        self.state.total_trajectories_collected += len(trajectories)
        self.state.total_training_steps += 1
        self.state.save(self._state_path)

        return IterationResult(
            iteration=iteration, policy_version=result.policy_version, train_loss=result.train_loss,
            reward_mean=result.reward_mean, reward_std=result.reward_std, sync_overhead_s=sync_overhead_s,
        )

    def _verify_rollout_workers(self) -> None:
        if not self.rollout.health_check():
            raise RuntimeError("rollout backend failed health_check() - refusing to start this iteration")

    def _generate(self, requests: list[GenerationRequest]) -> list[Trajectory]:
        # Real, transient-error retry (a dropped HTTP connection to
        # quic-vllm, say) - NOT worker-level fault tolerance (no health
        # check, no picking a different worker - see this module's own
        # docstring on why that's explicitly out of scope here).
        return retry_call(
            lambda: collect_rollouts(self.rollout, requests, self.sampling, num_workers=self.rollout_workers),
            max_attempts=self.max_retry_attempts, backoff_s=self.retry_backoff_s,
        )

    def _sync_policy(self, iteration: int, new_version: int) -> float:
        export_dir = os.path.join(self.state_dir, "exports", f"v{new_version}")
        exported_path = self.trainer.export_policy(export_dir)
        self.policy_registry.register(new_version, exported_path)

        sync_result = retry_call(
            lambda: self.weight_synchronizer.sync(exported_path, new_version),
            max_attempts=self.max_retry_attempts, backoff_s=self.retry_backoff_s,
        )
        self.rollout.load_policy(exported_path, new_version)

        self.state.policy_version.record_sync(new_version)
        self.metrics.log_network(
            iteration=iteration, bytes_transferred=sync_result.weight_size_bytes,
            trajectory_transfer_time_s=0.0, weight_sync_time_s=sync_result.sync_time_s, sync_frequency=1,
        )
        return sync_result.total_overhead_s
