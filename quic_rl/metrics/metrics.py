"""Plain JSONL metrics logging - one JSON object per line, deliberately
the same convention as quic-train's own `training_utils.ExperimentLogger`
(this ecosystem's established choice: no W&B/TensorBoard dependency by
default, trivially `pandas.read_json(lines=True)`-able or greppable).

Category names match the prompt's own metrics spec exactly (Training/
Rollout/Network/System/RL) so a reader can trace every tracked field back
to the requirement that asked for it. Fault-tolerance-only fields (worker
availability/failures/retries/failed-rollouts/checkpoint-recovery-time)
are not included - see ARCHITECTURE.md for why that whole category is
out of scope for this repo.

W&B is strictly OPT-IN and additive - pass `wandb_project` to also mirror
every record there (namespaced `category/field`, matching the reference
run's own tracker: `jaygala24/Qwen3-1.7B-GRPO-math-reasoning`'s
`training_config.yaml` uses `wandb.use_wandb: true`). JSONL stays the
always-on source of truth either way; nothing about the default (no
`wandb_project`) path changes."""
from __future__ import annotations

import json
import time


class MetricsLogger:
    def __init__(
        self, path: str | None,
        wandb_project: str | None = None, wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
    ) -> None:
        self.path = path
        if path:
            import os

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._wandb_run = None
        if wandb_project:
            import wandb

            self._wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config or {})

    def _write(self, category: str, **fields) -> None:
        iteration = fields.get("iteration")
        if self.path:
            record = {"ts": time.time(), "category": category, **fields}
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if self._wandb_run is not None:
            payload = {f"{category}/{k}": v for k, v in fields.items() if k != "iteration" and v is not None}
            if payload:
                self._wandb_run.log(payload, step=iteration)

    def finish(self) -> None:
        """Closes the W&B run, if one is open - a no-op otherwise. Call
        once at the end of a run (Controller.run() does not call this
        itself, since a caller may run Controller.run() more than once
        against the same MetricsLogger)."""
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def log_training(self, iteration: int, train_loss: float, reward_mean: float, reward_std: float,
                      completion_rate: float, tokens_per_sec: float, step_time_s: float) -> None:
        self._write("training", iteration=iteration, train_loss=train_loss, reward_mean=reward_mean,
                     reward_std=reward_std, completion_rate=completion_rate, tokens_per_sec=tokens_per_sec,
                     step_time_s=step_time_s)

    def log_rollout(self, iteration: int, prompts_per_sec: float, tokens_per_sec: float,
                     generation_latency_s: float, worker_utilization: float | None = None) -> None:
        self._write("rollout", iteration=iteration, prompts_per_sec=prompts_per_sec, tokens_per_sec=tokens_per_sec,
                     generation_latency_s=generation_latency_s, worker_utilization=worker_utilization)

    def log_network(self, iteration: int, bytes_transferred: int, trajectory_transfer_time_s: float,
                     weight_sync_time_s: float | None = None, sync_frequency: int | None = None) -> None:
        self._write("network", iteration=iteration, bytes_transferred=bytes_transferred,
                     trajectory_transfer_time_s=trajectory_transfer_time_s, weight_sync_time_s=weight_sync_time_s,
                     sync_frequency=sync_frequency)

    def log_system(self, iteration: int, gpu_utilization: dict | None = None) -> None:
        self._write("system", iteration=iteration, gpu_utilization=gpu_utilization)

    def log_rl(self, iteration: int, policy_version: int, kl: float | None, reward_mean: float, reward_std: float,
               response_length_mean: float, group_size: int) -> None:
        self._write("rl", iteration=iteration, policy_version=policy_version, kl=kl, reward_mean=reward_mean,
                     reward_std=reward_std, response_length_mean=response_length_mean, group_size=group_size)
