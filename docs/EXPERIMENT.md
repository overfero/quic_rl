# Experiment

**Status: not yet run.** This document's structure is fixed now so real
results have a place to land; every section below is either the planned
configuration (known now) or explicitly marked as pending a real run —
nothing here is a claimed result without a run backing it.

## Model

`Qwen/Qwen3-1.7B-Base`.

## Dataset

Main: DAPO-Math-17k. Sanity check (smaller, faster iteration during
development): GSM8K.

## Reward

`quic_rl.reward.math_verifier.MathVerifierReward` — deterministic answer
verification (`\boxed{}` / `#### ` extraction + normalized comparison),
no LLM judge. See its own docstring in
`quic_rl/reward/math_verifier.py` for exactly what normalization it does
and doesn't attempt.

## GRPO configuration

`configs/qwen3_1.7b_grpo.yaml` — `rollout.samples_per_prompt` (group
size), `rollout.workers`, and the underlying quic-train `GRPOConfig`
fields (`kl_coef`, `max_new_tokens`, `temperature`, `lr`) will be
recorded here with their final, actually-used values once a real run
happens (pending Phase E).

## Hardware / network topology

Available now: this local session plus 3 SSH-reachable machines (2×T4
each, ~15GB/GPU), matching the real multi-Kaggle-session target this
architecture was designed for (no NVLink/InfiniBand/shared filesystem/
static IP assumed anywhere — see `ARCHITECTURE.md`). Exact machine-to-
role assignment (which machine(s) run quic-train's training ranks, which
run quic-vllm's pipeline stages) will be recorded here once a real
deployment happens (pending Phase C/E).

## Benchmark methodology

See `benchmarks/README.md` for the planned experiment matrix (Experiments
1-3 from the integration architecture — scaling from 1 training/1
rollout worker up to whatever the available machine count supports).
Experiments 4-5 (heterogeneous worker availability, rollout worker
failure) are out of scope — this repo doesn't implement worker-level
fault tolerance, see `ARCHITECTURE.md`'s "Failure handling" section.

## Results

Pending a real run.

## Limitations

- Worker-level fault tolerance (a dead rollout worker being detected and
  replaced mid-run) is not implemented — a deliberate scope decision,
  not an oversight. See `ARCHITECTURE.md`.
- Live weight hot-swap across a distributed vLLM deployment is not
  possible with this ecosystem's current transport design (NAT
  incompatibility with vLLM's own NCCL-based weight-transfer engines —
  see `ARCHITECTURE.md`'s integration finding #3). Weight sync is
  restart-based, which has real, measured overhead per iteration —
  reported honestly in Results above once measured, not hidden.
- Only GRPO has a real `TrainerBackend` implementation; PPO/RLOO/
  REINFORCE stay documented-but-unbuilt future work.
