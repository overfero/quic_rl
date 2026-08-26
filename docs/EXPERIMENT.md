# Experiment

**Status: not yet run.** This document's structure is fixed now so real
results have a place to land; every section below is either the planned
configuration (known now) or explicitly marked as pending a real run —
nothing here is a claimed result without a run backing it.

## Model

`Qwen/Qwen3-1.7B` (instruction-tuned, not `-Base`) - matches the real
reproduction target's own base model
(`jaygala24/Qwen3-1.7B-GRPO-math-reasoning`'s `training_config.yaml`:
`model_path: Qwen/Qwen3-1.7B`). An earlier version of this document
assumed `-Base`; corrected once that config file was actually read
rather than guessed.

## Dataset

`gsm8k_train` + `math_train`, combined (see
`quic_rl.dataset.math_dataset.load_combined_math_examples`) - matches
the reproduction target's own real `train_dataset_names: [gsm8k_train,
math_train]`, read directly from its `training_config.yaml`. Earlier
versions of this document assumed DAPO-Math-17k, before that config was
actually read. Training hyperparameters are NOT required to match the
reference exactly - only model and dataset need to.

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

Two dedicated real machines (2×T4 each, ~15GB/GPU), matching the real
multi-Kaggle-session target this architecture was designed for (no
NVLink/InfiniBand/shared filesystem/static IP assumed anywhere — see
`ARCHITECTURE.md`): one runs quic-train's GRPO training ranks
(`full_finetune=True`, `world_size=2`), the other runs quic-vllm's
rollout deployment. Weight sync between them uses `quic_dist`'s own QUIC
transport (`QuicWeightSynchronizer`), not scp/ssh — see
`ARCHITECTURE.md`'s "Weight synchronization" section.

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
