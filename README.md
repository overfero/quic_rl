# quic-rl

The orchestration/integration layer between **quic-train** (distributed
LLM training over QUIC) and **quic-vllm** (distributed inference over
QUIC), for online RL. First target: **distributed GRPO with external
vLLM rollouts** on `Qwen/Qwen3-1.7B-Base`, mathematical reasoning.

```
quic-train  ↕  quic-rl  ↕  quic-vllm
```

## Why a separate repo

`quic-rl` does **not** implement GRPO's math and does **not** implement
LLM inference — both already exist, real and working, in the other two
projects. Its job is orchestration only: rollout scheduling, trajectory
collection, reward dispatch, policy-version management, weight
synchronization, and training-iteration coordination. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full integration
architecture and the real, verified findings about both other repos'
actual APIs that shaped every design choice here — nothing below was
invented without first checking what already exists.

## Architecture

```
quic-rl orchestrator
  ├─ RolloutBackend (QuicVLLMRollout)  → HTTP client to quic-vllm's driver
  │                                       machine's OpenAI-compatible API
  ├─ TrainerBackend (QuicTrainBackend) → launches quic-train's GRPO ranks,
  │                                       feeds them externally-generated rollouts
  ├─ RewardBackend (MathVerifierReward) → deterministic answer verification
  └─ WeightSynchronizer                 → merge → shard → transfer → restart
```

## Quick start (local mock example, no GPU needed)

```bash
pip install -e ".[dev]"
python3 examples/mock_loop.py
```

Runs the full orchestration loop (rollout → reward → train → weight sync
→ repeat) against `Mock*Backend` stand-ins — no model, no GPU, no
network call — and prints a coherent multi-iteration run plus a
machine-readable `metrics.jsonl`. This is the same loop the real
backends below plug into, unchanged.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Zero GPU/network dependency — trajectory serialization, policy-version
mismatch detection, reward calculation (including the real
`MathVerifierReward`), rollout/reward worker dispatch, orchestrator
state transitions and checkpoint/resume, retry behavior, weight-sync
orchestration, and config validation, all against mocks or pure
functions.

## Real distributed example

*(Phase B/C — not wired up yet. Will document the real launch sequence
across quic-train's training ranks and quic-vllm's pipeline-parallel
stages once implemented; see `docs/ARCHITECTURE.md`'s phasing.)*

## Kaggle deployment

*(Phase C/E — will document the real multi-Kaggle-session topology and
launch scripts once a real run has actually been done; see
`docs/EXPERIMENT.md` for real hardware/results once available.)*

## Scope

Synchronous orchestration only for now (rollout → reward → train →
sync, one iteration fully completing before the next starts) — the
prompt this architecture was built against is explicit: don't implement
asynchronous RL first. GRPO only, for now — `RolloutBackend`/
`TrainerBackend`/`RewardBackend` are algorithm-general `Protocol`s so
PPO/RLOO/REINFORCE can plug in later without changing the orchestrator,
but only GRPO has a real `TrainerBackend` implementation today. Worker-
level fault tolerance (heartbeat, health-check-driven dead-worker
detection/replacement) is explicitly out of scope — see
`docs/ARCHITECTURE.md`'s "failure handling" section for what IS in
scope (orchestrator-level checkpoint/resume, transient-error retry) and
why the rest was deliberately left out for now.
