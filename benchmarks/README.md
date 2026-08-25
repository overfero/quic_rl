# Benchmarks

**Status: not yet run** (pending Phase C/D — real quic-train/quic-vllm
backends need to exist before there's anything real to benchmark; the
mock backends in `quic_rl/*/mock.py` have no meaningful throughput of
their own to measure).

Planned experiment matrix (Experiments 1-3 from the integration
architecture; 4-5 — heterogeneous worker availability and rollout worker
failure — are out of scope, this repo doesn't implement worker-level
fault tolerance, see `docs/ARCHITECTURE.md`):

| Experiment | Training workers | Rollout workers |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 3 | as many as available machines support | as many as available machines support |

Measured (never claimed without the actual number behind it):
end-to-end RL iteration time, rollout throughput, training throughput,
weight-synchronization overhead, communication overhead, scaling
efficiency. Real numbers land in `docs/EXPERIMENT.md` once these runs
actually happen.
