# Architecture

## Why three separate repos

**quic-train** (a real `torch.distributed.ProcessGroup` backend over
QUIC) owns training. Its GRPO implementation
(`rlhf.py::run_grpo_training`) already contains the real loss math -
group-relative advantage, the reference/policy forced-forward pass, the
KL penalty. **quic-vllm** (a forked vLLM whose worker-to-worker
communication is swappable - TCP/UDP/QUIC/quic-shared) owns inference.
Its client-facing API is 100% vanilla, unmodified vLLM - the OpenAI-
compatible HTTP surface (`vllm/entrypoints/openai/*.py`) has zero
references to the custom transport; only the *worker-to-worker* traffic
on the pipeline-parallel stages uses it. `quic-rl` owns neither - it
coordinates when to generate, when to train, and when to propagate
updated weights, without reimplementing either.

## Integration findings (verified, not assumed)

These came from directly reading both repos' code, not from
documentation or assumption - see each finding's file:line citations.

### 1. quic-train's GRPO has no external-rollout hook

`rlhf.py::run_grpo_training` is monolithic: it calls `pipeline_generate`
itself (in-process, using quic-train's own pipeline-parallel model
instance), scores with a pluggable `reward_fn`, computes group-relative
advantage, then runs the real GRPO loss and `optimizer.step()` - all in
one function, one process group. There was no existing way to feed it
externally-generated `(prompt, response, reward)` triples.

**Resolution**: `run_grpo_training_from_rollouts(rank, signaling_url,
config, rollout_source, job_id=...)` was added to quic-train's `rlhf.py`,
next to (not replacing) `run_grpo_training`. The existing loop's loss-
computation body (`forced_forward`, advantage, `pg_loss`/`kl`, backward,
`optimizer.step()`) was factored into a shared helper both functions
call - the new one loops over `rollout_source` (already-tokenized
`(prompt_ids, generated_ids, reward)` batches) instead of calling
`pipeline_generate`. `run_grpo_training`'s existing behavior, tests, and
call sites are completely unchanged. This is the smallest integration
surface: no GRPO math duplicated into `quic_rl`, quic-train's public API
grows by exactly one function.

`rollout_source`'s handoff mechanism: `quic_rl.trainer.quic_train`
writes each rollout batch as a `torch.save()`'d file to a directory
shared with (or `scp`'d to) the machine running quic-train rank 0; the
new function polls that directory. Matches this ecosystem's existing
convention (checkpoints/configs are already file-based) - no new network
protocol invented.

### 2. quic-vllm's client-facing API needs no QUIC awareness

Only ONE machine in a distributed quic-vllm deployment (the pipeline-
parallel "driver", the one passing `--serve`) exposes the standard
OpenAI-compatible HTTP API, with real `logprobs`/`prompt_logprobs`/
`return_token_ids` support already there (vanilla vLLM feature, not
added by the fork). This makes `quic_rl.rollout.quic_vllm.QuicVLLMRollout`
a plain HTTP client - it never touches QUIC/UDP/the transport layer at
all, that's entirely internal to how quic-vllm's own worker processes
talk to each other.

### 3. Live weight hot-swap doesn't fit this topology - and can't cheaply

Vanilla vLLM has a real RLHF weight-transfer HTTP API
(`/update_weights`, `/pause`, `/resume`, etc.) built on NCCL/IPC weight-
transfer engines. quic-vllm's `TransportExecutor` forwards exactly 4
self-contained per-step RPCs to remote pipeline stages
(`execute_model`, `sample_tokens`, `execute_dummy_batch`,
`take_draft_token_ids`) - weight-transfer calls are not among them.

Extending that forwarding list is real, precedented, small work (the
receiving side, `stage_server.py`'s RPC loop, is already a generic
`getattr(model_executor, method)(*args, **kwargs)` dispatcher). But the
underlying mechanism doesn't fit regardless: `init_weight_transfer_engine`'s
NCCL backend needs every worker across *every machine* to rendezvous
into one shared process group via a raw TCP `StatelessProcessGroup`,
requiring **direct, un-NAT'd TCP reachability between the trainer and
every remote GPU worker** - exactly what the whole QUIC/hole-punch
transport exists to avoid needing. LoRA loading has the same shape
problem from a different angle: `LoRARequest.lora_path` is read as a
local filesystem path on each worker, never pushed as bytes over RPC.

**Resolution**: weight synchronization uses the restart-based "export →
transfer → reload" approach - not a compromise, the only mechanism that
doesn't require solving NAT traversal for a second network this
ecosystem's whole transport layer exists to route around. This needs
**zero code changes inside quic-vllm** - `scripts/launch_pp_stage.py`/
`stage_server.py` are used exactly as they exist today (`--model <path>`
already means "each stage loads its own local checkpoint shard
independently"; restart-with-new-path is the deployment's native
operating mode). quic-vllm's existing checkpoint stage-extractor tool
(present in the repo, not reinvented) shards a full checkpoint per PP
stage before transfer.

### 4. LoRA export fits naturally

quic-train's GRPO trains a peft/LoRA adapter (small, trainable-params-
only, matching `training_utils.py`'s existing checkpoint convention).
`WeightSynchronizer` merges it into the base
(`peft.PeftModel.merge_and_unload()`) to produce one full checkpoint,
then reuses quic-vllm's stage-extractor to shard it, then transfers
shards and restarts each stage. No LoRA-over-RPC path needed anywhere.

## Rollout lifecycle

```
prompt dataset
  → orchestrator partitions prompts across rollout_workers (workers/rollout_worker.py)
  → each dispatches a partition to RolloutBackend.generate()
  → results stamped with prompt_id + policy_version, collected, verified
    (no fabricated/dropped prompt_ids, no short counts - fails loudly otherwise)
  → handed to reward_workers (workers/reward_worker.py) for scoring
  → group_statistics() computes per-prompt reward mean/std (GRPO's group)
  → training batch ready for TrainerBackend.train()
```

## Policy lifecycle

```
policy_v0 → rollout (stamped v0) → GRPO update → policy_v1 → sync → rollout with v1 → ...
```

`orchestrator/state.py::PolicyVersionState` tracks three DISTINCT
fields, never conflated: `current_policy_version` (canonical, right
now), `rollout_policy_version` (what a just-collected batch was actually
generated under), `training_policy_version` (what the trainer just
trained FROM). Any mismatch - mixed versions within one batch, a stale
rollout worker, a sync that silently didn't take effect, a trainer that
doesn't advance the version - raises `PolicyVersionMismatchError`
immediately. Nothing in this repo catches and silently ignores that
exception.

## Weight synchronization

`WeightSynchronizer.sync()` performs merge → shard (quic-vllm's
stage-extractor) → transfer → restart, and its `SyncResult` structurally
requires every cost field the integration architecture measures: weight
size, transfer time, sync time, reload time, total overhead. Nothing
about this repo's benchmarking can accidentally omit these.

**`QuicWeightSynchronizer`'s real `transfer` step uses `quic_dist`'s QUIC
`ProcessGroup`** (`quic_rl/synchronization/quic_transfer.py`), NOT
`vllm/transport/quic_transport.py` and NOT `scp`/`ssh`. Why not vLLM's
own transport: it can only be imported through the full `vllm` package,
which eagerly resolves the CUDA platform at import time - a real,
working quic-vllm build has to already be present, which the training
machine has no reason to carry. Why not scp/ssh: this project's SSH
tunnels have a documented ~5GB/day budget that a repeated multi-GB
checkpoint sync would exhaust in roughly one sync. `quic_dist`, by
contrast, is genuinely standalone (no vLLM dependency - see
`process_group.py`'s own module docstring) and is already deployed on
the training machine for actual training traffic, so reusing it for a
second purpose (an ad-hoc, throwaway 2-rank process group per sync call,
torn down immediately after) costs nothing extra to deploy. SSH is used
only to start the remote side of each transfer (`quic_transfer.py`'s own
CLI, `nohup ... & disown` over a control-plane SSH command) - the weight
bytes themselves travel entirely over the QUIC path, so a sync's cost
never touches the SSH budget regardless of checkpoint size. Real-world
throughput measured between this project's actual two machines (Council
Bluffs, Iowa and Groningen, Netherlands - genuinely different continents)
was ~0.94 MB/s, byte-exact (sha256-verified) - slow enough that sync
frequency, not correctness, is the real constraint on this path for a
multi-GB checkpoint.

Real bug found deploying this: `mkdir -p X && CMD < /dev/null > log 2>&1
& disown` over SSH does NOT fully detach - the I/O redirection after
`CMD` only applies to `CMD`, not to `mkdir`, so `mkdir`'s own stdio stays
attached to the SSH session's pty and the whole channel hangs open
indefinitely even though the visible command already returned. Fixed by
wrapping the whole chain in a brace group (`{ mkdir -p X && CMD; } <
/dev/null > log 2>&1 & disown`) so the redirection covers the entire
group, not just its last simple command.

**`QuicTrainBackend.export_policy()`'s real implementation** turned out
simpler than the LoRA-merge plan above once the project moved to
`full_finetune=True`: no adapter, so no merge step - the live model
already IS the full checkpoint. It does NOT load a second model copy in
the orchestrator process (a real RAM-exhaustion bug found running this
for real: quic-train's rank subprocesses never exit between `train()`
calls, so a second full-model load in a separate process while both
stayed resident pushed a real 31GB-RAM box over its ceiling - not a
GPU/VRAM problem, the GPUs had headroom the whole time). Instead it
writes a file-based export *request* that rank 0 services directly from
its own already-GPU-resident model (`peft_model.save_pretrained()`),
checked both right after each real checkpoint and during the rank's own
idle poll for the next batch (the latter closes a real deadlock: without
it, a request arriving while the rank is idly waiting for the next
training batch - which is exactly when `export_policy()` is normally
called - would never get serviced).

**Known limitation, found running this for real**: the exported
checkpoint does not preserve Qwen3's `embed_tokens`/`lm_head` weight
tying - `save_pretrained()` on the live model wrote both as independent
full tensors (confirmed directly via the `.safetensors` file: both
`lm_head.weight` and `model.embed_tokens.weight` present at full size,
151936×2048 each), inflating the reported param count by exactly that
tensor's size (311,164,928) and the checkpoint by ~600MB (~18%) versus a
properly-tied save. Believed harmless for this repo's actual use of the
export (inference serving only - vLLM just needs numerically-correct
values, not object-level tying, and nothing ever reloads an export back
into training), but not yet fixed at the source.

## Failure handling

**In scope**: orchestrator-level checkpoint/resume
(`OrchestratorState.save()`/`.load()`, called every iteration - if the
orchestrator PROCESS itself crashes, restarting it picks up from the
last completed iteration, not from scratch) and plain transient-error
retry (`quic_rl/retry.py` - an HTTP call or file operation that failed
once gets retried a bounded number of times before the exception is
allowed to propagate for real).

**Explicitly out of scope**: worker-level fault tolerance - heartbeat,
health-check-driven dead-worker detection, worker re-registration, and
automatically replacing a dead rollout/reward worker mid-run. A backend
call that raises propagates and stops the run; it is not caught and
retried against a *different* worker. This was a deliberate scope
decision, not an oversight - the first milestone is making GRPO work
well with a fixed set of workers, not building elastic worker
management. A future iteration of this repo could add it back in as
genuinely new work, cleanly layered on top of the `RolloutBackend`/
`workers/rollout_worker.py` abstractions that already exist - it would
not require redesigning them.

## Future architecture (documented, not built)

- **Async rollout**: the `RolloutBackend`/`TrainerBackend` Protocols
  don't assume synchronous calling - a future async orchestrator loop
  (generation overlapping with training, rather than one iteration fully
  completing before the next starts) could implement the same
  interfaces without a redesign. Not attempted now per the prompt's own
  explicit "start synchronous" instruction.
- **Other RL algorithms** (PPO, RLOO, REINFORCE): quic-train already has
  real implementations of all three (`rlhf.py`). Each would need its own
  `run_<algo>_training_from_rollouts` decomposition (same shape as
  GRPO's), and its own `TrainerBackend` implementation - the
  orchestrator/rollout/reward abstractions here are already algorithm-
  general and would not need to change.
- **Worker-level fault tolerance**: see "Failure handling" above.
