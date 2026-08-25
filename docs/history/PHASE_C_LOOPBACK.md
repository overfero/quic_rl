# Phase C: real quic-vllm wiring, local 2-stage loopback validation

First real (non-mock) validation of `QuicVLLMRollout` against a real
quic-vllm pipeline-parallel deployment. Single machine, 2 GPUs, TCP
transport (no signaling server needed for loopback), Qwen3-1.7B-Base split
into 2 stages (14 layers each, `scripts/extract_stage_checkpoint_qwen3.py`).

## What was validated

- `LocalTcpStageLauncher.restart()`: real `subprocess.Popen` of
  `scripts/stage_server.py` (stage 0, non-driver) and
  `scripts/launch_pp_stage.py --serve` (stage 1, driver), TCP transport,
  loopback.
- `QuicVLLMRollout.load_policy()`: real restart + health-poll cycle,
  blocking until `/health` returns 200.
- `QuicVLLMRollout.generate()`: real `/v1/completions` round trip,
  `n>1` (group sampling), `logprobs=1` + `return_token_ids=true` both
  populate as the architecture doc assumed. Two prompts, group sizes 3
  and 2, produced 5 correctly-shaped `Trajectory` objects with correct
  `policy_version` stamping and `reward=None`.
- Correctness, not just plumbing: "9 plus 6" answered 15, "largest planet"
  answered Jupiter - real model output through the full 2-stage pipeline.
- Clean shutdown: `LocalTcpStageLauncher.shutdown()` frees both GPUs (confirmed via `nvidia-smi`).

## Two real environment bugs hit and fixed

**1. `openai` package too old for this vLLM fork's tool-parser import
chain.** `vllm/tool_parsers/utils.py` imports `NamespaceTool` from
`openai.types.responses`, which doesn't exist in `openai==2.15.0` (the
version this fresh environment had installed, despite
`requirements/common.txt` only pinning `>=2.0.0`). Only the driver stage
hits this (`launch_pp_stage.py --serve` execs the full `vllm serve` CLI,
which imports the OpenAI-compatible API server module chain;
`stage_server.py`, run by non-driver stages, does not). Fixed by
upgrading to `openai==3.3.1`. Will need to be fixed the same way on every
SSH machine before running Phase C there.

**2. The installed vLLM checkout's "V2" model runner
(`vllm/v1/worker/gpu/model_runner.py`) is incompatible with this
project's transport-backed synthetic PP group.** `GPUModelRunner.__init__`
gates construction of `self.pp_handler` behind `self.use_pp`, which reads
`parallel_config.pipeline_parallel_size` (always `1` here, deliberately -
see `scripts/launch_pp_stage.py`'s own docstring for why) - so
`pp_handler` never gets built, even after `TransportPPWorker` swaps in
the real multi-stage `_PP` group post-init. Every non-last rank's
`sample_tokens()` then hits `assert self.pp_handler is not None` during
warmup. Even if patched to construct `pp_handler` post-swap (mirroring
the existing `is_first_pp_rank`/`is_last_pp_rank` refresh already in
`vllm/transport/pp_worker.py`), `PPHandler.__init__` calls
`get_pp_group().make_sibling_device_group()`, which itself calls
`torch.distributed.new_group()` - a real cross-machine NCCL rendezvous
the synthetic transport group cannot provide by design. This is a deeper
incompatibility than the two bugs already patched in `pp_worker.py`'s
comments, not a one-line fix.

The older "V1" model runner (`vllm/v1/worker/gpu_model_runner.py`, still
present in this checkout) has no `pp_handler`/`make_sibling_device_group`
dependency at all - it's almost certainly what this project's earlier
validated 3-4 machine real deployments actually ran on, before upstream
vLLM added the V2 runner + `PPHandler` broadcast optimization as a new
default. Workaround: set `VLLM_USE_V2_MODEL_RUNNER=0` on every stage
process (`LocalTcpStageLauncher` does this automatically). A real fix
inside quic-vllm - either implementing a transport-backed
`make_sibling_device_group`, or making V1-selection automatic when a
transport PP group is active - is future work, not required for Phase C
to proceed.

## Commands (for reference / reproduction)

```bash
# stage 0 (non-driver)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 python3 -u scripts/stage_server.py \
  --pp-rank 0 --pp-world-size 2 --self-name Stage0 --next-name Stage1 --driver-name Stage1 \
  --transport tcp --tcp-port-base 31000 --tcp-connect-host-next 127.0.0.1 \
  --rpc-port 40000 --rpc-listen-host 0.0.0.0 \
  --model /data/models/qwen3-1.7b-stage0 --tensor-parallel-size 1 --dtype float16 \
  --gpu-memory-utilization 0.5 --max-model-len 2048 --max-num-seqs 4 \
  --num-gpu-blocks-override 2000

# stage 1 (driver)
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V2_MODEL_RUNNER=0 python3 -u scripts/launch_pp_stage.py \
  --pp-rank 1 --pp-world-size 2 --self-name Stage1 --prev-name Stage0 \
  --transport tcp --tcp-port-base 31001 --tcp-connect-host-prev 127.0.0.1 \
  --model /data/models/qwen3-1.7b-stage1 --tensor-parallel-size 1 --dtype float16 \
  --gpu-memory-utilization 0.5 --max-model-len 2048 --max-num-seqs 4 \
  --serve --host 0.0.0.0 --port 8080 \
  --remote-stage-names Stage0 --remote-stage-hosts 127.0.0.1 --rpc-port 40000 \
  --num-gpu-blocks-override 2000
```

Note the TCP port asymmetry: `establish_pp_transports()`
(`vllm/transport/pipeline_bootstrap.py`) computes each side's port from
its own `is_prev` viewpoint (`port_offset = local_rank*2 + (0 if is_prev
else 1)`), which differs by 1 for the two ends of the same link with
TP=1. `--tcp-port-base` on stage 1 is set to stage 0's base + 1 to
compensate (`31000` / `31001` above) - not a bug, just something to get
right when hand-constructing TCP loopback commands; UDP transport (the
real multi-machine target) doesn't have this concern since it rendezvous
via the signaling server instead of a fixed listen/connect port pair.

## Not yet done

- Multi-machine deployment (the 3 SSH machines) - not attempted yet.
- UDP transport (loopback here used TCP for simplicity, matching the
  plan's stated ordering).
- `WeightSynchronizer`'s real merge/shard/transfer implementation -
  `LocalTcpStageLauncher.restart()` in this test pointed directly at
  pre-extracted stage checkpoints via symlinks, not a real
  LoRA-merge-then-shard pipeline.
