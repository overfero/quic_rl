"""PERSISTENT server process for `LocalVLLMRollout` - run inside
vllm-tpu's own venv (a separate pinned jax+torch_xla stack from the
orchestrator's own). Boots the vLLM engine EXACTLY ONCE, pinned to a
real chip-count-sized rectangle of the physical TPU topology STARTING
AT CHIP 0 (via TPU_VISIBLE_CHIPS/TPU_CHIPS_PER_PROCESS_BOUNDS, set
before `vllm`/`jax` is ever imported - order matters, these are read at
first device query), then polls a directory for request files and
writes result files - the SAME file-based, atomic-rename-on-write
protocol `grpo_external_rollout_rank.py` already uses on the quic-dist
side (see that script's own module docstring), not a new IPC mechanism.

Why persistent, not one-shot-per-call (an earlier version of this
worker was launched fresh for every `generate()` call): confirmed
directly that a torch_xla process and a concurrent JAX process CAN each
claim their own disjoint chip rectangle of this host's real physical
topology at the same time (see quic_dist.training_utils.resolve_device's
`chip_offset` param and `_factor_chip_bounds()` docstrings for the exact
validated split - chips [0, tensor_parallel_size) here, chips
[tensor_parallel_size, total) for quic-train's own rank process(es), via
its OWN `tpu_chip_offset` config field) - so this engine can stay
resident across the orchestrator's entire run without ever conflicting
with quic-train's own TPU chip claim, and pays this repo's own real
first-boot cost (~99s: weight load + full JAX/XLA shape compilation on
Qwen3-1.7B) exactly ONCE instead of once per iteration.

Protocol: polls `{work_dir}/request_*.json` in strictly increasing
numeric order (matching the writer's own atomic write-then-rename
convention), writes `{work_dir}/result_{N}.json` for each, and exits
cleanly on `{work_dir}/shutdown.json` appearing.

Usage: python3 _local_vllm_worker.py <work_dir> <model_path>
  <tensor_parallel_size> <max_model_len> <dtype> <max_lora_rank>
"""
from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    work_dir, model_path = sys.argv[1], sys.argv[2]
    tensor_parallel_size = int(sys.argv[3])
    max_model_len = int(sys.argv[4])
    dtype = sys.argv[5]
    max_lora_rank = int(sys.argv[6])

    # MUST happen before the first `vllm`/`jax` import - these env vars
    # are read at first device query, not re-checked afterward. Chips
    # [0, tensor_parallel_size), a real rectangle of this host's
    # physical topology - see this module's own docstring.
    if tensor_parallel_size < 8:  # < the whole host - see resolve_device's
                                   # own whole-host-vs-subset distinction;
                                   # claiming everything needs no restriction
        os.environ.setdefault("TPU_VISIBLE_CHIPS", ",".join(str(c) for c in range(tensor_parallel_size)))
        px, py = 2, 4  # this project's only validated TPU shape so far (v5e-8) - see
                        # quic_dist.training_utils._factor_chip_bounds's identical fallback
        bounds = "1,1,1"
        for a in range(min(px, tensor_parallel_size), 0, -1):
            if tensor_parallel_size % a == 0 and (tensor_parallel_size // a) <= py:
                bounds = f"{a},{tensor_parallel_size // a},1"
                break
        os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", bounds)
        os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
        dtype=dtype, enable_lora=True, max_lora_rank=max_lora_rank,
    )
    print(f"[_local_vllm_worker] engine ready, polling {work_dir}", flush=True)

    current_lora: LoRARequest | None = None
    next_n = 1
    while True:
        shutdown_path = os.path.join(work_dir, "shutdown.json")
        if os.path.exists(shutdown_path):
            print("[_local_vllm_worker] shutdown requested, exiting", flush=True)
            return

        req_path = os.path.join(work_dir, f"request_{next_n:06d}.json")
        if not os.path.exists(req_path):
            # Also check for a policy-update request, which can arrive
            # between generate() calls (a real ordering requirement:
            # load_policy() must take effect before the NEXT generate(),
            # not retroactively on one already in flight).
            policy_path = os.path.join(work_dir, "load_policy.json")
            if os.path.exists(policy_path):
                with open(policy_path) as f:
                    p = json.load(f)
                if p.get("is_lora"):
                    current_lora = LoRARequest(
                        lora_name=f"policy_v{p['policy_version']}", lora_int_id=p["policy_version"],
                        lora_path=p["policy_path"],
                    )
                else:
                    current_lora = None
                os.remove(policy_path)
                done_path = os.path.join(work_dir, f"load_policy_done_{p['policy_version']}.json")
                with open(done_path, "w") as f:
                    json.dump({}, f)
            time.sleep(0.2)
            continue

        with open(req_path) as f:
            req = json.load(f)
        sampling = SamplingParams(
            n=1, max_tokens=req["sampling"]["max_tokens"], temperature=req["sampling"]["temperature"],
            top_p=req["sampling"]["top_p"], logprobs=1,
        )
        outputs = llm.generate(req["prompts"], sampling, lora_request=current_lora)

        completions = []
        for output in outputs:
            choice = output.outputs[0]
            logprobs_list = None
            if choice.logprobs:
                logprobs_list = [next(iter(lp.values())).logprob for lp in choice.logprobs if lp]
            completions.append({"text": choice.text, "token_ids": list(choice.token_ids), "logprobs": logprobs_list})

        result_path = os.path.join(work_dir, f"result_{next_n:06d}.json")
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"completions": completions}, f)
        os.rename(tmp_path, result_path)  # atomic - the orchestrator never sees a partial result file
        next_n += 1


if __name__ == "__main__":
    main()
