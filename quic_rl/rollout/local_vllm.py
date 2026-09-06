"""Real `RolloutBackend`: vLLM-TPU running IN-PROCESS on the same host
as the orchestrator, via vLLM's offline `LLM.generate()` API - not
`QuicVLLMRollout`'s HTTP-client-to-a-remote-driver-machine model
(quic_vllm.py), because this project's actual first topology (see
docs/EXPERIMENT.md) is one shared Kaggle TPU VM, not separate rollout/
training machines. quic-train's own TPU chips and this backend's chips
are never claimed concurrently - see `docs/ARCHITECTURE.md`'s
sequential-handoff note and quic_dist/examples/vllm_generate_rollout.py's
module docstring for the real, measured reason (partitioning a TPU host's
chips between two concurrently-running SPMD-sharded JAX/XLA processes
hits an untested torch_xla TPU_CHIPS_PER_PROCESS_BOUNDS factorization
issue for strict subsets; two processes sequentially claiming ALL local
chips each has none of that risk).

Uses vLLM's LoRA HOT-SWAP (`enable_lora=True` + a fresh `LoRARequest`
per `load_policy()` call), NOT a full model reload/restart, because
quic-train's real training here is LoRA (`quic_dist.finetune.PipelineConfig`/
`rlhf.GRPOConfig`'s default `full_finetune=False`) - `policy_path` is the
adapter directory peft's own `save_pretrained()` writes (a few MB), not
a full checkpoint. This is real, measured savings: this repo's own first
engine boot (weights + JAX/XLA compilation across every traced shape)
took ~99s; a LoRA swap needs none of that - the base model stays loaded,
only the adapter tensors change. `QuicWeightSynchronizer`'s restart-
based full-checkpoint-reload strategy (synchronization/weights.py) is
the right choice for `full_finetune=True` (there IS no small adapter to
hot-swap - see that module's own docstring) but would be real, avoidable
overhead here."""
from __future__ import annotations

from dataclasses import dataclass, field

from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.trajectory import Trajectory


@dataclass
class LocalVLLMRollout:
    model_path: str
    tensor_parallel_size: int = 8
    max_model_len: int = 4096
    dtype: str = "bfloat16"
    max_lora_rank: int = 8

    _llm: object = field(default=None, init=False, repr=False)
    _current_lora_request: object = field(default=None, init=False, repr=False)
    _policy_version: int | None = field(default=None, init=False, repr=False)

    def _ensure_engine(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            enable_lora=True,
            max_lora_rank=self.max_lora_rank,
        )

    def load_policy(self, policy_path: str, policy_version: int) -> None:
        """`policy_path`: a peft LoRA adapter directory (base model
        weights never change - see this module's own docstring). The
        FIRST call also does the one real heavy engine boot (weight
        load + full shape-compilation, ~seconds to low-minutes on this
        model size); every call after that is a real hot-swap, no
        engine restart, no recompilation of the base model."""
        self._ensure_engine()
        from vllm.lora.request import LoRARequest

        self._current_lora_request = LoRARequest(
            lora_name=f"policy_v{policy_version}", lora_int_id=policy_version, lora_path=policy_path,
        )
        self._policy_version = policy_version

    def generate(self, requests: list[GenerationRequest], sampling: SamplingParams) -> list[Trajectory]:
        if self._policy_version is None:
            raise RuntimeError("LocalVLLMRollout.generate() called before load_policy()")
        from vllm import SamplingParams as VLLMSamplingParams

        vllm_sampling = VLLMSamplingParams(
            n=1,  # one GenerationRequest per desired sample - see the expansion below, matching
                  # QuicVLLMRollout's own convention of one prompt per API call rather than n>1
            max_tokens=sampling.max_tokens, temperature=sampling.temperature, top_p=sampling.top_p,
            logprobs=1,
        )
        # Expand num_samples>1 into repeated prompts (rather than n=N)
        # so each Trajectory's sample_index is unambiguous and every
        # request keeps its own metadata (ground truth, etc.) attached -
        # matches GenerationRequest's own per-request metadata contract.
        prompts: list[str] = []
        owners: list[GenerationRequest] = []
        for req in requests:
            for _ in range(req.num_samples):
                prompts.append(req.prompt)
                owners.append(req)

        outputs = self._llm.generate(prompts, vllm_sampling, lora_request=self._current_lora_request)

        out: list[Trajectory] = []
        counters: dict[str, int] = {}
        for owner, output in zip(owners, outputs):
            choice = output.outputs[0]
            idx = counters.get(owner.prompt_id, 0)
            counters[owner.prompt_id] = idx + 1
            logprobs_list = None
            if choice.logprobs:
                logprobs_list = [
                    next(iter(lp.values())).logprob for lp in choice.logprobs if lp
                ]
            out.append(
                Trajectory(
                    prompt_id=owner.prompt_id,
                    policy_version=self._policy_version,
                    prompt=owner.prompt,
                    response=choice.text,
                    token_ids=list(choice.token_ids),
                    logprobs=logprobs_list,
                    metadata={**owner.metadata, "sample_index": idx, "backend": "local_vllm"},
                )
            )
        return out

    def health_check(self) -> bool:
        return self._llm is not None

    def get_status(self) -> dict:
        return {
            "model_path": self.model_path,
            "policy_version": self._policy_version,
            "engine_loaded": self._llm is not None,
            "backend": "local_vllm",
        }

    def unload_policy(self) -> None:
        self._current_lora_request = None
        self._policy_version = None
