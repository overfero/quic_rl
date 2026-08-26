"""Real dataset for the Qwen3-1.7B-GRPO-math-reasoning reproduction:
`gsm8k_train` + `math_train` combined - this repo's own decision, matching
the reference checkpoint's actual `training_config.yaml` exactly
(`train_dataset_names: [gsm8k_train, math_train]`, read directly from that
file, not guessed - DAPO-Math-17k was this project's earlier assumption
before that config was actually read).

Ground truth is extracted with the SAME `extract_answer()`/`normalize_answer()`
`MathVerifierReward` itself uses to grade a response - so
`Trajectory.metadata["ground_truth"]` is already in the exact normalized
form a rollout's answer gets compared against, not raw dataset text a
second normalization pass could subtly diverge from.

System prompt matches the reference's own `actor.system_prompt` verbatim
(read directly from that file's raw YAML, not guessed): "Please reason
step by step, and put your final answer within \\boxed{}." - applied via
the tokenizer's own chat template (Qwen3-1.7B is instruction-tuned, unlike
this project's earlier Qwen3-1.7B-Base work - a raw completion prompt
without the chat template would not reliably produce boxed answers)."""
from __future__ import annotations

import itertools
from typing import Callable, Iterator

from quic_rl.reward.math_verifier import extract_answer, normalize_answer
from quic_rl.rollout.base import GenerationRequest

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _gsm8k_examples() -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    for ex in ds:
        gt = extract_answer(ex["answer"])  # "#### N" trailing marker -> N
        if gt is not None:
            yield ex["question"], normalize_answer(gt)


def _math_examples() -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    # qwedsacf/competition_math: a real, currently-loadable mirror of the
    # MATH dataset (the original hendrycks/competition_math repo is gone
    # from the Hub) - verified directly (12500 real examples, `solution`
    # field ends in `\boxed{...}` same as GSM8K's `#### N`), not assumed.
    ds = load_dataset("qwedsacf/competition_math", split="train")
    for ex in ds:
        gt = extract_answer(ex["solution"])
        if gt is not None:
            yield ex["problem"], normalize_answer(gt)


def load_combined_math_examples(num_examples: int | None = None) -> list[tuple[str, str]]:
    """(problem_text, ground_truth) pairs from gsm8k_train + math_train,
    INTERLEAVED (not concatenated) so a `num_examples` slice - e.g. for a
    quick sanity run - still draws from both datasets rather than only
    ever seeing gsm8k first. An example whose answer can't be extracted
    at all (rare) is skipped outright - training against a ground truth
    we can't even parse ourselves would be worse than one fewer example."""
    out: list[tuple[str, str]] = []
    for pair in itertools.zip_longest(_gsm8k_examples(), _math_examples()):
        for item in pair:
            if item is not None:
                out.append(item)
                if num_examples is not None and len(out) >= num_examples:
                    return out
    return out


def build_prompt_source(
    tokenizer, samples_per_prompt: int, prompts_per_iteration: int = 8, num_examples: int | None = None,
) -> Callable[[], list[GenerationRequest]]:
    """Returns the zero-arg callable `Controller`'s `prompt_source` expects.
    Cycles through the combined dataset indefinitely (real multi-iteration
    training needs more passes than one epoch through ~20k examples at a
    handful of prompts/iteration) - never raises `StopIteration`.

    `tokenizer.apply_chat_template()` renders each problem with
    `SYSTEM_PROMPT` (matching the reference's real recipe) into the exact
    prompt string sent to `QuicVLLMRollout` - that backend's own
    `/v1/completions` call takes the string as-is, so the chat template
    is applied HERE rather than needing a second HTTP endpoint."""
    examples = load_combined_math_examples(num_examples)
    if not examples:
        raise RuntimeError("load_combined_math_examples() returned no usable examples")
    counter = itertools.count()
    cursor = itertools.cycle(examples)

    def source() -> list[GenerationRequest]:
        batch = []
        for _ in range(prompts_per_iteration):
            problem, ground_truth = next(cursor)
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True,
            )
            batch.append(
                GenerationRequest(
                    prompt_id=f"prompt-{next(counter)}", prompt=rendered, num_samples=samples_per_prompt,
                    metadata={"ground_truth": ground_truth},
                )
            )
        return batch

    return source
