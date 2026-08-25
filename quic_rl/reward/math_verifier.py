"""`MathVerifierReward`: deterministic answer verification for
mathematical-reasoning rollouts - the prompt's own explicit first
implementation, and explicitly NOT an LLM judge (no model call, no
network call, pure string/number comparison - trivially fast and
reproducible, exactly what a reward signal for RL needs to be stable).

Expects the ground-truth answer for a trajectory to travel in
`Trajectory.metadata["ground_truth"]` - set when the prompt/dataset
example is turned into a `GenerationRequest` (see `examples/math_grpo.py`
and `Trajectory`'s own docstring on optional-but-present-when-relevant
metadata). A trajectory with no ground truth in its metadata is a
configuration error, not a "reward 0" case - it fails loudly via
`score()` raising, not a silent wrong answer.

Answer extraction supports the two real formats this project's target
datasets actually use:
- MATH-style `\\boxed{...}` (DAPO-Math-17k, MATH-500, AIME, AMC prompts
  typically ask for this format) - LAST boxed expression in the response
  wins (a model that "shows its work" and states intermediate boxed
  values, then a final one, should be graded on the final one).
- GSM8K-style `#### <answer>` trailing marker.
Falls back to the last standalone number in the response if neither
marker is present - a real, if weaker, signal rather than an automatic
zero for a model that didn't follow the requested format yet (useful
early in training, before the policy has learned formatting)."""
from __future__ import annotations

import re

from quic_rl.trajectory import Trajectory

_GSM8K_MARKER = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_boxed(text: str) -> str | None:
    """Finds the LAST `\\boxed{...}` in `text`, handling nested braces
    (a naive regex like `\\\\boxed\\{(.*?)\\}` breaks on
    `\\boxed{\\frac{1}{2}}` - this walks brace depth manually instead)."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    content_start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unbalanced braces - malformed, not a match
    return text[content_start : i - 1]


def extract_gsm8k_answer(text: str) -> str | None:
    matches = _GSM8K_MARKER.findall(text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    return matches[-1] if matches else None


def normalize_answer(s: str) -> str:
    """Real, minimal normalization - NOT a full CAS/symbolic-equality
    checker (that's a materially bigger undertaking than "deterministic
    verification" implies, and the target datasets' answers are
    overwhelmingly plain numbers or simple expressions where this is
    sufficient): strips `$`/whitespace/thousands-commas, drops a
    trailing `.0`/redundant trailing zeros after a decimal point, and
    lowercases (so `\\text{...}` wrapper text and simple word answers
    still compare sensibly)."""
    s = s.strip().strip("$").strip()
    s = s.replace(",", "").replace(" ", "")
    s = s.lower()
    if re.fullmatch(r"-?\d+\.\d*?0*", s) and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_answer(response: str) -> str | None:
    for extractor in (extract_boxed, extract_gsm8k_answer, extract_last_number):
        found = extractor(response)
        if found is not None:
            return found
    return None


class MathVerifierReward:
    def __init__(self, correct_reward: float = 1.0, incorrect_reward: float = -1.0, no_answer_reward: float = -1.0) -> None:
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.no_answer_reward = no_answer_reward

    def score(self, trajectory: Trajectory) -> float:
        if "ground_truth" not in trajectory.metadata:
            raise ValueError(
                f"trajectory {trajectory.prompt_id!r} has no 'ground_truth' in metadata - "
                "MathVerifierReward needs the expected answer to compare against"
            )
        ground_truth = normalize_answer(str(trajectory.metadata["ground_truth"]))

        predicted = extract_answer(trajectory.response)
        if predicted is None:
            return self.no_answer_reward

        return self.correct_reward if normalize_answer(predicted) == ground_truth else self.incorrect_reward

    def batch_score(self, trajectories: list[Trajectory]) -> list[float]:
        return [self.score(t) for t in trajectories]
