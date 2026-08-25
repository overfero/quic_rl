"""Reward calculation (prompt's testing item 4) - both the mock backend
and the real MathVerifierReward."""
from __future__ import annotations

import pytest

from quic_rl.reward.math_verifier import (
    MathVerifierReward,
    extract_answer,
    extract_boxed,
    extract_gsm8k_answer,
    normalize_answer,
)
from quic_rl.reward.mock import MockRewardBackend
from quic_rl.trajectory import Trajectory


def test_mock_reward_is_deterministic():
    backend = MockRewardBackend(target_length=4)
    t = Trajectory(prompt_id="a", policy_version=0, prompt="p", response="one two three four")
    assert backend.score(t) == backend.score(t) == 0.0


def test_mock_reward_penalizes_length_deviation():
    backend = MockRewardBackend(target_length=4)
    exact = Trajectory(prompt_id="a", policy_version=0, prompt="p", response="one two three four")
    short = Trajectory(prompt_id="b", policy_version=0, prompt="p", response="one")
    assert backend.score(exact) > backend.score(short)


def test_extract_boxed_simple():
    assert extract_boxed("the answer is \\boxed{42}") == "42"


def test_extract_boxed_handles_nested_braces():
    assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_picks_last_occurrence():
    text = "first try \\boxed{1}, actually \\boxed{2}"
    assert extract_boxed(text) == "2"


def test_extract_boxed_returns_none_when_absent():
    assert extract_boxed("no boxed answer here") is None


def test_extract_gsm8k_marker():
    assert extract_gsm8k_answer("reasoning...\n#### 128") == "128"


def test_extract_answer_prefers_boxed_over_number_fallback():
    assert extract_answer("compute 1+1, answer \\boxed{2}, note: step 3 was used") == "2"


def test_extract_answer_falls_back_to_last_number():
    assert extract_answer("I think the answer is roughly 17") == "17"


@pytest.mark.parametrize("raw,expected", [
    ("$42$", "42"), ("1,000", "1000"), ("3.0", "3"), ("3.50", "3.5"), (" 5 ", "5"),
])
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_math_verifier_correct_answer():
    reward = MathVerifierReward()
    t = Trajectory(prompt_id="a", policy_version=0, prompt="2+2=?", response="\\boxed{4}",
                    metadata={"ground_truth": "4"})
    assert reward.score(t) == reward.correct_reward


def test_math_verifier_incorrect_answer():
    reward = MathVerifierReward()
    t = Trajectory(prompt_id="a", policy_version=0, prompt="2+2=?", response="\\boxed{5}",
                    metadata={"ground_truth": "4"})
    assert reward.score(t) == reward.incorrect_reward


def test_math_verifier_no_answer_found():
    reward = MathVerifierReward()
    t = Trajectory(prompt_id="a", policy_version=0, prompt="2+2=?", response="I don't know",
                    metadata={"ground_truth": "4"})
    assert reward.score(t) == reward.no_answer_reward


def test_math_verifier_requires_ground_truth_in_metadata():
    reward = MathVerifierReward()
    t = Trajectory(prompt_id="a", policy_version=0, prompt="2+2=?", response="\\boxed{4}")
    with pytest.raises(ValueError, match="ground_truth"):
        reward.score(t)


def test_math_verifier_normalizes_before_comparing():
    reward = MathVerifierReward()
    t = Trajectory(prompt_id="a", policy_version=0, prompt="p", response="\\boxed{1,000.0}",
                    metadata={"ground_truth": "1000"})
    assert reward.score(t) == reward.correct_reward


def test_math_verifier_batch_score_matches_individual():
    reward = MathVerifierReward()
    trajectories = [
        Trajectory(prompt_id="a", policy_version=0, prompt="p", response="\\boxed{4}", metadata={"ground_truth": "4"}),
        Trajectory(prompt_id="b", policy_version=0, prompt="p", response="\\boxed{5}", metadata={"ground_truth": "4"}),
    ]
    assert reward.batch_score(trajectories) == [reward.score(t) for t in trajectories]
