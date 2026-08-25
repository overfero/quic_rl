"""Trajectory serialization + ID consistency (prompt's testing items 1-2)."""
from __future__ import annotations

from quic_rl.trajectory import Trajectory


def test_to_dict_from_dict_roundtrip():
    t = Trajectory(
        prompt_id="p0", policy_version=3, prompt="2+2=?", response="4",
        token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3], reward=1.0, metadata={"k": "v"},
    )
    restored = Trajectory.from_dict(t.to_dict())
    assert restored == t


def test_from_dict_missing_optional_fields_defaults_cleanly():
    minimal = {"prompt_id": "p1", "policy_version": 0, "prompt": "hi", "response": "there"}
    t = Trajectory.from_dict(minimal)
    assert t.token_ids is None
    assert t.logprobs is None
    assert t.reward is None
    assert t.metadata == {}


def test_prompt_id_is_preserved_exactly_through_dict_roundtrip():
    ids = ["a", "a-1", "prompt_with_underscore", "123"]
    for pid in ids:
        t = Trajectory(prompt_id=pid, policy_version=0, prompt="x", response="y")
        assert Trajectory.from_dict(t.to_dict()).prompt_id == pid
