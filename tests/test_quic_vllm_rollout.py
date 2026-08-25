"""GPU-free tests for `QuicVLLMRollout`'s request/response translation and
protocol conformance - the real HTTP round trip against a live quic-vllm
deployment is validated separately (see docs/history/PHASE_C_LOOPBACK.md),
not repeated here since it needs 2 real GPUs and several minutes of model
load. These tests fake the HTTP layer to exercise the translation logic
fast and without a network."""
from __future__ import annotations

import pytest

from quic_rl.rollout.base import GenerationRequest, RolloutBackend, SamplingParams
from quic_rl.rollout.local_launcher import LocalTcpStageLauncher
from quic_rl.rollout.quic_vllm import QuicVLLMRollout


class _FakeStageLauncher:
    def __init__(self) -> None:
        self.restarted_with: list[str] = []

    def restart(self, policy_path: str) -> None:
        self.restarted_with.append(policy_path)


def _make_rollout(monkeypatch, health_sequence=None, completions_response=None):
    launcher = _FakeStageLauncher()
    rollout = QuicVLLMRollout(driver_url="http://fake:8080", stage_launcher=launcher, health_poll_interval=0.0)

    health_sequence = list(health_sequence or [200])

    def fake_get(path, timeout):
        if path == "/health":
            status = health_sequence.pop(0) if len(health_sequence) > 1 else health_sequence[0]
            return status, b""
        if path == "/v1/models":
            return 200, b'{"data": [{"id": "fake-model"}]}'
        raise AssertionError(f"unexpected GET {path}")

    def fake_post(path, payload, timeout):
        assert path == "/v1/completions"
        return completions_response(payload)

    monkeypatch.setattr(rollout, "_get", fake_get)
    monkeypatch.setattr(rollout, "_post_json", fake_post)
    return rollout, launcher


def test_quic_vllm_rollout_satisfies_protocol():
    rollout = QuicVLLMRollout(driver_url="http://x", stage_launcher=_FakeStageLauncher())
    assert isinstance(rollout, RolloutBackend)


def test_load_policy_restarts_and_polls_health_until_ready(monkeypatch):
    rollout, launcher = _make_rollout(monkeypatch, health_sequence=[0, 0, 200])
    rollout.load_policy("/policies/v3", policy_version=3)
    assert launcher.restarted_with == ["/policies/v3"]
    status = rollout.get_status()
    assert status["policy_version"] == 3
    assert status["policy_path"] == "/policies/v3"


def test_load_policy_raises_if_never_healthy(monkeypatch):
    rollout, _ = _make_rollout(monkeypatch, health_sequence=[0])
    rollout._health_poll_timeout = 0.01
    with pytest.raises(RuntimeError, match="never became healthy"):
        rollout.load_policy("/policies/v1", policy_version=1)


def test_generate_before_load_policy_raises(monkeypatch):
    rollout, _ = _make_rollout(monkeypatch)
    with pytest.raises(RuntimeError, match="before load_policy"):
        rollout.generate([GenerationRequest(prompt_id="p", prompt="hi")], SamplingParams())


def test_generate_translates_requests_and_responses(monkeypatch):
    seen_payloads = []

    def fake_completions(payload):
        seen_payloads.append(payload)
        n = payload["n"]
        return {
            "choices": [
                {
                    "index": i,
                    "text": f"resp-{i}",
                    "token_ids": [1, 2, 3],
                    "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3]},
                    "finish_reason": "length",
                }
                for i in range(n)
            ]
        }

    rollout, _ = _make_rollout(monkeypatch, completions_response=fake_completions)
    rollout.load_policy("/policies/v7", policy_version=7)

    requests = [
        GenerationRequest(prompt_id="a", prompt="prompt-a", num_samples=2),
        GenerationRequest(prompt_id="b", prompt="prompt-b", num_samples=1),
    ]
    trajs = rollout.generate(requests, SamplingParams(temperature=0.5, max_tokens=10, top_p=0.9))

    assert len(trajs) == 3
    assert [t.prompt_id for t in trajs] == ["a", "a", "b"]
    assert all(t.policy_version == 7 for t in trajs)
    assert all(t.token_ids == [1, 2, 3] for t in trajs)
    assert all(t.logprobs == [-0.1, -0.2, -0.3] for t in trajs)
    assert all(t.reward is None for t in trajs)
    assert seen_payloads[0]["n"] == 2
    assert seen_payloads[0]["model"] == "fake-model"
    assert seen_payloads[0]["logprobs"] == 1
    assert seen_payloads[0]["return_token_ids"] is True


def test_health_check_reflects_http_status(monkeypatch):
    rollout, _ = _make_rollout(monkeypatch, health_sequence=[200])
    assert rollout.health_check() is True


def test_local_tcp_stage_launcher_requires_stage_subdirs(tmp_path):
    launcher = LocalTcpStageLauncher(vllm_repo_dir="/does/not/matter")
    with pytest.raises(FileNotFoundError):
        launcher.restart(str(tmp_path))
