"""Real `RolloutBackend`: an HTTP client against quic-vllm's driver-stage
OpenAI-compatible API (vanilla, unmodified vLLM - see docs/ARCHITECTURE.md's
"quic-vllm's client-facing API is 100% vanilla" finding, confirmed by
reading `vllm/entrypoints/openai/*.py` directly). `generate()` never
launches or manages vLLM processes itself - restarting the stage processes
on a new policy version is `load_policy()`'s job, delegated to an injected
`StageLauncher` (keeps subprocess/SSH plumbing out of the HTTP-client half,
matching the same protocol-based separation as `RewardBackend`/
`TrainerBackend`).

Validated against a real local 2-stage TCP-loopback deployment
(Qwen3-1.7B-Base, stage0+stage1 on 2 local GPUs, `VLLM_USE_V2_MODEL_RUNNER=0`
- see this repo's own real-run notes for why that env var is required with
the currently-installed vLLM checkout): `/v1/completions` with `n>1`,
`logprobs`, and `return_token_ids` all confirmed to populate exactly as
this module assumes.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

from quic_rl.rollout.base import GenerationRequest, SamplingParams
from quic_rl.trajectory import Trajectory


class StageLauncher(Protocol):
    def restart(self, policy_path: str) -> None:
        """Blocking: stop the current stage process(es), start new ones
        pointed at `policy_path`, and return once every stage's local
        process has been launched - NOT necessarily once the driver's HTTP
        API is healthy yet. `QuicVLLMRollout.load_policy()` polls health
        itself after this returns, so a launcher only needs to get the
        processes running, not wait for full model load."""
        ...


class QuicVLLMRollout:
    def __init__(
        self,
        driver_url: str,
        stage_launcher: StageLauncher,
        model_name: str | None = None,
        request_timeout: float = 120.0,
        health_poll_interval: float = 2.0,
        health_poll_timeout: float = 600.0,
    ) -> None:
        self._driver_url = driver_url.rstrip("/")
        self._stage_launcher = stage_launcher
        self._model_name = model_name
        self._request_timeout = request_timeout
        self._health_poll_interval = health_poll_interval
        self._health_poll_timeout = health_poll_timeout
        self._policy_version: int | None = None
        self._policy_path: str | None = None

    def _get(self, path: str, timeout: float) -> tuple[int, bytes]:
        req = urllib.request.Request(f"{self._driver_url}{path}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, OSError):
            return 0, b""

    def _post_json(self, path: str, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._driver_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _resolve_model_name(self) -> str:
        if self._model_name:
            return self._model_name
        status, body = self._get("/v1/models", timeout=self._request_timeout)
        if status != 200:
            raise RuntimeError(f"QuicVLLMRollout: GET /v1/models failed with status {status}")
        models = json.loads(body).get("data", [])
        if not models:
            raise RuntimeError("QuicVLLMRollout: /v1/models returned no models")
        self._model_name = models[0]["id"]
        return self._model_name

    def load_policy(self, policy_path: str, policy_version: int) -> None:
        self._stage_launcher.restart(policy_path)
        deadline = time.monotonic() + self._health_poll_timeout
        while time.monotonic() < deadline:
            status, _ = self._get("/health", timeout=5.0)
            if status == 200:
                self._policy_path = policy_path
                self._policy_version = policy_version
                # The restarted process may serve a different --model path;
                # re-resolve rather than trust a stale cached name.
                self._model_name = None
                self._resolve_model_name()
                return
            time.sleep(self._health_poll_interval)
        raise RuntimeError(
            f"QuicVLLMRollout.load_policy: driver at {self._driver_url} never became healthy "
            f"within {self._health_poll_timeout}s after restarting for policy_version={policy_version}"
        )

    def generate(self, requests: list[GenerationRequest], sampling: SamplingParams) -> list[Trajectory]:
        if self._policy_version is None:
            raise RuntimeError("QuicVLLMRollout.generate() called before load_policy()")
        model_name = self._resolve_model_name()
        out: list[Trajectory] = []
        for req in requests:
            payload = {
                "model": model_name,
                "prompt": req.prompt,
                "n": req.num_samples,
                "max_tokens": sampling.max_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "logprobs": 1,
                "return_token_ids": True,
            }
            response = self._post_json("/v1/completions", payload, timeout=self._request_timeout)
            for choice in response["choices"]:
                logprobs_obj = choice.get("logprobs") or {}
                out.append(
                    Trajectory(
                        prompt_id=req.prompt_id,
                        policy_version=self._policy_version,
                        prompt=req.prompt,
                        response=choice["text"],
                        token_ids=choice.get("token_ids"),
                        logprobs=logprobs_obj.get("token_logprobs"),
                        metadata={
                            **req.metadata,
                            "sample_index": choice["index"],
                            "finish_reason": choice.get("finish_reason"),
                            "backend": "quic_vllm",
                        },
                    )
                )
        return out

    def health_check(self) -> bool:
        status, _ = self._get("/health", timeout=5.0)
        return status == 200

    def get_status(self) -> dict:
        return {
            "driver_url": self._driver_url,
            "policy_version": self._policy_version,
            "policy_path": self._policy_path,
            "healthy": self.health_check(),
        }

    def unload_policy(self) -> None:
        pass
