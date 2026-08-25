"""Retry behavior (prompt's testing item 12) - plain transient-error
retry, not worker-level fault tolerance (see `quic_rl/retry.py`'s own
docstring)."""
from __future__ import annotations

import pytest

from quic_rl.retry import retry_call


def test_retry_call_succeeds_on_first_attempt():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert retry_call(fn, max_attempts=3) == "ok"
    assert len(calls) == 1


def test_retry_call_succeeds_after_transient_failures():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert retry_call(fn, max_attempts=3, backoff_s=0.0) == "ok"
    assert attempts["n"] == 3


def test_retry_call_reraises_after_exhausting_attempts():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ConnectionError("persistent")

    with pytest.raises(ConnectionError, match="persistent"):
        retry_call(fn, max_attempts=3, backoff_s=0.0)
    assert attempts["n"] == 3


def test_retry_call_only_retries_matching_exception_types():
    def fn():
        raise ValueError("not retriable in this call")

    with pytest.raises(ValueError):
        retry_call(fn, max_attempts=3, retry_on=(ConnectionError,))


def test_retry_call_rejects_invalid_max_attempts():
    with pytest.raises(ValueError):
        retry_call(lambda: None, max_attempts=0)
