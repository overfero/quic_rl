"""Plain transient-error retry - NOT worker-level fault tolerance (no
health checks, no worker replacement, no re-registration - see
`docs/ARCHITECTURE.md` for why that whole category is out of scope for
this repo). This is the much smaller, ordinary thing: a single call that
failed once (a dropped HTTP connection, a momentary file-lock
contention) gets retried a bounded number of times before the exception
is allowed to propagate for real."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    max_attempts: int = 3,
    backoff_s: float = 0.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Calls `fn()` up to `max_attempts` times, sleeping `backoff_s *
    attempt_number` between attempts (0 by default - tests pass 0 so
    they run instantly and deterministically). Re-raises the LAST
    exception if every attempt fails - never silently swallows a
    persistent failure."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: BLE001 - re-raised below if this was the last attempt
            last_exc = exc
            if attempt < max_attempts and backoff_s > 0:
                time.sleep(backoff_s * attempt)
    assert last_exc is not None
    raise last_exc
