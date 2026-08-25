"""`PolicyRegistry`: maps a policy_version to the on-disk export that
represents it, and prunes old ones - the disk-usage-conscious convention
this whole ecosystem already uses (quic-train's own
`training_utils.save_checkpoint`'s `keep_last` pruning is the direct
precedent, not reinvented differently here)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolicyRegistry:
    root_dir: str
    keep_last: int = 2
    _by_version: dict[int, str] = field(default_factory=dict)

    def register(self, policy_version: int, path: str) -> None:
        self._by_version[policy_version] = path
        self._prune()

    def path_for(self, policy_version: int) -> str | None:
        return self._by_version.get(policy_version)

    def latest_version(self) -> int | None:
        return max(self._by_version) if self._by_version else None

    def _prune(self) -> None:
        if self.keep_last <= 0:
            return
        versions = sorted(self._by_version)
        for stale in versions[:-self.keep_last]:
            path = Path(self._by_version.pop(stale))
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
