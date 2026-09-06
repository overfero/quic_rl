"""`NoOpStageLauncher`: paired with `QuicVLLMRollout` (rollout/quic_vllm.py)
in this project's real cross-machine GRPO topology (examples/gpu_math_grpo.py)
- the ACTUAL restart of the GPU machine's inference server happens inside
`synchronization.weights.QuicWeightSynchronizer.sync()`, via its own
`stage_launcher` (a real `ssh_launcher.SshMultiMachineStageLauncher`,
launching this project's own custom quic-vllm fork - `scripts/
stage_server.py`/`launch_pp_stage.py`, real QUIC transport - not plain
vanilla `vllm serve`). `QuicVLLMRollout.load_policy()` calls
`self._stage_launcher.restart(policy_path)` a SECOND time right after
that, with the WRONG path (whatever `Controller._sync_policy()` passed
as `exported_path` - a path on the TRAINING machine's own disk, meaningless
on the GPU machine) - this would be both redundant (a second, wasted
full server reboot) and incorrect if given a real launcher. Passing THIS
no-op instead skips only that second restart call; `QuicVLLMRollout`'s
own post-restart `/health` poll and model-name re-resolution stay real
and useful regardless, confirming the restart `QuicWeightSynchronizer`
already triggered actually succeeded."""
from __future__ import annotations


class NoOpStageLauncher:
    def restart(self, policy_path: str) -> None:
        pass
