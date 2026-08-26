#!/bin/bash
# One-time setup for the orchestrator machine (quic-rl itself - runs
# alongside whichever machine drives training/inference, no compiled
# extensions of its own). Idempotent - safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== [1/3] quic_rl package (mock-backend core - no GPU/model needed) ==="
pip install -e ".[dev]" -q

echo "=== [2/3] real backends (quic-train/quic-vllm integration) ==="
pip install -e ".[real]" -q

echo "=== [3/3] W&B experiment tracking (opt-in) ==="
pip install -e ".[tracking]" -q
echo "Run 'wandb login <api-key>' once to authenticate this machine (stores"
echo "the key in ~/.netrc, never in the repo) - see MetricsLogger's own"
echo "docstring for how to enable it in a run (wandb_project=... on Controller)."

echo "=== verify ==="
python3 -m pytest tests/ -q

echo "=== done ==="
echo "quic_rl is ready. examples/mock_loop.py runs with no GPU; examples/math_grpo.py"
echo "needs the training/inference machines' own setup scripts to have run first."
