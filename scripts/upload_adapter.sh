#!/usr/bin/env bash
# Re-upload a saved adapter to a fresh Modal volume (the 2026-10-18 volume was deleted to stop storage cost).
# Usage: scripts/upload_adapter.sh [local_adapter_dir] [remote_run_id]
# Default: tonight's RL adapter saved on the box.
set -euo pipefail
SRC="${1:-/workspace/bio-rl-artifacts/2026-10-18/runs/grpo-qwen3b-sft150/adapter}"
RUN="${2:-grpo-qwen3b-sft150}"
MODAL="${MODAL:-modal}"
test -f "$SRC/adapter_model.safetensors" || { echo "no adapter at $SRC" >&2; exit 1; }
$MODAL volume create fdagym-checkpoints 2>/dev/null || true
$MODAL volume put fdagym-checkpoints "$SRC" "/runs/$RUN/adapter"
$MODAL volume ls fdagym-checkpoints "/runs/$RUN/adapter"
echo "Uploaded. In the pipeline use init adapter /vol/runs/$RUN/adapter"
