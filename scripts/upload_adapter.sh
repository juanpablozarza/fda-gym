#!/usr/bin/env bash
# Upload a saved adapter. Requires a local adapter directory.
# Usage: scripts/upload_adapter.sh <local_adapter_dir> [remote_run_id]
set -euo pipefail
SRC="${1:?usage: scripts/upload_adapter.sh <local_adapter_dir> [remote_run_id]}"
RUN="${2:-grpo-qwen3b-sft150}"
MODAL="${MODAL:-modal}"
test -f "$SRC/adapter_model.safetensors" || { echo "no adapter at $SRC" >&2; exit 1; }
$MODAL volume create fdagym-checkpoints 2>/dev/null || true
$MODAL volume put fdagym-checkpoints "$SRC" "/runs/$RUN/adapter"
$MODAL volume ls fdagym-checkpoints "/runs/$RUN/adapter"
echo "Uploaded. Pipeline init adapter: runs/$RUN/adapter"
