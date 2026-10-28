#!/usr/bin/env bash
# Pin the GPU stack on a fresh RunPod pod (Ubuntu, CUDA, one A100 80GB) and stop.
# Does not start probe, GRPO, or eval. Does not call Modal.
#
# On the pod:
#   git clone https://github.com/juanpablozarza/fda-gym.git
#   cd bio-rl
#   bash scripts/runpod_setup.sh
#
# Then the commands in docs/train.md, section "Tool-mode jobs on a RunPod pod".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -f data/snapshots/2026-10-15/manifest.jsonl

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

# Frozen solve of the [gpu] extra. Same direct pins Modal's image installs.
uv sync --extra gpu --frozen --no-dev

if [[ -n "${HF_HOME:-}" ]]; then
  echo "HF_HOME is already ${HF_HOME}. Jobs keep it. Unset HF_HOME to use <runs-dir>/../hf-cache."
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is set"
else
  echo "WANDB_API_KEY is not set (training still runs; W&B logging stays off)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set (public Qwen weights do not need it)"
fi

uv run python - <<'PY'
import torch
import vllm
from fdagym.train.gpu_stack import gpu_requirements

assert torch.cuda.is_available(), "no CUDA device"
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))

def base(version: str) -> str:
    return version.split("+", 1)[0]

installed = {
    "torch": base(torch.__version__),
    "vllm": base(vllm.__version__),
}
import accelerate
import datasets
import peft
import transformers
import trl

installed["accelerate"] = base(accelerate.__version__)
installed["datasets"] = base(datasets.__version__)
installed["peft"] = base(peft.__version__)
installed["transformers"] = base(transformers.__version__)
installed["trl"] = base(trl.__version__)

for requirement in gpu_requirements():
    name, pinned = requirement.split("==", 1)
    got = installed[name]
    if got != pinned:
        raise SystemExit(f"{name} {got} != pin {pinned}")
    print(f"{name} {got}")
PY

echo "git $(git rev-parse HEAD)"
cat <<'EOF'

Setup finished. No job was started.
Next, from this directory, with the venv uv just created:

  uv run python scripts/run_gpu.py --help

Commands, outputs, and how to copy them off: docs/train.md
  "Tool-mode jobs on a RunPod pod"
EOF
