#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
mode="${1:-}"
if [[ "$#" -gt 1 || ( -n "$mode" && "$mode" != --sources-only ) ]]; then
  echo "Usage: bash scripts/bootstrap.sh [--sources-only]" >&2
  exit 2
fi
mkdir -p .deps
fetch_pin() {
  local repo="$1" revision="$2" target="$3"
  if [[ ! -d "$target/.git" ]]; then
    git init -q "$target"
    git -C "$target" remote add origin "$repo"
  fi
  [[ "$(git -C "$target" remote get-url origin)" == "$repo" ]]
  if ! git -C "$target" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$target" fetch --depth 1 origin "$revision"
    git -C "$target" checkout --detach "$revision"
  fi
  [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]
  [[ -z "$(git -C "$target" status --porcelain --untracked-files=no)" ]]
}
fetch_pin https://github.com/hiyouga/LLaMA-Factory.git 7af909522a951e3ad9f022ea6f88b6755257eaa5 .deps/llamafactory
fetch_pin https://github.com/MeiGen-AI/GenEvolve.git 23c847c559ccc0f95bbf4b3d8925898463822f4c .deps/genevolve
if [[ "$mode" == --sources-only ]]; then
  exit 0
fi

base_python="${GENEVOLVE_PYTHON:-python3.11}"
if [[ "$(uname -s)" != Linux ]]; then
  echo "Use Linux for the training and serving environments." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q 'release 12.8'; then
  echo "Activate CUDA 12.8 toolkit (nvcc) for FlashAttention-2." >&2
  exit 1
fi
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
"$base_python" -m venv .venv-train
train=.venv-train/bin/python
"$train" -m pip install --upgrade pip wheel setuptools ninja packaging psutil
"$train" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$train" -m pip install --no-build-isolation -r requirements-train.txt
"$train" -m pip install -c requirements-train.txt -e .deps/llamafactory
MAX_JOBS="${GENEVOLVE_BUILD_JOBS:-4}" "$train" -m pip install --no-build-isolation flash-attn==2.8.3
"$train" -m pip check
"$train" -c 'import deepspeed, flash_attn, torch; assert torch.version.cuda == "12.8"'

"$base_python" -m venv .venv-rollout
rollout=.venv-rollout/bin/python
"$rollout" -m pip install --upgrade pip wheel setuptools
"$rollout" -m pip install -r requirements-rollout.txt
"$rollout" -m pip check
"$rollout" -c 'import vllm, torch; assert torch.version.cuda == "12.8"'
