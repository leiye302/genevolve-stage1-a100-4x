#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/run_all.sh WORK_DIR [--resume]" >&2
  exit 2
fi
work="$(realpath -m -- "$1")"
shift
cd "$(dirname -- "$0")/.."
if [[ "${GENEVOLVE_ALLOW_LIVE_SEARCH:-0}" != 1 || -z "${SERPER_API_KEY:-}" ]]; then
  echo "Set SERPER_API_KEY privately and GENEVOLVE_ALLOW_LIVE_SEARCH=1 to authorize live search." >&2
  exit 2
fi
if [[ ! -x .venv-train/bin/python || ! -x .venv-rollout/bin/python ]]; then
  echo "Run bash scripts/bootstrap.sh first." >&2
  exit 2
fi
.venv-train/bin/python run.py --work "$work" prepare
.venv-train/bin/python run.py --work "$work" sft "$@"
.venv-rollout/bin/python run.py --work "$work" collect --allow-live-search
