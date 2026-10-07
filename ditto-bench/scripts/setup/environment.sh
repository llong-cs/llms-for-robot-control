#!/usr/bin/env bash
# Install the suite in separate framework and simulator environments.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
framework_root="${DITTO_FRAMEWORK_ROOT:-$suite_root/../agentic-framework}"
if [[ -n "${AGENTIC_PROJECT_ROOT:-}" ]]; then
  release_root="$AGENTIC_PROJECT_ROOT"
  if [[ "$release_root" == "~" ]]; then
    release_root="$HOME"
  elif [[ "$release_root" == "~/"* ]]; then
    release_root="$HOME/${release_root:2}"
  fi
  if [[ "$release_root" != /* ]]; then
    release_root="$PWD/$release_root"
  fi
else
  release_root="$(cd -- "$suite_root/.." && pwd)"
fi

usage() {
  cat <<'USAGE'
Usage: environment.sh framework|sim

framework  Install the framework and suite into AGENTIC_ENV
           (default: <release>/envs/agentic-framework).
sim        Install pinned simulator dependencies and the suite into
           AGENTIC_MANISKILL_ENV (default: <release>/envs/agentic-framework-maniskill).

Robot assets, upstream source, and model weights are prepared separately.
USAGE
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  framework|sim) mode="$1"; shift ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi
command -v uv >/dev/null || { echo "Install uv before using the setup scripts." >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$release_root/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$release_root/cache/python}"
export HF_HOME="${HF_HOME:-$release_root/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$release_root/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$release_root/cache/pip}"
if [[ "$mode" == framework ]]; then
  suite_env="${AGENTIC_ENV:-$release_root/envs/agentic-framework}"
  AGENTIC_ENV="$suite_env" bash "$framework_root/scripts/setup/setup.sh"
else
  suite_env="${AGENTIC_MANISKILL_ENV:-$release_root/envs/agentic-framework-maniskill}"
  AGENTIC_MANISKILL_ENV="$suite_env" bash "$framework_root/scripts/setup/setup_maniskill.sh"
fi
uv pip install --python "$suite_env/bin/python" --no-deps -e "$suite_root"
PYTHONDONTWRITEBYTECODE=1 "$suite_env/bin/python" -B -c \
  'from ditto.catalog import TASKS; print(f"Ditto Bench: {len(TASKS)} tasks available")'
