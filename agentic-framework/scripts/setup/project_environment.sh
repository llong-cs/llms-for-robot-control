#!/usr/bin/env bash
# Shared project-relative locations; explicitly supplied environment variables take precedence.
framework_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -n "${AGENTIC_PROJECT_ROOT:-}" ]]; then
  project_root="$AGENTIC_PROJECT_ROOT"
  if [[ "$project_root" == "~" ]]; then
    project_root="$HOME"
  elif [[ "$project_root" == "~/"* ]]; then
    project_root="$HOME/${project_root:2}"
  fi
  if [[ "$project_root" != /* ]]; then
    project_root="$PWD/$project_root"
  fi
else
  project_root="$(cd -- "$framework_root/.." && pwd)"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$project_root/cache/python}"
export HF_HOME="${HF_HOME:-$project_root/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$project_root/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$project_root/cache/pip}"
