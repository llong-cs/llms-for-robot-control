#!/usr/bin/env bash
# Install the framework and benchmark in the lightweight framework environment.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project_root=${AGENTIC_PROJECT_ROOT:-$repo_root}
framework_env=${AGENTIC_ENV:-$project_root/envs/agentic-framework}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$project_root/cache/uv}
bash "$repo_root/agentic-framework/scripts/setup/setup.sh"
uv pip install --python "$framework_env/bin/python" --no-deps -e "$repo_root/ditto-bench"
