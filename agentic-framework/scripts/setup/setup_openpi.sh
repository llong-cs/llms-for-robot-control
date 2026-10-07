#!/usr/bin/env bash
# Install the pinned OpenPi runtime in its own Python environment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/project_environment.sh"
openpi_source=${OPENPI_SOURCE:-$project_root/third_party/openpi}
openpi_env=${OPENPI_ENV:-$project_root/envs/agentic-framework-openpi}
export UV_PROJECT_ENVIRONMENT="$openpi_env"
export GIT_LFS_SKIP_SMUDGE=1
python3 "$framework_root/scripts/setup/fetch_sources.py" openpi --destination "$openpi_source"
uv sync --project "$openpi_source" --frozen --no-dev --python 3.11
uv pip check --python "$openpi_env/bin/python"
