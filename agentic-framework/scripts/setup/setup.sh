#!/usr/bin/env bash
# Install the lightweight framework; model and simulator runtimes remain separate.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/project_environment.sh"
framework_env=${AGENTIC_ENV:-$project_root/envs/agentic-framework}
if [[ ! -x "$framework_env/bin/python" ]]; then
  uv venv --python 3.11 "$framework_env"
fi
uv pip install --python "$framework_env/bin/python" -r "$framework_root/requirements.lock"
uv pip install --python "$framework_env/bin/python" --no-deps \
  -e "$framework_root/vendor/inspect-robots" -e "$framework_root"
uv pip check --python "$framework_env/bin/python"
PYTHONDONTWRITEBYTECODE=1 "$framework_env/bin/python" -B -m agentic_framework --help >/dev/null
