#!/usr/bin/env bash
# Install the isolated official MolmoAct2 inference runtime.
# This installs no simulator and does not download model weights.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/project_environment.sh"
umask 0077

molmoact2_env="${MOLMOACT2_ENV:-$project_root/envs/agentic-framework-molmoact2}"

python3 "$framework_root/scripts/setup/fetch_sources.py" molmoact2 --destination "${MOLMOACT2_SOURCE:-$project_root/third_party/molmoact2}"
if [[ -x "$molmoact2_env/bin/python" ]]; then
  "$molmoact2_env/bin/python" -c 'import sys; assert (3, 11, 0) <= sys.version_info[:3] < (3, 12, 0), "MolmoAct2 service requires Python 3.11"'
else
  uv venv "$molmoact2_env" --python 3.11
fi
uv pip install --python "$molmoact2_env/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$framework_root/requirements-molmoact2.lock"
uv pip check --python "$molmoact2_env/bin/python"
