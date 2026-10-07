#!/usr/bin/env bash
set -euo pipefail
umask 0077
source "$(dirname -- "${BASH_SOURCE[0]}")/project_environment.sh"
framework_sim_env="${AGENTIC_MANISKILL_ENV:-$project_root/envs/agentic-framework-maniskill}"
python3 "$framework_root/scripts/setup/fetch_sources.py" molmoact2 --destination "${MOLMOACT2_SOURCE:-$project_root/third_party/molmoact2}"
if [[ ! -x "$framework_sim_env/bin/python" ]]; then
  uv venv "$framework_sim_env" --python 3.11
fi
if [[ -f "$framework_root/requirements-maniskill.lock" ]]; then
  uv pip sync --python "$framework_sim_env/bin/python" "$framework_root/requirements-maniskill.lock" --index-strategy unsafe-best-match
else
  uv pip install --python "$framework_sim_env/bin/python" -r "$framework_root/requirements-maniskill.in" --index-strategy unsafe-best-match
fi
"$framework_sim_env/bin/python" -c 'import torch, mani_skill; print("torch", torch.__version__); print("ManiSkill import OK")'
