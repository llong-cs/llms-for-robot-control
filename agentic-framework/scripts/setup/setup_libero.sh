#!/usr/bin/env bash
# Install the exact LIBERO simulator used by the framework in a separate environment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/project_environment.sh"
openpi_source=${OPENPI_SOURCE:-$project_root/third_party/openpi}
libero_env=${AGENTIC_LIBERO_ENV:-$project_root/envs/agentic-framework-libero}
libero_config=${LIBERO_CONFIG_PATH:-$project_root/.config/libero}
python3 "$framework_root/scripts/setup/fetch_sources.py" openpi --destination "$openpi_source"
if [[ ! -x "$libero_env/bin/python" ]]; then
  uv venv "$libero_env" --python 3.8
fi
uv pip install --python "$libero_env/bin/python" \
  -r "$openpi_source/examples/libero/requirements.txt" \
  -r "$openpi_source/third_party/libero/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
"$libero_env/bin/python" - "$openpi_source/third_party/libero" "$libero_config" "$project_root/data/libero" <<'PY'
import pathlib, sys, sysconfig, yaml
source = pathlib.Path(sys.argv[1]).resolve()
configuration = pathlib.Path(sys.argv[2]).expanduser()
configuration.mkdir(parents=True, exist_ok=True)
# Upstream LIBERO is a namespace package; its setuptools wheel omits the tree.
(pathlib.Path(sysconfig.get_paths()['purelib']) / 'agentic-libero-source.pth').write_text(str(source) + '\n')
benchmark = source / 'libero/libero'
paths = {key: str(benchmark / value) for key, value in {
    'benchmark_root': '.', 'bddl_files': 'bddl_files', 'init_states': 'init_files', 'assets': 'assets'
}.items()}
paths['datasets'] = str(pathlib.Path(sys.argv[3]).resolve())
(configuration / 'config.yaml').write_text(yaml.safe_dump(paths))
print('LIBERO configuration:', configuration)
PY
printf 'LIBERO environment ready: %s\n' "$libero_env"
printf 'Headless rendering requires the system OSMesa runtime, or MUJOCO_GL=egl on a supported GPU.\n'
