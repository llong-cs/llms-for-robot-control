"""Portable paths and native model server commands for the evaluation launcher."""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "agentic-framework"
SUITE = ROOT / "ditto-bench"
# Resource variables of child processes and their defaults below the project root. A value
# the shell already sets is kept, unless the experiment's paths group sets it.
RESOURCE_DEFAULTS = {
    "HF_HOME": "cache/huggingface",
    "TORCH_HOME": "cache/torch",
    "UV_CACHE_DIR": "cache/uv",
    "PIP_CACHE_DIR": "cache/pip",
    "MS_ASSET_DIR": "data/maniskill",
    "LIBERO_CONFIG_PATH": ".config/libero",
    "UV_PYTHON_INSTALL_DIR": "cache/python",
    "OPENPI_DATA_HOME": "models/openpi",
}
# The variable each single-path paths key sets; cache_root sets every cache variable below it.
PATH_VARIABLES = {
    "maniskill_asset_dir": "MS_ASSET_DIR",
    "ditto_asset_dir": "DITTO_ASSET_DIR",
    "libero_config": "LIBERO_CONFIG_PATH",
    "openpi_data_home": "OPENPI_DATA_HOME",
}
CACHE_VARIABLES = {
    "HF_HOME": "huggingface",
    "TORCH_HOME": "torch",
    "UV_CACHE_DIR": "uv",
    "PIP_CACHE_DIR": "pip",
    "UV_PYTHON_INSTALL_DIR": "python",
}

def registry():
    if str(FRAMEWORK / "src") not in sys.path:
        sys.path.insert(0, str(FRAMEWORK / "src"))
    from agentic_framework.configuration.profiles import list_model_profiles

    return list_model_profiles()



def without_credentials(env):
    """Return env without model credentials and every registered provider's variables."""
    if str(FRAMEWORK / "src") not in sys.path:
        sys.path.insert(0, str(FRAMEWORK / "src"))
    from agentic_framework.common.environment import without_credentials as strip

    return strip(env)



def resolve_model(name):
    if str(FRAMEWORK / "src") not in sys.path:
        sys.path.insert(0, str(FRAMEWORK / "src"))
    from agentic_framework.configuration.profiles import load_model_profile

    profile = load_model_profile(name)
    return profile["id"], profile



def workspace_home():
    """Return the project root for mutable resources."""
    return Path(os.environ.get("AGENTIC_PROJECT_ROOT", ROOT)).expanduser().absolute()



def environment(name):
    return workspace_home() / "envs" / name / "bin/python"



def run_output(kind, model, root=None):
    """Return a new timestamped run directory below root, by default the project outputs/."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    return Path(root or workspace_home() / "outputs") / f"{stamp}-{kind}-{model}"



def path_environment(paths):
    """Return the child-process variables that an experiment's paths group assigns."""
    paths = paths or {}
    env = {name: paths[key] for key, name in PATH_VARIABLES.items() if key in paths}
    if "cache_root" in paths:
        env.update({name: str(Path(paths["cache_root"]) / relative)
                    for name, relative in CACHE_VARIABLES.items()})
    return env



def child_environment(gpu=None, overrides=None, *, defaults=True):
    """Return the environment of a launched child process.

    Resource variables keep a value set in the shell, else get their project default;
    ``overrides`` (from :func:`path_environment`) are assigned outright, so a paths group takes
    precedence over the shell. ``defaults=False`` leaves unset resource variables unset, so the
    launcher re-executed with this environment still sees which ones the shell set.
    """
    home = workspace_home()
    env = os.environ.copy()
    # Use the two canonical source trees regardless of old editable installations.
    env["PYTHONPATH"] = os.pathsep.join(
        map(
            str,
            (
                FRAMEWORK / "src",
                SUITE / "src",
                FRAMEWORK / "vendor/inspect-robots/src",
            ),
        )
    ) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["DITTO_FRAMEWORK_ROOT"] = str(FRAMEWORK)
    env["AGENTIC_PROJECT_ROOT"] = str(home)
    if defaults:
        for key, relative in RESOURCE_DEFAULTS.items():
            env.setdefault(key, str(home / relative))
    env.update(overrides or {})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env



def checked_output(value):
    output = Path(value).expanduser().resolve()
    if output in (ROOT, FRAMEWORK, SUITE, workspace_home()):
        raise ValueError("Select a run subdirectory, for example outputs/<run>")
    return output



def gpu_ids(value):
    if not re.fullmatch(r"\d+(,\d+)*", value):
        raise ValueError("--gpu requires comma-separated nonnegative GPU indices")
    ids = value.split(",")
    if len(set(ids)) != len(ids):
        raise ValueError("GPU indices must be unique")
    return ids



def command(args, extra, overrides=None):
    """Build a native server command without loading a checkpoint.

    ``overrides`` are resource variables assigned outright, as in :func:`child_environment`.
    """
    key, spec = resolve_model(args.model)
    candidate = Path(args.model).expanduser()
    model_selector = str(candidate.resolve()) if candidate.is_file() else args.model
    env = child_environment(overrides=overrides)
    if args.command == "serve":
        server, transport = spec["server"], spec["transport"]
        if not args.check_only:
            if args.gpu is None:
                raise ValueError(
                    "serve requires --gpu with explicitly selected available GPU indices"
                )
            if len(gpu_ids(args.gpu)) != server["gpus"]:
                raise ValueError(f"{key} requires {server['gpus']} GPU(s) in this launcher")
            env = child_environment(args.gpu, overrides)
        output = checked_output(args.output_dir or run_output("server", key))
        from agentic_framework.configuration.profiles import profile_path
        cmd = [str(args.python or profile_path(server["environment"]) / "bin/python")]
        if server["gpus"] > 1 and not args.check_only:
            cmd += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={server['gpus']}",
            ]
        cmd += [
            str(FRAMEWORK / "scripts" / server["script"]),
            "--model-profile",
            model_selector,
            "--port",
            str(args.port or transport["port"]),
            "--output-dir",
            str(output),
        ]
        if args.check_only:
            cmd += ["--check-only"]
        env.pop("PYTHONPATH", None)
        # A native model server sends no LLM requests.
        env = without_credentials(env)
        if spec["family"] == "openpi":
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    else:
        raise ValueError("Only native model serving is supported by this helper")
    return cmd + extra, env
