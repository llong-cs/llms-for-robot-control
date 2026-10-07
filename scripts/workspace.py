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

def registry():
    if str(FRAMEWORK / "src") not in sys.path:
        sys.path.insert(0, str(FRAMEWORK / "src"))
    from agentic_framework.configuration.profiles import list_model_profiles

    return list_model_profiles()



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



def run_output(kind, model):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    return workspace_home() / "outputs" / f"{stamp}-{kind}-{model}"



def child_environment(gpu=None):
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
    for key, relative in {
        "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch",
        "UV_CACHE_DIR": "cache/uv",
        "PIP_CACHE_DIR": "cache/pip",
        "MS_ASSET_DIR": "data/maniskill",
        "LIBERO_CONFIG_PATH": ".config/libero",
        "UV_PYTHON_INSTALL_DIR": "cache/python",
        "OPENPI_DATA_HOME": "models/openpi",
    }.items():
        env.setdefault(key, str(home / relative))
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



def command(args, extra):
    """Build a native server command without loading a checkpoint."""
    key, spec = resolve_model(args.model)
    candidate = Path(args.model).expanduser()
    model_selector = str(candidate.resolve()) if candidate.is_file() else args.model
    env = child_environment()
    if args.command == "serve":
        server, transport = spec["server"], spec["transport"]
        if not args.check_only:
            if args.gpu is None:
                raise ValueError(
                    "serve requires --gpu with explicitly selected available GPU indices"
                )
            if len(gpu_ids(args.gpu)) != server["gpus"]:
                raise ValueError(f"{key} requires {server['gpus']} GPU(s) in this launcher")
            env = child_environment(args.gpu)
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
        if spec["family"] == "openpi":
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    else:
        raise ValueError("Only native model serving is supported by this helper")
    return cmd + extra, env
