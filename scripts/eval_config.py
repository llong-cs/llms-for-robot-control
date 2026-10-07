"""Load experiment JSON without importing a model, simulator, or framework package."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

# This boundary intentionally excludes simulator and controller implementation details.
_GROUP_FIELDS = {
    "run": {"mode": str, "preview_context": bool, "preview_dx": float, "repeat_id": int},
    "environment": {
        "benchmark": str, "suite": str, "task_ids": str, "trials": int,
        "init_start": int, "seed": int, "category": str, "python": str,
        "difficulty": str, "molmoact2_source": str, "maniskill_assets": str,
    },
    "execution": {
        "h": int, "k": int, "control_interface": str, "motion_time_scale": int,
        "max_motion_steps": int, "max_steps": int,
    },
    "policy": {
        "model_profile": str, "inference_timeout": float, "family": str,
        "profile": str, "host": str, "port": int, "url": str,
        "metadata": str, "seed": int,
    },
    "llm": {
        "model": str, "backend": str, "base_url": str, "api_key_env": str, "env_file": str,
        "backend_options": dict, "max_output_tokens": int,
        "supported_efforts": str, "reasoning_mode": str, "max_response_chars": int,
        "max_attempts": int, "augmentations": dict,
    },
    "observation": {
        "max_images": int, "max_context_chars": int, "cameras": str,
        "profile": str, "demo": bool, "demo_path": str, "demo_mode": str,
        "demo_content": str,
        "demo_image_max_side": int,
    },
    "output": {
        "directory": str, "record_trajectory": bool,
    },
}
_NULLABLE = frozenset({
    "run.preview_dx", "run.repeat_id", "environment.suite", "environment.seed",
    "environment.category", "environment.python", "execution.max_steps",
    "policy.family", "policy.model_profile", "policy.host", "policy.port",
    "policy.url", "policy.metadata", "policy.seed", "llm.base_url",
    "llm.api_key_env", "llm.env_file", "llm.max_output_tokens",
    "llm.supported_efforts", "llm.reasoning_mode",
    "observation.demo_path", "output.directory",
})
_FIXED_FIELDS = frozenset({
    "environment.control_hz", "environment.sim_hz", "environment.gpu",
    "environment.sim_backend", "environment.sim_shader",
    "execution.action_tools", "execution.approver", "execution.auto_motion",
    "execution.done_verification_seconds",
})
_LAUNCHER_FIELDS = {
    "bench": str, "model": str, "mode": str, "server": str,
    "task_randomize": bool, "server_timeout": float,
    "server_args": list, "repeats": int,
}
_BENCH_ALIASES = {
    "libero-all": "libero_all", "libero_all": "libero_all",
    "ditto": "ditto",
    "maniskill": "molmoact2-maniskill", "molmoact2-maniskill": "molmoact2-maniskill",
    **{name.replace("_", "-"): name for name in (
        "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
    )},
    **{name: name for name in (
        "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
    )},
}
_PATH_FIELDS = (
    ("output", "directory"), ("observation", "demo_path"),
    ("environment", "python"), ("environment", "molmoact2_source"),
    ("environment", "maniskill_assets"), ("policy", "metadata"),
    ("llm", "env_file"),
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object from pairs, rejecting ambiguous duplicate fields."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject the nonstandard numeric constant named by value."""
    raise ValueError(f"Configuration numbers must be finite; found {value}")


def _check_type(value: Any, expected: type, field: str) -> None:
    """Validate value against a JSON type, identifying errors by field."""
    if value is None and field in _NULLABLE:
        return
    valid = type(value) in (int, float) if expected is float else type(value) is expected
    if not valid:
        labels = {str: "string", int: "integer", float: "number", bool: "boolean",
                  dict: "object", list: "array"}
        raise ValueError(f"{field} must be a JSON {labels[expected]}")
    if expected is float:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{field} must be finite")


def _validate_augmentations(value: dict[str, Any]) -> None:
    """Validate the nested memory and reasoning groups in value."""
    schemas = {
        "memory": {"history_length": int},
        "reasoning": {"enabled": bool, "effort": str, "notes": bool, "hindsight": bool},
    }
    for group, fields in value.items():
        prefix = f"llm.augmentations.{group}"
        if group not in schemas:
            raise ValueError(f"Unknown configuration field: {prefix}")
        _check_type(fields, dict, prefix)
        for name, setting in fields.items():
            field = f"{prefix}.{name}"
            if name not in schemas[group]:
                raise ValueError(f"Unknown configuration field: {field}")
            _check_type(setting, schemas[group][name], field)


def _resolve_path(value: str, base: Path, *, follow_symlinks: bool = True) -> str:
    """Expand a config-relative path, preserving interpreter symlinks when requested."""
    if not value.strip():
        raise ValueError("Configured paths must be nonempty strings")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    return str(path.resolve()) if follow_symlinks else os.path.abspath(path)


def _profile_is_path(value: str) -> bool:
    """Return whether value denotes a profile path rather than a registered ID."""
    return value.endswith(".json") or value.startswith(("/", "./", "../", "~"))


def load_experiment(path: Path) -> dict[str, Any]:
    """Read an experiment file and separate launcher settings from framework groups.

    Args:
        path: JSON file to read. Relative file-valued settings resolve against its
            containing directory; credential files are never opened by this function.

    Returns:
        A dictionary containing ``launcher``, ``framework``, and the absolute ``path``.
        Only explicitly supplied settings are returned. Framework semantic validation
        remains the responsibility of the framework configuration parser.

    Raises:
        ValueError: The JSON structure, field names, or value types are invalid, or a
            GPU, simulator-frequency, or fixed controller setting is supplied.
        OSError: The experiment file cannot be read.
    """
    resolved = Path(path).expanduser().resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(document, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    unknown = set(document) - (_GROUP_FIELDS.keys() | {"evaluation"})
    if unknown:
        raise ValueError(f"Unknown experiment configuration groups: {sorted(unknown)}")
    launcher = copy.deepcopy(document.get("evaluation", {}))
    _check_type(launcher, dict, "evaluation")
    for name, value in launcher.items():
        field = f"evaluation.{name}"
        if name not in _LAUNCHER_FIELDS:
            raise ValueError(f"Unknown configuration field: {field}")
        _check_type(value, _LAUNCHER_FIELDS[name], field)
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
    if launcher.get("repeats", 1) < 1:
        raise ValueError("evaluation.repeats must be a positive integer")
    if "bench" in launcher:
        bench = launcher["bench"]
        if bench not in _BENCH_ALIASES:
            raise ValueError(f"Unknown evaluation.bench: {bench!r}")
        launcher["bench"] = _BENCH_ALIASES[bench]
    if "mode" in launcher and launcher["mode"] not in {"auto", "agent", "vla", "preview"}:
        raise ValueError("evaluation.mode must be auto, agent, vla, or preview")
    if "server" in launcher and launcher["server"] not in {"auto", "external"}:
        raise ValueError("evaluation.server must be auto or external")
    if "server_timeout" in launcher and launcher["server_timeout"] <= 0:
        raise ValueError("evaluation.server_timeout must be positive")
    for index, argument in enumerate(launcher.get("server_args", [])):
        _check_type(argument, str, f"evaluation.server_args[{index}]")
        if "\x00" in argument:
            raise ValueError(f"evaluation.server_args[{index}] must not contain NUL")
    framework = copy.deepcopy({key: value for key, value in document.items() if key != "evaluation"})
    for group, fields in framework.items():
        _check_type(fields, dict, group)
        for name, value in fields.items():
            field = f"{group}.{name}"
            if field in _FIXED_FIELDS:
                suffix = ("Set CUDA_VISIBLE_DEVICES to select GPUs."
                          if name == "gpu" else "The benchmark uses its standard settings.")
                raise ValueError(f"{field} is not configurable through eval.py. {suffix}")
            if name not in _GROUP_FIELDS[group]:
                raise ValueError(f"Unknown configuration field: {field}")
            _check_type(value, _GROUP_FIELDS[group][name], field)
            if field == "observation.demo_mode" and value not in {"id", "ood"}:
                raise ValueError("observation.demo_mode must be id or ood")
            if field == "observation.demo_content" and value not in {"observations", "full"}:
                raise ValueError("observation.demo_content must be observations or full")
            if field == "llm.augmentations":
                _validate_augmentations(value)
    for group, name in _PATH_FIELDS:
        value = framework.get(group, {}).get(name)
        if value is not None:
            # Resolving a venv's bin/python symlink selects its base interpreter
            # and loses the isolated simulator's site-packages.
            # Credential symlinks must reach the send-time permission checks intact.
            framework[group][name] = _resolve_path(
                value, resolved.parent,
                follow_symlinks=(group, name) not in {
                    ("environment", "python"), ("llm", "env_file"),
                },
            )
    profile = framework.get("policy", {}).get("model_profile")
    if profile is not None and (_profile_is_path(profile) or "/" in profile):
        framework["policy"]["model_profile"] = _resolve_path(profile, resolved.parent)
    model = launcher.get("model")
    if model is not None and _profile_is_path(model):
        launcher["model"] = _resolve_path(model, resolved.parent)
    return {"launcher": launcher, "framework": framework, "path": str(resolved)}


def infer_bench(framework: dict[str, Any]) -> str | None:
    """Infer a supported canonical benchmark from grouped framework settings.

    Args:
        framework: Grouped configuration containing optional environment benchmark
            and suite fields. An omitted benchmark is inferred only from a known suite.

    Returns:
        The canonical launcher benchmark, or ``None`` when the selection is absent,
        contradictory, or cannot be represented by a supported launcher benchmark.
    """
    environment = framework.get("environment", {})
    if not isinstance(environment, dict):
        return None
    suite = environment.get("suite")
    benchmark = environment.get("benchmark")
    if benchmark == "libero" and (suite is None or (isinstance(suite, str) and suite.strip() == "all")):
        return "libero_all"
    if not isinstance(suite, str):
        return None
    suite = suite.strip()
    if benchmark in (None, "libero"):
        alias = _BENCH_ALIASES.get(suite)
        if alias is not None and alias.startswith("libero_"):
            return alias
        parts = [part.strip() for part in suite.split(",")]
        if len(parts) == 4 and set(parts) == {
            "libero_spatial", "libero_object", "libero_goal", "libero_10"
        }:
            return "libero_all"
    if benchmark in (None, "maniskill"):
        if suite == "ditto":
            return "ditto"
        if suite == "DroidPutEverythingInBox-v1":
            return "molmoact2-maniskill"
    return None
