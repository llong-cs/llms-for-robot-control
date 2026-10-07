"""Data-only model profiles shared by policies, servers and the root launcher.

This module deliberately imports only the standard library. Profiles identify
weights and deployment defaults; inference and physical conversion live in the
model family and its observation/action adapter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from agentic_framework.common.project import project_path


def config_directory() -> Path:
    """Locate configuration in a source checkout or a packaged wheel."""
    source = Path(__file__).resolve().parents[3] / "configs"
    return source if (source / "models").is_dir() else Path(__file__).resolve().parent / "data"


MODEL_PROFILES_ROOT = config_directory() / "models"


def _validate(value: object, source: Path) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Model profile must be an object: {source}")
    for name in ("id", "label", "family", "adapter"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError(f"Model profile requires nonempty {name}: {source}")
    if value["family"] not in ("openpi", "molmoact2"):
        raise ValueError("Model profile family must be openpi (pi05) or molmoact2")
    aliases = value.get("aliases", [])
    if not isinstance(aliases, list) or any(
        not isinstance(a, str) or not a.strip() for a in aliases
    ):
        raise ValueError("Model profile aliases must be a list of nonempty strings")
    for name in ("defaults", "environment", "transport", "model", "server"):
        if not isinstance(value.get(name), dict):
            raise ValueError(f"Model profile requires an object {name}")
    defaults = value["defaults"]
    if set(defaults) != {"h", "k", "control_hz", "max_steps", "inference_timeout"}:
        raise ValueError(
            "Model profile defaults require exactly h, k, control_hz, max_steps and inference_timeout"
        )
    for name in ("h", "max_steps"):
        if type(defaults.get(name)) is not int or defaults[name] < 1:
            raise ValueError(f"Model profile defaults.{name} must be a positive integer")
    if type(defaults.get("k")) is not int or defaults["k"] < 1:
        raise ValueError("Native model profile defaults.k must be a positive integer")
    hz = defaults.get("control_hz")
    if type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("Model profile defaults.control_hz must be positive and finite")
    timeout = defaults.get("inference_timeout")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Model profile defaults.inference_timeout must be positive and finite")
    for name in ("benchmark", "control_mode", "observation_profile"):
        if not isinstance(value["environment"].get(name), str) or not value["environment"][name]:
            raise ValueError(f"Model profile requires environment.{name}")
    if value["environment"]["benchmark"] not in ("libero", "maniskill"):
        raise ValueError("Model profile benchmark must be libero or maniskill")
    if value["family"] == "openpi" and (
        value["model"].get("pi05") is not True
        or value["model"].get("config_name") not in ("pi05_droid", "pi05_libero")
    ):
        raise ValueError("OpenPi model profiles must select pi05_droid or pi05_libero")
    transport = value["transport"]
    if transport.get("kind") not in ("websocket", "http"):
        raise ValueError("Model profile transport.kind must be websocket or http")
    if "port" in transport and (
        type(transport["port"]) is not int or not 1 <= transport["port"] <= 65535
    ):
        raise ValueError("Model profile transport.port must be an integer in [1, 65535]")
    for name in ("host", "url"):
        if name in transport and (
            not isinstance(transport[name], str) or not transport[name].strip()
        ):
            raise ValueError(f"Model profile transport.{name} must be nonempty")
    if transport["kind"] == "http" and not transport.get("url"):
        raise ValueError("HTTP model profile requires transport.url")
    if transport["kind"] == "websocket" and not (transport.get("host") and transport.get("port")):
        raise ValueError("WebSocket model profile requires transport.host and transport.port")
    model = value["model"]
    if not isinstance(model.get("checkpoint"), str) or not model["checkpoint"].strip():
        raise ValueError("Model profile requires a local model.checkpoint path")
    if "://" in model["checkpoint"]:
        raise ValueError("model.checkpoint must be a local path, not a remote URI")
    if not isinstance(model.get("expected_metadata"), dict) or not model["expected_metadata"]:
        raise ValueError("Model profile requires nonempty model.expected_metadata")
    if not isinstance(model.get("optional_metadata", {}), dict):
        raise ValueError("Model profile model.optional_metadata must be an object")
    expected = model["expected_metadata"]
    optional = model.get("optional_metadata", {})
    if (
        type(expected.get("action_horizon")) is not int
        or expected["action_horizon"] != defaults["h"]
    ):
        raise ValueError("Model profile defaults.h must match expected_metadata.action_horizon")
    if "config_name" in model and model["config_name"] != expected.get("model_config"):
        raise ValueError("Model profile config_name must match expected_metadata.model_config")
    for values in (expected, optional):
        if "replan_steps" in values and values["replan_steps"] != defaults["k"]:
            raise ValueError("Model profile replan_steps must match defaults.k")
    metadata = value["server"].get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Model profile server.metadata must be an object")
    for key in metadata.keys() & expected.keys():
        if metadata[key] != expected[key]:
            raise ValueError(
                f"Model profile server.metadata conflicts with expected_metadata.{key}"
            )
    for name in ("script", "environment", "source", "checkpoint_marker"):
        if not isinstance(value["server"].get(name), str) or not value["server"][name].strip():
            raise ValueError(f"Model profile requires server.{name}")
    if type(value["server"].get("gpus")) is not int or value["server"]["gpus"] < 1:
        raise ValueError("Model profile server.gpus must be a positive integer")
    return value


def _read(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"Model profile contains nonfinite number {value}")

    return _validate(json.loads(path.read_text(), parse_constant=reject_constant), path)


def list_model_profiles() -> dict[str, dict]:
    """Return freshly loaded built-in profiles, keyed by their canonical IDs."""
    result, names = {}, set()
    for path in sorted(MODEL_PROFILES_ROOT.glob("*.json")):
        value = _read(path)
        if value["id"] != path.stem:
            raise ValueError(f"Model profile id must match its filename: {path}")
        for name in (value["id"], *value.get("aliases", [])):
            if name.casefold() in names:
                raise ValueError(f"Ambiguous model profile id or alias: {name}")
            names.add(name.casefold())
        result[value["id"]] = value
    return result


def load_model_profile(id_or_path: str | Path) -> dict:
    """Load a canonical ID, alias, or JSON file without importing any model code."""
    if not isinstance(id_or_path, (str, Path)) or not str(id_or_path).strip():
        raise ValueError("model_profile must be a profile ID or JSON path")
    candidate = Path(id_or_path).expanduser()
    if candidate.is_file():
        return _read(candidate)
    name = str(id_or_path)
    if isinstance(id_or_path, Path) or candidate.suffix == ".json" or "/" in name:
        raise FileNotFoundError(candidate)
    for value in list_model_profiles().values():
        if name.casefold() in (
            entry.casefold() for entry in (value["id"], *value.get("aliases", []))
        ):
            return value
    raise ValueError(f"Unknown model profile: {name}")


def validate_native_horizon(model_profile: dict, h: int) -> None:
    """A weight's output horizon is fixed, including custom execution schedules."""
    expected = model_profile["defaults"]["h"]
    if type(h) is not int or h != expected:
        raise ValueError(
            f"native {model_profile['id']} requires H={expected} from its model profile; "
            f"got H={h!r}. H cannot be overridden, including with profile='custom'."
        )


def profile_path(value: str | Path) -> Path:
    """Resolve intrinsic resource paths against the project root without accessing them."""
    return project_path(value)
