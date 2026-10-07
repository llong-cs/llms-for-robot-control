"""Select evaluation workers only within explicitly visible CUDA devices.

This module validates caller-selected device identifiers and constructs worker
environments without inspecting hardware or changing process-global variables.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)
_INDEX = re.compile(r"[0-9]+")


def _devices(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("Provide a nonempty sequence of CUDA device identifiers")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("CUDA device identifiers must be strings")
        token = value.strip()
        if token.upper().startswith("MIG"):
            raise ValueError("MIG devices are not supported by the evaluation rendering backends")
        if _INDEX.fullmatch(token):
            token = str(int(token))
        elif _UUID.fullmatch(token):
            token = "GPU-" + token[4:].lower()
        else:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES requires nonnegative GPU indices or complete GPU UUIDs; "
                "empty entries, -1, and abbreviated UUIDs are not supported"
            )
        if token in result:
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU identifiers")
        result.append(token)
    return result


def visible_devices(environ: Mapping[str, str] | None = None) -> list[str]:
    """Read the explicit allocation without inspecting GPUs or loading CUDA."""
    environ = os.environ if environ is None else environ
    value = environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Set CUDA_VISIBLE_DEVICES explicitly to select evaluation devices"
        )
    return _devices(value.split(","))


def allocate_devices(devices: Sequence[str], model_gpus: int = 1) -> list[list[str]]:
    """Partition every selected GPU into complete, ordered model replicas.

    API workers use the default single GPU for their simulator. Native replicas
    use the model profile's required GPU count, sharing their first GPU with the
    simulator. No GPU is discarded when a model requires more than one device.
    """
    devices = _devices(devices)
    if type(model_gpus) is not int or model_gpus < 1:
        raise ValueError("The model GPU count must be a positive integer")
    if len(devices) % model_gpus:
        raise ValueError(
            f"The {len(devices)} visible GPUs cannot form complete groups of {model_gpus}; "
            "adjust CUDA_VISIBLE_DEVICES to the model profile's GPU requirement"
        )
    return [devices[start:start + model_gpus] for start in range(0, len(devices), model_gpus)]


def device_environment(base_env: Mapping[str, str], group: Sequence[str]) -> dict[str, str]:
    """Return a child environment restricted to one group from the parent scope.

    Framework simulator configuration must leave ``gpu`` unset: passing logical
    device 0 there would overwrite this physical/UUID allocation in its worker.
    LIBERO currently uses OSMesa, and ManiSkill chooses a visible CUDA device.
    Stale EGL selectors can contradict the child's visibility even with OSMesa,
    so remove them instead of guessing a physical EGL index from a CUDA ordinal.
    """
    visible = visible_devices(base_env)
    group = _devices(group)
    if not set(group).issubset(visible):
        raise ValueError("A worker GPU group must be a subset of CUDA_VISIBLE_DEVICES")
    result = dict(base_env)
    result["CUDA_VISIBLE_DEVICES"] = ",".join(group)
    result.pop("MUJOCO_EGL_DEVICE_ID", None)
    result.pop("EGL_DEVICE_ID", None)
    return result

