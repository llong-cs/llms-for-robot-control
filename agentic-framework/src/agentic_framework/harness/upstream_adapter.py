"""Small driver boundary around the vendored Inspect displacement compiler.

The upstream compiler treats every component as a displacement. Binary/command
semantics for a robot's gripper remain the controller's responsibility: use a
separate scalar split to determine a finite duration, then repeat the original
command, never send the divided values as purported gripper positions.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from agentic_framework._vendor.inspect_robots_agent.tools import Toolset
from agentic_framework.harness.types import ActionSpec


def _compiler(low: np.ndarray, high: np.ndarray, control_hz: float) -> Toolset:
    return Toolset(
        absolute=False,
        labels=tuple(str(i) for i in range(len(low))),
        state_key=None,
        state_labels=None,
        control_hz=control_hz,
        bounds_text="",
        low=low,
        high=high,
        step_limits=np.zeros_like(low),
    )


def playout_cap_steps(control_hz: float) -> int:
    """The original compiler permits at most ten seconds per motion."""
    if isinstance(control_hz, bool) or not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if not math.isfinite(10.0 * control_hz):
        raise ValueError("control_hz is too large to derive the upstream duration cap")
    return math.ceil(10.0 * control_hz)


def split_normalized_delta(
    delta: Any, *, low: Any, high: Any, control_hz: float
) -> tuple[np.ndarray, ...]:
    """Compile a finite numeric displacement with the exact upstream splitter.

    Bounds are in the same units as ``delta``; the name denotes our usual native
    normalized OSC input. No clipping, state feedback, arrival test, retry or
    extra settling occurs. The upstream ten-second duration cap raises ValueError.
    A zero displacement produces the upstream minimum of one zero action.
    """
    value = np.asarray(delta, dtype=np.float64)
    lower, upper = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if value.ndim != 1 or not value.size:
        raise ValueError("delta must be a nonempty vector")
    if lower.shape != value.shape or upper.shape != value.shape:
        raise ValueError("delta and bounds must have identical shapes")
    if not all(np.isfinite(v).all() for v in (value, lower, upper)):
        raise ValueError("delta and bounds must be finite")
    if np.any(lower > 0) or np.any(upper < 0) or np.any(lower > upper):
        raise ValueError("displacement bounds must contain zero")
    playout_cap_steps(control_hz)
    result = _compiler(lower, upper, float(control_hz))._move_displacement(
        value.copy(), list(range(value.size))
    )
    if result.error:
        raise ValueError(result.error)
    if result.chunk is None:
        raise RuntimeError("upstream displacement compiler returned no chunk")
    return tuple(
        np.asarray(action.data, dtype=np.float64).copy() for action in result.chunk.actions
    )


def build_motion_segment(pose_delta6: Any, spec: ActionSpec) -> tuple[np.ndarray, ...]:
    """Convert declared physical pose increments once and compile the six arm axes."""
    delta = np.asarray(pose_delta6, dtype=np.float64)
    scale = np.asarray(spec.pose_scale, dtype=np.float64)
    if delta.shape != (6,) or scale.shape != (6,):
        raise ValueError("pose delta and pose_scale must have six dimensions")
    if not np.isfinite(delta).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("pose delta must be finite and pose_scale positive and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = delta / (scale * np.asarray(spec.pose_sign, dtype=np.float64))
    low = np.asarray(spec.input_low[:6], dtype=float)
    high = np.asarray(spec.input_high[:6], dtype=float)
    result = split_normalized_delta(normalized, low=low, high=high, control_hz=spec.control_hz)
    if spec.rotation_norm_limit is not None:
        # Native ManiSkill clips the rotation vector by its joint norm before
        # Euler conversion. Preserve that limit when choosing an automatic chunk.
        norm = math.hypot(*normalized[3:])
        required = norm / spec.rotation_norm_limit / (1.0 - 1e-6)
        cap = playout_cap_steps(spec.control_hz)
        if not math.isfinite(required) or required > cap:
            raise ValueError("requested rotation exceeds the upstream playout cap")
        count = max(len(result), math.ceil(required))
        if count > len(result):
            part = normalized / count
            if np.any((normalized != 0) & (part == 0)):
                raise ValueError("motion is too small to split into executable steps")
            result = tuple(part.copy() for _ in range(count))
    return result


def upstream_tool_fields() -> dict[str, dict]:
    """Reuse upstream human-facing note and termination fields, without its loop."""
    schemas = _compiler(np.array([-1.0]), np.array([1.0]), 20.0).schemas()
    functions = {schema["function"]["name"]: schema["function"] for schema in schemas}
    return copy.deepcopy(
        {
            "note": functions["move_by"]["parameters"]["properties"]["note"],
            "done": functions["done"],
            "give_up": functions["give_up"],
        }
    )
