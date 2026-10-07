"""Shared pose envelopes for automatic targets and single-step chunk targets.

Environment step limits constrain goal offsets, with H scaling for move_by.
They do not change native controller scales or promise arrival within H steps.
"""

import math
from decimal import Decimal, localcontext

from agentic_framework.harness.types import ActionSpec, ScheduleConfig

POSE_FIELDS = ("dx", "dy", "dz", "rx", "ry", "rz")


def _decimal_product(*values: float | int) -> float:
    """Multiply declared decimal bounds before converting once to a JSON number.

    Direct binary multiplication makes endpoints such as 0.022 * 5 smaller than
    the advertised 0.11. Decimal arithmetic preserves the configuration's decimal
    intent without adding validation tolerances or accepting out-of-range values.
    """
    operands = tuple(Decimal(str(value)) for value in values)
    with localcontext() as context:
        context.prec = max(28, sum(len(value.as_tuple().digits) for value in operands))
        return float(math.prod(operands, start=Decimal(1)))


def single_step_bounds(spec: ActionSpec) -> tuple[tuple[float, float], ...]:
    endpoints = (
        (_decimal_product(low, scale, sign), _decimal_product(high, scale, sign))
        for low, high, scale, sign in zip(
            spec.input_low[:6], spec.input_high[:6], spec.pose_scale, spec.pose_sign, strict=True
        )
    )
    return tuple(
        (max(min(pair), -limit), min(max(pair), limit))
        for pair, limit in zip(endpoints, spec.single_step_pose_limit, strict=True)
    )


def motion_pose_bounds(spec: ActionSpec, config: ScheduleConfig) -> tuple[tuple[float, float], ...]:
    """Return a pose envelope, keeping gripper commands outside displacement scaling."""
    multiplier = config.h if config.control_interface == "move_by" else 1
    try:
        bounds = tuple(
            (_decimal_product(low, multiplier), _decimal_product(high, multiplier))
            for low, high in single_step_bounds(spec)
        )
    except OverflowError as exc:
        raise ValueError("H-scaled motion bounds must be finite") from exc
    if any(not math.isfinite(value) for pair in bounds for value in pair):
        raise ValueError("H-scaled motion bounds must be finite")
    return bounds


def validate_motion_pose(values, spec: ActionSpec, config: ScheduleConfig) -> None:
    automatic = config.control_interface == "move_by"
    kind = f"H-scaled target limit (H={config.h})" if automatic else "single-step limit"
    for name, (low, high) in zip(POSE_FIELDS, motion_pose_bounds(spec, config), strict=True):
        if name in values and not low <= values[name] <= high:
            raise ValueError(f"{name} exceeds the {kind} [{low:g}, {high:g}]")


def validate_single_step_pose(values, spec: ActionSpec) -> None:
    """Validate a chunk target against its per-component tool limits."""
    validate_motion_pose(values, spec, ScheduleConfig(h=1, control_interface="move_by_chunk"))


def validate_gripper_bounds(value: float, spec: ActionSpec) -> None:
    if not spec.gripper_low <= value <= spec.gripper_high:
        raise ValueError(
            f"gripper is out of gripper bounds [{spec.gripper_low:g}, {spec.gripper_high:g}]"
        )
