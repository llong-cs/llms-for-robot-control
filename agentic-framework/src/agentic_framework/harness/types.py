"""Framework contracts; model wire formats never enter the controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from inspect_robots.types import Action, ActionChunk

Horizon = int

# Semantic version of LLM chunk waypoints, recorded for teacher compatibility.
CHUNK_TARGET_REFERENCE = "cumulative_from_call_start"


@dataclass(frozen=True)
class AutoMotionConfig:
    """Measured-state termination and bounded feedback for automatic motions."""

    position_tolerance_m: float = 0.001
    rotation_tolerance_rad: float = 0.02
    stable_steps: int = 3
    stall_steps: int = 20
    position_progress_m: float = 0.0002
    rotation_progress_rad: float = 0.002
    gripper_min_steps: int = 8
    gripper_stable_steps: int = 4

    def __post_init__(self):
        for name in ("stable_steps", "stall_steps", "gripper_min_steps", "gripper_stable_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"auto_motion.{name} must be a positive integer")
        for name in (
            "position_tolerance_m",
            "rotation_tolerance_rad",
            "position_progress_m",
            "rotation_progress_rad",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"auto_motion.{name} must be positive and finite")


@dataclass(frozen=True)
class ScheduleConfig:
    h: Horizon = 5
    k: int = 5
    max_motion_steps: int = 200
    max_steps: int = 220
    auto_motion: AutoMotionConfig = field(default_factory=AutoMotionConfig)
    control_interface: str = "move_by"
    motion_time_scale: int = 1

    def __post_init__(self):
        if not isinstance(self.auto_motion, AutoMotionConfig):
            raise ValueError("auto_motion must be an AutoMotionConfig")
        if type(self.h) is not int or self.h < 1:
            raise ValueError("h must be a positive integer")
        if self.control_interface not in ("move_by", "move_by_chunk"):
            raise ValueError("control_interface must be move_by or move_by_chunk")
        if type(self.k) is not int or self.k == 0 or self.k < -1:
            raise ValueError("k must be -1 or a positive integer")
        if self.k == -1 and self.control_interface != "move_by":
            raise ValueError("K=-1 requires control_interface=move_by; move_by_chunk requires positive K")
        for name in ("max_motion_steps", "max_steps", "motion_time_scale"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


# Termination of an episode one of whose decisions failed max_attempts consecutive
# attempts; such an episode is discarded and excluded from every result.
MAX_ATTEMPTS_TERMINATION = "max_attempts"
DISCARDED_STATUS = "discarded"


@dataclass(frozen=True)
class MemoryConfig:
    """Nonpositive disables memory; positive selects an API window or a Codex dialogue."""

    history_length: int = 2

    def __post_init__(self):
        if type(self.history_length) is not int:
            raise ValueError("memory.history_length must be an integer")


@dataclass(frozen=True)
class ReasoningConfig:
    enabled: bool = True
    effort: str = "low"
    notes: bool = True
    hindsight: bool = True

    def __post_init__(self):
        for name in ("enabled", "notes", "hindsight"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"reasoning.{name} must be a boolean")
        if not isinstance(self.effort, str) or not self.effort.strip():
            raise ValueError("reasoning.effort must be a nonempty string")

    @property
    def effective_effort(self):
        return self.effort if self.enabled else "lowest"


@dataclass(frozen=True)
class AugmentationConfig:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)

    def __post_init__(self):
        if not isinstance(self.memory, MemoryConfig) or not isinstance(
            self.reasoning, ReasoningConfig
        ):
            raise ValueError("memory and reasoning must be grouped config objects")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {
            "memory",
            "reasoning",
        }:
            raise ValueError("unknown augmentation group")
        fields = dict(value)
        for name, config in (("memory", MemoryConfig), ("reasoning", ReasoningConfig)):
            if name in fields:
                if not isinstance(fields[name], dict):
                    raise ValueError(f"{name} must be an object")
                if name == "memory" and set(fields[name]) - {"history_length"}:
                    raise ValueError(
                        "memory now accepts only history_length; remove selective history flags. "
                        "Use history_length<=0 to disable history; positive values select an API window or enable a persistent Codex dialogue."
                    )
                fields[name] = config(**fields[name])
        return cls(**fields)


@dataclass(frozen=True)
class ActionSpec:
    pose_scale: tuple[float, ...] = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    single_step_pose_limit: tuple[float, ...] = (0.001, 0.001, 0.001, 0.01, 0.01, 0.01)
    pose_sign: tuple[float, ...] = (1.0,) * 6
    control_hz: float = 20.0
    frame: str = "world"
    rotation: str = "axis_angle"
    input_low: tuple[float, ...] = (-1.0,) * 7
    input_high: tuple[float, ...] = (1.0,) * 7
    gripper_low: float = -1.0
    gripper_high: float = 1.0
    gripper_neutral: float = 0.0
    # command: a driver command with a fixed neutral value (LIBERO).
    # absolute: an absolute gripper joint target; omission dynamically holds it.
    gripper_mode: str = "command"
    gripper_open: float | None = None
    gripper_close: float | None = None
    control_notes: str = ""
    rotation_norm_limit: float | None = None
    gripper_state_key: str = "gripper_width"
    gripper_open_position: float = 0.08
    gripper_close_position: float = 0.0
    gripper_position_tolerance: float = 0.0015
    gripper_stability_tolerance: float = 0.0002
    # Named arms use consecutive seven-dimensional pose/gripper action blocks.
    # Empty preserves the single-arm action and observation contract.
    arm_names: tuple[str, ...] = ()
    gripper_units: str = "radians"
    # Optional physical joint limits expressed in the observed gripper units.
    # Servo completion clips its expected measurement, never the commanded target.
    gripper_measured_range: tuple[float, float] | None = None

    def __post_init__(self):
        if len(self.single_step_pose_limit) != 6 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            for value in self.single_step_pose_limit
        ):
            raise ValueError("single_step_pose_limit must contain six positive finite values")
        if not isinstance(self.gripper_state_key, str) or not self.gripper_state_key.strip():
            raise ValueError("gripper_state_key must be a nonempty string")
        for name in (
            "gripper_open_position",
            "gripper_close_position",
            "gripper_position_tolerance",
            "gripper_stability_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric data")
        if self.gripper_position_tolerance <= 0 or self.gripper_stability_tolerance <= 0:
            raise ValueError("gripper observation tolerances must be positive")
        if self.gripper_open_position == self.gripper_close_position:
            raise ValueError("gripper observed open and close positions must differ")
        if (
            len(self.pose_scale) != 6
            or not np.all(np.isfinite(self.pose_scale))
            or min(self.pose_scale) <= 0
        ):
            raise ValueError("pose_scale must contain six positive finite values")
        if len(self.pose_sign) != 6 or any(value not in (-1.0, 1.0) for value in self.pose_sign):
            raise ValueError("pose_sign must contain six signs, each -1 or +1")
        if self.frame not in ("world", "base") or self.rotation not in ("axis_angle", "euler_xyz"):
            raise ValueError(
                "action pose requires WORLD/base frame and axis_angle/euler_xyz rotation"
            )
        if (not isinstance(self.arm_names, tuple)
            or any(not isinstance(name, str) or not name.isidentifier() for name in self.arm_names)
            or len(set(self.arm_names)) != len(self.arm_names)):
            raise ValueError("arm_names must contain unique identifier strings")
        if not isinstance(self.gripper_units, str) or not self.gripper_units.strip():
            raise ValueError("gripper_units must be a nonempty string")
        if self.gripper_measured_range is not None:
            limits = self.gripper_measured_range
            if (not isinstance(limits, tuple) or len(limits) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) for value in limits)
                or limits[0] >= limits[1]):
                raise ValueError("gripper_measured_range must contain two ascending finite limits")
        dimensions = 7 * max(1, len(self.arm_names))
        if len(self.input_low) != dimensions or len(self.input_high) != dimensions:
            raise ValueError("action bounds must contain seven dimensions per arm")
        if self.arm_names and any(
            bounds[offset:offset + 7] != bounds[:7]
            for bounds in (self.input_low, self.input_high)
            for offset in range(7, dimensions, 7)
        ):
            raise ValueError("named arms must share pose and gripper action bounds")
        if not np.all(np.isfinite(self.input_low + self.input_high)) or any(
            a >= b for a, b in zip(self.input_low, self.input_high)
        ):
            raise ValueError("invalid action bounds")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        if not np.all(np.isfinite([self.gripper_low, self.gripper_high, self.gripper_neutral])):
            raise ValueError("gripper action bounds and neutral must be finite")
        if not self.gripper_low <= self.gripper_neutral <= self.gripper_high:
            raise ValueError("gripper neutral must lie within its action bounds")
        if self.gripper_mode not in ("command", "absolute"):
            raise ValueError("gripper_mode must be command or absolute")
        for label in ("gripper_open", "gripper_close"):
            value = getattr(self, label)
            if value is not None and (
                isinstance(value, bool)
                or not np.isfinite(value)
                or not self.gripper_low <= value <= self.gripper_high
            ):
                raise ValueError(f"{label} must lie within the gripper bounds")
        if not isinstance(self.control_notes, str):
            raise ValueError("control_notes must be a string")
        if self.rotation_norm_limit is not None and (
            isinstance(self.rotation_norm_limit, bool)
            or not np.isfinite(self.rotation_norm_limit)
            or self.rotation_norm_limit <= 0
        ):
            raise ValueError("rotation_norm_limit must be positive and finite")


@dataclass(frozen=True)
class MotionCommand:
    name: str
    values: Mapping[str, float | str]
    note: str | None = None
    arm: str | None = None


@dataclass(frozen=True)
class MotionPlan:
    plan_id: str
    commands: tuple[MotionCommand, ...]
    horizon: Horizon
    observation_step: int


@dataclass(frozen=True)
class Stop:
    name: str
    detail: str = ""
    hindsight: str = "none"


@dataclass(frozen=True)
class PlanStatus:
    plan_id: str
    steps_executed: int
    next_command_index: int
    remaining_commands: int
    state: str = "running"
    target: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionFeedback:
    plan_id: str
    status: str
    steps_executed: int
    tracking_steps: int = 0
    holding_steps: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    instructions: str
    text: str
    images: Mapping[str, np.ndarray]
    tools: tuple[Mapping[str, Any], ...] = ()
    reasoning_effort: str = "low"
    # Current observation only; backends own API windows or the Codex conversation.
    messages: tuple[Mapping[str, Any], ...] = ()
    history_length: int = 2
    tool_results: tuple[Mapping[str, Any], ...] = ()
    decision_id: str | None = None


@dataclass(frozen=True)
class ModelReply:
    payload: Mapping[str, Any]
    usage: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    output_items: tuple[Mapping[str, Any], ...] = ()
    # Visible text the model returned with its tool call; never reasoning.
    response_text: str = ""


class ModelBackend(Protocol):
    def generate(self, request: ModelRequest) -> ModelReply: ...
    def close(self) -> None: ...


PLAN_KEY = "agentic.motion_plan"


def pack_plan(plan: MotionPlan, *, action_size: int = 7) -> ActionChunk:
    if not plan.commands:
        raise ValueError("motion plan must not be empty")
    return ActionChunk([Action(np.zeros(action_size)) for _ in plan.commands], meta={PLAN_KEY: plan})


def unpack_plan(chunk: ActionChunk) -> MotionPlan | None:
    value = chunk.meta.get(PLAN_KEY)
    if value is not None and not isinstance(value, MotionPlan):
        raise ValueError("invalid motion plan envelope")
    return value
