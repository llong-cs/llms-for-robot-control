"""Finite action queues over Inspect's single-physical-step rollout contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from inspect_robots.errors import PolicyStop
from inspect_robots.types import Action, ActionChunk, Observation

from agentic_framework.common.geometry import (
    axis_angle_quaternion,
    euler_xyz_quaternion,
    pose,
    quaternion_multiply,
    quaternion_rotate,
    quaternion_to_euler_xyz,
    rotation_error,
    unit_quaternion,
    vector,
)
from agentic_framework.harness.motion_limits import (
    validate_gripper_bounds,
    validate_motion_pose,
)
from agentic_framework.harness.types import (
    CHUNK_TARGET_REFERENCE,
    ActionSpec,
    ExecutionFeedback,
    MotionCommand,
    MotionPlan,
    ScheduleConfig,
    Stop,
    unpack_plan,
)

_STATE = "agentic.controller_state"
# Execution status of the plan still active when the rollout ends for any reason.
EPISODE_ENDED = "episode_ended"
# Recorded control action of a done/give_up call that ends the episode with one neutral step.
NEUTRAL_STOP_STEP = "neutral_stop_step"


@dataclass
class _Resolved:
    position: np.ndarray
    quaternion: np.ndarray
    start_position: np.ndarray
    start_quaternion: np.ndarray
    delta: np.ndarray
    gripper: float
    explicit_gripper: bool
    held_gripper: float
    rotation_frame: np.ndarray
    arm: str | None = None


@dataclass
class _Active:
    plan_id: str
    start_step: int
    plan: MotionPlan | None = None
    sequence: tuple[Action, ...] = ()
    waypoints: tuple[_Resolved, ...] = ()
    target: _Resolved | None = None
    cursor: int = 0
    steps: int = 0
    tracking_steps: int = 0
    holding_steps: int = 0
    gripper_steps: int = 0
    automatic: bool = False
    completion: str | None = None
    pose_stable: int = 0
    position_best: float = float("inf")
    rotation_best: float = float("inf")
    position_progress_step: int = 0
    rotation_progress_step: int = 0
    position_in_tolerance: bool = False
    rotation_in_tolerance: bool = False
    gripper_target: float | None = None
    gripper_previous: float | None = None
    gripper_stable: int = 0
    gripper_at_target: int = 0
    gripper_status: str = "not_requested"
    gripper_best_error: float = float("inf")
    gripper_progress_step: int = 0

    @property
    def length(self) -> int:
        return (
            1
            if self.automatic
            else len(self.sequence)
            if self.sequence
            else len(self.waypoints)
        )


@dataclass
class _Pending:
    t: int
    action: Action
    observation: Observation
    kind: str


@dataclass
class _State:
    active: _Active | None = None
    pending: _Pending | None = None
    native_sequence: int = 0
    total_steps: int = 0
    stop_steps: int = 0
    stop_requested: bool = False
    finalized: bool = False


class FrameworkController:
    """The outer rollout alone steps physics.

    Automatic motions servo a frozen pose and observed gripper target. Motion
    chunks integrate and subdivide fixed waypoints before execution and track
    each substep for one control period; native chunks are untouched.
    """

    supports_motion_plans = True

    def __init__(self, config: ScheduleConfig | None = None, *, spec: ActionSpec):
        self.config = config or ScheduleConfig()
        if not isinstance(spec, ActionSpec):
            # No default: motion plans must be enforced against the same measured
            # single-step limits the policy validated them with.
            raise TypeError("FrameworkController requires the embodiment's explicit ActionSpec")
        self.spec = spec
        self._last_state: _State | None = None
        self._scale = vector(self.spec.pose_scale, 6, "pose_scale") * np.asarray(
            self.spec.pose_sign
        )
        if self.spec.frame not in ("world", "base") or self.spec.rotation not in (
            "axis_angle",
            "euler_xyz",
        ):
            raise ValueError("FrameworkController requires declared WORLD/base pose increments")

    def _arm_observation(self, obs: Observation, arm: str | None) -> Observation:
        """Project one arm's robot sensors into the single-arm feedback contract."""
        if not self.spec.arm_names:
            if arm is not None:
                raise ValueError("single-arm motions do not accept an arm selector")
            return obs
        if arm not in self.spec.arm_names:
            raise ValueError("motion requires a declared arm selector")
        prefix = arm + "_"
        state = {key[len(prefix):]: value for key, value in obs.state.items()
                 if key.startswith(prefix)}
        extra = {"env_step": obs.extra.get("env_step", 0)}
        for name in ("robot_base_quat", "gripper_target"):
            if prefix + name in obs.extra:
                extra[name] = obs.extra[prefix + name]
        return replace(obs, state=state, extra=extra)

    def _expand_action(self, data: np.ndarray, obs: Observation, arm: str | None) -> np.ndarray:
        if not self.spec.arm_names:
            return data
        result = self._neutral(obs)
        offset = 7 * self.spec.arm_names.index(arm)
        result[offset:offset + 7] = data
        return result

    def _held_gripper(self, obs: Observation) -> float:
        if self.spec.gripper_mode != "absolute":
            return self.spec.gripper_neutral
        # Robot/controller state only. Never invent a zero target: zero may open
        # an absolute-position gripper and release an object on Stop or omission.
        value = obs.extra.get("gripper_target")
        if value is None:
            value = obs.state.get("gripper_target")
        if value is None:
            raw = obs.extra.get("raw_droid", {})
            value = raw.get("gripper_target") if isinstance(raw, dict) else None
        if value is None:
            value = obs.state.get("gripper_pos")
        if value is None:
            raise ValueError("absolute gripper hold requires observed gripper target or position")
        return float(vector(np.asarray(value).reshape(-1), 1, "gripper hold target")[0])

    def _neutral(self, obs: Observation) -> np.ndarray:
        if self.spec.arm_names:
            return np.concatenate([
                np.r_[np.zeros(6), self._held_gripper(self._arm_observation(obs, arm))]
                for arm in self.spec.arm_names
            ])
        return np.r_[np.zeros(6), self._held_gripper(obs)]

    def _gripper(self, value: Any) -> float:
        # Internal compatibility only: the model-facing tools use numeric values.
        if isinstance(value, str):
            legacy = {
                "open": self.spec.gripper_low
                if self.spec.gripper_open is None
                else self.spec.gripper_open,
                "close": self.spec.gripper_high
                if self.spec.gripper_close is None
                else self.spec.gripper_close,
                "hold": self.spec.gripper_neutral,
            }
            if value not in legacy:
                raise ValueError("unknown gripper command")
            if value == "hold" and self.spec.gripper_mode == "absolute":
                raise ValueError("omit an absolute gripper command to hold its observed target")
            value = legacy[value]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
            raise ValueError("gripper command must be finite numeric data")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("gripper command must be finite numeric data")
        return value

    def _validate_command(self, command: MotionCommand) -> None:
        fields = {"dx", "dy", "dz", "rx", "ry", "rz", "gripper"}
        if not isinstance(command, MotionCommand) or command.name != "move_by":
            raise ValueError("motion commands must be move_by")
        if (self.spec.arm_names and command.arm not in self.spec.arm_names) or (
            not self.spec.arm_names and command.arm is not None
        ):
            raise ValueError("motion arm does not match the declared action contract")
        if not command.values or set(command.values) - fields:
            raise ValueError("invalid motion fields")
        for name, value in command.values.items():
            if name == "gripper":
                self._gripper(value)
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric data")
        validate_motion_pose(command.values, self.spec, self.config)
        if "gripper" in command.values:
            validate_gripper_bounds(self._gripper(command.values["gripper"]), self.spec)

    def _delta_quaternion(self, delta, rotation_frame):
        local = (
            euler_xyz_quaternion(delta)
            if self.spec.rotation == "euler_xyz"
            else axis_angle_quaternion(delta)
        )
        if self.spec.frame == "world":
            return local
        inverse = np.r_[-rotation_frame[:3], rotation_frame[3]]
        return quaternion_multiply(quaternion_multiply(rotation_frame, local), inverse)

    def _resolve(
        self, command: MotionCommand, obs: Observation, previous: _Resolved | None = None
    ) -> _Resolved:
        """Resolve a goal from the call's measured start or a prior planned waypoint."""
        obs = self._arm_observation(obs, command.arm)
        start_position, start_quaternion = pose(obs)
        values = command.values
        delta = np.array([values.get(k, 0.0) for k in ("dx", "dy", "dz", "rx", "ry", "rz")])
        if self.spec.frame == "base":
            if "robot_base_quat" not in obs.extra:
                raise ValueError(
                    "base-frame motion requires observed robot_base_quat in xyzw order"
                )
            rotation_frame = unit_quaternion(obs.extra["robot_base_quat"])
            world_delta = quaternion_rotate(rotation_frame, delta[:3])
        else:
            rotation_frame = np.array([0.0, 0.0, 0.0, 1.0])
            world_delta = delta[:3]
        reference_position = start_position if previous is None else previous.position
        reference_quaternion = start_quaternion if previous is None else previous.quaternion
        destination = reference_position + world_delta
        quaternion = quaternion_multiply(
            self._delta_quaternion(delta[3:], rotation_frame), reference_quaternion
        )
        vector(destination, 3, "resolved target")
        vector(delta / self._scale, 6, "normalized action")
        return _Resolved(
            destination,
            quaternion,
            start_position.copy(),
            start_quaternion.copy(),
            delta,
            self._gripper(values["gripper"]) if "gripper" in values else self._held_gripper(obs),
            "gripper" in values,
            self._held_gripper(obs),
            rotation_frame,
            command.arm,
        )

    def _gripper_position(self, obs: Observation) -> float:
        value = obs.state.get(self.spec.gripper_state_key)
        if value is None:
            raise ValueError(
                f"automatic gripper completion requires observed {self.spec.gripper_state_key}"
            )
        return float(vector(np.asarray(value).reshape(-1), 1, "observed gripper position")[0])

    def _prepare_auto(self, active: _Active, target: _Resolved, obs: Observation) -> None:
        obs = self._arm_observation(obs, target.arm)
        active.automatic = True
        active.target = target
        active.position_best = float(np.linalg.norm(target.position - target.start_position))
        active.rotation_best = float(
            np.linalg.norm(rotation_error(target.quaternion, target.start_quaternion))
        )
        active.position_in_tolerance = (
            active.position_best <= self.config.auto_motion.position_tolerance_m
        )
        active.rotation_in_tolerance = (
            active.rotation_best <= self.config.auto_motion.rotation_tolerance_rad
        )
        requested = target.explicit_gripper and (
            self.spec.gripper_mode == "absolute" or target.gripper != self.spec.gripper_neutral
        )
        if requested:
            if self.spec.gripper_mode == "absolute":
                if not self.spec.gripper_low <= target.gripper <= self.spec.gripper_high:
                    raise ValueError("automatic absolute gripper target is outside declared bounds")
                active.gripper_target = target.gripper
                if self.spec.gripper_measured_range is not None:
                    # Some native drivers command beyond a physical joint stop.
                    # Keep that command (and its holding force), but use the
                    # reachable robot measurement for completion feedback.
                    active.gripper_target = float(np.clip(
                        active.gripper_target, *self.spec.gripper_measured_range
                    ))
            else:
                active.gripper_target = (
                    self.spec.gripper_close_position
                    if target.gripper > self.spec.gripper_neutral
                    else self.spec.gripper_open_position
                )
            active.gripper_previous = self._gripper_position(obs)
            active.gripper_best_error = abs(active.gripper_target - active.gripper_previous)
            active.gripper_status = "pending"

    def _auto_errors(self, active: _Active, obs: Observation):
        obs = self._arm_observation(obs, active.target.arm)
        position, quaternion = pose(obs)
        position_error = active.target.position - position
        angular_error = rotation_error(active.target.quaternion, quaternion)
        return position_error, angular_error

    def _update_auto(self, active: _Active, obs: Observation, action: Action) -> None:
        if active.completion in ("stalled", "timeout", "gripper_stalled"):
            return
        active.completion = None
        config = self.config.auto_motion
        pos, rot = self._auto_errors(active, obs)
        distance, angle = float(np.linalg.norm(pos)), float(np.linalg.norm(rot))
        pose_ok = distance <= config.position_tolerance_m and angle <= config.rotation_tolerance_rad
        active.pose_stable = active.pose_stable + 1 if pose_ok else 0
        position_inside = distance <= config.position_tolerance_m
        rotation_inside = angle <= config.rotation_tolerance_rad
        # A pose component may start at zero error and be disturbed by motion in
        # another component. Start a fresh progress window when it leaves the
        # tolerance band, instead of comparing recovery to a historical zero.
        if (
            position_inside
            or active.position_in_tolerance
            or distance < active.position_best - config.position_progress_m
        ):
            active.position_best = distance
            active.position_progress_step = active.steps
        if (
            rotation_inside
            or active.rotation_in_tolerance
            or angle < active.rotation_best - config.rotation_progress_rad
        ):
            active.rotation_best = angle
            active.rotation_progress_step = active.steps
        active.position_in_tolerance = position_inside
        active.rotation_in_tolerance = rotation_inside
        if active.gripper_target is not None and (
            action.meta.get("auto_gripper_active") or active.gripper_status == "reached"
        ):
            measured = self._gripper_position(self._arm_observation(obs, active.target.arm))
            steady = (
                abs(measured - active.gripper_previous) <= self.spec.gripper_stability_tolerance
            )
            active.gripper_previous = measured
            active.gripper_stable = active.gripper_stable + 1 if steady else 0
            gripper_error = abs(measured - active.gripper_target)
            if gripper_error < active.gripper_best_error - self.spec.gripper_stability_tolerance:
                active.gripper_best_error = gripper_error
                active.gripper_progress_step = active.gripper_steps
            endpoint = gripper_error <= self.spec.gripper_position_tolerance
            active.gripper_at_target = active.gripper_at_target + 1 if endpoint else 0
            if not endpoint and active.gripper_status == "reached":
                active.gripper_status = "pending"
                active.gripper_stable = 0
                active.gripper_best_error = gripper_error
                active.gripper_progress_step = active.gripper_steps
            if active.gripper_steps >= config.gripper_min_steps:
                if (
                    active.gripper_at_target >= config.gripper_stable_steps
                    and active.gripper_stable >= config.gripper_stable_steps
                ):
                    active.gripper_status = "reached"
                elif (
                    not endpoint
                    and active.gripper_stable >= config.gripper_stable_steps
                    and active.gripper_steps - active.gripper_progress_step >= config.stall_steps
                ):
                    # Finger stability does not establish contact or a grasp.
                    active.gripper_status = "stalled"
                elif active.gripper_status == "stalled":
                    active.gripper_status = "pending"
        required_pose_samples = (
            1
            if not np.any(active.target.delta) and active.gripper_target is None
            else config.stable_steps
        )
        if active.pose_stable >= required_pose_samples:
            if active.gripper_target is None or active.gripper_status == "reached":
                active.completion = "reached"
            elif active.gripper_status == "stalled":
                active.completion = "gripper_stalled"
        if active.completion is None:
            if (
                distance > config.position_tolerance_m
                and active.steps - active.position_progress_step >= config.stall_steps
            ) or (
                angle > config.rotation_tolerance_rad
                and active.steps - active.rotation_progress_step >= config.stall_steps
            ):
                active.completion = "stalled"
            elif active.steps >= self.config.max_motion_steps:
                active.completion = "timeout"

    def _pose_action(self, target: _Resolved, obs: Observation) -> np.ndarray:
        """Convert the full WORLD pose error; the environment owns native processing."""
        obs = self._arm_observation(obs, target.arm)
        position, quaternion = pose(obs)
        pos_error = target.position - position
        rot_error = rotation_error(target.quaternion, quaternion)
        error_quaternion = axis_angle_quaternion(rot_error)
        if self.spec.frame == "base":
            if "robot_base_quat" not in obs.extra:
                raise ValueError(
                    "base-frame motion requires observed robot_base_quat in xyzw order"
                )
            frame = unit_quaternion(obs.extra["robot_base_quat"])
            inverse = np.r_[-frame[:3], frame[3]]
            pos_error = quaternion_rotate(inverse, pos_error)
            error_quaternion = quaternion_multiply(
                quaternion_multiply(inverse, error_quaternion), frame
            )
        rotation = (
            quaternion_to_euler_xyz(error_quaternion)
            if self.spec.rotation == "euler_xyz"
            else rotation_error(error_quaternion, [0, 0, 0, 1])
        )
        normalized = np.r_[pos_error, rotation] / self._scale
        return vector(normalized, 6, "pose tracking action")

    def _auto_action(self, active: _Active, obs: Observation) -> Action:
        if active.completion is not None:
            raise RuntimeError("completed automatic motion must replan before another action")
        normalized = self._pose_action(active.target, obs)
        full_observation = obs
        obs = self._arm_observation(obs, active.target.arm)
        drive_gripper = (
            active.gripper_target is not None
            and active.gripper_status != "reached"
        )
        # Once an absolute target arrives, keep commanding that same target.
        gripper = (
            active.target.gripper
            if drive_gripper
            or (self.spec.gripper_mode == "absolute" and active.gripper_status == "reached")
            else self._held_gripper(obs)
        )
        command = active.plan.commands[0]
        data = self._expand_action(np.r_[normalized, gripper], full_observation, command.arm)
        return Action(
            data,
            meta={
                "plan_id": active.plan_id,
                "command_index": 0,
                **({"arm": command.arm} if command.arm is not None else {}),
                "motion_phase": "joint" if active.gripper_target is not None else "arm",
                "tool": command.name,
                **({"note": command.note} if command.note is not None else {}),
                "action_source": "servo",
                "raw_action": data.tolist(),
                "requested_delta_pose": (normalized * self._scale).tolist(),
                "auto_gripper_active": drive_gripper,
                "target_status": "running",
                **(
                    {"gripper_hold": not drive_gripper}
                    if self.spec.gripper_mode == "absolute"
                    else {}
                ),
            },
        )

    def _prepare(self, reply: Any, obs: Observation, t: int, state: _State) -> _Active:
        if isinstance(reply, MotionPlan):
            plan, native = reply, ()
        elif isinstance(reply, ActionChunk):
            plan = unpack_plan(reply)
            native = () if plan is not None else tuple(reply.actions)
        else:
            raise ValueError("planning must return a MotionPlan or ActionChunk")
        if plan is None:
            if len(native) != self.config.h:
                raise ValueError(
                    f"native action chunk must contain exactly H={self.config.h} actions"
                )
            for action in native:
                vector(action.data, len(self.spec.input_low), "native action")
            return _Active(f"native-{state.native_sequence}", t, sequence=native)
        if not isinstance(plan.plan_id, str) or not plan.plan_id:
            raise ValueError("motion plan needs a nonempty plan_id")
        expected = 1 if self.config.control_interface == "move_by" else self.config.h
        if plan.horizon != self.config.h or len(plan.commands) != expected:
            raise ValueError(
                f"motion plan must use H={self.config.h} and contain {expected} commands"
            )
        # Freeze the complete path before any prefix is executed. Each arm has
        # its own chain, anchored to the pose and base frame observed at this call.
        waypoints = []
        previous_by_arm = {}
        for command in plan.commands:
            self._validate_command(command)
            previous = previous_by_arm.get(command.arm)
            with np.errstate(over="ignore", invalid="ignore"):
                resolved = self._resolve(command, obs, previous)
            if self.config.control_interface == "move_by_chunk":
                # Subdivide the original segment in the frozen call frame. In
                # particular, dividing Euler components would change its endpoint.
                start_position = previous.position if previous else resolved.start_position
                start_quaternion = previous.quaternion if previous else resolved.start_quaternion
                scale = self.config.motion_time_scale
                if scale > 1:
                    translation = resolved.position - start_position
                    rotation = rotation_error(resolved.quaternion, start_quaternion)
                    for substep in range(1, scale):
                        fraction = substep / scale
                        waypoints.append(replace(
                            resolved,
                            position=start_position + fraction * translation,
                            quaternion=quaternion_multiply(
                                axis_angle_quaternion(fraction * rotation), start_quaternion
                            ),
                        ))
            # Preserve the exact original endpoint, including at time scale 1.
            waypoints.append(resolved)
            previous_by_arm[command.arm] = resolved
        active = _Active(plan.plan_id, t, plan=plan, waypoints=tuple(waypoints))
        if self.config.control_interface == "move_by":
            self._prepare_auto(active, resolved, obs)
        return active

    def validate_motion_plan(self, plan: MotionPlan, observation: Observation, t: int) -> None:
        """Check complete compilation without accepting a plan or changing execution state."""
        self._prepare(plan, observation, t, _State())

    def _sync(
        self, policy: Any, obs: Observation, t: int, state: _State, *, reason="stopped"
    ) -> None:
        pending = state.pending
        if pending is None or t == pending.t:
            return
        if t != pending.t + 1:
            raise ValueError("rollout step count must advance exactly once per controller action")
        if pending.kind == "stop":
            state.stop_steps += 1
        else:
            if state.active is None:
                raise ValueError("executed motion has no active plan")
            active = state.active
            active.steps += 1
            if pending.kind == "holding":
                active.holding_steps += 1
            else:
                active.tracking_steps += 1
                if not active.automatic:
                    active.cursor += 1
            applied_gripper = (
                not pending.action.meta.get("gripper_hold", False)
                if self.spec.gripper_mode == "absolute"
                else pending.action.data[
                    7 * self.spec.arm_names.index(pending.action.meta["arm"]) + 6
                    if self.spec.arm_names and "arm" in pending.action.meta else 6
                ] != self.spec.gripper_neutral
            )
            if pending.kind != "holding" and applied_gripper:
                active.gripper_steps += 1
            if active.automatic:
                self._update_auto(active, obs, pending.action)
        state.total_steps += 1
        state.pending = None
        recorder = getattr(policy, "record_action", None)
        if recorder is not None:
            recorder(pending.t, pending.action, pending.observation)
        if pending.kind == "stop":
            outcome_recorder = getattr(policy, "record_stop_execution_result", None)
            if outcome_recorder is not None:
                outcome_recorder(obs, t, reason, 1, control_action=NEUTRAL_STOP_STEP)

    def _finish(self, policy, obs, t, reason, store, state, *, rollout_termination_reason=None):
        active = state.active
        if active is None:
            return None
        state.active = None
        extra = {
            "start_step": active.start_step,
            "end_step": t,
            "source": "native" if active.plan is None else "motion_plan",
            "compiled_motion_steps": active.length,
            "remaining_steps": max(0, active.length - active.cursor),
            "next_command_index": active.cursor,
            "remaining_commands": max(0, active.length - active.cursor),
            "remaining_control_steps": max(0, self.config.max_steps - t),
            "gripper_steps": active.gripper_steps,
            "gripper_applied": active.gripper_steps > 0,
        }
        measured_obs = (
            self._arm_observation(obs, active.target.arm) if active.target is not None else obs
        )
        if active.target is not None and active.target.arm is not None:
            extra["arm"] = active.target.arm
        if active.plan is not None and self.config.motion_time_scale > 1:
            extra["motion_time_scale"] = self.config.motion_time_scale
        if active.waypoints and not active.automatic:
            extra["chunk_target_reference"] = CHUNK_TARGET_REFERENCE
            if self.config.motion_time_scale > 1:
                command_index, substep_index = divmod(active.cursor, self.config.motion_time_scale)
                extra.update(
                    next_command_index=command_index,
                    next_substep_index=substep_index,
                    next_sequence_index=active.cursor,
                    remaining_commands=max(0, len(active.plan.commands) - command_index),
                )
        if rollout_termination_reason is not None:
            # Audit only: the policy receives {plan_id, status: accepted}, and recorded
            # execution results never reach a model.
            extra["rollout_termination_reason"] = rollout_termination_reason
        if active.automatic:
            extra.update(
                target_status=active.completion or "interrupted",
                gripper_status=active.gripper_status,
                gripper_target_position=active.gripper_target,
                motion_step_limit=self.config.max_motion_steps,
                remaining_steps=max(0, self.config.max_motion_steps - active.steps),
                remaining_commands=0 if active.completion else 1,
                compiled_motion_steps=None,
            )
            if active.gripper_target is not None:
                extra["gripper_measured_position"] = self._gripper_position(measured_obs)
                extra["gripper_error"] = active.gripper_target - extra["gripper_measured_position"]
        if "eef_pos" in measured_obs.state and "eef_quat" in measured_obs.state:
            position, quaternion = pose(measured_obs)
            extra.update(actual_position=position.tolist(), actual_quaternion=quaternion.tolist())
            if active.target is not None:
                target = active.target
                extra.update(
                    target_position=target.position.tolist(),
                    target_quaternion=target.quaternion.tolist(),
                    start_position=target.start_position.tolist(),
                    start_quaternion=target.start_quaternion.tolist(),
                    actual_displacement_m=(position - target.start_position).tolist(),
                    position_error_vector_m=(target.position - position).tolist(),
                    position_error_m=float(np.linalg.norm(target.position - position)),
                    orientation_error_rad=float(
                        np.linalg.norm(rotation_error(target.quaternion, quaternion))
                    ),
                )
        if "gripper_width" in measured_obs.state:
            extra["gripper_width_m"] = float(
                vector(measured_obs.state["gripper_width"], 1, "gripper_width")[0]
            )
        feedback = ExecutionFeedback(
            active.plan_id, reason, active.steps, active.tracking_steps, active.holding_steps, extra
        )
        store.setdefault("execution_feedback", []).append(asdict(feedback))
        recorder = getattr(policy, "record_execution", None)
        if recorder is not None:
            recorder(feedback)
        outcome_recorder = getattr(policy, "record_execution_result", None)
        if outcome_recorder is not None:
            outcome_recorder(feedback, obs, t)
        return feedback

    def _stop(self, stop: Stop, policy, obs, t, store, state) -> Action:
        if stop.name not in ("done", "give_up"):
            raise ValueError("unknown termination tool")
        self._finish(policy, obs, t, stop.name, store, state)
        state.stop_requested = True
        store["policy_stop"] = asdict(stop)
        neutral = self._neutral(obs)
        meta = {
            "request_stop": True,
            "stop_reason": stop.name,
            "stop_detail": stop.detail,
            "motion_phase": "stop",
            "action_source": "stop",
            "raw_action": neutral.tolist(),
        }
        if stop.hindsight.strip() and stop.hindsight.strip().lower() != "none":
            meta["stop_hindsight"] = stop.hindsight.strip()
        return Action(neutral, meta=meta)

    def _request(self, policy, obs, t, store, state):
        validate_before_acceptance = getattr(policy, "supports_motion_plan_validation", False)
        had_validator = hasattr(policy, "motion_plan_validator")
        previous_validator = getattr(policy, "motion_plan_validator", None)
        if validate_before_acceptance:

            def validate(plan):
                if previous_validator is not None:
                    previous_validator(plan)
                self.validate_motion_plan(plan, obs, t)

            policy.motion_plan_validator = validate
        try:
            try:
                reply = policy.act(obs)
            finally:
                if validate_before_acceptance:
                    if had_validator:
                        policy.motion_plan_validator = previous_validator
                    else:
                        del policy.motion_plan_validator
        except PolicyStop as exc:
            store.setdefault("_controller_inferences", []).append(
                (getattr(policy, "_last_inference_latency_s", None), 0)
            )
            if exc.name not in ("done", "give_up"):
                raise
            return self._stop(
                Stop(exc.name, exc.detail, exc.hindsight), policy, obs, t, store, state
            )
        latency = getattr(
            reply, "inference_latency_s", getattr(policy, "_last_inference_latency_s", None)
        )
        count = (
            len(reply)
            if isinstance(reply, ActionChunk)
            else len(reply.commands)
            if isinstance(reply, MotionPlan)
            else 0
        )
        store.setdefault("_controller_inferences", []).append((latency, count))
        if isinstance(reply, Stop):
            return self._stop(reply, policy, obs, t, store, state)
        prepared = self._prepare(reply, obs, t, state)
        state.active = prepared
        if prepared.plan is None:
            state.native_sequence += 1
        store.setdefault("controller_decisions", []).append(
            {
                "kind": "plan",
                "step": t,
                "result": "plan",
                "plan_id": state.active.plan_id,
            }
        )
        return None

    def _action(self, active, obs):
        if active.automatic:
            return self._auto_action(active, obs)
        if active.cursor >= active.length:
            raise RuntimeError("exhausted action sequence must replan before another action")
        if active.sequence:
            original = active.sequence[active.cursor]
            data = vector(original.data, len(self.spec.input_low), "compiled action").copy()
            if original.meta.get("gripper_hold") and active.plan is not None:
                data[6] = self._held_gripper(obs)
            return Action(
                data,
                meta={
                    **original.meta,
                    "plan_id": active.plan_id,
                    "sequence_index": active.cursor,
                    "motion_phase": original.meta.get("motion_phase", "native"),
                    "action_source": "native" if active.plan is None else "compiled",
                    "raw_action": data.tolist(),
                },
            )
        command_index, substep_index = divmod(active.cursor, self.config.motion_time_scale)
        command = active.plan.commands[command_index]
        target = active.waypoints[active.cursor]
        active.target = target
        normalized = self._pose_action(target, obs)
        gripper = (
            target.gripper if target.explicit_gripper
            else self._held_gripper(self._arm_observation(obs, command.arm))
        )
        data = np.r_[normalized, gripper]
        vector(data, 7, "compiled action")
        data = self._expand_action(data, obs, command.arm)
        return Action(
            data,
            meta={
                "plan_id": active.plan_id,
                "command_index": command_index,
                **(
                    {
                        "motion_time_scale": self.config.motion_time_scale,
                        "sequence_index": active.cursor,
                        "substep_index": substep_index,
                    }
                    if self.config.motion_time_scale > 1 else {}
                ),
                **({"arm": command.arm} if command.arm is not None else {}),
                "motion_phase": "single_step",
                "tool": "move_by_chunk",
                **({"note": command.note} if command.note is not None else {}),
                "action_source": "servo",
                "raw_action": data.tolist(),
                "requested_delta_pose": target.delta.tolist(),
                "tracking_delta_pose": (normalized * self._scale).tolist(),
                "target_position": target.position.tolist(),
                "target_quaternion": target.quaternion.tolist(),
                "chunk_target_reference": CHUNK_TARGET_REFERENCE,
                **(
                    {"gripper_hold": not target.explicit_gripper}
                    if self.spec.gripper_mode == "absolute"
                    else {}
                ),
            },
        )

    def record_applied(self, action: Action) -> None:
        state = self._last_state
        if state is None or state.pending is None:
            raise RuntimeError("no controller action is pending")
        state.pending.action = Action(
            vector(action.data, len(self.spec.input_low), "applied action").copy(),
            meta={**state.pending.action.meta, **action.meta},
        )

    def next_action(
        self, policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        if type(t) is not int or t < 0:
            raise ValueError("step must be a nonnegative integer")
        state = store.setdefault(_STATE, _State())
        self._last_state = state
        if state.finalized:
            raise RuntimeError("controller already finalized for this trial")
        self._sync(policy, observation, t, state)
        if t >= self.config.max_steps:
            raise PolicyStop("budget_exhausted", "No physical control steps remain", "none")
        if state.pending is not None:
            raise RuntimeError("previous controller action has not been executed")
        if state.stop_requested:
            raise RuntimeError("rollout must terminate after a request_stop action")
        extra = {**observation.extra, "env_step": t}
        obs = replace(observation, extra=extra)
        active = state.active
        action = None
        if active is not None:
            k = self.config.k
            if active.plan is not None and k > 0:
                k *= self.config.motion_time_scale
            # Completion and queue exhaustion end a plan independently of K.
            # A positive K is an upper bound, never a minimum execution duration.
            completion = (
                active.completion
                if active.automatic
                else "exhausted" if active.cursor >= active.length else None
            )
            if completion is not None or (k > 0 and active.steps >= k):
                self._finish(policy, obs, t, completion or "replan_boundary", store, state)
        if state.active is None and action is None:
            action = self._request(policy, obs, t, store, state)
        if action is None:
            action = self._action(state.active, obs)
        kind = (
            "stop"
            if action.meta.get("request_stop")
            else "holding"
            if action.meta.get("motion_phase") == "holding"
            else "motion"
        )
        state.pending = _Pending(t, action, obs, kind)
        return action

    def finalize(
        self, policy, observation: Observation, t: int, reason: str, store: dict[str, Any],
        *, stop_control_action: str = NEUTRAL_STOP_STEP,
    ):
        """End the trial; an unexecuted stop is recorded with zero executed steps.

        ``stop_control_action`` names what replaced an unexecuted stop step: the
        neutral step that never completed (approver rejection or a simulator
        fault), or ``done_verification`` when done verification follows a done call.
        """
        state = store.get(_STATE)
        if state is None or state.finalized:
            return None
        self._sync(policy, observation, t, state, reason=reason)
        if state.pending is not None and state.pending.kind == "stop":
            outcome_recorder = getattr(policy, "record_stop_execution_result", None)
            if outcome_recorder is not None:
                outcome_recorder(observation, t, reason, 0, control_action=stop_control_action)
        state.pending = None
        state.finalized = True
        # The plan active at the episode end was not ended by its own execution;
        # the rollout reason stays in audit-only extra fields.
        feedback = self._finish(
            policy, observation, t, EPISODE_ENDED, store, state,
            rollout_termination_reason=reason,
        )
        store["controller_totals"] = {
            "actual_steps": state.total_steps,
            "tracking_steps": sum(
                row["tracking_steps"] for row in store.get("execution_feedback", [])
            ),
            "holding_steps": sum(
                row["holding_steps"] for row in store.get("execution_feedback", [])
            ),
            "stop_steps": state.stop_steps,
        }
        return feedback
