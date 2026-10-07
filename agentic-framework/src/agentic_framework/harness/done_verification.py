"""Hold phase after a model's done call; the outer rollout remains the only stepper.

A model's ``done`` is not success. When done verification is enabled, the done
call starts a fixed number of ordinary, recorded control steps without model
calls. The environment keeps judging success after every step: success inside
the hold counts, and a hold that runs out without success is a task failure.
Only a model done call starts it: give_up, reaching max_steps, exhausting
max_attempts and errors end the episode exactly as without verification.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from inspect_robots.errors import PolicyStop
from inspect_robots.types import Action, Observation

from agentic_framework.common.geometry import (
    pose,
    quaternion_multiply,
    quaternion_rotate,
    quaternion_to_euler_xyz,
    rotation_error,
    unit_quaternion,
    vector,
)
from agentic_framework.harness.types import ActionSpec

POLICY_PHASE = "policy"
VERIFICATION_PHASE = "verification"
ACTION_SOURCE = "done_verification"
# Recorded as the done call's control action when verification replaces the neutral stop step.
CONTROL_ACTION = "done_verification"
TIMEOUT_TERMINATION = "done_verification_timeout"
MAX_STEPS_TERMINATION = "max_steps"
BUDGET_STOP_DETAIL = "Policy control budget (max_steps) exhausted; budget stop, not a model call"


def verification_steps(seconds: float, control_hz: float) -> int:
    """Return ceil(seconds * control_hz) hold steps; 0 disables verification.

    The product is rounded first so that, e.g., 0.1 s at 30 Hz is 3 steps, not 4.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("done_verification_seconds must be a finite nonnegative number")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("done_verification_seconds must be a finite nonnegative number")
    if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)):
        raise ValueError("control_hz must be positive and finite")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    return math.ceil(round(seconds * control_hz, 9))


def driven_gripper_target(observation: Observation) -> float:
    """Return the gripper controller's driven target, never the measured opening.

    Holding the measured finger angle collapses the grasp force on a held object.
    """
    value = observation.extra.get("gripper_target")
    if value is None:
        raw = observation.extra.get("raw_droid")
        value = raw.get("gripper_target") if isinstance(raw, Mapping) else None
    if value is None:
        raise ValueError(
            "done verification requires the gripper controller's driven target in "
            "observation.extra['gripper_target'] or observation.extra['raw_droid']"
            "['gripper_target']; the measured gripper opening is never held"
        )
    return float(vector(np.asarray(value, dtype=float).reshape(-1), 1, "driven gripper target")[0])


def episode_phases(record: Any, *, oracle_success: bool | None) -> dict[str, Any]:
    """Policy/verification step accounting and success phase of one rollout record."""
    totals = record.metadata.get("controller_totals", {})
    executed = record.metadata.get(
        "executed_control_steps", totals.get("actual_steps", len(record.steps))
    )
    hold = totals.get(
        "verification_control_steps",
        sum(step.action.meta.get("evaluation_phase") == VERIFICATION_PHASE for step in record.steps),
    )
    verification = totals.get("done_verification") or {}
    success_phase = None
    if oracle_success:
        last = record.steps[-1].action.meta if record.steps else {}
        success_phase = (
            VERIFICATION_PHASE if last.get("evaluation_phase") == VERIFICATION_PHASE
            else POLICY_PHASE
        )
    return {
        "policy_control_steps": executed - hold,
        "verification_control_steps": hold,
        "verification_triggered": bool(verification.get("triggered", False)),
        "success_phase": success_phase,
    }


class DoneVerificationController:
    """Wrap the LLM FrameworkController with a measured-pose hold after done.

    Policy-phase actions are capped at ``action_budget`` (the trial's max_steps);
    reaching it without success ends the episode as ``max_steps``. A model done
    call ends the policy phase without its neutral stop step and starts exactly
    ``steps`` hold actions. Each hold is an ordinary outer rollout action with
    the same mode, bounds, contacts and success oracle as the policy; no model is
    called. The arm servos back to the end-effector pose measured at the done
    call, within the embodiment's single-step and input limits; the gripper
    keeps the controller's driven target. Success on any hold step ends the
    episode as success; otherwise the last hold requests the stop
    ``done_verification_timeout``, a task failure.
    """

    def __init__(self, inner, *, action_budget: int, steps: int, control_hz: float,
                 seconds: float, spec: ActionSpec):
        if type(action_budget) is not int or action_budget < 1:
            raise ValueError("action_budget must be a positive integer")
        if type(steps) is not int or steps < 1:
            raise ValueError("done verification requires a positive number of hold steps")
        if not isinstance(spec, ActionSpec):
            raise TypeError("done verification requires the embodiment's explicit ActionSpec")
        if spec.gripper_mode != "absolute" or spec.arm_names:
            raise ValueError("done verification requires a single-arm absolute-gripper embodiment")
        if spec.frame not in ("world", "base") or spec.rotation not in ("axis_angle", "euler_xyz"):
            raise ValueError("done verification requires declared WORLD/base pose increments")
        self.inner = inner
        self.action_budget = action_budget
        self.max_hold_steps = steps
        self.control_hz = float(control_hz)
        self.seconds = float(seconds)
        self.spec = spec
        self.supports_motion_plans = getattr(inner, "supports_motion_plans", False)
        self.start_step: int | None = None
        self.hold_steps = 0
        self.policy_steps = 0
        self.hold_target: dict[str, Any] | None = None
        self._pending: str | None = None
        self._inner_finalized = False
        self._finalized = False

    @property
    def triggered(self) -> bool:
        return self.start_step is not None

    def _begin(self, policy, observation: Observation, t: int, store: dict[str, Any]) -> None:
        position, quaternion = pose(observation)
        hold_target = {
            "position": position.tolist(),
            "quaternion_xyzw": quaternion.tolist(),
            "gripper_target": driven_gripper_target(observation),
        }
        # The done call's neutral stop step is replaced by verification; its
        # receipt records that no step executed and that verification followed.
        self.inner.finalize(policy, observation, t, "done", store,
                            stop_control_action=CONTROL_ACTION)
        self._inner_finalized = True
        self.hold_target = hold_target
        self.start_step = t

    def _hold(self, observation: Observation) -> np.ndarray:
        position, quaternion = pose(observation)
        target = np.asarray(self.hold_target["quaternion_xyzw"])
        delta_position = np.asarray(self.hold_target["position"]) - position
        error = quaternion_multiply(target, np.r_[-quaternion[:3], quaternion[3]])
        if self.spec.frame == "base":
            if "robot_base_quat" not in observation.extra:
                raise ValueError("base-frame hold requires observed robot_base_quat in xyzw order")
            frame = unit_quaternion(observation.extra["robot_base_quat"])
            inverse = np.r_[-frame[:3], frame[3]]
            delta_position = quaternion_rotate(inverse, delta_position)
            error = quaternion_multiply(quaternion_multiply(inverse, error), frame)
        rotation = (quaternion_to_euler_xyz(error) if self.spec.rotation == "euler_xyz"
                    else rotation_error(error, [0, 0, 0, 1]))
        scale = np.asarray(self.spec.pose_scale) * np.asarray(self.spec.pose_sign)
        normalized = np.r_[delta_position, rotation] / scale
        # The same physical single-step servo limits as the policy; never a new task motion.
        limit = np.asarray(self.spec.single_step_pose_limit) / np.abs(scale)
        normalized = np.clip(normalized, -limit, limit)
        normalized = np.clip(normalized, self.spec.input_low[:6], self.spec.input_high[:6])
        if self.spec.rotation_norm_limit is not None:
            norm = np.linalg.norm(normalized[3:])
            if norm > self.spec.rotation_norm_limit:
                normalized[3:] *= self.spec.rotation_norm_limit / norm
        return np.r_[normalized, self.hold_target["gripper_target"]]

    def next_action(self, policy, observation: Observation, t: int,
                    store: dict[str, Any]) -> Action:
        if self._finalized or self._pending is not None:
            raise RuntimeError("done verification has an unexecuted action or was finalized")
        if not self.triggered:
            if t >= self.action_budget:
                # Policy actions stay capped at max_steps; the rollout limit also
                # covers a hold that starts on the last budgeted step. The rollout
                # records this as metadata.policy_stop, but it is a budget stop:
                # the model called neither done nor give_up.
                raise PolicyStop(MAX_STEPS_TERMINATION, BUDGET_STOP_DETAIL, "none")
            try:
                action = self.inner.next_action(policy, observation, t, store)
            except PolicyStop as stop:
                if stop.name != "done":
                    raise
                self._begin(policy, observation, t, store)
            else:
                if action.meta.get("request_stop") and action.meta.get("stop_reason") == "done":
                    self._begin(policy, observation, t, store)
                else:
                    self._pending = POLICY_PHASE
                    return Action(action.data, {**action.meta, "evaluation_phase": POLICY_PHASE})
        if self.hold_steps >= self.max_hold_steps:
            raise PolicyStop(TIMEOUT_TERMINATION, "done verification hold completed", "none")
        data = self._hold(observation)
        index = self.hold_steps + 1
        self._pending = VERIFICATION_PHASE
        return Action(data, {
            "evaluation_phase": VERIFICATION_PHASE,
            "action_source": ACTION_SOURCE,
            "motion_phase": VERIFICATION_PHASE,
            "verification_step": index,
            "verification_steps": self.max_hold_steps,
            "raw_action": data.tolist(),
            **({"request_stop": True, "stop_reason": TIMEOUT_TERMINATION}
               if index >= self.max_hold_steps else {}),
        })

    def record_applied(self, action: Action) -> None:
        if self._pending is None:
            raise RuntimeError("no done verification action is pending")
        phase, self._pending = self._pending, None
        if phase == VERIFICATION_PHASE:
            # A hold is not a model decision: it is never attributed to a model turn.
            self.hold_steps += 1
            return
        self.policy_steps += 1
        record = getattr(self.inner, "record_applied", None)
        if callable(record):
            record(action)

    def finalize(self, policy, observation: Observation, t: int, reason: str,
                 store: dict[str, Any]):
        if self._finalized:
            return None
        self._finalized = True
        feedback = None
        if not self._inner_finalized:
            feedback = self.inner.finalize(policy, observation, t, reason, store)
        self._pending = None
        totals = store.setdefault("controller_totals", {})
        totals.update(
            actual_steps=self.policy_steps + self.hold_steps,
            policy_control_steps=self.policy_steps,
            verification_control_steps=self.hold_steps,
            done_verification={
                "enabled": True,
                "seconds": self.seconds,
                "max_steps": self.max_hold_steps,
                "triggered": self.triggered,
                "trigger": "done" if self.triggered else None,
                "start_step": self.start_step,
                "steps": self.hold_steps,
                "hold_target": self.hold_target,
                "outcome": reason if self.triggered else None,
                "new_model_calls": 0,
            },
        )
        return feedback
