"""Physical contracts for native model inference; no checkpoint defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyInfo
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Observation

from agentic_framework._vendor.openpi_client import image_tools
from agentic_framework.models.vla.common import DROID_COMMIT, _numeric, droid_images, droid_state


def _quat2axisangle(value: Any) -> np.ndarray:
    """Match examples/libero/main.py; input is the hand-body XYZW quaternion."""
    quat = np.asarray(value, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("robot0_eef_quat must have four finite XYZW components")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / den


def preprocess_observation(
    observation: Observation, *, image_size: int | None = 224
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return official 224 RGB views and float32 state8; never rotate twice.

    The embodiment already applies raw[::-1, ::-1]. Body quaternion and finger
    positions are mandatory raw LIBERO fields: substituting grip_site attitude
    or scalar jaw width would change the checkpoint's state representation.
    """
    views = []
    for name in ("agentview", "robot0_eye_in_hand"):
        if name not in observation.images:
            raise ValueError(f"OpenPI requires camera {name}")
        value = np.asarray(observation.images[name])
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError(f"{name} must be an HWC uint8 RGB image")
        views.append(
            np.ascontiguousarray(
                image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(value, image_size, image_size)
                )
                if image_size is not None
                else value.copy()
            )
        )
    raw = observation.extra.get("raw_libero")
    if not isinstance(raw, Mapping):
        raise ValueError("OpenPI requires observation.extra['raw_libero']")
    vectors = {}
    for key, shape in (("robot0_eef_pos", (3,)), ("robot0_gripper_qpos", (2,))):
        value = np.asarray(raw.get(key), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key} must have shape {shape} and finite values")
        vectors[key] = value
    rotation = _quat2axisangle(raw.get("robot0_eef_quat"))
    state = np.concatenate(
        (vectors["robot0_eef_pos"], rotation, vectors["robot0_gripper_qpos"])
    ).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("OpenPI state cannot be represented in float32")
    return views[0], views[1], state


class DroidJointAdapter:
    execution_kind = "droid"
    action_dim = 8
    requires_bind = True
    requires_droid_controller = False
    raw_action_semantics = "absolute_joint_position_7_and_native_knuckle_position_1"

    @staticmethod
    def initial_info(name, control_hz):
        return PolicyInfo(
            name,
            Box(
                shape=(8,),
                low=-np.ones(8),
                high=np.ones(8),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=tuple(f"joint_{i}" for i in range(7)) + ("gripper",),
                ),
            ),
            ObservationSpace(),
            control_hz,
        )

    @staticmethod
    def bind(self, embodiment_info):
        space, sem = (embodiment_info.action_space, embodiment_info.action_space.semantics)
        if space.shape != (8,) or sem is None or sem.control_mode != "joint_pos":
            raise ValueError("DROID native execution requires eight-dimensional pd_joint_pos")
        low = _numeric(space.low, (8,), "joint position lower bounds")
        high = _numeric(space.high, (8,), "joint position upper bounds")
        if not np.all(low < high):
            raise ValueError("native joint position bounds must be ordered")
        if not math.isclose(float(embodiment_info.control_hz), self.control_hz):
            raise ValueError("DROID policy and embodiment control_hz disagree")
        self.gripper_low, self.gripper_high = (float(low[-1]), float(high[-1]))
        self.info = PolicyInfo(
            self.profile, space, embodiment_info.observation_space, self.control_hz
        )
        self._bound = True
        self._metadata["native_gripper_bounds"] = [self.gripper_low, self.gripper_high]

    @staticmethod
    def sim_time(observation, step, control_hz):
        return step / control_hz

    @staticmethod
    def metadata(model_profile):
        return {
            "executed_control_mode": "pd_joint_pos",
            "comparison_note": "native cadence differs between models; not a matched-control comparison",
        }

    @staticmethod
    def native_action(policy, raw, observation):
        return raw.copy(), {}


class OpenPiDroidAdapter(DroidJointAdapter):
    raw_action_semantics = "droid_normalized_joint_velocity_7_and_normalized_closed_fraction_1"
    requires_droid_controller = True

    @staticmethod
    def request(self, observation, instruction):
        external, wrist = droid_images(observation, resize=True)
        state = droid_state(observation)
        grip = (state[7] - self.gripper_low) / (self.gripper_high - self.gripper_low)
        return {
            "observation/exterior_image_1_left": external,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": state[:7].copy(),
            "observation/gripper_position": np.array([grip], dtype=np.float32),
            "prompt": instruction,
        }

    @staticmethod
    def request_state(self, request):
        return np.concatenate(
            (request["observation/joint_position"], request["observation/gripper_position"])
        )

    @staticmethod
    def native_action(self, raw, observation):
        clipped = np.clip(raw[:7], -1.0, 1.0)
        bounded = clipped / max(1.0, float(np.max(np.abs(clipped))))
        delta = 0.2 * bounded
        q = droid_state(observation)[:7].astype(np.float64)
        grip = self.gripper_high if raw[7] > 0.5 else self.gripper_low
        return (
            np.concatenate((q + delta, [grip])),
            {
                "droid_measured_joint_position": q.tolist(),
                "droid_clipped_velocity": bounded.tolist(),
                "droid_joint_delta": delta.tolist(),
                "droid_gripper_closed_fraction": float(raw[7] > 0.5),
            },
        )

    @staticmethod
    def metadata(model_profile):
        return {
            **DroidJointAdapter.metadata(model_profile),
            "sim_mapping": {
                "source_commit": DROID_COMMIT,
                "joint_delta_per_normalized_unit": 0.2,
                "joint_target_reference": "current measured q at each executed action",
                "gripper_state": "linear native knuckle closed fraction; explicit sim approximation to real DROID width fraction",
                "gripper_action": "official >0.5 binary closed fraction, mapped to runtime native knuckle endpoints",
                "cadence": "0.2 rad per normalized action, unchanged at custom Hz; custom cadence is not official real-DROID timing",
            },
        }


class OpenPiLiberoAdapter:
    execution_kind = "libero"
    action_dim = 7
    requires_bind = False
    requires_droid_controller = False
    raw_action_semantics = "native normalized OSC; no bridge clipping or gripper rewriting"

    @staticmethod
    def initial_info(name, control_hz):
        return PolicyInfo(
            name,
            Box(
                shape=(7,),
                low=-np.ones(7),
                high=np.ones(7),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pose",
                    rotation_repr="axis_angle",
                    gripper="continuous",
                    frame="world",
                    dim_labels=("dx", "dy", "dz", "rx", "ry", "rz", "gripper"),
                ),
            ),
            ObservationSpace(),
            control_hz,
        )

    @staticmethod
    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        box = embodiment_info.action_space
        sem = box.semantics
        if (
            box.shape != (7,)
            or sem is None
            or sem.control_mode != "eef_delta_pose"
            or (sem.frame != "world")
            or (sem.rotation_repr != "axis_angle")
            or (box.low is None)
            or (box.high is None)
            or (not np.array_equal(box.low, -np.ones(7)))
            or (not np.array_equal(box.high, np.ones(7)))
        ):
            raise ValueError(
                "OpenPI requires the native normalized seven-dimensional LIBERO OSC space"
            )
        if not math.isclose(float(embodiment_info.control_hz), self.control_hz):
            raise ValueError("LIBERO policy and embodiment control_hz disagree")
        self.info = PolicyInfo(
            self.info.name, box, embodiment_info.observation_space, self.control_hz
        )

    @staticmethod
    def request(policy, observation, instruction):
        image, wrist, state = preprocess_observation(observation)
        return {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": instruction,
        }

    @staticmethod
    def request_state(policy, request):
        return request["observation/state"]

    @staticmethod
    def sim_time(observation, step, control_hz):
        return float(observation.state_time)

    @staticmethod
    def metadata(model_profile):
        return {
            "preprocessing_commit": model_profile["model"]["expected_metadata"]["openpi_commit"],
            "observation_profile": model_profile["environment"]["observation_profile"],
            "action_representation": OpenPiLiberoAdapter.raw_action_semantics,
            "executed_control_mode": "eef_delta_pose",
        }

    @staticmethod
    def native_action(policy, raw, observation):
        return raw.copy(), {}


class MolmoAct2LiberoAdapter(OpenPiLiberoAdapter):
    """Official LIBERO state8 and RGB views; processor owns image resizing."""

    @staticmethod
    def request(policy, observation, instruction):
        image, wrist, state = preprocess_observation(observation, image_size=None)
        # Pinned LeRobot LIBERO processor computes rotation in float32.
        quat = np.asarray(observation.extra["raw_libero"]["robot0_eef_quat"], dtype=np.float32)
        w = np.clip(quat[3], np.float32(-1), np.float32(1))
        den = np.sqrt(np.maximum(np.float32(1) - w * w, np.float32(0)))
        state[3:6] = (quat[:3] / den) * (np.float32(2) * np.arccos(w)) if den > 1e-10 else 0
        return {
            "external_cam": image,
            "wrist_cam": wrist,
            "state": state,
            "instruction": instruction,
        }

    @staticmethod
    def request_state(policy, request):
        return request["state"]

    @staticmethod
    def metadata(model_profile):
        return {
            "preprocessing_commit": model_profile["model"]["expected_metadata"]["molmoact2_commit"],
            "observation_profile": model_profile["environment"]["observation_profile"],
            "action_representation": OpenPiLiberoAdapter.raw_action_semantics,
            "executed_control_mode": "eef_delta_pose",
            "image_preprocessing": "environment rotates once; original resolution to official processor",
        }
