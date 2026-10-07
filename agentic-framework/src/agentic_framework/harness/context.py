"""Current observations and fixed demonstrations; backends own live history."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from inspect_robots.types import Observation

from agentic_framework.environments.privilege import (
    LEGACY_PRIVILEGE_KEYS,
    PRIVILEGE_KEY,
    validate_privilege,
)
from agentic_framework.harness.demonstration import (
    DEFAULT_DEMO_CONTENT,
    DEFAULT_DEMO_IMAGE_MAX_SIDE,
    DEFAULT_DEMO_MODE,
    Demonstration,
    demonstration_image_names,
    validate_demo_content,
    validate_demo_mode,
    validate_step_budget,
)
from agentic_framework.harness.types import ActionSpec, AugmentationConfig

_STATE_KEYS = frozenset(
    {"eef_pos", "eef_quat", "joint_pos", "gripper_width", "gripper_pos", "gripper_target"}
)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported context value: {type(value).__name__}")


def _step(observation: Observation) -> int:
    value = observation.extra.get("env_step", 0)
    if type(value) is not int or value < 0:
        raise ValueError("observation env_step must be a nonnegative integer")
    return value


def _feedback(value: Any) -> dict:
    value = asdict(value) if is_dataclass(value) else value
    if not isinstance(value, Mapping):
        raise TypeError("execution feedback must be a mapping or dataclass")
    # Keep scheduling, completion diagnostics and counts in internal audit logs.
    # The model infers movement from successive observations, not executor summaries.
    result = {"plan_id": _plain(value.get("plan_id")), "status": "accepted"}
    json.dumps(result, allow_nan=False)
    return result


def _rounded_text(value: Any) -> Any:
    """Compact model-facing numbers without changing owned observations or receipts."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_text(item) for item in value]
    return value


class ContextBuilder:
    """Encode a fresh observation with an optional complete fixed demonstration."""

    def __init__(
        self,
        augmentation=None,
        spec=None,
        *,
        observation_profile="control",
        cameras=(),
        max_images=16,
        max_context_chars=64000,
        demo=False,
        demo_path=None,
        demo_mode=DEFAULT_DEMO_MODE,
        demo_content=DEFAULT_DEMO_CONTENT,
        demo_image_max_side=DEFAULT_DEMO_IMAGE_MAX_SIDE,
    ):
        self.augmentation = augmentation or AugmentationConfig()
        if not isinstance(self.augmentation, AugmentationConfig):
            raise TypeError("augmentation must be AugmentationConfig")
        self.memory = self.augmentation.memory
        self.spec = spec or ActionSpec()
        if observation_profile not in ("control", "openpi_matched", "privileged"):
            raise ValueError("unknown observation profile")
        if type(demo) is not bool:
            raise TypeError("demo must be a boolean")
        if demo and demo_path is None:
            raise ValueError("demo=true requires demo_path")
        if type(demo_image_max_side) is not int or demo_image_max_side < 0:
            raise ValueError("demo_image_max_side must be a nonnegative integer (0 keeps original resolution)")
        if type(max_images) is not int or max_images < 1:
            raise ValueError("max_images must be a positive integer")
        if type(max_context_chars) is not int or max_context_chars < 1:
            raise ValueError("max_context_chars must be a positive integer")
        if not isinstance(cameras, (tuple, list)) or any(not isinstance(c, str) for c in cameras):
            raise ValueError("cameras must be a sequence of camera names")
        if len(set(cameras)) != len(cameras):
            raise ValueError("camera names must be unique")
        self.observation_profile = observation_profile
        self.cameras = tuple(cameras)
        self.max_images, self.max_context_chars = max_images, max_context_chars
        self.demo = demo
        self.demo_path = demo_path
        self.demo_mode = validate_demo_mode(demo_mode)
        self.demo_content = validate_demo_content(demo_content)
        self.demo_image_max_side = demo_image_max_side
        # A disabled demo never reads its artifact, keeping the ablation independent
        # of whether the demonstration file exists on the evaluation machine. The
        # Full transcripts require the same profile; observation sequences strip
        # privileged channels and do not depend on the teacher action interface.
        self.demonstration = (
            Demonstration(
                demo_path, observation_profile=observation_profile,
                image_max_side=demo_image_max_side,
                demo_mode=self.demo_mode,
                demo_content=self.demo_content,
            )
            if demo else None
        )
        self.reset()

    def reset(self):
        self.last_audit = {}

    def _snapshot(self, observation):
        if self.observation_profile == "openpi_matched":
            from agentic_framework.models.vla.openpi import preprocess_observation

            image, wrist, state = preprocess_observation(observation)
            images = {"agentview": image, "robot0_eye_in_hand": wrist}
            fields = {"openpi_state": _plain(state)}
        else:
            images = dict(observation.images)
            allowed = (
                {f"{arm}_{key}" for arm in self.spec.arm_names for key in _STATE_KEYS}
                if self.spec.arm_names else _STATE_KEYS
            )
            fields = {
                key: _plain(value) for key, value in observation.state.items() if key in allowed
            }
        selected = self.cameras or tuple(images)
        missing = [name for name in selected if name not in images]
        if missing:
            raise ValueError(f"Requested cameras unavailable: {', '.join(missing)}")
        frames = {}
        for name in selected:
            frame = np.asarray(images[name])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Camera {name!r} must be an HWC uint8 RGB image")
            frames[name] = frame.copy()
        # Validate the internal clock; only the derived step budget reaches the model.
        _step(observation)
        snapshot = {"state": fields}
        if self.observation_profile == "privileged":
            if any(key in observation.extra for key in LEGACY_PRIVILEGE_KEYS):
                raise ValueError("Legacy live privilege channels are unsupported; use privileged schema v2")
            if PRIVILEGE_KEY not in observation.extra:
                raise ValueError("Privileged observations require the privileged schema v2 sensor")
            snapshot[PRIVILEGE_KEY] = validate_privilege(observation.extra[PRIVILEGE_KEY])
        json.dumps(snapshot, allow_nan=False)
        return snapshot, frames

    def build(self, observation, status=None, *, step_budget=None):
        """Encode current input only; status is supplied separately as a native tool result.

        ``step_budget`` ({steps_remaining, max_steps}) is the policy's remaining
        physical step budget; every model request carries it in current_observation.
        """
        current, current_images = self._snapshot(observation)
        if len(current_images) > self.max_images:
            raise ValueError("max_images is smaller than the current selected camera count")
        if step_budget is not None:
            current["step_budget"] = validate_step_budget(step_budget)
        payload = {
            "instruction": observation.instruction or "",
            "current_observation": current,
        }
        if self.observation_profile == "openpi_matched":
            payload["observation_spec"] = {
                "field": "current_observation.state.openpi_state",
                "dim_labels": [
                    "eef_x",
                    "eef_y",
                    "eef_z",
                    "hand_body_rx",
                    "hand_body_ry",
                    "hand_body_rz",
                    "gripper_joint_0",
                    "gripper_joint_1",
                ],
                "units": ["m", "m", "m", "rad", "rad", "rad", "m", "m"],
                "frame": "world",
                "position": "raw robot0_eef_pos",
                "orientation": "absolute hand-body WORLD axis-angle from robot0_eef_quat; not grip-site, not Euler, not an action delta",
                "gripper": "two signed robot0_gripper_qpos joint positions; not jaw width or commands",
            }
        images = {f"current/{name}": value for name, value in current_images.items()}
        current["images"] = list(images)
        public = (
            _rounded_text(payload) if self.observation_profile != "openpi_matched" else payload
        )
        if self.demonstration is not None:
            demonstration, demo_images = self.demonstration.snapshot()
            # Preserve every teacher request and frame, with bounded image
            # resolution, independently of live history.
            public = {"demonstration": demonstration, **public}
            images = {**demo_images, **images}
            if len(images) > self.max_images:
                raise ValueError(
                    f"Complete demonstration plus current cameras requires {len(images)} images "
                    f"but max_images={self.max_images}; increase the budget, no frames were removed"
                )
        text = json.dumps(public, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(text) > self.max_context_chars:
            if self.demonstration is not None:
                raise ValueError(
                    f"Complete demonstration plus current observation requires {len(text)} "
                    f"characters but max_context_chars={self.max_context_chars}; "
                    "increase the budget, no requests were removed"
                )
            raise ValueError("Current observation/instruction exceeds max_context_chars")
        self.last_audit = {
            "format": "native_current_observation_v1",
            "output_format": "native_tool_calls",
            "history_owner": "backend",
            "configured_history_length": self.memory.history_length,
            "image_count": len(images),
            "text_chars": len(text),
        }
        if self.demonstration is not None:
            self.last_audit.update(
                demo=True,
                demo_mode=self.demo_mode,
                demo_content=self.demo_content,
                demo_sha256=self.demonstration.sha256,
                demo_requests=self.demonstration.request_count,
                demo_images=self.demonstration.image_count,
                demo_image_max_side=self.demo_image_max_side,
                demo_resized_images=self.demonstration.resized_image_count,
            )
        return text, images


def context_messages(text: str, images: Mapping[str, np.ndarray]) -> tuple[dict, ...]:
    """Build multimodal input with chronological demo frames before current frames."""
    payload = json.loads(text)
    names = payload["current_observation"].get("images", [])
    demonstration = payload.get("demonstration")
    if demonstration is not None:
        names = demonstration_image_names(demonstration) + names
    if len(names) != len(set(names)) or set(names) != set(images):
        raise ValueError("Every current image must appear exactly once in the observation")
    return (
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                *({"type": "image", "name": name} for name in names),
            ],
        },
    )
