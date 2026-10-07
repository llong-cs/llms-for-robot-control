"""Isolated official MolmoAct2 / ManiSkill DROID embodiment.

The main environment never imports ManiSkill, torch, or SAPIEN. The existing
LIBERO RPC transport supplies request attribution, deadlines and cleanup.
"""
from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

from agentic_framework.common.project import environment_python, project_path
from agentic_framework.environments.libero.embodiment import LiberoEmbodiment, SimulatorError
from agentic_framework.environments.privilege import (
    LEGACY_PRIVILEGE_KEYS,
    PRIVILEGE_KEY,
    validate_privilege,
)

OFFICIAL_COMMIT = "66b87e64efd99dfd103241418113955cf64dfa9c"
ENV_ID = "DroidPutEverythingInBox-v1"
CAMERAS = ("external_cam", "wrist_cam")
MODES = {
    "pd_ee_delta_pose": (7, "eef_delta_pose", 30),
    "pd_joint_pos": (8, "joint_pos", 30),
    "pd_joint_vel": (8, "joint_vel", 15),
}
MODE_ALIASES = {
    "ee_delta_pose": "pd_ee_delta_pose",
    "joint_position": "pd_joint_pos",
    "joint_velocity": "pd_joint_vel",
}


def canonical_mode(mode: str) -> str:
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"Unsupported DROID control mode: {mode!r}")
    return mode


def validate_action(action: Any, control_mode: str) -> np.ndarray:
    """Check shape/finiteness only; native controllers own their transformations."""
    dimension = MODES[canonical_mode(control_mode)][0]
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise ValueError(f"{control_mode} action must contain {dimension} finite numbers")
    return value.copy()


class ManiSkillEmbodiment(LiberoEmbodiment):
    """One official DROID simulator in an independent interpreter.

    A scene selects ``env_id`` (currently only DroidPutEverythingInBox-v1) and
    optional ``instruction``. Initial conditions come from the explicit reset
    seed; no LIBERO init-state index, settling actions, or image flips apply.
    ``action_spec`` is available after metadata() or reset, once native controllers have been
    inspected. Native joint policies bind to the resulting eight-dimensional
    action space without passing through the motion-tool layer.
    """
    _worker_label = "ManiSkill"

    def __init__(
        self,
        *,
        control_mode: str = "pd_ee_delta_pose",
        observation_profile: str = "control",
        control_hz: float | None = None,
        sim_hz: int = 150,
        max_episode_steps: int = 2000,
        shader_pack: str = "rt-fast",
        sim_backend: str = "cpu",
        source_root: str | Path | None = None,
        assets_root: str | Path | None = None,
        python: str | Path | None = None,
        worker_env: Mapping[str, str] | None = None,
        request_timeout_s: float = 600,
    ) -> None:
        if observation_profile not in ("control", "privileged"):
            raise ValueError("unknown ManiSkill observation profile")
        self.control_mode = canonical_mode(control_mode)
        dimension, semantics, default_hz = MODES[self.control_mode]
        self.control_hz = float(default_hz if control_hz is None else control_hz)
        if (
            not np.isfinite(self.control_hz) or self.control_hz <= 0
            or not self.control_hz.is_integer()
            or type(sim_hz) is not int or sim_hz <= 0
            or sim_hz % int(self.control_hz) != 0
        ):
            raise ValueError("sim_hz must be a positive integer multiple of control_hz")
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if sim_backend not in ("cpu", "gpu"):
            raise ValueError("sim_backend must be cpu or gpu")
        super().__init__(
            python=python or environment_python("agentic-framework-maniskill"),
            worker_env=worker_env, request_timeout_s=request_timeout_s,
            observation_profile=observation_profile,
        )
        self.benchmark = "maniskill"
        self.source_root = project_path(source_root or "third_party/molmoact2")
        self.assets_root = project_path(assets_root or "data/molmoact2-sim-eval-assets")
        self.sim_hz = sim_hz
        self.max_episode_steps = max_episode_steps
        self.shader_pack = shader_pack
        self.sim_backend = sim_backend
        self.controller_metadata: dict[str, Any] = {}
        labels = (("dx", "dy", "dz", "rx", "ry", "rz") if dimension == 7
                  else tuple(f"q{i + 1}" for i in range(7))) + ("gripper",)
        self.info = EmbodimentInfo(
            name="maniskill-franka-droid",
            action_space=Box(
                shape=(dimension,),
                semantics=ActionSemantics(
                    control_mode=semantics,
                    rotation_repr="euler_xyz" if dimension == 7 else "none",
                    gripper="continuous", frame="base",
                    dim_labels=labels,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 360, 640) for name in CAMERAS),
                state=StateSpec(fields=(
                    StateField("eef_pos", (3,), "m"),
                    StateField("eef_quat", (4,), "unit_quat"),
                    StateField("joint_pos", (7,), "rad"),
                    StateField("gripper_pos", (1,), "rad"),
                )),
            ),
            control_hz=self.control_hz, is_simulated=True,
            capabilities=frozenset({
                "seedable", "resettable", "auto_reset", "privileged_success", "renderable",
                "recording_state",
            }),
        )

    def _start(self) -> None:
        if self._process is not None:
            return
        if not (self.source_root / "sim_eval/robots/franka_droid.py").is_file():
            raise FileNotFoundError(f"Official MolmoAct2 source missing: {self.source_root}")
        env = os.environ.copy()
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
            env.pop(name, None)
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MS_ASSET_DIR": str(project_path(os.environ.get("MS_ASSET_DIR", "data/maniskill"))),
        })
        env.update(self.worker_env)
        self._read_buffer.clear()
        self._process = subprocess.Popen(
            [self.python, "-B", "-u", str(Path(__file__).with_name("worker.py")),
             "--source-root", str(self.source_root), "--assets-root", str(self.assets_root),
             "--control-mode", self.control_mode, "--control-hz", str(int(self.control_hz)),
             "--sim-hz", str(self.sim_hz), "--max-episode-steps", str(self.max_episode_steps),
             "--shader-pack", self.shader_pack, "--sim-backend", self.sim_backend,
             "--observation-profile", self.observation_profile],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env, bufsize=0,
        )

    def list_tasks(
        self, suite: str = ENV_ID, task_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        if suite not in (ENV_ID, "maniskill", "droid"):
            raise ValueError(f"Unsupported ManiSkill suite: {suite}")
        if task_ids is not None and any(type(x) is not int or x != 0 for x in task_ids):
            raise ValueError("The official DROID suite contains task_id 0 only")
        return self._rpc("list_tasks") if task_ids != [] else []

    @property
    def motion_profile(self):
        from agentic_framework.configuration.motion_profiles import load_motion_profile

        return load_motion_profile(
            environment="maniskill", embodiment=self.info.name,
            control_mode=self.control_mode, control_hz=self.control_hz,
            frame="base", rotation="euler_xyz",
        )

    @property
    def action_spec(self):
        from agentic_framework.harness.types import ActionSpec
        if self.control_mode != "pd_ee_delta_pose":
            raise ValueError("Native joint policies use info.action_space, not a motion ActionSpec")
        if not self.controller_metadata:
            raise SimulatorError("metadata() or reset() is required before reading the actual controller ActionSpec")
        values = dict(self.controller_metadata["action_spec"])
        for key in ("pose_scale", "pose_sign", "input_low", "input_high", "single_step_pose_limit"):
            values[key] = tuple(values[key])
        return ActionSpec(**values)

    def _observation(self, data: Mapping[str, Any]) -> Observation:
        images = {}
        for name in CAMERAS:
            encoded = data["images"][name]
            shape = tuple(encoded["shape"])
            if shape != (360, 640, 3):
                raise SimulatorError(f"Unexpected native {name} image shape: {shape}")
            raw = base64.b64decode(encoded["data"], validate=True)
            if len(raw) != int(np.prod(shape)):
                raise SimulatorError(f"Malformed {name} image payload")
            images[name] = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        state = {}
        for field in self.info.observation_space.state.fields:
            values = np.asarray(data["state"][field.key], dtype=np.float64)
            if values.shape != field.shape or not np.isfinite(values).all():
                raise SimulatorError(f"Invalid measured state field: {field.key}")
            state[field.key] = values
        stamp = float(data["sim_time"])
        if not np.isfinite(stamp) or stamp < 0:
            raise SimulatorError("Invalid simulator timestamp")
        extra = dict(data.get("extra", {}))
        # Scene truth is available only through the explicitly opted-in v2 channel.
        if self.observation_profile != "privileged":
            for key in (PRIVILEGE_KEY, *LEGACY_PRIVILEGE_KEYS):
                extra.pop(key, None)
        else:
            if any(key in extra for key in LEGACY_PRIVILEGE_KEYS):
                raise SimulatorError("Legacy live privilege channels are unsupported; use privileged schema v2")
            try:
                extra[PRIVILEGE_KEY] = validate_privilege(
                    extra.get(PRIVILEGE_KEY), expected_adapter="maniskill",
                )
            except (TypeError, ValueError) as error:
                raise SimulatorError(f"Invalid privileged ManiSkill observation: {error}") from error
        return Observation(
            images=images, state=state, instruction=data["instruction"], state_time=stamp,
            image_times={name: stamp for name in images}, extra=extra,
        )

    def metadata(self) -> dict[str, Any]:
        result = self._rpc("metadata")
        self._update_controller_metadata(result["controller"])
        return result

    def _update_controller_metadata(self, metadata):
        space = self.info.action_space
        low = np.asarray(metadata["input_low"], dtype=np.float64)
        high = np.asarray(metadata["input_high"], dtype=np.float64)
        if (low.shape != space.shape or high.shape != space.shape
                or not np.isfinite(low).all() or not np.isfinite(high).all()
                or np.any(low >= high)):
            raise SimulatorError("Worker returned invalid native action bounds")
        if metadata["control_mode"] != self.control_mode:
            raise SimulatorError("Worker returned a different control mode")
        if float(metadata["control_hz"]) != self.control_hz:
            raise SimulatorError("Worker returned a different control frequency")
        if self.control_mode == "pd_ee_delta_pose":
            profile = self.motion_profile
            metadata["motion_profile"] = profile
            metadata["action_spec"]["single_step_pose_limit"] = profile["single_step_pose_limit"]
        self.controller_metadata = metadata
        self.info = replace(self.info, action_space=replace(space, low=low, high=high))

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._has_reset = False
        result = self._rpc(
            "reset", env_id=scene.metadata.get("env_id", ENV_ID),
            seed=seed if seed is not None else scene.init_seed,
            instruction=scene.metadata.get("instruction"),
        )
        self._update_controller_metadata(result["info"]["controller"])
        observation = self._observation(result["observation"])
        self.last_reset_info = result["info"]
        self._has_reset = True
        self._terminated = bool(result["info"]["success"])
        return observation

    def step(self, action: Action) -> StepResult:
        if not self._has_reset:
            raise SimulatorError("reset(scene) must be called before step(action)")
        if self._terminated:
            raise SimulatorError("Episode terminated; reset before further actions")
        declared_mode = action.meta.get("control_mode")
        if declared_mode is not None and canonical_mode(declared_mode) != self.control_mode:
            raise ValueError("Action control mode differs from the active native controller")
        values = validate_action(action.data, self.control_mode)
        result = self._rpc("step", action=values.tolist())
        self._terminated = bool(result["terminated"] or result["truncated"])
        return StepResult(
            observation=self._observation(result["observation"]), reward=result["reward"],
            terminated=result["terminated"], termination_reason=result["termination_reason"],
            truncated=result["truncated"], info=result["info"],
        )
