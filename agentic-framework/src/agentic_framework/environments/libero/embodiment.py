"""Inspect Robots embodiment backed by a persistent, isolated LIBERO process.

The simulation worker deliberately remains compatible with Python 3.8 while
the policy process can use the Python version required by Inspect Robots.
"""

from __future__ import annotations

import base64
import json
import os
import selectors
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
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

from agentic_framework.common.project import environment_python, project_path, project_root
from agentic_framework.environments.privilege import (
    LEGACY_PRIVILEGE_KEYS,
    PRIVILEGE_KEY,
    validate_privilege,
)

POSE_SCALE = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float64)
CAMERAS = ("agentview", "robot0_eye_in_hand")


class SimulatorError(RuntimeError):
    """A simulator worker failed, timed out, or violated its RPC protocol."""


def validate_action(action: Any) -> np.ndarray:
    """Validate native OSC input without modifying controller or gripper semantics.

    Bounds describe the nominal normalized input space. Native LIBERO applies
    its own saturation and gripper formatting; this bridge never clips,
    rescales, hardens a sign, or replaces zero with the previous command.
    """
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("LIBERO action must be seven finite numbers")
    return value.copy()


class LiberoEmbodiment(EmbodimentBase):
    """Run one LIBERO simulator per embodiment, importing it only in the worker.

    ``Scene.metadata`` requires ``suite``, ``task_id``, and ``init_state_index``.
    Scene/``reset`` seeds govern simulator randomness, independently from the
    selected official initial state. Defaults use separate project-local simulator environments.
    """

    _worker_label = "LIBERO"

    def __init__(
        self,
        benchmark: str = "libero",
        python: str | Path | None = None,
        image_size: int = 256,
        worker_env: Mapping[str, str] | None = None,
        *,
        request_timeout_s: float = 600,
        observation_profile: str = "control",
    ) -> None:
        if benchmark != "libero":
            raise ValueError("benchmark must be 'libero'")
        if image_size < 1 or request_timeout_s <= 0:
            raise ValueError("image_size and request_timeout_s must be positive")
        if observation_profile not in ("control", "openpi_matched", "privileged"):
            raise ValueError("unknown LIBERO observation profile")
        self.benchmark = benchmark
        self.observation_profile = observation_profile
        self.python = str(project_path(python) if python else environment_python("agentic-framework-" + benchmark))
        self.image_size = image_size
        self.request_timeout_s = request_timeout_s
        self.worker_env = dict(worker_env or {})
        self.last_reset_info: dict[str, Any] = {}
        self._process: subprocess.Popen[bytes] | None = None
        self._read_buffer = bytearray()
        self._lock = threading.RLock()
        self._request_id = 0
        self._has_reset = False
        self._terminated = False

        bounds = np.ones(7, dtype=np.float64)
        self.info = EmbodimentInfo(
            name=benchmark,
            action_space=Box(
                shape=(7,),
                low=-bounds,
                high=bounds,
                semantics=ActionSemantics(
                    control_mode="eef_delta_pose",
                    rotation_repr="axis_angle",
                    gripper="continuous",
                    frame="world",
                    dim_labels=("dx", "dy", "dz", "rx", "ry", "rz", "gripper"),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, image_size, image_size) for name in CAMERAS),
                state=StateSpec(
                    fields=(
                        StateField("eef_pos", (3,), "m"),
                        StateField("eef_quat", (4,), "unit_quat"),
                        StateField("gripper_width", (1,), "m"),
                        StateField("joint_pos", (7,), "rad"),
                    )
                ),
            ),
            control_hz=20.0,
            is_simulated=True,
            capabilities=frozenset(
                {"seedable", "resettable", "auto_reset", "privileged_success", "renderable",
                 "recording_state"}
            ),
        )

    @property
    def motion_profile(self):
        from agentic_framework.configuration.motion_profiles import load_motion_profile

        return load_motion_profile(
            environment=self.benchmark, embodiment=self.info.name,
            control_mode="osc_pose", control_hz=self.info.control_hz,
            frame="world", rotation="axis_angle",
        )

    @property
    def action_spec(self):
        from agentic_framework.configuration.motion_profiles import motion_pose_limits
        from agentic_framework.harness.types import ActionSpec

        return ActionSpec(
            pose_scale=tuple(POSE_SCALE), control_hz=self.info.control_hz,
            single_step_pose_limit=motion_pose_limits(self.motion_profile),
        )

    def _start(self) -> None:
        if self._process is not None:
            return
        env = os.environ.copy()
        # No model credentials are needed inside the simulator process.
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
            env.pop(name, None)
        renderer = self.worker_env.get("MUJOCO_GL", env.get("MUJOCO_GL", "osmesa"))
        defaults = {
            "MUJOCO_GL": renderer,
            "PYOPENGL_PLATFORM": self.worker_env.get(
                "PYOPENGL_PLATFORM", env.get(
                    "PYOPENGL_PLATFORM", "egl" if renderer == "egl" else "osmesa"
                )
            ),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH", str(project_root() / ".config" / self.benchmark)),
        }
        env.update(defaults)
        env.update(self.worker_env)
        config_file = Path(env["LIBERO_CONFIG_PATH"]) / "config.yaml"
        if not config_file.is_file():
            raise FileNotFoundError(
                f"LIBERO config missing: {config_file}; set worker_env['LIBERO_CONFIG_PATH'] "
                "to an existing config directory (to avoid LIBERO's interactive setup)"
            )
        self._read_buffer.clear()
        self._process = subprocess.Popen(
            [
                self.python,
                "-u",
                str(Path(__file__).with_name("worker.py")),
                "--benchmark",
                self.benchmark,
                "--image-size",
                str(self.image_size),
                "--observation-profile",
                self.observation_profile,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
            bufsize=0,
        )

    def _rpc(self, method: str, **params: Any) -> Any:
        with self._lock:
            self._start()
            process = self._process
            assert process is not None and process.stdin is not None and process.stdout is not None
            self._request_id += 1
            request_id = self._request_id
            try:
                data = json.dumps(
                    {"id": request_id, "method": method, "params": params}, allow_nan=False
                )
                process.stdin.write((data + "\n").encode())
                process.stdin.flush()
                deadline = time.monotonic() + self.request_timeout_s
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while b"\n" not in self._read_buffer:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise SimulatorError(f"{self._worker_label} worker timed out during {method}")
                        chunk = os.read(process.stdout.fileno(), 1024 * 1024)
                        if not chunk:
                            raise SimulatorError(
                                f"{self._worker_label} worker exited during {method} (exit={process.poll()}); see stderr"
                            )
                        self._read_buffer.extend(chunk)
                line, _, remainder = self._read_buffer.partition(b"\n")
                self._read_buffer = bytearray(remainder)
                response = json.loads(line)
                if response.get("id") != request_id:
                    raise SimulatorError(f"{self._worker_label} worker returned an out-of-order response")
            except (OSError, ValueError, SimulatorError) as exc:
                self._stop_process()
                raise SimulatorError(str(exc)) from exc
            if "error" in response:
                error = response["error"]
                raise SimulatorError(
                    f"{method}: {error['type']}: {error['message']}\n{error.get('traceback', '')}"
                )
            return response["result"]

    def list_tasks(self, suite: str, task_ids: Sequence[int] | None = None) -> list[dict[str, Any]]:
        """Return task/instruction/path metadata without constructing a simulator.

        Explicit ``task_ids`` also loads those initial state files and returns
        their exact ``num_init_states``. With ``None``, counts are null to avoid
        loading initial-state tensors simply to enumerate tasks.
        """
        return self._rpc(
            "list_tasks", suite=suite, task_ids=None if task_ids is None else list(task_ids)
        )

    def metadata(self) -> dict[str, Any]:
        """Get simulator dependency versions and the actual data/configuration paths."""
        return {**self._rpc("metadata"), "motion_profile": self.motion_profile}

    def recording_state(self) -> dict[str, Any]:
        """Read the shared measured-scene contract, never a policy observation."""
        if not self._has_reset:
            raise SimulatorError("reset(scene) must precede recording_state()")
        return self._rpc("recording_state")

    def _observation(self, data: Mapping[str, Any]) -> Observation:
        images = {}
        for name, encoded in data["images"].items():
            raw = base64.b64decode(encoded["data"], validate=True)
            shape = tuple(encoded["shape"])
            if shape != (self.image_size, self.image_size, 3):
                raise SimulatorError(f"Unexpected {name} image shape: {shape}")
            images[name] = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        state = {name: np.asarray(value, dtype=np.float64) for name, value in data["state"].items()}
        stamp = float(data["sim_time"])
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
                    extra.get(PRIVILEGE_KEY), expected_adapter="libero",
                )
            except (TypeError, ValueError) as error:
                raise SimulatorError(f"Invalid privileged LIBERO observation: {error}") from error
        return Observation(
            images=images,
            state=state,
            instruction=data["instruction"],
            state_time=stamp,
            image_times={name: stamp for name in images},
            extra=extra,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        for key in ("suite", "task_id", "init_state_index"):
            if key not in scene.metadata:
                raise ValueError(f"LIBERO scene metadata requires {key!r}")
        self._has_reset = False
        result = self._rpc(
            "reset",
            suite=scene.metadata["suite"],
            task_id=scene.metadata["task_id"],
            init_state_index=scene.metadata["init_state_index"],
            seed=seed if seed is not None else scene.init_seed,
        )
        self.last_reset_info = result["info"]
        self._has_reset = True
        self._terminated = bool(result["info"]["success"])
        return self._observation(result["observation"])

    @property
    def initially_solved(self) -> bool:
        return bool(self.last_reset_info.get("success", False))

    def step(self, action: Action) -> StepResult:
        if not self._has_reset:
            raise SimulatorError("reset(scene) must be called before step(action)")
        if self._terminated:
            raise SimulatorError("Episode already succeeded; reset before further actions")
        values = validate_action(action.data)
        result = self._rpc("step", action=values.tolist())
        self._terminated = bool(result["terminated"])
        return StepResult(
            observation=self._observation(result["observation"]),
            reward=result["reward"],
            terminated=result["terminated"],
            termination_reason=result["termination_reason"],
            truncated=result["truncated"],
            info=result["info"],
        )

    def _stop_process(self) -> None:
        process, self._process = self._process, None
        self._has_reset = False
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def close(self) -> None:
        with self._lock:
            if self._process is not None:
                try:
                    if self._process.poll() is None:
                        self._rpc("close")
                finally:
                    self._stop_process()

    def __enter__(self) -> LiberoEmbodiment:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
