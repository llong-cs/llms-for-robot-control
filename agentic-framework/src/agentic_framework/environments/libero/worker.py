"""Python 3.8-compatible LIBERO worker; stdin/stdout carry newline-delimited JSON.

Run this file directly using the benchmark's existing simulation interpreter.
Library output (including native file descriptor 1) is redirected to stderr;
only the duplicated RPC output descriptor may emit protocol records.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import random
import sys
import traceback
from pathlib import Path

CAMERAS = ("agentview", "robot0_eye_in_hand")
POSE_SCALE = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
SETTLE_STEPS = 10

def _component(value, label):
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ValueError("%s must be a nonempty path component" % label)
    if any(char in value for char in ("/", "\\", "\x00")):
        raise ValueError("%s must be a single path component" % label)
    return value


def _require_file(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def resolve_task_files(bddl_root, init_root, task):
    """Return exact BDDL and initial-state paths from the native task definition."""
    folder = _component(task.problem_folder, "problem_folder")
    bddl_file = _component(task.bddl_file, "bddl_file")
    init_file = _component(task.init_states_file, "init_states_file")
    if not bddl_file.endswith(".bddl") or not init_file.endswith(".pruned_init"):
        raise ValueError("Unexpected task file extension")
    bddl = _require_file(Path(bddl_root) / folder / bddl_file)
    initial = _require_file(Path(init_root) / folder / init_file)
    return bddl, bddl, initial, "exact"


class Simulator:
    def __init__(self, benchmark, image_size, observation_profile="control"):
        if observation_profile not in ("control", "openpi_matched", "privileged"):
            raise ValueError("unknown LIBERO observation profile")
        # Imports happen only after main has installed the stdio guard.
        import numpy as np
        import torch
        from libero.libero import benchmark as benchmark_module
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.transform_utils import mat2quat

        self.np = np
        self._mat2quat = mat2quat
        self.torch = torch
        self.OffScreenRenderEnv = OffScreenRenderEnv
        self.benchmark_module = benchmark_module
        self.benchmark = benchmark
        self.observation_profile = observation_profile
        self.image_size = image_size
        self.paths = {
            key: get_libero_path(key)
            for key in (
                "benchmark_root",
                "bddl_files",
                "init_states",
                "assets",
                "datasets",
            )
        }
        self.suites = {}
        self.env = None
        self.env_key = None
        self.instruction = None
        self.step_count = 0
        self.success = False
        self.ready = False
        self.current_info = {}
        self._recording_task_feedback = None
        self._init_cache_path = None
        self._init_cache = None

    def _suite(self, name):
        if name not in self.suites:
            mapping = self.benchmark_module.get_benchmark_dict()
            if name not in mapping:
                raise ValueError("Unknown suite %r; available: %s" % (name, sorted(mapping)))
            with contextlib.redirect_stdout(io.StringIO()):
                self.suites[name] = mapping[name](task_order_index=0)
        return self.suites[name]

    def _task(self, suite, task_id):
        if not isinstance(task_id, int) or isinstance(task_id, bool):
            raise ValueError("task_id must be an integer")
        selected = self._suite(suite)
        if not 0 <= task_id < selected.n_tasks:
            raise ValueError("task_id %s outside [0, %s)" % (task_id, selected.n_tasks))
        return selected.get_task(task_id)

    def _files(self, task):
        return resolve_task_files(
            self.paths["bddl_files"],
            self.paths["init_states"],
            task,
        )

    def _initial_states(self, path):
        if str(path) != self._init_cache_path:
            # Trusted benchmark tensors include NumPy objects. torch 2.6
            # changes weights_only's default, while torch 1.11 lacks it.
            kwargs = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(self.torch.load).parameters:
                kwargs["weights_only"] = False
            states = self.torch.load(str(path), **kwargs)
            if getattr(states, "ndim", 0) == 1:
                states = states.reshape(1, -1)
            if len(states) == 0:
                raise ValueError("Initial state file is empty: %s" % path)
            self._init_cache = states
            self._init_cache_path = str(path)
        return self._init_cache

    def _task_info(self, suite, task_id, include_count=True):
        task = self._task(suite, task_id)
        bddl, source, initial, strategy = self._files(task)
        instruction = task.language.strip()
        if not instruction:
            raise ValueError("Empty benchmark instruction in %s" % source)
        return {
            "benchmark": self.benchmark,
            "suite": suite,
            "task_id": task_id,
            "name": task.name,
            "instruction": instruction,
            "num_init_states": len(self._initial_states(initial)) if include_count else None,
            "bddl_file": str(bddl),
            "bddl_source_path": str(source),
            "init_states_file": task.init_states_file,
            "init_states_path": str(initial),
            "init_states_strategy": strategy,
        }

    def list_tasks(self, suite, task_ids=None):
        selected = self._suite(suite)
        ids = range(selected.n_tasks) if task_ids is None else task_ids
        return [self._task_info(suite, index, task_ids is not None) for index in ids]

    def metadata(self):
        versions = {}
        for name in ("numpy", "torch", "robosuite", "mujoco", "libero", "bddl"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        libero_module = sys.modules["libero.libero"]
        benchmark_file = Path(self.benchmark_module.__file__)
        candidates = {
            "libero_module": Path(libero_module.__file__),
            "benchmark_module": benchmark_file,
            "task_map": benchmark_file.with_name("libero_suite_task_map.py"),
            "environment_wrapper": Path(sys.modules[self.OffScreenRenderEnv.__module__].__file__),
            "libero_config": Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml",
        }
        files = {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in candidates.items()
            if path.is_file()
        }
        return {
            "benchmark": self.benchmark,
            "python": sys.executable,
            "libero_module_file": str(libero_module.__file__),
            "files": files,
            "python_version": sys.version,
            "versions": versions,
            "paths": self.paths,
            "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
            "renderer": os.environ.get("MUJOCO_GL"),
            "image_size": self.image_size,
            "observation_profile": self.observation_profile,
            "image_transform": "rotate_180: observation[::-1, ::-1]",
            "control_hz": 20,
            "settle_steps": SETTLE_STEPS,
            "pose_scale": POSE_SCALE,
            "gripper": "native values unchanged; negative=open; positive=close; native zero semantics",
            "action_representation": "native normalized OSC, no bridge scaling or clipping",
            "suites": sorted(self.benchmark_module.get_benchmark_dict()),
        }

    def _check_controller(self):
        # Fail visibly instead of silently using an incompatible OSC scaling.
        controller = self.env.robots[0].controller
        for attr, expected in (
            ("output_max", POSE_SCALE),
            ("output_min", [-x for x in POSE_SCALE]),
            ("input_max", [1] * 6),
            ("input_min", [-1] * 6),
        ):
            actual = self.np.asarray(getattr(controller, attr))
            if not self.np.allclose(actual, expected, rtol=0, atol=1e-8):
                raise ValueError("Unsupported OSC %s: %s" % (attr, actual))
        if not controller.use_delta:
            raise ValueError("LIBERO requires a delta OSC_POSE controller")
        native = getattr(self.env, "env", self.env)
        low, high = native.action_spec
        if not (
            self.np.asarray(low).shape == (7,)
            and self.np.asarray(high).shape == (7,)
            and self.np.allclose(low, [-1.0] * 7, rtol=0, atol=1e-8)
            and self.np.allclose(high, [1.0] * 7, rtol=0, atol=1e-8)
        ):
            raise ValueError("Unsupported native seven-dimensional action bounds")
        if native.control_freq != 20:
            raise ValueError("Unsupported LIBERO control frequency")
        if type(self.env.robots[0].gripper).__name__ != "PandaGripper":
            raise ValueError(
                "Gripper neutral/sign semantics require the checked PandaGripper driver"
            )

    def _privileged_observation(self):
        """Read generic scene truth only when explicitly requested."""
        if not hasattr(self, "_privilege_adapter"):
            path = Path(__file__).with_name("privilege.py")
            spec = importlib.util.spec_from_file_location("_libero_scene_privilege", path)
            self._privilege_adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._privilege_adapter)
        return self._privilege_adapter.read_privileged_state(self.env, self._mat2quat)

    def recording_state(self):
        """Separate read-only RPC; never included in a policy Observation."""
        if not self.ready:
            raise RuntimeError("Reset an episode before reading recording state")
        if not hasattr(self, "_recording_adapter"):
            path = Path(__file__).with_name("trajectory_state.py")
            spec = importlib.util.spec_from_file_location("_libero_recording_state", path)
            self._recording_adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._recording_adapter)
        return self._recording_adapter.read_recording_state(
            self.env, self.step_count, task=getattr(self, "_recording_task_feedback", None)
        )

    def _cache_recording_task(self, phase, reward, done, info, reset_info=None):
        # Use only feedback already produced by the ordinary reset/step path.
        # No success check, predicate evaluation or custom progress computation.
        self._recording_task_feedback = {
            "availability": "available",
            "source": "cached LIBERO env.step return and existing env.check_success result",
            "data": {
                "phase": phase, "info": copy.deepcopy(info),
                "success": self.success,
                "reward": None if reward is None else float(reward),
                "terminated": self.success,
                "truncated": None if done is None else bool(done and not self.success),
                "native_done": None if done is None else bool(done),
                "field_sources": {
                    "info": "env.step returned info" if phase == "step" else "final reset stabilization env.step returned info",
                    "success": "existing env.check_success() call on ordinary reset/step path",
                    "reward": "env.step returned reward" if phase == "step" else "final reset stabilization env.step returned reward",
                    "terminated": "adapter termination derived from cached native success",
                    "truncated": "adapter bool(native_done and not success)",
                    "native_done": "env.step returned done",
                },
            },
        }
        if phase == "reset":
            self._recording_task_feedback["data"]["reset_info"] = copy.deepcopy(reset_info)
            self._recording_task_feedback["data"]["reset_stabilization_steps"] = SETTLE_STEPS

    def _observation(self, observation):
        np = self.np
        images = {}
        for camera in CAMERAS:
            raw = np.asarray(observation[camera + "_image"])
            if raw.dtype != np.uint8 or raw.shape != (self.image_size, self.image_size, 3):
                raise ValueError("Unexpected %s RGB image: %s/%s" % (camera, raw.shape, raw.dtype))
            rgb = np.ascontiguousarray(raw[::-1, ::-1])
            images[camera] = {
                "shape": list(rgb.shape),
                "data": base64.b64encode(rgb.tobytes()).decode("ascii"),
            }
        qpos = np.asarray(observation["robot0_gripper_qpos"], dtype=float)
        # robosuite's default eef quaternion is the robot hand BODY orientation,
        # while OSC controls the gripper SITE. Publish one consistent pose.
        site_id = self.env.robots[0].eef_site_id
        eef_pos = np.asarray(self.env.sim.data.site_xpos[site_id], dtype=float)
        eef_rotation = np.asarray(self.env.sim.data.site_xmat[site_id]).reshape(3, 3)
        state = {
            "eef_pos": eef_pos.tolist(),
            "eef_quat": self._mat2quat(eef_rotation).astype(float).tolist(),
            "gripper_width": [float(abs(qpos[0] - qpos[1]))],
            "joint_pos": np.asarray(observation["robot0_joint_pos"], dtype=float).tolist(),
        }
        result = {
            "images": images,
            "state": state,
            "instruction": self.instruction,
            "sim_time": (self.step_count + SETTLE_STEPS) / 20.0,
            "extra": {
                "raw_libero": {
                    "robot0_eef_pos": np.asarray(
                        observation["robot0_eef_pos"], dtype=float
                    ).tolist(),
                    "robot0_eef_quat": np.asarray(
                        observation["robot0_eef_quat"], dtype=float
                    ).tolist(),
                    "robot0_gripper_qpos": qpos.tolist(),
                },
            },
        }

        # The standard profile never even reads privileged simulator fields.
        if getattr(self, "observation_profile", "control") == "privileged":
            result["extra"]["privileged"] = self._privileged_observation()
        return result

    def reset(self, suite, task_id, init_state_index, seed=None):
        self.ready = False
        self._recording_task_feedback = None
        info = self._task_info(suite, task_id)
        states = self._initial_states(Path(info["init_states_path"]))
        if not isinstance(init_state_index, int) or isinstance(init_state_index, bool):
            raise ValueError("init_state_index must be an integer")
        if not 0 <= init_state_index < len(states):
            raise ValueError(
                "init_state_index %s outside [0, %s)" % (init_state_index, len(states))
            )
        resolved_seed = 0 if seed is None else seed
        random.seed(resolved_seed)
        self.np.random.seed(resolved_seed)
        key = (suite, task_id)
        if key != self.env_key:
            self.close()
            self.env = self.OffScreenRenderEnv(
                bddl_file_name=info["bddl_file"],
                camera_heights=self.image_size,
                camera_widths=self.image_size,
                camera_names=list(CAMERAS),
                control_freq=20,
                horizon=100000,
                ignore_done=True,
            )
            self.env_key = key
        self.env.seed(resolved_seed)
        native_reset = self.env.reset()
        reset_info = native_reset[1] if isinstance(native_reset, tuple) and len(native_reset) == 2 else None
        self._check_controller()
        observation = self.env.set_init_state(states[init_state_index])
        settle_reward, settle_done, settle_info = None, None, {}
        for _ in range(SETTLE_STEPS):
            observation, settle_reward, settle_done, settle_info = self.env.step([0.0] * 6 + [-1.0])
        self.instruction = info["instruction"]
        self.step_count = 0
        self.success = bool(self.env.check_success())
        self._cache_recording_task("reset", settle_reward, settle_done, settle_info, reset_info)
        self.ready = True
        self.current_info = dict(
            info,
            init_state_index=init_state_index,
            initial_state_sha256=hashlib.sha256(
                self.np.asarray(states[init_state_index], dtype="<f8").tobytes(order="C")
            ).hexdigest(),
            seed=resolved_seed,
            settle_steps=SETTLE_STEPS,
            success=self.success,
        )
        return {"observation": self._observation(observation), "info": self.current_info}

    def _native_action_audit(self, values):
        """Record derived input saturation separately from observed controller state.

        This reads the controller after the native step. It never writes goals,
        repeats an action, or claims a computed clipped vector was intercepted.
        """
        np = self.np
        audit = {"input_semantics": "raw normalized OSC input sent unchanged"}
        try:
            robot = self.env.robots[0]
            controller = robot.controller
        except (AttributeError, IndexError, TypeError):
            audit["observed_controller_state_available"] = False
            return audit
        audit["observed_controller_state_available"] = True
        low = np.asarray(controller.input_min, dtype=float)
        high = np.asarray(controller.input_max, dtype=float)
        expected = np.clip(values[:6], low, high)
        audit["expected_clipped_pose_input"] = expected.tolist()
        audit["expected_input_saturation"] = bool(np.any(expected != values[:6]))
        audit["saturation_value_source"] = (
            "computed from native input_min/input_max; not an intercepted action"
        )
        for key, value in (
            ("observed_goal_pos", getattr(controller, "goal_pos", None)),
            ("observed_goal_ori", getattr(controller, "goal_ori", None)),
            ("observed_gripper_current_action", getattr(robot.gripper, "current_action", None)),
        ):
            if value is not None:
                array = np.asarray(value, dtype=float)
                if np.isfinite(array).all():
                    audit[key] = array.tolist()
        return audit

    def step(self, action):
        if not self.ready:
            raise RuntimeError("Reset an episode before stepping")
        if self.success:
            raise RuntimeError("Episode already succeeded")
        values = self.np.asarray(action, dtype=float)
        if values.shape != (7,) or not self.np.isfinite(values).all():
            raise ValueError("Expected seven finite normalized actions")
        observation, reward, done, env_info = self.env.step(values)
        self.step_count += 1
        self.success = bool(self.env.check_success())
        self._cache_recording_task("step", reward, done, env_info)
        return {
            "observation": self._observation(observation),
            "reward": float(reward),
            "terminated": self.success,
            "termination_reason": "success" if self.success else None,
            "truncated": bool(done and not self.success),
            "info": {
                "success": self.success,
                "sim_steps": self.step_count,
                "settle_steps": SETTLE_STEPS,
                "normalized_action": values.tolist(),
                "controller_audit": self._native_action_audit(values),
            },
        }

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
                self.env_key = None
        self.ready = False
        self._recording_task_feedback = None
        return {"closed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero",), required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--observation-profile",
        choices=("control", "openpi_matched", "privileged"),
        default="control",
    )
    args = parser.parse_args()
    # Preserve a private protocol fd, then redirect Python and C-library stdout.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    simulator = None
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                method = request["method"]
                if method not in ("metadata", "list_tasks", "reset", "step", "recording_state", "close"):
                    raise ValueError("Unknown RPC method: %s" % method)
                if simulator is None:
                    simulator = Simulator(args.benchmark, args.image_size)
                    simulator.observation_profile = args.observation_profile
                result = getattr(simulator, method)(**request.get("params", {}))
                response = {"id": request["id"], "result": result}
            except Exception as exc:
                response = {
                    "id": request.get("id"),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=12),
                    },
                }
            output.write(json.dumps(response, allow_nan=False) + "\n")
            output.flush()
            if request.get("method") == "close":
                break
    finally:
        if simulator is not None:
            simulator.close()
        output.close()


if __name__ == "__main__":
    main()
