"""Official MolmoAct2 ManiSkill environment behind the framework JSON RPC.

Run in the independent ManiSkill interpreter. The only external-source
adaptation changes the robot class's asset path in memory; source files, native
controllers, observations, and task success logic are not rewritten.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

OFFICIAL_COMMIT = "66b87e64efd99dfd103241418113955cf64dfa9c"
ENV_ID = "DroidPutEverythingInBox-v1"
INSTRUCTION = "put everything into the box"
CAMERAS = ("external_cam", "wrist_cam")
MODES = {"pd_ee_delta_pose": 7, "pd_joint_pos": 8, "pd_joint_vel": 8}
ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


def array(value, *, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def vector(value, dimension, label):
    result = array(value)
    if result.shape == (1, dimension):
        result = result[0]
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {dimension} finite numbers")
    return result


def scalar(value, label):
    result = array(value)
    if result.size != 1 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite scalar for one environment")
    return result.item()


def resolve_robot_assets(root):
    """Accept the external snapshot root or its assets/ child, without copying."""
    root = Path(root).resolve()
    relative = Path("franka_description/urdfs/fr3_robotiq.urdf")
    for base in (root / "assets", root):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Official Franka DROID URDF missing under {root}")


def encode_camera(image, name):
    image = array(image, dtype=np.uint8)
    if image.shape != (360, 640, 3):
        raise ValueError(f"Official {name} RGB must be 360x640x3, received {image.shape}")
    image = np.ascontiguousarray(image)
    return {"shape": list(image.shape), "data": base64.b64encode(image.tobytes()).decode()}


ASSET_REVISION = "9332a64224ff0a813d9f77bd377b845270232513"
ASSET_SOURCE = "https://huggingface.co/datasets/TreeePlanter/molmoact2-sim-eval-assets"
YCB_SOURCE = "https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root, paths):
    entries, aggregate, total = {}, hashlib.sha256(), 0
    # The resource downloader hashes sorted Path objects (component order).
    for relative in sorted(paths, key=lambda value: Path(value).parts):
        path = root / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            raise ValueError(f"Unsafe resource inventory path: {relative}")
        size, digest = path.stat().st_size, sha256_file(path)
        entries[relative] = {"bytes": size, "sha256": digest}
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())
        total += size
    return entries, aggregate.hexdigest(), total


def _download_cache(relative):
    parts = Path(relative).parts
    return parts[:3] == (".cache", "huggingface", "download") or parts == (
        ".cache", "huggingface", ".gitignore",
    )


def verify_asset_inventory(root, *, source_uri, revision=None):
    """Read and verify every asset byte once before creating the simulator.

    Only Hugging Face's local download bookkeeping is excluded. This does not
    trust the manifest's declared tree hash without re-reading actual files.
    """
    root = Path(root).resolve(strict=True)
    manifest_path = root / "VLA_SOURCE.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_uri") != source_uri:
        raise ValueError(f"Asset source URI mismatch: {root}")
    if revision is not None and manifest.get("revision") != revision:
        raise ValueError(f"Asset source revision mismatch: {root}")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"Asset manifest has no file inventory: {root}")
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _download_cache(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in asset inventory: {relative}")
        if path.is_file() and path != manifest_path:
            actual.add(relative)
    if actual != set(expected):
        raise ValueError(
            f"Asset inventory differs from manifest: {root}; "
            f"missing={sorted(set(expected) - actual)}, added={sorted(actual - set(expected))}"
        )
    entries, tree, total = _inventory(root, actual)
    for relative, current in entries.items():
        if current != expected[relative]:
            raise ValueError(f"Asset integrity mismatch: {root / relative}")
    if tree != manifest.get("tree_sha256") or total != manifest.get("total_bytes"):
        raise ValueError(f"Asset aggregate integrity mismatch: {root}")
    return {
        "root": str(root), "source_uri": source_uri,
        "revision": manifest.get("revision"),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "tree_sha256": tree, "files_verified": len(entries), "bytes_verified": total,
        "integrity": "all file paths, SHA-256 values and byte counts verified before environment creation",
    }


def package_fingerprint(root):
    """Fingerprint installed ManiSkill Python and bundled visual/collision assets.

    These are actual installed bytes for repeatability/provenance checks, not a
    claim that package contents were verified against an upstream wheel.
    """
    root = Path(root).resolve(strict=True)
    paths = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_file() and (path.suffix == ".py" or "assets" in relative.parts):
            paths.add(relative.as_posix())
    if not paths:
        raise ValueError(f"Installed ManiSkill source/assets are missing: {root}")
    entries, tree, total = _inventory(root, paths)
    return {"root": str(root), "tree_sha256": tree, "total_bytes": total,
            "files": entries, "file_count": len(entries),
            "scope": "all installed mani_skill Python files and assets directories; excludes __pycache__",
            "integrity": "actual runtime files fingerprinted; not an upstream wheel verification"}


def source_fingerprint(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10,
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"Expected MolmoAct2 {OFFICIAL_COMMIT}, found {commit}")
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", commit, "sim_eval"],
        timeout=10,
    ).decode().split("\0")
    paths = {name for name in tracked if name.endswith(".py")}
    actual = {path.relative_to(root).as_posix() for path in (root / "sim_eval").rglob("*.py")
              if "__pycache__" not in path.parts}
    if not paths or paths != actual:
        raise ValueError("Official simulator Python file inventory differs from the pinned commit")
    hashes = {}
    for relative in sorted(paths):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Official source file escapes the pinned checkout: {relative}")
        expected = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"], timeout=10,
        )
        current = sha256_file(path)
        if current != hashlib.sha256(expected).hexdigest():
            raise ValueError(f"Official simulator source differs from the pinned commit: {relative}")
        hashes[relative] = current
    return {"commit": commit, "source_root": str(root), "sha256": hashes,
            "source_files_verified": len(hashes), "matches_pinned_commit": True}


class Simulator:
    def __init__(
        self, source_root, assets_root, control_mode="pd_ee_delta_pose", control_hz=30,
        sim_hz=150, max_episode_steps=2000, shader_pack="rt-fast", sim_backend="cpu",
        observation_profile="control",
    ):
        if observation_profile not in ("control", "privileged"):
            raise ValueError(f"Unsupported observation profile {observation_profile!r}")
        self.observation_profile = observation_profile
        if control_mode not in MODES:
            raise ValueError(f"Unsupported native control mode {control_mode!r}")
        if (type(control_hz) is not int or control_hz < 1 or type(sim_hz) is not int
                or sim_hz < 1 or sim_hz % control_hz):
            raise ValueError("sim_hz must be a positive multiple of control_hz")
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.source = source_fingerprint(source_root)
        self.urdf_path = resolve_robot_assets(assets_root)
        asset_root = Path(assets_root).resolve()
        if not (asset_root / "VLA_SOURCE.json").is_file() and asset_root.name == "assets":
            asset_root = asset_root.parent
        if not self.urdf_path.resolve().is_relative_to(asset_root):
            raise ValueError("Robot URDF is outside the verified asset snapshot")
        robot_assets = verify_asset_inventory(
            asset_root, source_uri=ASSET_SOURCE, revision=ASSET_REVISION,
        )
        sys.path.insert(0, str(Path(source_root).resolve()))
        import gymnasium as gym
        import mani_skill
        import torch
        # Importing modules does not create physics; validate remaining resources
        # before gym.make, including the YCB files and table visuals.
        ycb_assets = verify_asset_inventory(
            mani_skill.ASSET_DIR / "assets/mani_skill2_ycb", source_uri=YCB_SOURCE,
        )
        runtime_package = package_fingerprint(Path(mani_skill.__file__).parent)
        self.resource_integrity = {
            "robot_assets": robot_assets, "ycb_assets": ycb_assets,
            "maniskill_package": runtime_package,
        }
        from sim_eval.inference.common import droid_state_adapter, extract_camera, extract_qpos
        from sim_eval.robots.franka_droid import FrankaDROID
        # The official properties read self.urdf_path for both articulation and IK.
        # Change only this in-memory path; the immutable checkout contains no assets.
        FrankaDROID.urdf_path = str(self.urdf_path)
        import sim_eval.tasks  # noqa: F401  register official environments
        self.gym = gym
        self.torch = torch
        self.mani_skill = mani_skill
        self._state_adapter = droid_state_adapter
        self._extract_camera = extract_camera
        self._extract_qpos = extract_qpos
        self.control_mode = control_mode
        self.control_hz = control_hz
        self.sim_hz = sim_hz
        self.max_episode_steps = max_episode_steps
        self.shader_pack = shader_pack
        self.sim_backend = sim_backend
        self.env = None
        self.ready = False
        self.finished = False
        self.step_count = 0
        self.instruction = INSTRUCTION
        self.current_info = {}
        self.native_task_info = {}
        self.task_receipt = None
        self.controller_info = {}

    def _ensure_env(self, env_id=ENV_ID):
        if env_id != ENV_ID:
            raise ValueError(f"Only the official DROID task {ENV_ID} is supported")
        if self.env is None:
            # Constructor reset initializes spaces, as in the official evaluator.
            # This creates no extra policy-controlled step and does not settle.
            self.env = self.gym.make(
                env_id, obs_mode="rgb", control_mode=self.control_mode,
                render_mode="rgb_array", max_episode_steps=self.max_episode_steps,
                reward_mode="none", sensor_configs=dict(shader_pack=self.shader_pack),
                sim_config=dict(sim_freq=self.sim_hz, control_freq=self.control_hz),
                sim_backend=self.sim_backend, num_envs=1,
            )
        self.controller_info = self._inspect_controller()

    def _inspect_controller(self):
        env = self.env.unwrapped
        agent = env.agent
        robot = agent.robot
        if agent.uid != "franka_droid":
            raise ValueError("Expected the official franka_droid robot")
        names = tuple(j.name for j in robot.get_active_joints())
        if len(names) != 13 or names[:7] != ARM_JOINTS or names[7] != "left_outer_knuckle_joint":
            raise ValueError(f"Unexpected DROID qpos ordering: {names}")
        if env.control_freq != self.control_hz or env.sim_freq != self.sim_hz:
            raise ValueError("Native simulation/control frequencies differ from requested values")
        controller = agent.controller
        arm = controller.controllers["arm"]
        gripper = controller.controllers["gripper"]
        if gripper.config.normalize_action or gripper.config.use_delta:
            raise ValueError("DROID gripper must use unnormalized absolute joint targets")
        space = env.single_action_space
        low = vector(space.low, MODES[self.control_mode], "native action lower bounds")
        high = vector(space.high, MODES[self.control_mode], "native action upper bounds")
        if np.any(low >= high):
            raise ValueError("Degenerate native action bounds")
        gripper_low = float(vector(gripper.single_action_space.low, 1, "gripper lower")[0])
        gripper_high = float(vector(gripper.single_action_space.high, 1, "gripper upper")[0])
        if low[-1] != gripper_low or high[-1] != gripper_high:
            raise ValueError("Combined action space does not preserve native gripper bounds")
        root_quat = vector(robot.pose.q, 4, "robot base quaternion wxyz")
        if not np.allclose(np.abs(root_quat), [1, 0, 0, 0], atol=1e-6):
            raise ValueError("Official world-aligned DROID base rotation changed")
        result = {
            "control_mode": self.control_mode, "control_hz": self.control_hz,
            "sim_hz": self.sim_hz, "physics_steps_per_control": self.sim_hz // self.control_hz,
            "input_low": low.tolist(), "input_high": high.tolist(),
            "joint_names": list(names), "eef_reference": "fr3_link8",
            "gripper_joint_lower": gripper_low, "gripper_joint_upper": gripper_high,
            "gripper_mode": "absolute", "gripper_unit": "rad",
            "gripper_open": gripper_low, "gripper_close": gripper_high,
            "arm_normalize_action": bool(arm.config.normalize_action),
            "gripper_normalize_action": False,
            "native_arm_controller": type(arm).__name__,
            "native_gripper_controller": type(gripper).__name__,
            "robot_base_quaternion_xyzw": root_quat[[1, 2, 3, 0]].tolist(),
            "robot_base_position": vector(robot.pose.p, 3, "robot base position").tolist(),
        }
        if self.control_mode == "pd_ee_delta_pose":
            config = arm.config
            if (not config.normalize_action or not config.use_delta or config.use_target
                    or config.frame != "root_translation:root_aligned_body_rotation"
                    or config.ee_link != "fr3_link8"):
                raise ValueError("Unexpected official DROID end-effector controller configuration")
            if (not np.allclose(low[:6], -1) or not np.allclose(high[:6], 1)
                    or not np.allclose(config.pos_lower, -0.1)
                    or not np.allclose(config.pos_upper, 0.1)
                    or config.rot_lower != -0.1 or config.rot_upper != 0.1):
                raise ValueError("Official DROID end-effector control ranges changed")
            result.update({
                "pose_scale": [0.1] * 6, "pose_sign": [1, 1, 1, -1, -1, -1],
                "rotation": "euler_xyz", "frame": "base",
                "native_frame": config.frame,
                "rotation_clipping": "normalized vector L2 norm capped at 1 before rot_lower scaling",
                "position_clipping": "normalized components clipped to [-1,1] before scaling",
                "zero_pose_behavior": "target equals current measured pose (use_target=False)",
            })
            result["action_spec"] = {
                "pose_scale": (0.1,) * 6, "pose_sign": (1, 1, 1, -1, -1, -1),
                "control_hz": self.control_hz, "frame": "base", "rotation": "euler_xyz",
                "input_low": tuple(low), "input_high": tuple(high),
                "gripper_low": gripper_low, "gripper_high": gripper_high,
                "gripper_neutral": gripper_low, "gripper_mode": "absolute",
                "gripper_open": gripper_low, "gripper_close": gripper_high,
                "rotation_norm_limit": 1.0,
                "gripper_state_key": "gripper_pos",
                "gripper_open_position": gripper_low,
                "gripper_close_position": gripper_high,
                "gripper_position_tolerance": 0.01,
                "gripper_stability_tolerance": 0.002,
                "control_notes": (
                    "Native ManiSkill pd_ee_delta_pose uses root-frame translation and "
                    "left-composed Euler XYZ rotation; the official robot root is world-aligned. "
                    "Rotation input is norm-clipped and multiplied by rot_lower=-0.1. "
                    "Zero pose targets the current measured fr3_link8 pose. "
                    "The controlled fr3_link8 reference is not the midpoint between the fingers. "
                    "external_cam is external and wrist_cam is wrist-mounted; both are native "
                    "RGB views without the LIBERO image rotation. "
                    "Gripper values are absolute outer-knuckle joint targets in radians."
                ),
            }
        elif self.control_mode == "pd_joint_pos":
            if arm.config.normalize_action or arm.config.use_delta:
                raise ValueError("Molmo DROID joint-position actions must pass through unnormalized")
            result["arm_unit"] = "absolute rad"
        else:
            result["arm_unit"] = "native normalized joint velocity (not OpenPi DROID velocity)"
        return result

    def list_tasks(self):
        return [{
            "suite": "maniskill", "task_id": 0, "env_id": ENV_ID, "instruction": INSTRUCTION,
            "num_init_states": None, "initialization": "env.reset(seed=seed)",
            "official_evaluator_max_steps": 2000, "registered_max_steps": 400,
            "configured_max_steps": self.max_episode_steps,
        }]

    def metadata(self):
        self._ensure_env()
        versions = {}
        for package in ("mani-skill", "sapien", "torch", "numpy", "gymnasium"):
            versions[package] = importlib.metadata.version(package)
        return {
            "benchmark": "maniskill", "env_id": ENV_ID, "versions": versions,
            "official_source": self.source, "resource_integrity": self.resource_integrity,
            "robot_urdf": str(self.urdf_path),
            "robot_urdf_sha256": hashlib.sha256(self.urdf_path.read_bytes()).hexdigest(),
            "maniskill_asset_dir": str(self.mani_skill.ASSET_DIR),
            "controller": self.controller_info, "max_episode_steps": self.max_episode_steps,
            "shader_pack": self.shader_pack, "sim_backend": self.sim_backend,
            "observation_profile": self.observation_profile,
            "cameras": {name: {"width": 640, "height": 360, "rgb_transform": "identity"}
                        for name in CAMERAS},
            "settle_steps": 0,
        }

    def _gripper_target(self):
        controller = self.env.unwrapped.agent.controller.controllers["gripper"]
        target = array(controller._target_qpos).reshape(-1)
        indices = array(controller.control_joint_indices, dtype=np.int64).reshape(-1)
        if len(indices) != 1 or len(target) < 1 or not np.isfinite(target).all():
            raise ValueError("Invalid native gripper controller target")
        return float(target[indices[0]])

    def _recording_adapter(self):
        module = getattr(self, "_recording_state_adapter", None)
        if module is None:
            path = Path(__file__).resolve().with_name("trajectory_state.py")
            spec = importlib.util.spec_from_file_location("_agentic_maniskill_recording", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._recording_state_adapter = module
        return module

    def _cache_task_info(self, info, *, phase, success, reward, terminated, truncated):
        self.native_task_info = self._recording_adapter().native_info(info)
        self.task_receipt = {
            "availability": "available",
            "source": f"env.{phase}() returned info (cached; no reevaluation)",
            "data": {
                "info": self.native_task_info, "success": success, "reward": reward,
                "terminated": terminated, "truncated": truncated, "phase": phase,
            },
        }

    def recording_state(self):
        if not self.ready:
            raise RuntimeError("Reset an episode before reading recording state")
        return self._recording_adapter().read_recording_state(
            self.env.unwrapped, self.step_count, self.control_hz,
            task=getattr(self, "task_receipt", None),
        )

    def _privileged_observation(self):
        """Read the generic adapter only for an explicitly privileged episode."""
        collect = getattr(self, "_collect_privilege", None)
        if collect is None:
            # The simulator and task-suite workers run by path in independent
            # environments that need not have agentic_framework installed.
            path = Path(__file__).resolve().with_name("privilege.py")
            spec = importlib.util.spec_from_file_location("_agentic_maniskill_privilege", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            collect = self._collect_privilege = module.collect
        return collect(self.env.unwrapped)

    def _observation(self, observation):
        qpos = vector(self._extract_qpos(observation), 13, "DROID qpos13")
        # Call the official adapter, including its original float32 conversion.
        state8 = vector(self._state_adapter(qpos), 8, "official DROID state8")
        agent = self.env.unwrapped.agent
        tcp = agent.tcp.pose
        position = vector(tcp.p, 3, "fr3_link8 position")
        quat = vector(tcp.q, 4, "fr3_link8 quaternion wxyz")[[1, 2, 3, 0]]
        if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
            raise ValueError("Measured end-effector quaternion is not unit length")
        gripper_target = self._gripper_target()
        raw = {
            "qpos": qpos.tolist(), "qpos13": qpos.tolist(), "state8": state8.tolist(),
            "control_mode": self.control_mode, "gripper_target": gripper_target,
            "gripper_joint_lower": self.controller_info["gripper_joint_lower"],
            "gripper_joint_upper": self.controller_info["gripper_joint_upper"],
            "joint_names": self.controller_info["joint_names"],
        }
        extra = {"raw_droid": raw, "gripper_target": gripper_target,
                 "robot_base_quat": self.controller_info["robot_base_quaternion_xyzw"],
                 "robot_base_position": self.controller_info["robot_base_position"]}
        # Control observations never inspect scene actors or task predicates.
        if getattr(self, "observation_profile", "control") == "privileged":
            extra["privileged"] = self._privileged_observation()
        return {
            "images": {name: encode_camera(self._extract_camera(observation, name), name)
                       for name in CAMERAS},
            "state": {"eef_pos": position.tolist(), "eef_quat": quat.tolist(),
                      "joint_pos": qpos[:7].tolist(), "gripper_pos": qpos[7:8].tolist()},
            "sim_time": self.step_count / self.control_hz,
            "instruction": self.instruction,
            "extra": extra,
        }

    def reset(self, env_id=ENV_ID, seed=None, instruction=None):
        self.ready = False
        if seed is None:
            seed = 42
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32-1]")
        if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError("instruction must be a nonempty string")
        self._ensure_env(env_id)
        observation, info = self.env.reset(seed=seed)
        # Match the official evaluator's order exactly: env.reset, then external RNG seeds.
        self.torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.controller_info = self._inspect_controller()
        self.step_count = 0
        self.instruction = instruction or INSTRUCTION
        success = bool(scalar(info.get("success", False), "initial success"))
        self.finished = success
        self._cache_task_info(
            info, phase="reset", success=success if "success" in info else None, reward=None,
            terminated=None, truncated=None,
        )
        self.current_info = {
            "env_id": env_id, "instruction": self.instruction, "seed": seed,
            "initialization": "official env.reset(seed)", "settle_steps": 0,
            "success": success, "controller": self.controller_info,
            "max_episode_steps": self.max_episode_steps, "official_commit": OFFICIAL_COMMIT,
            "native_task_info": self.native_task_info,
        }
        encoded = self._observation(observation)
        self.ready = True
        return {"observation": encoded, "info": self.current_info}

    def _controller_audit(self):
        controllers = self.env.unwrapped.agent.controller.controllers
        result = {"control_mode": self.control_mode, "gripper_target": self._gripper_target()}
        arm = controllers["arm"]
        if getattr(arm, "_target_qpos", None) is not None:
            result["observed_arm_target_qpos"] = vector(arm._target_qpos, 7, "arm target").tolist()
        if getattr(arm, "_target_pose", None) is not None:
            result["observed_arm_target_pose_base_wxyz"] = vector(
                arm._target_pose.raw_pose, 7, "arm target pose",
            ).tolist()
        return result

    def step(self, action):
        if not self.ready:
            raise RuntimeError("Reset an episode before stepping")
        if self.finished:
            raise RuntimeError("Episode terminated; reset before stepping again")
        values = vector(action, MODES[self.control_mode], "native DROID action")
        # Exactly one native control step; no settling, repeats, gripper rewrite,
        # joint integration, controller switching, or hidden clipping in this bridge.
        observation, reward, terminated, truncated, info = self.env.step(values)
        self.step_count += 1
        success = bool(scalar(info.get("success", False), "success"))
        native_terminated = bool(scalar(terminated, "terminated"))
        terminated = native_terminated or success
        truncated = bool(scalar(truncated, "truncated"))
        self.finished = terminated or truncated
        if success:
            reason = "success"
        elif terminated:
            reason = "environment"
        elif truncated and self.step_count >= self.max_episode_steps:
            # The TimeLimit is the framework step budget: report it exactly like
            # the rollout's own budget end so every benchmark labels it max_steps.
            reason = "max_steps"
        else:
            reason = None
        reward = scalar(reward, "reward")
        self._cache_task_info(
            info, phase="step", success=success if "success" in info else None, reward=reward,
            terminated=native_terminated, truncated=truncated,
        )
        try:
            audit = self._controller_audit()
        except Exception as exc:
            # Diagnostics must not discard an already executed physical step.
            # Required observation/state failures still terminate the trial.
            audit = {"observed_controller_state_available": False,
                     "error": {"type": type(exc).__name__, "message": str(exc)}}
        return {
            "observation": self._observation(observation), "reward": reward,
            "terminated": terminated, "truncated": truncated, "termination_reason": reason,
            "info": {"success": success, "sim_steps": self.step_count, "settle_steps": 0,
                     "native_action": values.tolist(), "controller_audit": audit,
                     "native_task_info": self.native_task_info,
                     "n_in_box": scalar(info["n_in_box"], "n_in_box") if "n_in_box" in info else None,
                     "n_total": scalar(info["n_total"], "n_total") if "n_total" in info else None},
        }

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            self.ready = False
            self.finished = True
        return {"closed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--control-mode", choices=tuple(MODES), default="pd_ee_delta_pose")
    parser.add_argument("--control-hz", type=int, default=30)
    parser.add_argument("--sim-hz", type=int, default=150)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--shader-pack", default="rt-fast")
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--observation-profile", choices=("control", "privileged"),
                        default="control")
    args = parser.parse_args()
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
                    raise ValueError(f"Unknown RPC method: {method}")
                if simulator is None:
                    simulator = Simulator(**vars(args))
                result = getattr(simulator, method)(**request.get("params", {}))
                response = {"id": request["id"], "result": result}
            except Exception as exc:
                response = {"id": request.get("id"), "error": {
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }}
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
