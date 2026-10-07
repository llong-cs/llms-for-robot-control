"""Shared DROID environment, observation boundary, and time-based success latch."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from .assets import pose
from .difficulties import DIFFICULTIES, SCORING_VERSION, normalize_difficulty
from .paths import asset_root, molmo_root, robot_asset_root


def prepare_robot():
    source = molmo_root()
    if not (source / "sim_eval/robots/franka_droid.py").is_file():
        raise FileNotFoundError(f"Set MOLMOACT2_SOURCE_ROOT to the official checkout; missing {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from sim_eval.robots.franka_droid import FrankaDROID
    root = robot_asset_root()
    candidates = [root / "assets/franka_description/urdfs/fr3_robotiq.urdf",
                  root / "franka_description/urdfs/fr3_robotiq.urdf"]
    urdf = next((p for p in candidates if p.is_file()), None)
    if urdf is None:
        raise FileNotFoundError(f"Set DROID_ASSET_ROOT to downloaded official FR3/Robotiq assets: {root}")
    FrankaDROID.urdf_path = str(urdf)
    return FrankaDROID


FrankaDROID = prepare_robot()

# Reuse the reference camera definitions; do not maintain a second calibration.
from sim_eval.tasks.droid_tasks.droid_put_everything_in_box import (  # noqa: E402
    DroidPutEverythingInBoxEnv,
)


class SuiteTaskEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["franka_droid"]
    SUCCESS_HOLD_SECONDS = 1.0
    robot_base_p = tuple(DroidPutEverythingInBoxEnv.robot_base_p)
    ALIGNMENT_PROFILE = "molmoact2-official-box-v1"
    LAYOUT_VERSION = 5
    TASK_HISTORY_FIELDS = ()
    instruction = "Complete the physical manipulation task."
    _HISTORY_FIELDS = (
        "_elapsed_steps", "_hold_steps", "_ever_lifted", "_previous_opening",
        "_recent_tool_contact", "_tool_opening", "_hand_opened", "_rod_contact_seen",
        "_progress_peak", "_progress_raw_at_peak", "_progress_peak_step", "_progress_steps", "_progress_raw_min", "_progress_raw_max",
        "_progress_initial", "_progress_initial_raw",
    )

    def __init__(self, *args, robot_uids="franka_droid", randomize=False, asset_dir=None,
                 difficulty="easy", robot_init_qpos_noise=0.02,
                 reset_robot_qpos=True, **kwargs):
        from .catalog import get_task
        difficulty = normalize_difficulty(difficulty)
        self.instruction = get_task(self.TASK_ID).instruction
        self.randomize = bool(randomize)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reset_robot_qpos = reset_robot_qpos
        self.difficulty = difficulty
        self.asset_dir = Path(asset_dir) if asset_dir else asset_root()
        self.asset_records = []
        self.task_objects = []
        self.failure_objects = []
        self.task_spec = {}
        kwargs.setdefault("reconfiguration_freq", 1)
        kwargs.setdefault("reward_mode", "none")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # Official MolmoAct2 evaluation uses 150/30 Hz over ManiSkill's
        # unmodified SceneConfig (including contact/solver/material defaults).
        return SimConfig(sim_freq=150, control_freq=30)

    def _load_agent(self, options):
        super()._load_agent(options, pose(self.robot_base_p))

    def _build_table_scene(self):
        """Build the reference table; tasks may supply a physical aperture."""
        table = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        table.build()
        return table

    def _load_scene(self, options):
        self.asset_records = []
        self.task_objects = []
        self.failure_objects = []
        self.table_scene = self._build_table_scene()
        self.build_task()
        self.task_spec.update(
            alignment_profile=self.ALIGNMENT_PROFILE,
            layout_version=self.LAYOUT_VERSION,
            instruction=self.instruction,
            difficulty=self.difficulty,
            scoring_version=SCORING_VERSION,
            progress_reporting={
                "baseline": "Per-environment reset state, retained across snapshot restoration",
                "normalization": "clip((measure - initial_measure) / (1 - initial_measure), 0, 1)",
                "initially_saturated": "Zero improvement when the initial measure is already one",
                "primary": "Maximum relative progress reached during real simulation steps",
                "final": "Relative progress at the current state; may decrease",
            },
        )
        identity = {"task_spec": self.task_spec,
                    "assets": [record["spec_sha256"] for record in self.asset_records]}
        self.geometry_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.task_spec["geometry_fingerprint"] = self.geometry_fingerprint

    _default_sensor_configs = DroidPutEverythingInBoxEnv._default_sensor_configs
    _default_human_render_camera_configs = (
        DroidPutEverythingInBoxEnv._default_human_render_camera_configs
    )

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(pose(self.robot_base_p))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
            if not hasattr(self, "_hold_steps") or len(self._hold_steps) != self.num_envs:
                self._hold_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._hold_steps[env_idx] = 0
            self.initialize_task(env_idx, options)
            for name, dtype in (("_progress_peak", torch.float32),
                                ("_progress_initial", torch.float32),
                                ("_progress_initial_raw", torch.float32),
                                ("_progress_raw_at_peak", torch.float32),
                                ("_progress_raw_min", torch.float32),
                                ("_progress_raw_max", torch.float32),
                                ("_progress_peak_step", torch.int64),
                                ("_progress_steps", torch.int64)):
                if not hasattr(self, name) or len(getattr(self, name)) != self.num_envs:
                    setattr(self, name, torch.zeros(self.num_envs, dtype=dtype, device=self.device))
                getattr(self, name)[env_idx] = 0
            metrics = self.task_metrics()
            self._progress_initial[env_idx] = metrics["progress"][env_idx].clamp(0, 1)
            self._progress_initial_raw[env_idx] = metrics["progress_raw"][env_idx]
            self._progress_raw_at_peak[env_idx] = metrics["progress_raw"][env_idx]
            self._progress_raw_min[env_idx] = metrics["progress_raw"][env_idx]
            self._progress_raw_max[env_idx] = metrics["progress_raw"][env_idx]


    def batch_pose(self, actor, xyz, quaternion=(1, 0, 0, 0), env_idx=None):
        xyz = torch.as_tensor(xyz, device=self.device, dtype=torch.float32)
        quat = torch.as_tensor(quaternion, device=self.device, dtype=torch.float32)
        n = len(env_idx) if env_idx is not None else self.num_envs
        if xyz.ndim == 1:
            xyz = xyz.expand(n, -1)
        if quat.ndim == 1:
            quat = quat.expand(n, -1)
        actor.set_pose(Pose.create_from_pq(xyz, quat))
        actor.set_linear_velocity(torch.zeros((n, 3), device=self.device))
        actor.set_angular_velocity(torch.zeros((n, 3), device=self.device))

    def world_points(self, actor, localpoints):
        points = torch.as_tensor(localpoints, device=self.device, dtype=torch.float32)
        if points.ndim == 1:
            points = points[None, :]
        transform = actor.pose.to_transformation_matrix()
        return points[None] @ transform[:, :3, :3].transpose(1, 2) + transform[:, None, :3, 3]

    def contact_force_vector(self, a, b):
        return self.scene.get_pairwise_contact_forces(a, b)

    def contact_force(self, a, b):
        return torch.linalg.vector_norm(self.contact_force_vector(a, b), dim=-1)

    def robot_contact_force(self, actor):
        force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.links:
            force = force + self.contact_force(link, actor)
        return force

    def is_released(self, actor):
        return self.robot_contact_force(actor) < 0.01

    def is_still(self, actor, linear=0.025, angular=0.25):
        return ((torch.linalg.vector_norm(actor.linear_velocity, dim=-1) < linear)
                & (torch.linalg.vector_norm(actor.angular_velocity, dim=-1) < angular))

    def update_task_state(self):
        pass

    def relative_progress(self, progress):
        """Query improvement from this trial's reset state, without updating history.

        Tasks may already express a physical quantity relative to their reset
        state (for example distance reduction or the two standing components).
        Their initial measure is zero, so this common normalization is an identity.
        """
        current = progress.to(dtype=self._progress_initial.dtype).clamp(0, 1)
        remaining = 1 - self._progress_initial
        return torch.where(
            remaining > 1e-7,
            ((current - self._progress_initial) / remaining.clamp(min=1e-7)).clamp(0, 1),
            torch.zeros_like(current),
        )

    def _after_control_step(self):
        # The native hook runs before the normal GPU fetch; read the current state.
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self.update_task_state()
        metrics = self.task_metrics()
        self._progress_steps += 1
        self._progress_raw_min = torch.minimum(self._progress_raw_min, metrics["progress_raw"])
        self._progress_raw_max = torch.maximum(self._progress_raw_max, metrics["progress_raw"])
        current = self.relative_progress(metrics["progress"])
        improved = current > self._progress_peak
        self._progress_peak = torch.maximum(self._progress_peak, current)
        self._progress_peak_step = torch.where(improved, self._progress_steps, self._progress_peak_step)
        self._progress_raw_at_peak = torch.where(improved, metrics["progress_raw"], self._progress_raw_at_peak)
        condition = metrics["goal_reached"].to(dtype=torch.bool)
        self._hold_steps = torch.where(condition, self._hold_steps + 1,
                                       torch.zeros_like(self._hold_steps))

    def evaluate(self):
        metrics = dict(self.task_metrics())
        measure = metrics["progress"].clone()
        current = self.relative_progress(measure)
        required = max(1, math.ceil(self.SUCCESS_HOLD_SECONDS * self.control_freq))
        fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for actor in self.failure_objects:
            fail |= actor.pose.p[:, 2] < -0.08
        metrics.update(success=(self._hold_steps >= required) & ~fail,
                       fail=fail,
                       hold_seconds=self._hold_steps.float() / self.control_freq,
                       progress_score=self._progress_peak.clone(),
                       progress=current,
                       final_progress_score=current.clone(),
                       progress_measure=measure,
                       initial_progress_measure=self._progress_initial.clone(),
                       initial_progress_raw=self._progress_initial_raw.clone(),
                       progress_peak_step=self._progress_peak_step.clone(),
                       progress_peak_seconds=self._progress_peak_step.float() / self.control_freq,
                       peak_progress_raw=self._progress_raw_at_peak.clone(),
                       progress_raw_min=self._progress_raw_min.clone(),
                       progress_raw_max=self._progress_raw_max.clone())
        return metrics

    def get_privileged_task_metadata(self):
        """Optional declared sections shared by all tasks; never advance physics."""
        from .privilege import task_privilege_metadata
        return task_privilege_metadata(self.task_spec)

    def _history_fields(self):
        return tuple(dict.fromkeys(name for name in (*self._HISTORY_FIELDS, *self.TASK_HISTORY_FIELDS) if hasattr(self, name)))

    def get_state_dict(self):
        """Snapshot physics and task history, with independent tensor storage.

        Preserve this dictionary for restoration. ManiSkill's flat state
        reconstruction drops custom task fields and is deliberately unsupported.
        Controller/physics behavior otherwise follows ManiSkill's own snapshot
        contract; policies still supply the next control action after restoring.
        """
        from .catalog import get_task
        state = copy.deepcopy(super().get_state_dict())
        task_id = get_task(self.TASK_ID).task_id
        state["ditto_task"] = {
            "schema_version": torch.full((self.num_envs,), 3, dtype=torch.int64, device=self.device),
            "task_id": torch.full((self.num_envs,), task_id, dtype=torch.int64, device=self.device),
            "control_freq": torch.full((self.num_envs,), self.control_freq, dtype=torch.int64, device=self.device),
            "sim_freq": torch.full((self.num_envs,), self.sim_freq, dtype=torch.int64, device=self.device),
            "difficulty_id": torch.full((self.num_envs,), DIFFICULTIES.index(self.difficulty), dtype=torch.int64, device=self.device),
            "geometry_id": torch.full((self.num_envs,), int(self.geometry_fingerprint[:15], 16), dtype=torch.int64, device=self.device),
            "history": {name: getattr(self, name).detach().clone() for name in self._history_fields()},
        }
        return state

    def set_state_dict(self, state, env_idx=None):
        """Restore complete task dictionaries; reject lossy physics-only states.

        For a partial restore, each saved tensor must contain exactly the rows
        selected by env_idx, matching ManiSkill's native state-dictionary API.
        All custom fields are validated before any physical state is changed.
        """
        from .catalog import get_task
        if not isinstance(state, dict) or not isinstance(state.get("ditto_task"), dict):
            raise ValueError("State restore requires the complete dictionary from get_state_dict(), including ditto_task history")
        indices = (torch.arange(self.num_envs, device=self.device) if env_idx is None
                   else torch.as_tensor(env_idx, device=self.device))
        if (indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)
                or not len(indices) or bool(((indices < 0) | (indices >= self.num_envs)).any())
                or len(torch.unique(indices)) != len(indices)):
            raise ValueError("env_idx must contain distinct in-range integer environment indices")
        saved = state["ditto_task"]
        expected_keys = {"schema_version", "task_id", "control_freq", "sim_freq", "history", "difficulty_id", "geometry_id"}
        if set(saved) != expected_keys:
            raise ValueError("Incomplete or unsupported ditto_task state schema")
        identity = {"schema_version": 3, "task_id": get_task(self.TASK_ID).task_id,
                    "control_freq": self.control_freq, "sim_freq": self.sim_freq,
                    "difficulty_id": DIFFICULTIES.index(self.difficulty),
                    "geometry_id": int(self.geometry_fingerprint[:15], 16)}
        for name, expected in identity.items():
            value = torch.as_tensor(saved[name], device=self.device)
            if (value.dtype != torch.int64 or value.shape != (len(indices),)
                    or not bool((value == expected).all())):
                raise ValueError(f"Saved task state has incompatible {name}")
        history = saved["history"]
        if not isinstance(history, dict) or set(history) != set(self._history_fields()):
            raise ValueError("Saved task history must include every counter for this task")
        prepared = {}
        for name in self._history_fields():
            target = getattr(self, name)
            value = torch.as_tensor(history[name], device=self.device)
            if value.shape != target[indices].shape or value.dtype != target.dtype:
                raise ValueError(f"Saved task counter {name} has incompatible shape or dtype")
            if (not bool(torch.isfinite(value).all())
                    or (value.dtype != torch.bool and not value.is_floating_point() and bool((value < 0).any()))):
                raise ValueError(f"Saved task counter {name} contains invalid values")
            prepared[name] = value.detach().clone()
        physics_state = {key: value for key, value in state.items() if key != "ditto_task"}
        expected_physics = super().get_state_dict()
        if set(physics_state) != set(expected_physics):
            raise ValueError("Saved state must include complete physics as well as task history")
        for group in ("actors", "articulations"):
            if group in expected_physics and (
                not isinstance(physics_state[group], dict)
                or set(physics_state[group]) != set(expected_physics[group])
            ):
                raise ValueError(f"Saved state has an incompatible {group} inventory")
        super().set_state_dict(physics_state, env_idx=env_idx)
        for name, value in prepared.items():
            getattr(self, name)[indices] = value

    def get_state(self):
        raise ValueError("Flat task state loses multidimensional task history; use get_state_dict()")

    def set_state(self, state, env_idx=None):
        raise ValueError("Flat state restore is unsupported because it loses task history; use get_state_dict()/set_state_dict()")

    def _get_obs_extra(self, info):
        # No task geometry, object poses, grasp sites, or oracle diagnostics in policy observations.
        return {"tcp_pose": self.agent.tcp.pose.raw_pose}

    def build_task(self):
        raise NotImplementedError

    def initialize_task(self, env_idx, options):
        raise NotImplementedError

    def task_metrics(self):
        raise NotImplementedError
