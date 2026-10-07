"""Contact, stability and tool-use tasks with physical success gates.

All distances are metres. Privileged geometry is used only for evaluation,
never added to the policy observation. Objects are passive rigid bodies.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..assets import (
    add_parts,
    box,
    build_actor,
    capsule_between,
    export_actor,
    pose,
    quat_euler,
)
from ..base import SuiteTaskEnv
from .contact_geometry import (
    STAND_BASE_TO_TOP_AXIS,
    drawer_handle_parts,
    resting_origin,
    stand_parts,
    surface_vertices,
    tool_holder_parts,
    tool_parts,
)

WOOD = (0.55, 0.31, 0.14, 1.0)
DARK = (0.18, 0.22, 0.28, 1.0)
CREAM = (0.88, 0.85, 0.74, 1.0)


def initialize_free_actor(env, actor, xyz, euler, env_idx):
    """Small seeded in-plane variations, preserving valid initial support."""
    xyz = np.tile(np.asarray(xyz, dtype=float), (len(env_idx), 1))
    angles = np.tile(np.asarray(euler, dtype=float), (len(env_idx), 1))
    if env.randomize:
        xyz[:, :2] += env._batched_episode_rng[env_idx].uniform(-0.01, 0.01, size=2)
        angles[:, 2] += env._batched_episode_rng[env_idx].uniform(-math.pi / 18, math.pi / 18, size=1).reshape(-1)
    quats = np.asarray([quat_euler(*angle) for angle in angles])
    env.batch_pose(actor, xyz, quats, env_idx=env_idx)


def net_tool_opening_update(opening, previous, tool_contact, held, hand_contact,
                            recent, tool_credit, hand_credit, *, grace_steps=2):
    """Net causal displacement; closing erases credit and incidental touches add none."""
    increment = (opening - previous).clamp(min=0)
    closing = (previous - opening).clamp(min=0)
    recent = torch.where(tool_contact & held, grace_steps, (recent - 1).clamp(min=0))
    tool_credit = (tool_credit - closing).clamp(min=0) + torch.where(
        (recent > 0) & held & ~hand_contact, increment, torch.zeros_like(increment))
    hand_credit = (hand_credit - closing).clamp(min=0) + torch.where(
        hand_contact, increment, torch.zeros_like(increment))
    return recent, torch.minimum(tool_credit, opening.clamp(min=0)), hand_credit


def axis_tilt_deg(axis):
    """Angle in degrees between world-frame axis vectors ``(..., 3)`` and world +z.

    atan2 keeps the angle accurate over the whole 0 to 180 degree range,
    including the near-vertical angles where acos loses float32 precision.
    """
    horizontal = torch.linalg.vector_norm(axis[..., :2], dim=-1)
    return torch.rad2deg(torch.atan2(horizontal, axis[..., 2]))


def stand_sample_state(robot_contact_force, table_contact_force, upper_lowest_z,
                       base_down_tilt_deg, linear_speed, angular_speed, *, clearance=0.004,
                       contact_threshold=0.01, tilt_tolerance_deg=10.0,
                       still_linear_speed=0.015, still_angular_speed=0.16):
    """Task 4 standing predicate for one physics sample; the tabletop is at z = 0.

    The object stands when (a) no robot link touches it, (b) the lowest point of
    its upper part (all geometry above the gray base) is more than ``clearance``
    above the tabletop, (c) it touches the table, (d) its base-to-top axis is at
    most ``tilt_tolerance_deg`` from world +z, and (e) it is at rest: its linear
    speed is below ``still_linear_speed`` and its angular speed below
    ``still_angular_speed``.

    Resting flat on the gray base gives a tilt of about 0 degrees, lying on a side
    about 90 and upside down about 180, so (d) separates gray-base-down poses from
    every other pose while tolerating the small tilts of settling. Neither (d) nor
    the length of the run shows stability by itself: in the CPU PhysX simulation a
    narrow base released a few degrees off vertical can rock on its rim for more
    than 3 s, gaining energy and staying within the tilt tolerance, before it
    falls. (e) rejects such rocking, rolling, spinning and sliding at every
    sample, so a counted run is a run at rest on the gray base.
    """
    released = robot_contact_force < contact_threshold
    table_support = table_contact_force > contact_threshold
    upper_clear = upper_lowest_z > clearance
    base_down = base_down_tilt_deg <= tilt_tolerance_deg
    still = (linear_speed < still_linear_speed) & (angular_speed < still_angular_speed)
    return {"released": released, "table_support": table_support,
            "upper_body_clear": upper_clear, "base_down": base_down, "still": still,
            "standing": released & table_support & upper_clear & base_down & still}


def standing_run_update(samples, standing):
    """Count consecutive standing physics samples; any violating sample resets the run."""
    return torch.where(standing, samples + 1, torch.zeros_like(samples))


def standing_seconds(samples, sim_freq):
    """Time spanned by a run of consecutive standing samples taken at ``sim_freq``.

    A run of n samples spans (n - 1) / sim_freq seconds, so the first standing
    sample is not credited with a full physics interval it was not observed for.
    """
    return (samples - 1).clamp(min=0).float() / sim_freq


def stand_progress_components(tilt_deg, initial_tilt_deg, seconds, required_seconds):
    """Relative approach to gray-base-down vertical, plus the standing run.

    The signed base-to-top axis distinguishes upright from inverted. Orienting an
    object that started upright has no available orientation improvement; its
    orientation contribution stays zero instead of awarding reset-time credit.
    """
    orientation = torch.where(
        initial_tilt_deg > 1e-5,
        (initial_tilt_deg - tilt_deg) / initial_tilt_deg.clamp(min=1e-5),
        torch.zeros_like(tilt_deg),
    ).clamp(0, 1)
    duration = (seconds / required_seconds).clamp(0, 1)
    return orientation, duration


@register_env("DittoStandObject-v1", max_episode_steps=2000)
class StandObjectEnv(SuiteTaskEnv):
    TASK_ID = 3
    # The standing duration is tracked at physics frequency by this task, so the
    # base latch only confirms the control step in which it was completed.
    SUCCESS_HOLD_SECONDS = 0.0
    STANDING_SECONDS = 3.0
    UPPER_CLEARANCE_M = 0.004
    CONTACT_THRESHOLD_N = 0.01
    BASE_DOWN_TILT_TOLERANCE_DEG = 10.0
    STILL_LINEAR_SPEED_M_S = 0.015
    STILL_ANGULAR_SPEED_RAD_S = 0.16
    ORIENTATION_PROGRESS_WEIGHT = 0.5
    STANDING_TIME_PROGRESS_WEIGHT = 0.5
    HEIGHT = 0.151
    TASK_HISTORY_FIELDS = ("_standing_samples", "_longest_standing_samples",
                           "_standing_physics_steps", "_standing_peak_sample",
                           "_standing_goal_in_step", "_initial_base_down_tilt_deg",
                           "_minimum_base_down_tilt_deg", "_base_down_tilt_at_progress_peak",
                           "_standing_robot_contact_force_n", "_standing_table_contact_force_n")

    def build_task(self):
        parts, family = stand_parts(self.difficulty)
        # The gray base is the last part; everything else forms the upper part.
        self.upper_vertices = surface_vertices(parts[:-1])
        self._base_to_top_axis = torch.tensor(STAND_BASE_TO_TOP_AXIS, dtype=torch.float32,
                                              device=self.device)
        self.initial_quaternion = quat_euler(0, math.pi / 2, 0.12)
        self.initial_position = resting_origin(parts, self.initial_quaternion, (-0.47, -0.20))
        self.object = build_actor(self, "stand_object", parts,
                                  position=self.initial_position, quaternion=self.initial_quaternion,
                                  mass=0.20, friction=1.0)
        self.task_objects = self.failure_objects = [self.object]
        # A run of n samples spans (n - 1) physics intervals.
        self._required_standing_samples = round(self.STANDING_SECONDS * self.sim_freq) + 1
        self._standing_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._longest_standing_samples = torch.zeros_like(self._standing_samples)
        self._standing_physics_steps = torch.zeros_like(self._standing_samples)
        self._standing_peak_sample = torch.zeros_like(self._standing_samples)
        self._standing_goal_in_step = torch.zeros_like(self._standing_samples, dtype=torch.bool)
        self._initial_base_down_tilt_deg = torch.zeros(self.num_envs, device=self.device)
        self._minimum_base_down_tilt_deg = torch.zeros_like(self._initial_base_down_tilt_deg)
        self._base_down_tilt_at_progress_peak = torch.zeros_like(self._initial_base_down_tilt_deg)
        self._standing_robot_contact_force_n = torch.zeros_like(self._initial_base_down_tilt_deg)
        self._standing_table_contact_force_n = torch.zeros_like(self._initial_base_down_tilt_deg)
        seconds = f"{self.STANDING_SECONDS:g} s"
        clearance = f"{self.UPPER_CLEARANCE_M * 1000:g} mm"
        tilt = f"{self.BASE_DOWN_TILT_TOLERANCE_DEG:g} degrees"
        still = (f"linear speed below {self.STILL_LINEAR_SPEED_M_S:g} m/s and angular speed below "
                 f"{self.STILL_ANGULAR_SPEED_RAD_S:g} rad/s")
        self.task_spec = {
            "task_geometry_version": 5, "shape_family": family, "mass_kg": 0.20,
            "height_m": self.HEIGHT,
            "base_radius_m": 0.032 if self.difficulty == "easy" else 0.016,
            "base_thickness_m": 0.012, "stem_diameter_m": 0.024,
            "upper_body_dimensions_m": ([0.060, 0.060, 0.066] if self.difficulty in ("easy", "medium") else [0.100, 0.048, 0.066]),
            "upper_body_y_offset_m": -0.008 if self.difficulty == "xhard" else 0.0,
            "upper_part_definition": "all object geometry above the gray base (stem and upper body)",
            "upper_part_table_clearance_m": self.UPPER_CLEARANCE_M,
            "base_to_top_axis_local": list(STAND_BASE_TO_TOP_AXIS),
            "base_down_tilt_definition": (
                "angle between the object's base-to-top axis (from the gray base toward the top) "
                "and world +z: about 0 degrees resting on the gray base, about 90 lying on a side, "
                "about 180 upside down"),
            "base_down_tilt_tolerance_deg": self.BASE_DOWN_TILT_TOLERANCE_DEG,
            "still_definition": (
                f"object at rest: centre-of-mass {still}; rocking or rolling on the base rim, "
                "spinning and sliding are not at rest"),
            "still_linear_speed_threshold_m_s": self.STILL_LINEAR_SPEED_M_S,
            "still_angular_speed_threshold_rad_s": self.STILL_ANGULAR_SPEED_RAD_S,
            "contact_force_threshold_n": self.CONTACT_THRESHOLD_N,
            "contact_sampling": "Robot and table forces sampled each physics step and retained in snapshots",
            "hold_seconds": self.STANDING_SECONDS, "duration_sampling_hz": self.sim_freq,
            "required_consecutive_standing_samples": self._required_standing_samples,
            "progress_peak_sampling_hz": self.sim_freq,
            "success_standing_definition": (
                "no robot link touches the object, the object touches the table, the lowest "
                f"point of its upper part stays more than {clearance} above the tabletop, its "
                f"base-to-top axis stays within {tilt} of world +z (gray base down), and it is at "
                f"rest ({still}), at every {self.sim_freq} Hz physics sample spanning {seconds}"),
            "initial_state": "lying on its side",
            "goal": (f"standing gray base down and at rest on the table, base-to-top axis within "
                     f"{tilt} of vertical, untouched by the robot with its upper part clear of the "
                     f"tabletop for {seconds}"),
            "progress_definition": (
                "0.5 * clip((initial base-to-top tilt - current tilt) / initial tilt, 0, 1) "
                f"+ 0.5 * clip(continuous standing seconds / {self.STANDING_SECONDS:g}, 0, 1); "
                "orientation is measured relative to world +z and can improve while held; robot contact, "
                "upper-part table contact, loss of table contact, a base-to-top axis more than "
                f"{tilt} from world +z, or object motion at or above the at-rest speed thresholds "
                "resets only the duration; the combined score is sampled at one instant, not "
                "formed from independent component peaks"),
            "progress_component_weights": {"orientation": self.ORIENTATION_PROGRESS_WEIGHT,
                                           "released_standing_time": self.STANDING_TIME_PROGRESS_WEIGHT},
            "progress_orientation_raw_unit": "base-to-top tilt degrees from world +z",
            "progress_raw_unit": "continuous standing seconds",
            "mass_properties": self.asset_records[-1].get("mass_properties"),
        }

    def initialize_task(self, env_idx, options):
        initialize_free_actor(self, self.object, self.initial_position,
                              (0, math.pi / 2, 0.12), env_idx)
        self._standing_samples[env_idx] = 0
        self._longest_standing_samples[env_idx] = 0
        self._standing_physics_steps[env_idx] = 0
        self._standing_peak_sample[env_idx] = 0
        self._standing_goal_in_step[env_idx] = False
        rotation = self.object.pose.to_transformation_matrix()[env_idx, :3, :3]
        self._initial_base_down_tilt_deg[env_idx] = axis_tilt_deg(rotation @ self._base_to_top_axis)
        self._minimum_base_down_tilt_deg[env_idx] = self._initial_base_down_tilt_deg[env_idx]
        self._base_down_tilt_at_progress_peak[env_idx] = self._initial_base_down_tilt_deg[env_idx]
        self._standing_robot_contact_force_n[env_idx] = 0
        self._standing_table_contact_force_n[env_idx] = 0

    def _standing_state(self):
        # Native pose snapshots do not restore contact impulse buffers. Read the
        # saved last physical sample so a query after restore retains its duration.
        robot_force = self._standing_robot_contact_force_n.clone()
        table_force = self._standing_table_contact_force_n.clone()
        upper_lowest_z = self.world_points(self.object, self.upper_vertices)[..., 2].amin(dim=1)
        rotation = self.object.pose.to_transformation_matrix()[:, :3, :3]
        tilt = axis_tilt_deg(rotation @ self._base_to_top_axis)
        linear_speed = torch.linalg.vector_norm(self.object.linear_velocity, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.object.angular_velocity, dim=-1)
        state = stand_sample_state(robot_force, table_force, upper_lowest_z, tilt,
                                   linear_speed, angular_speed,
                                   clearance=self.UPPER_CLEARANCE_M,
                                   contact_threshold=self.CONTACT_THRESHOLD_N,
                                   tilt_tolerance_deg=self.BASE_DOWN_TILT_TOLERANCE_DEG,
                                   still_linear_speed=self.STILL_LINEAR_SPEED_M_S,
                                   still_angular_speed=self.STILL_ANGULAR_SPEED_RAD_S)
        return {**state, "robot_contact_force_n": robot_force,
                "table_contact_force_n": table_force, "upper_body_lowest_z_m": upper_lowest_z,
                "base_down_tilt_deg": tilt, "linear_speed_m_s": linear_speed,
                "angular_speed_rad_s": angular_speed}

    def _before_control_step(self):
        super()._before_control_step()
        self._standing_goal_in_step.fill_(False)

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._standing_robot_contact_force_n.copy_(self.robot_contact_force(self.object))
        self._standing_table_contact_force_n.copy_(self.contact_force(self.object, self.table_scene.table))
        state = self._standing_state()
        standing = state["standing"]
        self._minimum_base_down_tilt_deg = torch.minimum(
            self._minimum_base_down_tilt_deg, state["base_down_tilt_deg"])
        self._standing_physics_steps += 1
        self._standing_samples = standing_run_update(self._standing_samples, standing)
        self._longest_standing_samples = torch.maximum(self._longest_standing_samples, self._standing_samples)
        # The goal fires in the control step whose substep completes the duration,
        # even if a later substep of the same action breaks the standing state.
        self._standing_goal_in_step |= self._standing_samples >= self._required_standing_samples
        # Preserve peaks that occur and end within one 30 Hz control action.
        # The base control-step observer then sees the same or a lower value.
        if hasattr(self, "_progress_peak"):
            raw = standing_seconds(self._standing_samples, self.sim_freq)
            orientation, duration = stand_progress_components(
                state["base_down_tilt_deg"], self._initial_base_down_tilt_deg,
                raw, self.STANDING_SECONDS)
            current = (self.ORIENTATION_PROGRESS_WEIGHT * orientation
                       + self.STANDING_TIME_PROGRESS_WEIGHT * duration)
            current = self.relative_progress(current)
            improved = current > self._progress_peak
            self._progress_peak = torch.maximum(self._progress_peak, current)
            self._progress_raw_at_peak = torch.where(improved, raw, self._progress_raw_at_peak)
            self._progress_raw_max = torch.maximum(self._progress_raw_max, raw)
            self._progress_raw_min = torch.minimum(self._progress_raw_min, raw)
            self._standing_peak_sample = torch.where(improved, self._standing_physics_steps, self._standing_peak_sample)
            self._base_down_tilt_at_progress_peak = torch.where(
                improved, state["base_down_tilt_deg"], self._base_down_tilt_at_progress_peak)
            control_step = (self._standing_physics_steps + self._sim_steps_per_control - 1) // self._sim_steps_per_control
            self._progress_peak_step = torch.where(improved, control_step, self._progress_peak_step)

    def task_metrics(self):
        state = self._standing_state()
        run = torch.where(state["standing"], self._standing_samples,
                          torch.zeros_like(self._standing_samples))
        seconds = standing_seconds(run, self.sim_freq)
        orientation, duration = stand_progress_components(
            state["base_down_tilt_deg"], self._initial_base_down_tilt_deg,
            seconds, self.STANDING_SECONDS)
        goal = self._standing_goal_in_step | (run >= self._required_standing_samples)
        return {**state, "goal_reached": goal,
                "progress": (self.ORIENTATION_PROGRESS_WEIGHT * orientation
                             + self.STANDING_TIME_PROGRESS_WEIGHT * duration),
                "progress_raw": seconds,
                "orientation_progress": orientation,
                "standing_time_progress": duration,
                "initial_base_down_tilt_deg": self._initial_base_down_tilt_deg.clone(),
                "minimum_base_down_tilt_deg": self._minimum_base_down_tilt_deg.clone(),
                "base_down_tilt_at_progress_peak_deg": self._base_down_tilt_at_progress_peak.clone(),
                "standing_seconds": seconds,
                "longest_standing_seconds": standing_seconds(self._longest_standing_samples, self.sim_freq)}

    def evaluate(self):
        metrics = super().evaluate()
        # Elapsed duration of the success condition, matching task_spec.hold_seconds.
        metrics["hold_seconds"] = metrics["standing_seconds"]
        metrics["progress_peak_seconds"] = self._standing_peak_sample.float() / self.sim_freq
        metrics["progress_peak_physics_step"] = self._standing_peak_sample.clone()
        return metrics


@register_env("DittoToolDrawer-v1", max_episode_steps=2000)
class ToolDrawerEnv(SuiteTaskEnv):
    TASK_ID = 4
    SUCCESS_HOLD_SECONDS = 0.5
    DRAWER_ORIGIN = (-0.08, 0.0, 0.0)
    ROD_INITIAL_POSITION = (-0.45, -0.23, 0.011)
    TOOL_FRICTION = 2.0
    OPEN_DISTANCE = 0.105
    TASK_HISTORY_FIELDS = ("_previous_opening", "_recent_tool_contact", "_tool_opening",
                           "_hand_opening", "_hand_opened", "_rod_contact_seen",
                           "_tool_relative_pose", "_grasp_candidate_steps", "_tool_held",
                           "_tool_contact_peak", "_hand_contact_peak",
                           "_gripper_cabinet_contact_peak", "_robot_drawer_contact_peak")

    def build_task(self):
        cabinet_parts = [
            box((0.17, 0.16, 0.012), center=(0, 0, 0.012), color=DARK),
            box((0.15, 0.006, 0.056), center=(0, -0.144, 0.080), color=WOOD),
            box((0.15, 0.006, 0.056), center=(0, 0.144, 0.080), color=WOOD),
            box((0.006, 0.138, 0.056), center=(0.144, 0, 0.080), color=WOOD),
            box((0.15, 0.15, 0.008), center=(0, 0, 0.144), color=WOOD),
        ]
        eave_enabled = self.difficulty in ("hard", "xhard")
        if eave_enabled:
            # The fixed roof covers the handle's vertical approach, but leaves
            # a real front opening for an oblique rod or edge-on board.
            cabinet_parts.append(box((0.025, 0.065, 0.008),
                                     center=(-0.165, 0, 0.144), color=WOOD))
        self.cabinet_collision_parts = cabinet_parts
        rail_x, rail_z = (-0.167, 0.026)
        cabinet_parts.append(capsule_between((rail_x, -0.126, rail_z),
                                             (rail_x, 0.126, rail_z), 0.008, color=DARK))
        drawer_parts = [
            box((0.132, 0.125, 0.005), center=(0, 0, 0.039), color=CREAM),
            box((0.132, 0.005, 0.027), center=(0, -0.125, 0.073), color=CREAM),
            box((0.132, 0.005, 0.027), center=(0, 0.125, 0.073), color=CREAM),
            box((0.005, 0.120, 0.027), center=(0.127, 0, 0.073), color=CREAM),
            box((0.007, 0.136, 0.048), center=(-0.143, 0, 0.078), color=CREAM),
        ]
        handles, handle_spec = drawer_handle_parts(self.difficulty)
        drawer_parts.extend(handles)
        self.drawer_collision_parts = drawer_parts
        heavy_slide = self.difficulty != "easy"
        drawer_density = 700 if heavy_slide else 350
        slide_friction = 0.15 if heavy_slide else 0.08
        slide_damping = 4.0 if heavy_slide else 2.0
        builder = self.scene.create_articulation_builder()
        builder.initial_pose = pose(self.DRAWER_ORIGIN)
        cabinet = builder.create_link_builder()
        cabinet.set_name("cabinet")
        add_parts(cabinet, cabinet_parts, density=650, friction=0.3, asset_dir=self.asset_dir)
        self.asset_records.append(export_actor("cabinet", cabinet_parts, density=650,
                                              friction=0.3, static=True, root=self.asset_dir))
        drawer = builder.create_link_builder(cabinet)
        drawer.set_name("drawer")
        add_parts(drawer, drawer_parts, density=drawer_density, friction=1.0, asset_dir=self.asset_dir)
        self.asset_records.append(export_actor("drawer", drawer_parts, density=drawer_density,
                                              friction=1.0, static=False, root=self.asset_dir))
        drawer.set_joint_name("drawer_slide")
        joint_q = quat_euler(0, 0, math.pi)
        drawer.set_joint_properties(type="prismatic", limits=[[0, 0.145]],
                                    pose_in_parent=pose((0, 0, 0), joint_q),
                                    pose_in_child=pose((0, 0, 0), joint_q),
                                    friction=slide_friction, damping=slide_damping)
        self.drawer_articulation = builder.build("tool_cabinet", fix_root_link=True)
        self.cabinet_link = next(link for link in self.drawer_articulation.links
                                 if link.name.endswith("cabinet"))
        self.drawer_link = next(link for link in self.drawer_articulation.links
                                if link.name.endswith("drawer"))
        physical_drawer = self.drawer_link._objs[0]
        drawer_mass_properties = {
            "mass_kg": float(physical_drawer.mass),
            "center_of_mass_m": physical_drawer.cmass_local_pose.p.tolist(),
            "inertia_frame_quaternion_wxyz": physical_drawer.cmass_local_pose.q.tolist(),
            "principal_inertia_kg_m2": physical_drawer.inertia.tolist(),
            "source": "SAPIEN collision-density mass properties of the moving link",
        }
        self.drawer_articulation.active_joints[0].set_drive_properties(0, slide_damping)
        # ManiSkill's builder currently omits JointRecord.friction while building
        # the physical joint. Apply it explicitly so resistance is real.
        self.drawer_articulation.active_joints[0].set_friction(slide_friction)
        holder_parts, holder_spec = tool_holder_parts(self.difficulty)
        holder_position = (*self.ROD_INITIAL_POSITION[:2], 0.0)
        holder_builder = self.scene.create_actor_builder()
        holder_builder.initial_pose = pose(holder_position)
        add_parts(holder_builder, holder_parts, density=650, friction=holder_spec["friction"],
                  asset_dir=self.asset_dir)
        # Kinematic fixtures stay fixed under contact, while supporting paired
        # randomized GPU resets and native state-dictionary restoration.
        self.tool_holder = holder_builder.build_kinematic("tool_holder")
        self.asset_records.append(export_actor(
            "tool_holder", holder_parts, density=650, friction=holder_spec["friction"],
            static=True, root=self.asset_dir))
        parts, family = tool_parts(self.difficulty)
        # Show the board's broad face in the external camera; the cylindrical
        # rod is axially symmetric. Both tools still require free physical grasping.
        yaw = 0.10 + (math.pi / 2 if self.difficulty == "xhard" else 0.0)
        lean = math.radians(holder_spec["initial_lean_deg"])
        self.tool_initial_euler = (0.0, -math.pi / 2 + lean, yaw)
        self.tool_initial_quaternion = quat_euler(*self.tool_initial_euler)
        self.tool_center_offset_m = 0.0
        tool_xy = (holder_position[0] + self.tool_center_offset_m * math.cos(yaw),
                   holder_position[1] + self.tool_center_offset_m * math.sin(yaw))
        self.tool_initial_position = resting_origin(
            parts, self.tool_initial_quaternion, tool_xy,
            clearance=holder_spec["floor_thickness_m"] + 0.0005)
        self.tool_actor = build_actor(self, family, parts, position=self.tool_initial_position,
                                      quaternion=self.tool_initial_quaternion, mass=0.060,
                                      friction=self.TOOL_FRICTION)
        self.rod = self.tool_actor  # Historical snapshot/integration alias.
        self.task_objects = self.failure_objects = [self.tool_actor]
        for name in ("_previous_opening", "_tool_opening", "_hand_opening",
                     "_tool_contact_peak", "_hand_contact_peak",
                           "_gripper_cabinet_contact_peak", "_robot_drawer_contact_peak"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        for name in ("_recent_tool_contact", "_grasp_candidate_steps"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
        for name in ("_hand_opened", "_rod_contact_seen", "_tool_held"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        self._tool_relative_pose = torch.eye(4, device=self.device)[None].repeat(self.num_envs, 1, 1)
        self.task_spec = {
            "task_geometry_version": 5, "tool_family": family, "tool_mass_kg": 0.060,
            "tool_friction": self.TOOL_FRICTION,
            "tool_dimensions_m": ([0.190, 0.012, 0.012] if self.difficulty != "xhard" else [0.190, 0.044, 0.006]),
            "tool_presentation": "upright_in_tall_passive_holder_v3",
            "tool_initial_lean_deg": holder_spec["initial_lean_deg"],
            "tool_center_offset_from_holder_m": self.tool_center_offset_m,
            "tool_initial_position_m": list(self.tool_initial_position),
            "tool_initial_quaternion_wxyz": list(self.tool_initial_quaternion),
            "tool_holder": {**holder_spec, "nominal_position_m": list(holder_position),
                            "reset_xy_jitter_m": 0.010,
                            "reset_yaw_jitter_deg": 10.0,
                            "randomization": "when enabled, holder and tool share XY jitter; upright tool yaw varies about the centered axis"},
            "cabinet_height_m": 0.152, "drawer_front_height_m": 0.096,
            "drawer_density_kg_m3": drawer_density,
            "drawer_mass_properties": drawer_mass_properties,
            "drawer_joint_friction": slide_friction, "drawer_joint_damping": slide_damping,
            "eave": {"enabled": eave_enabled, "front_x_m": -0.190,
                     "back_x_m": -0.140, "half_width_m": 0.065,
                     "underside_z_m": 0.136, "top_z_m": 0.152},
            "drawer_origin_m": list(self.DRAWER_ORIGIN), "drawer_travel_m": 0.145, "success_opening_m": self.OPEN_DISTANCE,
            "fixture_randomization": {
                "version": "planar-fixture-v1", "enabled": self.randomize,
                "xy_half_range_m": 0.010, "yaw_half_range_deg": 10.0,
                "pivot_world_m": list(self.DRAWER_ORIGIN),
                "assembly": "cabinet, drawer, handle and slide move together",
                "tool_placement": "tool and holder keep their independent paired XY jitter and tool yaw jitter",
                "sampling": "independent uniform world X, world Y and world yaw at reset",
            },
            "required_tool_attributed_opening_m": 0.060, "direct_hand_opening_tolerance_m": 0.002,
            "causal_sampling_hz": self.sim_freq, "contact_grace_control_steps": 2,
            "drawer_actuation": "passive prismatic joint; no motor",
            "tool_use_gate": "any physically held tool part may open drawer; net opening over 2 mm during direct robot-drawer or gripper-fixed-cabinet contact immediately fails episode",
            "hold_seconds": self.SUCCESS_HOLD_SECONDS, "progress_raw_unit": "tool-attributed opening metres",
            "progress_definition": "clip(net_tool_attributed_opening / 0.105, 0, 1)",
            **handle_spec,
        }

    def _relative_tool_pose(self):
        tcp = self.agent.tcp.pose.to_transformation_matrix()
        return torch.linalg.inv(tcp) @ self.tool_actor.pose.to_transformation_matrix()

    def initialize_task(self, env_idx, options):
        self.drawer_articulation.set_qpos(torch.zeros((len(env_idx), 1), device=self.device))
        self.drawer_articulation.set_qvel(torch.zeros((len(env_idx), 1), device=self.device))
        holder_position = np.tile([*self.ROD_INITIAL_POSITION[:2], 0.0], (len(env_idx), 1))
        tool_angles = np.tile(self.tool_initial_euler, (len(env_idx), 1))
        if self.randomize:
            holder_position[:, :2] += self._batched_episode_rng[env_idx].uniform(-0.01, 0.01, size=2)
            tool_angles[:, 2] += self._batched_episode_rng[env_idx].uniform(
                -math.pi / 18, math.pi / 18, size=1).reshape(-1)
        tool_position = holder_position.copy()
        tool_position[:, 0] += self.tool_center_offset_m * np.cos(tool_angles[:, 2])
        tool_position[:, 1] += self.tool_center_offset_m * np.sin(tool_angles[:, 2])
        tool_position[:, 2] = self.tool_initial_position[2]
        self.batch_pose(self.tool_actor, tool_position,
                        np.asarray([quat_euler(*angles) for angles in tool_angles]), env_idx=env_idx)
        self.batch_pose(self.tool_holder, holder_position, env_idx=env_idx)
        drawer_positions = np.tile(np.asarray(self.DRAWER_ORIGIN), (len(env_idx), 1))
        drawer_yaw = np.zeros(len(env_idx))
        if self.randomize:
            # Sample after tool placement so its established seeded layout is
            # preserved, while the complete drawer assembly varies independently.
            jitter = self._batched_episode_rng[env_idx].uniform(-1, 1, size=3)
            drawer_positions[:, :2] += jitter[:, :2] * 0.010
            drawer_yaw = jitter[:, 2] * math.radians(10)
        drawer_quaternions = np.asarray([quat_euler(0, 0, yaw) for yaw in drawer_yaw])
        self.drawer_articulation.set_pose(Pose.create_from_pq(
            torch.as_tensor(drawer_positions, dtype=torch.float32, device=self.device),
            torch.as_tensor(drawer_quaternions, dtype=torch.float32, device=self.device)))
        for name in self.TASK_HISTORY_FIELDS:
            if name != "_tool_relative_pose":
                getattr(self, name)[env_idx] = 0
        self._tool_relative_pose[env_idx] = self._relative_tool_pose()[env_idx]

    def _before_control_step(self):
        super()._before_control_step()
        self._tool_contact_peak.zero_()
        self._hand_contact_peak.zero_()
        self._gripper_cabinet_contact_peak.zero_()
        self._robot_drawer_contact_peak.zero_()

    def _direct_hand_contact_force(self, actor):
        # The direct-hand rule concerns the physical gripper (palm, knuckles, fingers and
        # pads). Arm-link collisions with the fixed cabinet are not hand use.
        force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.links:
            name = link.name.split("/")[-1]
            if name.startswith(("left_", "right_", "robotiq_")):
                force = force + self.contact_force(link, actor)
        return force

    def _held_evidence(self):
        relative = self._relative_tool_pose()
        consistent = (torch.linalg.vector_norm(relative[:, :3, 3] - self._tool_relative_pose[:, :3, 3], dim=-1) < 0.015)
        consistent &= torch.linalg.matrix_norm(relative[:, :3, :3] - self._tool_relative_pose[:, :3, :3]) < 0.25
        two_jaws = (self.contact_force(self.agent.finger1_link, self.tool_actor) > 0.03)
        two_jaws &= self.contact_force(self.agent.finger2_link, self.tool_actor) > 0.03
        return relative, two_jaws & consistent

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        tool_force = self.contact_force(self.tool_actor, self.drawer_link)
        drawer_force = self.robot_contact_force(self.drawer_link)
        cabinet_gripper_force = self._direct_hand_contact_force(self.cabinet_link)
        hand_force = drawer_force + cabinet_gripper_force
        self._gripper_cabinet_contact_peak = torch.maximum(self._gripper_cabinet_contact_peak, cabinet_gripper_force)
        self._robot_drawer_contact_peak = torch.maximum(self._robot_drawer_contact_peak, drawer_force)
        self._tool_contact_peak = torch.maximum(self._tool_contact_peak, tool_force)
        self._hand_contact_peak = torch.maximum(self._hand_contact_peak, hand_force)
        _, candidate = self._held_evidence()
        self._tool_held = self.agent.is_grasping(self.tool_actor) | (
            candidate & (self._grasp_candidate_steps >= 1))
        opening = self.drawer_articulation.qpos[:, 0]
        tool_contact, hand_contact = tool_force > 0.03, hand_force > 0.05
        self._rod_contact_seen |= tool_contact & self._tool_held
        self._recent_tool_contact, self._tool_opening, self._hand_opening = net_tool_opening_update(
            opening, self._previous_opening, tool_contact, self._tool_held, hand_contact,
            self._recent_tool_contact, self._tool_opening, self._hand_opening,
            grace_steps=2 * self._sim_steps_per_control)
        self._hand_opened |= self._hand_opening > 0.002
        self._previous_opening = opening.clone()

    def update_task_state(self):
        relative, candidate = self._held_evidence()
        self._grasp_candidate_steps = torch.where(candidate, self._grasp_candidate_steps + 1, 0)
        self._tool_held = self.agent.is_grasping(self.tool_actor) | (self._grasp_candidate_steps >= 2)
        self._tool_relative_pose = relative.clone()

    def task_metrics(self):
        opening = self.drawer_articulation.qpos[:, 0]
        speed = self.drawer_articulation.qvel[:, 0].abs()
        tool_used = self._rod_contact_seen & (self._tool_opening >= 0.060)
        open_enough = opening >= self.OPEN_DISTANCE
        return {
            "goal_reached": open_enough & tool_used & ~self._hand_opened & (speed < 0.04),
            "progress": (self._tool_opening / self.OPEN_DISTANCE).clamp(0, 1),
            "progress_raw": self._tool_opening.clone(),
            "drawer_opening": opening, "drawer_speed": speed, "open_enough": open_enough,
            "tool_contact_seen": self._rod_contact_seen.clone(), "tool_held": self._tool_held.clone(),
            "tool_attributed_opening": self._tool_opening.clone(), "tool_used": tool_used,
            "direct_hand_opening": self._hand_opened.clone(),
            "hand_attributed_opening_m": self._hand_opening.clone(),
            "tool_contact_peak_force_n": self._tool_contact_peak.clone(),
            "hand_contact_peak_force_n": self._hand_contact_peak.clone(),
            "gripper_cabinet_contact_peak_force_n": self._gripper_cabinet_contact_peak.clone(),
            "robot_drawer_contact_peak_force_n": self._robot_drawer_contact_peak.clone(),
        }

    def evaluate(self):
        metrics = super().evaluate()
        metrics["fail"] = metrics["fail"] | self._hand_opened
        metrics["success"] = metrics["success"] & ~metrics["fail"]
        return metrics
