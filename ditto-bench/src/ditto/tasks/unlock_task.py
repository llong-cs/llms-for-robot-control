"""Disengage a real open ring from a fixed closed bridge without scripted locks."""
from __future__ import annotations

import math

import numpy as np
import torch
from mani_skill.utils.registration import register_env
from transforms3d.quaternions import mat2quat, quat2mat

from ..assets import build_actor, quat_euler
from ..base import SuiteTaskEnv
from .unlock_geometry import (
    BEAM_HALF,
    BRIDGE_ORIGIN,
    GAP_HALF,
    HALF_THICKNESS,
    box_vertices,
    bridge_parts,
    convex_prism_box_separation,
    filled_prism,
    mouth_alignment_error,
    mouth_exit_distance,
    ring_parts_for_level,
    vertices,
)


@register_env("DittoUnlockRing-v1", max_episode_steps=2000)
class UnlockRingEnv(SuiteTaskEnv):
    TASK_ID = "DittoUnlockRing-v1"
    INSTRUCTION = "Take the blue open ring off the fixed bridge."
    instruction = INSTRUCTION
    SUCCESS_HOLD_SECONDS = 0.0
    TASK_HISTORY_FIELDS = ("_initial_exit_distance_m",)

    def build_task(self):
        self.fixture_quaternion = quat_euler(0, 0, math.pi/4)
        self.fixture_rotation = quat2mat(self.fixture_quaternion)
        fixture = bridge_parts()
        self.bridge = build_actor(self, "fixed_bridge", fixture, position=BRIDGE_ORIGIN,
                                  quaternion=self.fixture_quaternion, static=not self.randomize,
                                  kinematic=self.randomize, friction=.5)
        self.fixture_centers = np.asarray([np.asarray(p.center)+BRIDGE_ORIGIN for p in fixture])
        self.fixture_halves = np.asarray([p.size for p in fixture])
        self.beam_vertices = box_vertices(BEAM_HALF, BRIDGE_ORIGIN)
        parts, geometry = ring_parts_for_level(self.difficulty)
        self.ring_vertices = vertices(parts)
        self.fill_points, self.fill_edges, self.fill_normals = filled_prism(geometry["filled_outline_xz"])
        self.initial_quaternion = tuple(mat2quat(self.fixture_rotation @ quat2mat(quat_euler(0, geometry["initial_pitch_rad"], 0))))
        initial_vertices = self.ring_vertices @ quat2mat(self.initial_quaternion).T
        self.initial_position = np.asarray([*BRIDGE_ORIGIN[:2], .001-initial_vertices[:,2].min()])
        self.mouth_center_local_m = geometry["mouth_center_local_m"]
        self.mouth_z = geometry["mouth_center_local_m"][2]
        self.outer_half_width_m = geometry["outer_envelope_m"][0]/2
        self.inner_half_height_m = geometry["inner_envelope_m"][2]/2
        self.ring = build_actor(self, "blue_open_ring", parts, position=self.initial_position,
                               quaternion=self.initial_quaternion, mass=.120, friction=1.0)
        self.task_objects = self.failure_objects = [self.ring]
        self.task_spec = {
            "task_geometry_version": 6, "object_family": geometry["family"],
            "ring_outer_envelope_m": geometry["outer_envelope_m"], "ring_inner_envelope_m": geometry["inner_envelope_m"],
            "ring_mass_kg": .120, "ring_friction": 1.0,
            "mouth_width_m": 2*GAP_HALF, "mouth_center_local_m": geometry["mouth_center_local_m"],
            "initial_ring_pitch_deg": math.degrees(geometry["initial_pitch_rad"]),
            "initial_ring_position_m": self.initial_position.tolist(),
            "bridge_origin_m": BRIDGE_ORIGIN.tolist(), "bridge_world_yaw_deg": 45.0, "bridge_beam_dimensions_m": (2*BEAM_HALF).tolist(),
            "bridge_beam_height_m": float(BRIDGE_ORIGIN[2]), "bridge_friction": .5,
            "fixture_randomization": {
                "enabled": self.randomize,
                "frame": "world XY translation and world Z rotation about bridge origin",
                "xy_half_range_m": .010, "yaw_half_range_deg": 10.0,
                "pivot_world_m": BRIDGE_ORIGIN.tolist(),
                "assembly": "fixed bridge and initially threaded ring move together",
                "coupled_actors": ["fixed_bridge", "blue_open_ring"],
                "preserved_relation": "initial ring pose relative to bridge; no along-beam ring jitter",
                "bridge_motion": "fixed during physics; kinematic pose included in snapshots",
            },
            "required_separation_m": .002,
            "goal": "Whole ring, including its filled interior, clear of every bridge solid by 2 mm; holding allowed.",
            "progress_quantity": "minimum 3D straight-line distance from mouth center to either accessible beam-side exit centerline",
            "progress_raw_unit": "metres to nearest exit",
            "progress_definition": "clip((initial_mouth_exit_distance_m - mouth_exit_distance_m) / initial_mouth_exit_distance_m, 0, 1)",
            "exit_centerline_local_x_m": [-float(BEAM_HALF[0]), float(BEAM_HALF[0])],
            "exit_centerline_local_y_half_length_m": float(BEAM_HALF[1])-.020-HALF_THICKNESS,
            "exit_centerline_local_z_m": 0.0,
            "progress_boundary": "Reset is zero. Reaching an exit centerline can score 1 while still threaded; complete removal is checked independently.",
            "mass_properties": self.asset_records[-1].get("mass_properties"),
        }

    def initialize_task(self, env_idx, options):
        perturbation = np.zeros((len(env_idx), 3))
        if self.randomize:
            perturbation = self._batched_episode_rng[env_idx].uniform(-1, 1, size=3)
        delta_xy = perturbation[:, :2] * .010
        delta_yaw = perturbation[:, 2] * math.radians(10)
        rotation = np.stack([quat2mat(quat_euler(0, 0, angle)) for angle in delta_yaw])
        bridge_position = np.tile(BRIDGE_ORIGIN, (len(env_idx), 1))
        bridge_position[:, :2] += delta_xy
        bridge_quaternion = np.stack([mat2quat(r @ self.fixture_rotation) for r in rotation])
        ring_position = bridge_position + np.einsum(
            "bij,j->bi", rotation, self.initial_position - BRIDGE_ORIGIN)
        ring_rotation = rotation @ quat2mat(self.initial_quaternion)
        ring_quaternion = np.stack([mat2quat(r) for r in ring_rotation])
        if self.randomize:
            self.batch_pose(self.bridge, bridge_position, bridge_quaternion, env_idx=env_idx)
        self.batch_pose(self.ring, ring_position, ring_quaternion, env_idx=env_idx)
        if not hasattr(self, "_initial_exit_distance_m") or len(self._initial_exit_distance_m) != self.num_envs:
            self._initial_exit_distance_m = torch.zeros(self.num_envs, device=self.device)
        rotation, position = self._ring_pose_in_fixture_frame()
        distance, _, _, _ = mouth_exit_distance(rotation, position,
                                               mouth_center_local_m=self.mouth_center_local_m)
        self._initial_exit_distance_m[env_idx] = distance[env_idx]

    def fixture_frame_pose(self, position, quaternion=(1,0,0,0), env_idx=None):
        """Map canonical diagnostics through the live fixture pose, per environment.

        A single selected environment returns one position and quaternion;
        multiple environments return batches. Never used in observations.
        """
        transform = self.bridge.pose.to_transformation_matrix().detach().cpu().numpy()
        if env_idx is not None:
            if isinstance(env_idx, torch.Tensor):
                env_idx = env_idx.detach().cpu().numpy()
            transform = transform[np.asarray(env_idx).reshape(-1)]
        point = transform[:, :3, 3] + np.einsum(
            "bij,bj->bi", transform[:, :3, :3],
            np.broadcast_to(np.asarray(position)-BRIDGE_ORIGIN, (len(transform), 3)))
        quaternion = np.broadcast_to(np.asarray(quaternion), (len(transform), 4))
        rotation = transform[:, :3, :3] @ np.stack([quat2mat(q) for q in quaternion])
        result_quaternion = np.stack([mat2quat(r) for r in rotation])
        if len(transform) == 1:
            return point[0], tuple(result_quaternion[0])
        return point, result_quaternion

    def _ring_pose_in_fixture_frame(self):
        """Read the current relative pose; never cache a sampled fixture frame."""
        transform = self.ring.pose.to_transformation_matrix()
        rotation, position = transform[:,:3,:3], transform[:,:3,3]
        fixture = self.bridge.pose.to_transformation_matrix()
        frame = fixture[:, :3, :3]
        origin = torch.as_tensor(BRIDGE_ORIGIN,dtype=position.dtype,device=position.device)
        rotation = frame.transpose(1, 2) @ rotation
        position = ((position-fixture[:, :3, 3])[:, None] @ frame).squeeze(1) + origin
        return rotation, position

    def task_metrics(self):
        rotation, position = self._ring_pose_in_fixture_frame()
        separation = convex_prism_box_separation(self.fill_points,self.fill_edges,self.fill_normals,
                    rotation,position,self.fixture_centers,self.fixture_halves)
        minimum = separation.amin(-1)
        fully_clear = minimum >= .002
        beam = torch.as_tensor(self.beam_vertices,dtype=position.dtype,device=position.device)
        error, section = mouth_alignment_error(beam,rotation,position,self.mouth_z,
            outer_half_width_m=self.outer_half_width_m, inner_half_height_m=self.inner_half_height_m)
        distance, exit_distances, nearest_exit, mouth = mouth_exit_distance(rotation, position,
            mouth_center_local_m=self.mouth_center_local_m)
        progress = ((self._initial_exit_distance_m-distance)
                    / self._initial_exit_distance_m.clamp(min=1e-7)).clamp(0, 1)
        return {"goal_reached": fully_clear, "ring_fully_removed": fully_clear,
                "fixture_fill_separation_m": minimum, "beam_fill_separation_m": separation[:,0],
                "beam_intersects_ring_plane": section,
                "mouth_alignment_error_m": error,
                "mouth_exit_distance_m": distance,
                "initial_mouth_exit_distance_m": self._initial_exit_distance_m.clone(),
                "left_exit_distance_m": exit_distances[:, 0],
                "right_exit_distance_m": exit_distances[:, 1],
                "nearest_exit_side": nearest_exit*2-1,
                "mouth_center_in_bridge_frame_m": mouth-torch.as_tensor(BRIDGE_ORIGIN, dtype=position.dtype, device=position.device),
                "progress_raw": distance, "progress": progress}
