"""Free rigid-body extraction, unusual grasping and shape-matched insertion."""
from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import torch
from mani_skill.utils.registration import register_env
from transforms3d.quaternions import mat2quat, quat2mat

from ..assets import (
    box,
    build_actor,
    convex_polyhedron,
    extruded_polygon,
    part_mesh,
    prism,
    quat_euler,
    sphere,
)
from ..base import SuiteTaskEnv
from .geometry_profiles import insertion_profile, profile_contains_polygon, submerged_vertices

FIXTURE_GRAY = (0.47, 0.49, 0.51, 1.0)
BOLT_GRAY = (0.20, 0.21, 0.22, 1.0)
CREAM = (0.76, 0.71, 0.59, 1.0)
BLUE = (0.16, 0.39, 0.70, 1.0)
GOLD = (0.95, 0.64, 0.16, 1.0)
YELLOW = (0.95, 0.78, 0.08, 1.0)


def _box_vertices(half_size, center=(0.0, 0.0, 0.0)):
    return np.asarray([[center[i] + sign[i]*half_size[i] for i in range(3)]
                       for sign in itertools.product((-1.0, 1.0), repeat=3)], dtype=np.float32)



def _clip_convex_above_plane(vertices, normal, offset):
    """Intersect a convex vertex hull with dot(normal, point) >= offset.

    Retained vertices and pairwise plane crossings contain every clipped edge
    vertex. Extra crossings inside the plane are removed by the convex hull.
    """
    vertices = np.asarray(vertices, dtype=float)
    normal = np.asarray(normal, dtype=float)
    distance = vertices @ normal - offset
    retained = list(vertices[distance >= -1e-10])
    for i, j in itertools.combinations(range(len(vertices)), 2):
        if distance[i] * distance[j] < 0:
            retained.append(vertices[i] + (vertices[j]-vertices[i]) *
                            distance[i]/(distance[i]-distance[j]))
    clipped = np.asarray(retained, dtype=float).reshape(-1, 3)
    if len(clipped) < 4 or np.linalg.matrix_rank(clipped-clipped[0]) < 3:
        raise ValueError("Clipping must leave a nonempty three-dimensional convex solid")
    return clipped


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _frame_points(points, origin, rotation):
    return (points - torch.as_tensor(origin, device=points.device, dtype=points.dtype)) @ torch.as_tensor(
        rotation, device=points.device, dtype=points.dtype)


def _inside_xy(points, half_size, tolerance=0.0):
    limits = torch.as_tensor(half_size, device=points.device, dtype=points.dtype)
    return (points[..., :2].abs() <= limits+tolerance).all(dim=-1).all(dim=-1)


def _vertices(parts):
    return np.concatenate([np.asarray(part_mesh(p).vertices, dtype=np.float32) for p in parts])


def _history(env, name, dtype=torch.float32):
    if not hasattr(env, name):
        setattr(env, name, torch.zeros(env.num_envs, device=env.device, dtype=dtype))
    return getattr(env, name)


def _planar_sweep_bounds(points, pivot, frame_yaw, translation_bound, yaw_bound):
    """Exact rectangle bounds for a convex hull under bounded planar motion.

    Return bounds in the fixed frame with yaw ``frame_yaw``. Each rotated
    coordinate is sinusoidal, so interval endpoints and in-range stationary
    angles cover the continuous yaw range, not only sampled reset poses.
    """
    rotation = _rotation_z(frame_yaw).astype(float)
    pivot = np.asarray(pivot, dtype=float)
    relative = (np.asarray(points, dtype=float)-pivot) @ rotation
    projected_pivot = pivot @ rotation
    lower, upper = [], []
    for component in range(2):
        coefficients = ((relative[:, 0], -relative[:, 1]) if component == 0
                        else (relative[:, 1], relative[:, 0]))
        values = []
        for a, b in zip(*coefficients):
            angles = [-yaw_bound, yaw_bound]
            stationary = math.atan2(b, a)
            angles += [stationary+k*math.pi for k in range(-2, 3)
                       if -yaw_bound <= stationary+k*math.pi <= yaw_bound]
            values.extend(a*math.cos(angle)+b*math.sin(angle) for angle in angles)
        offset = translation_bound*np.abs(rotation[:2, component]).sum()
        lower.append(projected_pivot[component]+min(values)-offset)
        upper.append(projected_pivot[component]+max(values)+offset)
    return np.asarray(lower), np.asarray(upper)


@register_env("DittoSlantedBoard-v1", max_episode_steps=2000)
class SlantedBoardEnv(SuiteTaskEnv):
    """Four rigid channels with optional planar fixture placement variation."""
    TASK_ID = "DittoSlantedBoard-v1"
    INSTRUCTION = "Pull the blue board completely out of the slot."
    instruction = INSTRUCTION
    SUCCESS_HOLD_SECONDS = 0.0
    TASK_HISTORY_FIELDS = ("_reset_rear_edge",)
    FIXTURE_XY_JITTER_M = .010
    FIXTURE_YAW_JITTER_RAD = math.radians(10)

    def _configure_channel(self):
        level = getattr(self, "difficulty", "easy")
        pitch, yaw, roll, bore_w, bore_h, friction = {
            "easy": (0, 0, 0, .066, .022, .50),
            "medium": (28, 0, 0, .066, .022, .50),
            "hard": (28, 30, 30, .066, .022, .50),
            "xhard": (28, 30, 30, .062, .018, .50),
        }[level]
        self.board_half_size = (.170, .030, .008)
        self.channel_angle = math.radians(pitch)
        self.channel_quaternion = quat_euler(math.radians(roll), -self.channel_angle, math.pi+math.radians(yaw))
        self.channel_rotation = quat2mat(self.channel_quaternion).astype(np.float32)
        self.channel_mouth, self.channel_back = .10, -.115
        mouth_reference_rotation = _rotation_z(math.pi) @ _rotation_y(-math.radians(28))
        self.mouth_world = np.asarray((-.26, .20, .15), dtype=np.float32) + mouth_reference_rotation @ np.asarray((.10, 0, 0))
        self.mouth_world[2] = .105
        self.channel_origin = self.mouth_world - self.channel_rotation @ np.asarray((.10, 0, 0))
        self.channel_outer_half = (.1075, .040, .018)
        # The rectangular pocket contains only the buried rear of the complete
        # housing. Its footprint includes the near-table crossing and a 3 mm
        # installation margin; the board travels within the intact bore.
        rear_cap = .006
        shell_back = self.channel_back-rear_cap
        shell = _box_vertices(((self.channel_mouth-shell_back)/2, .040, .018),
                              center=((self.channel_mouth+shell_back)/2, 0, 0))
        world_shell = shell @ self.channel_rotation.T + self.channel_origin
        self.table_pocket = None
        if world_shell[:, 2].min() < 0:
            buried = _clip_convex_above_plane(world_shell, (0, 0, -1), -.002)
            pocket_yaw = math.pi+math.radians(yaw)
            rotation = _rotation_z(pocket_yaw)
            planar = buried @ rotation
            low, high = planar[:, :2].min(axis=0)-.003, planar[:, :2].max(axis=0)+.003
            if self.randomize:
                low, high = _planar_sweep_bounds(
                    buried, self.mouth_world, pocket_yaw,
                    SlantedBoardEnv.FIXTURE_XY_JITTER_M,
                    SlantedBoardEnv.FIXTURE_YAW_JITTER_RAD)
                low, high = low-.003, high+.003
            center = rotation[:2, :2] @ ((low+high)/2)
            self.table_pocket = {
                "cutout_center_xy": center.tolist(), "cutout_size_xy": (high-low).tolist(),
                "cutout_yaw": pocket_yaw, "pocket_floor_z": -.040,
            }
        return pitch, yaw, roll, bore_w, bore_h, friction

    def _build_table_scene(self):
        SlantedBoardEnv._configure_channel(self)
        if self.table_pocket is None:
            return super()._build_table_scene()
        from ..pocket_table import PocketTableSceneBuilder
        table = PocketTableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise,
                                       **self.table_pocket)
        table.build()
        return table

    def build_task(self):
        pitch, yaw, roll, bore_w, bore_h, friction = SlantedBoardEnv._configure_channel(self)
        half_w, half_h = bore_w/2, bore_h/2
        xmid, hx, outer_w, outer_h = -.0075, .1075, .040, .018
        rear_cap_thickness = .006
        sleeve_parts = [
            box((hx, outer_w, (outer_h-half_h)/2), center=(xmid, 0, -(outer_h+half_h)/2), color=FIXTURE_GRAY),
            box((hx, (outer_w-half_w)/2, half_h), center=(xmid, -(outer_w+half_w)/2, 0), color=FIXTURE_GRAY),
            box((hx, (outer_w-half_w)/2, half_h), center=(xmid, (outer_w+half_w)/2, 0), color=FIXTURE_GRAY),
            box((rear_cap_thickness/2, outer_w, outer_h), center=(self.channel_back-rear_cap_thickness/2, 0, 0), color=FIXTURE_GRAY),
        ]
        board_parts = [box(self.board_half_size, color=BLUE)]
        sleeve_parts.append(box((hx, outer_w, (outer_h-half_h)/2), center=(xmid, 0, (outer_h+half_h)/2), color=FIXTURE_GRAY))
        # Preserve the full closed sleeve, including its below-table rear.
        # The table builder removes the matching physical and visual pocket.
        self.channel = build_actor(self, "inclined_channel", sleeve_parts,
                                   position=self.channel_origin, quaternion=self.channel_quaternion,
                                   static=not self.randomize, kinematic=self.randomize,
                                   friction=friction)
        # A single solid plinth meets the tabletop directly. Start its lower
        # plane below every underside corner, then clip at world Z=0. Clamping
        # individual negative top corners would create material above the true
        # underside and could obstruct the low rear of the rolled bore.
        base_xy = self.channel_origin[:2]
        support_top = np.asarray([
            self.channel_origin + self.channel_rotation @ np.asarray((x, y, -outer_h))
            for x in (self.channel_back-rear_cap_thickness, self.channel_mouth)
            for y in (-outer_w, outer_w)
        ])
        support_bottom = support_top.copy()
        support_bottom[:, 2] = min(float(support_top[:, 2].min()), 0.0)-.010
        support_vertices = _clip_convex_above_plane(
            np.vstack((support_bottom, support_top)), (0, 0, 1), 0.0)
        support_vertices[:, :2] -= base_xy
        support_parts = [convex_polyhedron(support_vertices, color=FIXTURE_GRAY)]
        support_dimensions = np.ptp(support_vertices, axis=0)
        sleeve_world_vertices = _vertices(sleeve_parts) @ self.channel_rotation.T + self.channel_origin
        rear_bore = _box_vertices((0, half_w, half_h), center=(self.channel_back, 0, 0))
        rear_bore_world = rear_bore @ self.channel_rotation.T + self.channel_origin
        self.channel_support = build_actor(self, "channel_support", support_parts,
                                           position=(*base_xy, 0), static=not self.randomize,
                                           kinematic=self.randomize)
        self.board_initial_x = .065
        # Center the reset cross-section so the tighter bore starts without penetration.
        self.board_start = self.channel_origin + self.channel_rotation @ np.asarray((self.board_initial_x, 0, 0))
        self.board_corners = _vertices(board_parts)
        self.board = build_actor(self, "blue_board", board_parts, position=self.board_start,
                                 quaternion=self.channel_quaternion, density=550, friction=1.0)
        self.task_objects = self.failure_objects = [self.board]
        self.extraction_clearance = .002
        self.task_spec = {
            "task": self.TASK_ID, "instruction": self.INSTRUCTION, "asset_version": 5,
            "board_dimensions_m": [.340, .060, .016],
            "profile": "rectangle",
            "key_dimensions_m": None,
            "key_profile_clearance_m": None,
            "target_color": "blue", "fixture_color": "grey", "fixture_attachment": "Fixed to table/world",
            "static_fixture_actors": ["inclined_channel", "channel_support"],
            "sleeve_outer_cross_section_m": [.080, .036], "fixture_base_dimensions_m": support_dimensions.tolist(),
            "support_structure": "Solid plinth above the tabletop supports the complete sleeve; rear housing sits in a real table pocket",
            "geometry_revision": "recessed-105mm-slot-v5.11",
            "sleeve_length_m": self.channel_mouth-self.channel_back,
            "rear_cap_thickness_m": rear_cap_thickness,
            "fixture_lowest_underside_m": float(sleeve_world_vertices[:, 2].min()),
            "nominal_unclipped_underside_min_m": float(support_top[:, 2].min()),
            "exterior_clip_world_z_m": None, "raised_base_plate_height_m": 0.0,
            "table_pocket_parameters": self.table_pocket,
            "table_aperture": getattr(getattr(self, "table_scene", None), "aperture_metadata", {}),
            "minimum_rear_bore_height_m": float(rear_bore_world[:, 2].min()),
            "exposed_board_length_m": .135, "inserted_board_length_m": .205,
            "board_initial_axis_position_m": self.board_initial_x,
            "sleeve_internal_cross_section_m": [bore_w, bore_h],
            "nominal_side_clearance_m": [half_w-self.board_half_size[1], half_h-self.board_half_size[2]],
            "board_initial_cross_section_offset_m": [0.0, 0.0],
            "sleeve_angle_deg": pitch, "sleeve_yaw_offset_deg": yaw, "sleeve_roll_deg": roll,
            "sleeve_friction": friction, "board_friction": 1.0,
            "mouth_local_x_m": self.channel_mouth, "mouth_world_m": self.mouth_world.tolist(),
            "required_full_clearance_m": self.extraction_clearance,
            "grasp_position": (self.channel_origin+self.channel_rotation@np.asarray((.185, 0, 0))).tolist(),
            "goal_position": (self.channel_origin+self.channel_rotation@np.asarray((.30, 0, 0))).tolist(),
            "success": "Full trailing edge clears the intended mouth by 2 mm; holding and later lowering are allowed.",
            "progress_quantity": "axial trailing-edge extraction displacement", "progress_raw_unit": "m",
            "randomize": self.randomize,
            "fixture_randomization": {
                "kind": "rigid_fixture_world_xy_yaw",
                "moving_actors": ["inclined_channel", "channel_support", "blue_board"],
                "pivot_world_m": self.mouth_world.tolist(),
                "xy_half_range_m": SlantedBoardEnv.FIXTURE_XY_JITTER_M,
                "yaw_half_range_deg": math.degrees(SlantedBoardEnv.FIXTURE_YAW_JITTER_RAD),
                "assembly": "Channel, solid plinth and inserted board share a rigid planar transform",
                "board_relative_pose": "Unchanged; 205 mm nominal insertion with no axial jitter",
                "table_pocket": "Fixed pocket covers the complete bounded fixture sweep" if self.randomize else "Nominal fixed pocket",
            },
        }

    def initialize_task(self, env_idx, options):
        positions = np.tile(self.board_start, (len(env_idx), 1))
        quaternions = np.tile(self.channel_quaternion, (len(env_idx), 1))
        if self.randomize:
            jitter = self._batched_episode_rng[env_idx].uniform(-1, 1, size=3)
            translation = np.column_stack((jitter[:, :2]*self.FIXTURE_XY_JITTER_M,
                                           np.zeros(len(env_idx))))
            yaw = jitter[:, 2]*self.FIXTURE_YAW_JITTER_RAD
            rotations = np.asarray([_rotation_z(angle) for angle in yaw])

            def moved(points):
                return (rotations @ (np.asarray(points)-self.mouth_world)
                        + self.mouth_world+translation)

            positions = moved(self.board_start)
            quaternions = np.asarray([mat2quat(rotation @ self.channel_rotation)
                                      for rotation in rotations])
            self.batch_pose(self.channel, moved(self.channel_origin), quaternions, env_idx=env_idx)
            self.batch_pose(self.channel_support, moved((*self.channel_origin[:2], 0)),
                            np.asarray([quat_euler(0, 0, angle) for angle in yaw]), env_idx=env_idx)
        self.batch_pose(self.board, positions, quaternions, env_idx=env_idx)
        rear = self._board_in_channel_frame()[..., 0].amin(dim=-1)
        _history(self, "_reset_rear_edge")[env_idx] = rear[env_idx]

    def _board_in_channel_frame(self):
        transform = self.channel.pose.to_transformation_matrix()
        corners = self.world_points(self.board, self.board_corners)
        return (corners-transform[:, None, :3, 3]) @ transform[:, :3, :3]

    def task_metrics(self):
        corners = self.world_points(self.board, self.board_corners)
        rear = self._board_in_channel_frame()[..., 0].amin(dim=-1)
        clear = rear >= self.channel_mouth + self.extraction_clearance
        raw = (rear-self._reset_rear_edge).clamp(min=0)
        target = self.channel_mouth+self.extraction_clearance-self._reset_rear_edge
        return {"goal_reached": clear, "board_fully_extracted": clear,
                "rear_edge_past_mouth_m": rear-self.channel_mouth, "board_height_m": self.board.pose.p[:, 2],
                "board_bottom_height_m": corners[..., 2].amin(dim=-1),
                "progress": (raw/target.clamp(min=1e-6)).clamp(0, 1), "progress_raw": raw}


def football_vertices(radius=.056):
    """A regular truncated icosahedron, resting on a true hexagonal face.

    Truncating each directed icosahedron edge at one third yields the 60
    vertices of a soccer-ball polyhedron: 12 pentagons and 20 hexagons. All
    collision and visual geometry uses these vertices without a round shell.
    """
    phi = (1+math.sqrt(5))/2
    ico = np.asarray([(0, a, b*phi) for a, b in itertools.product((-1, 1), repeat=2)] +
                     [(a, b*phi, 0) for a, b in itertools.product((-1, 1), repeat=2)] +
                     [(b*phi, 0, a) for a, b in itertools.product((-1, 1), repeat=2)], dtype=float)
    distance = np.linalg.norm(ico[:, None]-ico[None], axis=-1)
    edges = np.isclose(distance, 2)
    vertices = np.asarray([(2*ico[i]+ico[j])/3 for i, j in np.argwhere(edges)])
    vertices *= radius/np.linalg.norm(vertices[0])
    face = next(ids for ids in itertools.combinations(range(12), 3)
                if all(edges[i, j] for i, j in itertools.combinations(ids, 2)))
    normal = ico[list(face)].sum(axis=0)
    normal /= np.linalg.norm(normal)
    down = np.array((0., 0., -1.))
    axis = np.cross(normal, down)
    skew = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    rotation = np.eye(3)+skew+skew@skew/(1+np.dot(normal, down))
    return vertices@rotation.T


def rounded_football_vertices(radius=.035, half_length=.080):
    """Sample a convex circular-ogive football with a rounded central belly.

    Revolve r(y) = sqrt(a*a - y*y) - b about Y, where
    a = (L*L + R*R)/(2*R) and b = a - R. The analytic profile has
    continuous central curvature, a zero central slope, and pointed ends.
    Nine rings and two unique tips give a shared 218-vertex physical hull.
    """
    if (not np.isfinite([radius, half_length]).all()
            or radius <= 0 or half_length <= radius):
        raise ValueError("The football requires a positive radius and a longer half-length")
    circle_radius = (half_length*half_length+radius*radius)/(2*radius)
    radial_offset = circle_radius-radius
    fractions = np.asarray((-.825, -.60, -.35, -.125, 0., .125, .35, .60, .825))
    angles = np.arange(24)*2*np.pi/24
    rings = []
    for fraction in fractions:
        y = fraction*half_length
        r = np.sqrt(circle_radius*circle_radius-y*y)-radial_offset
        rings.append(np.column_stack((r*np.cos(angles), np.full(24, y), r*np.sin(angles))))
    tips = np.asarray(((0., -half_length, 0.), (0., half_length, 0.)))
    return np.concatenate([tips, *rings])


def odd_geometry_parts(difficulty):
    """Smooth grasp-affordance contrasts; all contact behavior is physical.

    The trapezoid has parallel short/long bases in the XY footprint and steep
    tapered legs. The football has a rounded ogive belly along Y. The final
    body is a football polyhedron with one joined triangular prism whose two
    exposed triangular ends are normal to Y. A real hexagonal face supports it. No grasp is accepted/rejected in software.
    """
    if difficulty == "easy":
        parts = [sphere(.024, center=(0, 0, .024), color=BLUE)]
    elif difficulty == "medium":
        # Compact near-square footprint with real sloping waist faces. The
        # parallel Y end faces stay vertical; X narrows by 10 mm per side from
        # bottom to top, so side squeezing also creates a downward component.
        vertices = ((-.026, -.024, 0), (.026, -.024, 0), (.018, .024, 0), (-.018, .024, 0),
                    (-.016, -.024, .034), (.016, -.024, .034), (.008, .024, .034), (-.008, .024, .034))
        parts = [convex_polyhedron(vertices, color=BLUE,
                                  face_friction=(((0, -1, 0), .10), ((0, 1, 0), .10)))]
    elif difficulty == "hard":
        parts = [convex_polyhedron(rounded_football_vertices(), center=(0, 0, .035), color=BLUE)]
    else:
        # A full soccer-ball polyhedron rests on a true hexagon. The small
        # triangular ear is embedded above its top face, with two parallel
        # exposed end faces and no extra support, pedestal, or hidden joint.
        ear_frame = _rotation_y(math.radians(90)) @ _rotation_z(-math.pi/2)
        extrusion_frame = quat2mat(quat_euler(math.pi/2, 0, math.pi/2))
        triangle = ((-.050, -.016), (-.050, .016), (-.068, 0))
        parts = [convex_polyhedron(football_vertices(), center=(0, 0, .056), color=BLUE),
                 prism(triangle, .014, center=(0, 0, .056),
                       quat=mat2quat(ear_frame@extrusion_frame), color=BLUE)]
    vertices = _vertices(parts)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    shift = np.asarray((-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, -lo[2]))
    return [replace(p, center=tuple(np.asarray(p.center)+shift)) for p in parts]


@register_env("DittoOddGeometry-v1", max_episode_steps=2000)
class OddGeometryEnv(SuiteTaskEnv):
    TASK_ID = "DittoOddGeometry-v1"
    INSTRUCTION = "Pick up the blue object and put it in the basket."
    instruction = INSTRUCTION
    SUCCESS_HOLD_SECONDS = 0.0
    TASK_HISTORY_FIELDS = ("_ever_lifted", "_table_contact_force")

    def build_task(self):
        self.object_start = np.asarray((-.40, .23, .001), dtype=np.float32)
        self.object_initial_yaw = math.radians(45) if self.difficulty == "medium" else 0.0
        self.object_initial_quaternion = quat_euler(0, 0, self.object_initial_yaw)
        self.basket_origin = np.asarray((-.40, 0, 0), dtype=np.float32)
        self.basket_inner_half = (.120, .100)
        self.basket_floor_z, self.basket_rim_z = .010, .080
        parts = odd_geometry_parts(self.difficulty)
        self.object_corners = _vertices(parts)
        self._object_collision_support = None
        friction = .10 if self.difficulty == "easy" else .05
        self.odd_object = build_actor(self, "odd_geometry", parts, position=self.object_start,
                                      quaternion=self.object_initial_quaternion, mass=.30, friction=friction)
        self.basket_floor = build_actor(self, "basket_floor", [box((.136, .116, .005), center=(0, 0, .005), color=CREAM)],
                                       position=self.basket_origin, static=True)
        wall_half_height = (self.basket_rim_z-self.basket_floor_z)/2
        wall_center_z = (self.basket_rim_z+self.basket_floor_z)/2
        wall_parts = [box((.008, .116, wall_half_height), center=(x, 0, wall_center_z), color=CREAM) for x in (-.128, .128)]
        wall_parts += [box((.120, .008, wall_half_height), center=(0, y, wall_center_z), color=CREAM) for y in (-.108, .108)]
        self.basket = build_actor(self, "basket", wall_parts, position=self.basket_origin, static=True)
        self.task_objects = self.failure_objects = [self.odd_object]
        self.task_spec = {
            "task": self.TASK_ID, "instruction": self.INSTRUCTION, "asset_version": 5,
            "shape_family": {"easy": "small_sphere", "medium": "smooth_trapezoid", "hard": "rounded_pointed_football", "xhard": "football_polyhedron_triangular_prism"}[self.difficulty],
            "object_dimensions_m": np.ptp(self.object_corners, axis=0).tolist(), "object_mass_kg": .30,
            "material": "uniform density over solid union", "object_friction": friction, "marked_grasp_post": False,
            "surface_friction": {
                "default_static_and_dynamic": friction,
                "overrides": ([{"local_outward_normal": [0, sign, 0], "static_and_dynamic": .10,
                                "surface": "parallel vertical trapezoidal grasp end"} for sign in (-1, 1)]
                              if self.difficulty == "medium" else []),
                "geometry": "Exact original exterior; face materials use nonoverlapping face-to-center convex cells.",
            },
            "grasp_rules": "No software restrictions: any physically effective grasp is accepted.",
            **({"geometry_revision": "rotated-taper-v5.6", "nominal_initial_yaw_degrees": 45.0,
                "random_yaw_jitter_degrees": 10.0} if self.difficulty == "medium" else
               {"geometry_revision": "rounded-long-football-v5.7",
                "grasp_tolerance_status": "New geometry; the physical grasp tolerance has not been measured."}
               if self.difficulty == "hard" else
               {"geometry_revision": "football-compact-ear-v5.1"} if self.difficulty == "xhard" else {}),
            "shape_parameters_m": {
                "easy": {"sphere_diameter": .048},
                "medium": {"bottom_parallel_bases": [.036, .052], "top_parallel_bases": [.016, .032],
                           "base_separation": .048, "thickness": .034, "waist_inset_per_side": .010,
                           "end_faces": "parallel vertical Y faces", "waist_faces": "X tapers inward as Z increases"},
                "hard": {"principal_dimensions": [.070, .160, .070],
                         "equatorial_radius": .035, "long_axis_half_length": .080,
                         "profile_curve": "circular_ogive",
                         "radial_profile": "r(y) = sqrt(a*a - y*y) - b; a = (L*L + R*R)/(2*R); b = a - R",
                         "profile_circle_radius": (.080**2+.035**2)/(2*.035),
                         "profile_radial_offset": (.080**2+.035**2)/(2*.035)-.035,
                         "ring_positions_y": [-.066, -.048, -.028, -.010, 0., .010, .028, .048, .066],
                         "azimuth_samples": 24, "vertex_count": 218},
                "xhard": {"body_circumscribed_diameter": .112, "body_vertex_count": 60,
                          "body_face_count": 32, "body_face_pattern": {"pentagons": 12, "hexagons": 20},
                          "support_face": "hexagon", "prism_axis_length": .028,
                          "ear_elevation_degrees": 90, "ear_radial_extension": .012, "ear_height_above_top_face": .0167623306,
                          "prism_end_normal_world": [0, 1, 0],
                          "prism_triangle_relative_to_body_center": [[-.050, -.016], [-.050, .016], [-.068, 0]]},
            }[self.difficulty],
            "basket_inner_dimensions_m": [.240, .200, self.basket_rim_z-self.basket_floor_z],
            "basket_floor_height_m": self.basket_floor_z, "basket_rim_height_m": self.basket_rim_z,
            "fixture_geometry_revision": "table-basket-80mm-v5.10",
            "goal_position": (self.basket_origin+np.asarray((0, 0, self.basket_floor_z))).tolist(),
            "success": "Complete XY projection in basket and released, on the first eligible control step. No lift history, bottom support, stillness or sustained hold is required. Upper protrusions are allowed.",
            "success_hold_seconds": self.SUCCESS_HOLD_SECONDS,
            "progress_quantity": "lowest physical collision point clearance above tabletop; zero during tabletop contact",
            "progress_raw_unit": "m", "progress_geometry": "cooked convex collision vertices and analytic native sphere",
            "progress_table_contact_threshold_n": .001,
            "progress_target_bottom_height_m": self.basket_rim_z+.005, "randomize": self.randomize,
        }

    def initialize_task(self, env_idx, options):
        positions = np.tile(self.object_start, (len(env_idx), 1))
        quaternions = np.tile(self.object_initial_quaternion, (len(env_idx), 1)).astype(np.float32)
        if self.randomize:
            jitter = self._batched_episode_rng[env_idx].uniform(-1, 1, size=3)
            positions[:, :2] += jitter[:, :2]*.010
            quaternions = np.asarray([quat_euler(0, 0, self.object_initial_yaw+x*math.radians(10))
                                      for x in jitter[:, 2]])
        self.batch_pose(self.odd_object, positions, quaternions, env_idx=env_idx)
        _history(self, "_ever_lifted", torch.bool)[env_idx] = False
        _history(self, "_table_contact_force")[env_idx] = 0

    def _physical_bottom_height(self):
        """Lowest point of the actual collision solid, independent of roll.

        Convex cooking can change mesh vertices, and a native sphere must use
        its analytic radius rather than its tessellated visual mesh. This cache
        is immutable geometry, not episode history or a pose-dependent value.
        """
        if self._object_collision_support is None:
            support = []
            for shape in self.odd_object._bodies[0].get_collision_shapes():
                local_pose = shape.local_pose
                if hasattr(shape, "vertices"):
                    vertices = np.asarray(shape.vertices)*np.asarray(shape.scale)
                    vertices = vertices @ quat2mat(local_pose.q).T + local_pose.p
                    support.append((vertices.astype(np.float32), 0.0))
                elif type(shape).__name__ == "PhysxCollisionShapeSphere":
                    support.append((np.asarray(local_pose.p, dtype=np.float32)[None], float(shape.radius)))
                else:
                    raise ValueError("Task 2 requires convex mesh or sphere collision support geometry")
            self._object_collision_support = tuple(support)
        heights = [self.world_points(self.odd_object, points)[..., 2].amin(dim=-1)-radius
                   for points, radius in self._object_collision_support]
        return torch.stack(heights).amin(dim=0)

    def update_task_state(self):
        # Contact impulses are not part of ManiSkill's native pose snapshot.
        # Retain the last real step's force for read-only, restorable scoring.
        self._table_contact_force.copy_(self.contact_force(self.odd_object, self.table_scene.table))
        self._ever_lifted |= self._physical_bottom_height() > .055

    def task_metrics(self):
        corners = self.world_points(self.odd_object, self.object_corners)
        local = corners-torch.as_tensor(self.basket_origin, device=self.device)
        contained = _inside_xy(local, self.basket_inner_half, tolerance=.001)
        bottom, top = self._physical_bottom_height(), local[..., 2].amax(dim=-1)
        bottom_height_valid = (bottom >= self.basket_floor_z-.004) & (bottom <= self.basket_floor_z+.012)
        floor_force = self.contact_force(self.odd_object, self.basket_floor)
        supported = (floor_force > .02) & bottom_height_valid
        released, still = self.is_released(self.odd_object), self.is_still(self.odd_object)
        table_force = self._table_contact_force.clone()
        table_contact = table_force > .001
        raw = torch.where(table_contact, torch.zeros_like(bottom), bottom.clamp(min=0))
        # Lift history, bottom support and speed remain diagnostic measurements.
        return {"goal_reached": contained & released,
                "whole_object_in_basket": contained, "object_lifted": self._ever_lifted.clone(),
                "object_released": released, "object_still": still, "object_bottom_height_m": bottom,
                "object_top_height_m": top, "basket_bottom_supported": supported, "basket_bottom_contact_force_n": floor_force,
                "object_table_contact_force_n": table_force, "object_table_contact": table_contact,
                "object_geometric_clearance_m": bottom.clamp(min=0),
                "progress": (raw/(self.basket_rim_z+.005)).clamp(0, 1), "progress_raw": raw}


@register_env("DittoPrecisionInsert-v1", max_episode_steps=2000)
class PrecisionInsertEnv(SuiteTaskEnv):
    TASK_ID = "DittoPrecisionInsert-v1"
    INSTRUCTION = "Insert the yellow object fully into the matching slot."
    instruction = INSTRUCTION

    def _build_table_scene(self):
        from ..recessed_table import RecessedTableSceneBuilder
        table = RecessedTableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        table.build()
        return table

    def build_task(self):
        self.profile_kind, self.peg_profile, self.socket_profile, clearance = insertion_profile(self.difficulty)
        self.peg_half_height = .035
        self.peg_half_size = (*np.max(np.abs(self.peg_profile), axis=0), self.peg_half_height)
        self.socket_inner_half = tuple(np.max(np.abs(self.socket_profile), axis=0))
        self.socket_floor_z, self.socket_depth = -.055, .055
        self.socket_top_z = self.socket_floor_z+self.socket_depth
        self.socket_origin = np.asarray((-.40, 0, 0), dtype=np.float32)
        self.socket_yaw = math.radians(-20)
        self.socket_rotation = _rotation_z(self.socket_yaw)
        self.socket_quaternion = quat_euler(0, 0, self.socket_yaw)
        self.peg_start = np.asarray((-.40, -.20, self.peg_half_height+.0005), dtype=np.float32)
        self.peg_start_quaternion = quat_euler(0, 0, math.radians(30))
        if self.profile_kind == "l":
            # Two disjoint boxes retain the re-entrant corner in collision.
            parts = [box((.024, .009, self.peg_half_height), center=(0, -.015, 0), color=YELLOW),
                     box((.012, .015, self.peg_half_height), center=(-.012, .009, 0), color=YELLOW)]
        else:
            parts = [prism(self.peg_profile, self.peg_half_height, color=YELLOW)]
        meshes = [part_mesh(p) for p in parts]
        self.peg_parts_geometry = [(np.asarray(m.vertices, dtype=np.float32), np.asarray(m.edges_unique, dtype=np.int64)) for m in meshes]
        self.peg_corners = _vertices(parts)
        self.peg = build_actor(self, "rectangular_peg", parts, position=self.peg_start,
                               quaternion=self.peg_start_quaternion, mass=.120, friction=1.0)
        self.socket_base_half, self.socket_body_half = (.090, .070, .003), (.090, .070, self.socket_depth/2)
        outer = [(-.090, -.070), (.090, -.070), (.090, .070), (-.090, .070)]
        socket_parts = [box(self.socket_base_half, center=(0, 0, self.socket_floor_z-.003), color=FIXTURE_GRAY)]
        socket_parts += extruded_polygon(outer, self.socket_depth/2, holes=(self.socket_profile,),
                                        center=(0, 0, self.socket_floor_z+self.socket_depth/2), color=FIXTURE_GRAY)
        self.socket = build_actor(self, "rectangular_socket", socket_parts, position=self.socket_origin,
                                  quaternion=self.socket_quaternion, static=True, friction=.3)
        self.task_objects = self.failure_objects = [self.peg]
        self.task_spec = {
            "task": self.TASK_ID, "instruction": self.INSTRUCTION, "asset_version": 5,
            "geometry_revision": "recessed-insertion-v5.8",
            "object_mass_kg": .120, "object_friction": 1.0, "socket_friction": .30,
            "profile": self.profile_kind, "peg_profile_xy_m": self.peg_profile.tolist(), "socket_profile_xy_m": self.socket_profile.tolist(),
            "peg_dimensions_m": [*np.ptp(self.peg_profile, axis=0).tolist(), 2*self.peg_half_height],
            "target_color": "yellow", "fixture_color": "grey", "fixture_attachment": "Fixed to table/world",
            "static_fixture_actors": ["rectangular_socket"], "fixture_base_dimensions_m": [.180, .140, .006],
            "socket_outer_dimensions_m": [.180, .140, self.socket_depth+.006],
            "socket_rim_height_m": self.socket_top_z, "socket_floor_height_m": self.socket_floor_z,
            "fully_inserted_protrusion_m": 2*self.peg_half_height-self.socket_depth,
            "table_aperture": getattr(getattr(self, "table_scene", None), "aperture_metadata", {}),
            "socket_inner_dimensions_m": [*np.ptp(self.socket_profile, axis=0).tolist(), self.socket_depth],
            "side_clearance_m": clearance, "grasp_position": (self.peg_start+np.asarray((0, 0, .020))).tolist(),
            "goal_position": (self.socket_origin+np.asarray((0, 0, self.socket_floor_z+self.peg_half_height))).tolist(),
            "goal_quaternion": list(self.socket_quaternion),
            "success": "Actual submerged profile fits the bore and reaches the floor, released and stable for 1 s; shape symmetry is accepted.",
            "progress_quantity": "valid below-rim insertion depth", "progress_raw_unit": "m",
            "progress_target_depth_m": self.socket_depth, "randomize": self.randomize,
        }

    def initialize_task(self, env_idx, options):
        positions = np.tile(self.peg_start, (len(env_idx), 1))
        quaternions = np.tile(self.peg_start_quaternion, (len(env_idx), 1))
        if self.randomize:
            jitter = self._batched_episode_rng[env_idx].uniform(-1, 1, size=3)
            positions[:, :2] += jitter[:, :2]*.010
            quaternions = np.asarray([quat_euler(0, 0, math.radians(30)+x*math.radians(10)) for x in jitter[:, 2]])
        self.batch_pose(self.peg, positions, quaternions, env_idx=env_idx)

    def task_metrics(self):
        local = _frame_points(self.world_points(self.peg, self.peg_corners), self.socket_origin, self.socket_rotation)
        bottom = local[..., 2].amin(dim=-1)
        # Clip each convex body part to the socket top before testing aperture
        # containment. The uninserted upper body can remain outside the bore.
        fits, footprints = [], []
        local_np = local.detach().cpu().numpy()
        for row in local_np:
            start, fit, footprint = 0, True, True
            for vertices, edges in self.peg_parts_geometry:
                transformed = row[start:start+len(vertices)]
                start += len(vertices)
                below = submerged_vertices(transformed, edges, self.socket_top_z)
                fit &= profile_contains_polygon(below[:, :2], self.socket_profile, self.profile_kind)
                footprint &= profile_contains_polygon(transformed[:, :2], self.socket_profile, self.profile_kind)
            fits.append(fit)
            footprints.append(footprint)
        valid = torch.tensor(fits, dtype=torch.bool, device=self.device) & (bottom >= self.socket_floor_z-.003)
        footprint = torch.tensor(footprints, dtype=torch.bool, device=self.device)
        seated = (bottom >= self.socket_floor_z-.003) & (bottom <= self.socket_floor_z+.004)
        released, still = self.is_released(self.peg), self.is_still(self.peg, linear=.012, angular=.15)
        depth = (self.socket_top_z-bottom).clamp(0, self.socket_depth)*valid
        return {"goal_reached": valid & seated & released & still,
                "peg_footprint_inside": footprint, "peg_seated": seated, "peg_aligned": footprint,
                "peg_inserted_geometry_valid": valid, "peg_released": released, "peg_still": still,
                "peg_bottom_height_m": bottom, "insertion_depth_m": depth,
                "progress": (depth/self.socket_depth).clamp(0, 1), "progress_raw": depth}
