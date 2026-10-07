"""Official table geometry with a real aperture for recessed task fixtures.

The actor pose, outer collision bounds, ground, contact material, and robot reset
come from the official table scene. Only the selected rectangular aperture is
removed. Original wood textures and UV coordinates survive the visual clipping.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sapien
import trimesh
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from transforms3d.euler import euler2mat, euler2quat

from .assets import (
    GENERATOR_ID,
    _write_atomic,
    extruded_polygon,
    pose,
    primitive_mesh_path,
)
from .paths import asset_root

_TABLE_HEIGHT = 0.9196429
_TABLE_POSITION = np.asarray((-.12, 0.0, -_TABLE_HEIGHT))
_TABLE_ROTATION = euler2mat(0, 0, math.pi / 2)
_TABLE_QUATERNION = euler2quat(0, 0, math.pi / 2)
_TABLE_LOCAL_HALF_XY = np.asarray((2.418 / 2, 1.209 / 2))
_TABLE_VISUAL_BOUNDS = np.asarray((
    (-.7402168, -1.2148621, -.91964257),
    (.4688596, 1.2030163, 3.5762787e-7),
))
_MODULE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _aperture_geometry(center_xy, size_xy, yaw):
    """Return collision prisms and transforms without creating a simulator."""
    center_xy = np.asarray(center_xy, dtype=float)
    size_xy = np.asarray(size_xy, dtype=float)
    if (center_xy.shape != (2,) or size_xy.shape != (2,)
            or not np.all(np.isfinite(center_xy))
            or not np.all(np.isfinite(size_xy)) or np.any(size_xy <= 0)
            or not math.isfinite(yaw)):
        raise ValueError("The table aperture needs finite XY coordinates, positive dimensions, and a finite yaw")
    half = size_xy / 2
    opening_local = np.asarray(((-half[0], -half[1]), (half[0], -half[1]),
                                (half[0], half[1]), (-half[0], half[1])))
    aperture_rotation = euler2mat(0, 0, yaw)
    opening_world = opening_local @ aperture_rotation[:2, :2].T + center_xy
    opening_table = (opening_world - _TABLE_POSITION[:2]) @ _TABLE_ROTATION[:2, :2]
    if not np.all(np.abs(opening_table) < _TABLE_LOCAL_HALF_XY - 1e-6):
        raise ValueError("The table aperture must lie strictly inside the original table collision bounds")
    hx, hy = _TABLE_LOCAL_HALF_XY
    outer = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    parts = tuple(extruded_polygon(
        outer, _TABLE_HEIGHT / 2, holes=(opening_table,),
        center=(0, 0, _TABLE_HEIGHT / 2), visual=False,
    ))
    table_to_world = _transform(_TABLE_ROTATION, _TABLE_POSITION)
    aperture_to_world = _transform(aperture_rotation, (*center_xy, 0.0))
    return parts, opening_world, table_to_world, aperture_to_world


def _clip_table_visual(source_path, size_xy, table_to_world, aperture_to_world):
    """Remove the aperture while preserving source materials and interpolated UVs."""
    source = trimesh.load_scene(source_path, process=False)
    result = trimesh.Scene()
    half_x, half_y = np.asarray(size_xy, dtype=float) / 2
    # Four disjoint regions cover the complement of the aperture rectangle.
    regions = (
        ([[-1, 0, 0]], [[-half_x, 0, 0]]),
        ([[1, 0, 0]], [[half_x, 0, 0]]),
        ([[1, 0, 0], [-1, 0, 0], [0, -1, 0]],
         [[-half_x, 0, 0], [half_x, 0, 0], [0, -half_y, 0]]),
        ([[1, 0, 0], [-1, 0, 0], [0, 1, 0]],
         [[-half_x, 0, 0], [half_x, 0, 0], [0, half_y, 0]]),
    )
    visual_to_table = _transform(euler2mat(0, 0, math.pi / 2) * 1.75, (0, 0, 0))
    source_to_aperture = np.linalg.inv(aperture_to_world) @ table_to_world @ visual_to_table
    aperture_to_table = np.linalg.inv(table_to_world) @ aperture_to_world
    for node in sorted(source.graph.nodes_geometry):
        node_transform, geometry_name = source.graph[node]
        mesh = source.geometry[geometry_name].copy()
        mesh.apply_transform(source_to_aperture @ node_transform)
        for index, (normals, origins) in enumerate(regions):
            clipped = trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=normals, plane_origin=origins, cap=False,
            )
            if not len(clipped.faces):
                continue
            clipped.apply_transform(aperture_to_table)
            name = f"{geometry_name}-aperture-region-{index}"
            result.add_geometry(clipped, node_name=name, geom_name=name)
    if not result.geometry:
        raise ValueError("Clipping removed the entire table visual")
    return result


def recessed_table_assets(root=None, *, cutout_center_xy=(-.40, 0.0),
                          cutout_size_xy=(.180, .140), cutout_yaw=-math.pi / 9):
    """Create cached meshes and return collision parts, GLB path, and metadata.

    Generated assets use the regular suite asset cache. Returned metadata omits
    cache paths so relocating the cache does not change geometry fingerprints.
    """
    root = Path(root) if root is not None else asset_root()
    parts, opening_world, table_to_world, aperture_to_world = _aperture_geometry(
        cutout_center_xy, cutout_size_xy, cutout_yaw,
    )
    source_path = Path(inspect.getfile(TableSceneBuilder)).parent / "assets" / "table.glb"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    specification = {
        "schema_version": 1,
        "generator": {"source_sha256": _MODULE_HASH, "asset_generator": GENERATOR_ID},
        "source_table_model_sha256": source_hash,
        "cutout_center_xy_m": list(map(float, cutout_center_xy)),
        "cutout_size_xy_m": list(map(float, cutout_size_xy)),
        "cutout_yaw_rad": float(cutout_yaw),
        "cutout_world_xy_m": opening_world.tolist(),
        "tabletop_height_m": 0.0,
        "table_actor_position_m": _TABLE_POSITION.tolist(),
        "table_actor_quaternion_wxyz": _TABLE_QUATERNION.tolist(),
        "original_collision_local_half_xy_m": _TABLE_LOCAL_HALF_XY.tolist(),
        "original_collision_height_m": _TABLE_HEIGHT,
        "contact_material": "Unchanged simulator default material",
        "collision_parts": [asdict(part) for part in parts],
    }
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:24]
    folder = root / "fixtures" / f"recessed-table-{key}"
    visual_path = folder / "visual.glb"
    if not visual_path.is_file():
        visual = _clip_table_visual(source_path, cutout_size_xy, table_to_world, aperture_to_world)
        _write_atomic(visual_path, visual.export(file_type="glb"))
    _write_atomic(folder / "spec.json", json.dumps(specification, indent=2))
    metadata = {
        **specification,
        "asset_id": key,
        "spec_sha256": hashlib.sha256((folder / "spec.json").read_bytes()).hexdigest(),
        "visual_sha256": hashlib.sha256(visual_path.read_bytes()).hexdigest(),
        "collision_part_count": len(parts),
    }
    return parts, visual_path, metadata


class RecessedTableSceneBuilder(TableSceneBuilder):
    """Use the official table scene with an actual through-aperture for an insert.

    The inserted task fixture supplies its own bore walls and floor. Table
    collisions retain the official simulator material. Inherited initialization
    preserves the original table and robot reset poses, including batched resets.
    """

    def __init__(self, env, robot_init_qpos_noise=0.02, *,
                 cutout_center_xy=(-.40, 0.0), cutout_size_xy=(.180, .140),
                 cutout_yaw=-math.pi / 9):
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.cutout_center_xy = tuple(cutout_center_xy)
        self.cutout_size_xy = tuple(cutout_size_xy)
        self.cutout_yaw = float(cutout_yaw)

    def build(self):
        root = getattr(self.env, "asset_dir", None)
        self.collision_parts, self.visual_path, self.aperture_metadata = recessed_table_assets(
            root, cutout_center_xy=self.cutout_center_xy,
            cutout_size_xy=self.cutout_size_xy, cutout_yaw=self.cutout_yaw,
        )
        builder = self.scene.create_actor_builder()
        for part in self.collision_parts:
            # Omitting material inherits the same default as the official table.
            builder.add_convex_collision_from_file(
                filename=str(primitive_mesh_path(part, root)),
                pose=pose(part.center, part.quat),
            )
        builder.add_visual_from_file(filename=str(self.visual_path))
        builder.initial_pose = sapien.Pose(p=_TABLE_POSITION, q=_TABLE_QUATERNION)
        self.table = builder.build_kinematic(name="table-workspace")
        dimensions = _TABLE_VISUAL_BOUNDS[1] - _TABLE_VISUAL_BOUNDS[0]
        self.table_length, self.table_width, self.table_height = dimensions
        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        self.ground = build_ground(self.scene, floor_width=floor_width, altitude=-self.table_height)
        self.scene_objects = [self.table, self.ground]
