"""Official textured table with a finite-depth pocket for buried fixtures.

The table remains one actor with the official pose and default contact material.
Only the specified volume between the pocket floor and tabletop is removed.
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
from transforms3d.euler import euler2mat

from .assets import GENERATOR_ID, _write_atomic, pose, primitive_mesh_path, prism
from .paths import asset_root
from .recessed_table import (
    _MODULE_HASH as _APERTURE_MODULE_HASH,
)
from .recessed_table import (
    _TABLE_HEIGHT,
    _TABLE_LOCAL_HALF_XY,
    _TABLE_POSITION,
    _TABLE_QUATERNION,
    _TABLE_ROTATION,
    _TABLE_VISUAL_BOUNDS,
    _aperture_geometry,
    _clip_table_visual,
    _transform,
)

_MODULE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _pocket_geometry(center_xy, size_xy, yaw, pocket_floor_z=-.04):
    """Return disjoint collision parts and transforms without a simulator.

    The floor is a world Z coordinate. Original full-height prisms surround the
    opening; one prism fills its bottom up to the floor, preventing a through-hole.
    """
    floor_z = float(pocket_floor_z)
    if not math.isfinite(floor_z) or not -_TABLE_HEIGHT < floor_z < 0:
        raise ValueError("The pocket floor must be finite and strictly inside the table height")
    parts, opening_world, table_to_world, aperture_to_world = _aperture_geometry(
        center_xy, size_xy, yaw,
    )
    opening_table = (opening_world - _TABLE_POSITION[:2]) @ _TABLE_ROTATION[:2, :2]
    fill_height = _TABLE_HEIGHT + floor_z
    bottom_fill = prism(
        opening_table, fill_height / 2,
        center=(0, 0, fill_height / 2), visual=False,
    )
    return (*parts, bottom_fill), opening_world, table_to_world, aperture_to_world


def _pocket_visual(source_path, size_xy, floor_z, table_to_world, aperture_to_world):
    """Clip only the pocket and add wood-lined walls and a matching floor."""
    result = _clip_table_visual(source_path, size_xy, table_to_world, aperture_to_world)
    source = trimesh.load_scene(source_path, process=False)
    half_x, half_y = np.asarray(size_xy, dtype=float) / 2
    visual_to_table = _transform(euler2mat(0, 0, math.pi / 2) * 1.75, (0, 0, 0))
    source_to_aperture = np.linalg.inv(aperture_to_world) @ table_to_world @ visual_to_table
    aperture_to_table = np.linalg.inv(table_to_world) @ aperture_to_world
    material = None
    uv_from_xy = None
    for node in sorted(source.graph.nodes_geometry):
        node_transform, geometry_name = source.graph[node]
        mesh = source.geometry[geometry_name].copy()
        mesh.apply_transform(source_to_aperture @ node_transform)
        if mesh.visual.material.name == "TableTop":
            material = mesh.visual.material.copy()
            # Interpolate the source top's planar UV chart on the pocket floor.
            top_faces = mesh.faces[mesh.face_normals[:, 2] > .99]
            top_indices = np.unique(top_faces)
            coordinates = np.column_stack((mesh.vertices[top_indices, :2],
                                           np.ones(len(top_indices))))
            uv_from_xy = np.linalg.lstsq(coordinates, mesh.visual.uv[top_indices], rcond=None)[0]
        # The through-aperture helper removes an entire column. Restore original
        # surfaces below the finite floor, including any leg crossing that column.
        clipped = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=[[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]],
            plane_origin=[[-half_x, 0, 0], [half_x, 0, 0], [0, -half_y, 0],
                          [0, half_y, 0], [0, 0, floor_z]],
            cap=False,
        )
        if len(clipped.faces):
            clipped.apply_transform(aperture_to_table)
            name = f"{geometry_name}-below-pocket"
            result.add_geometry(clipped, node_name=name, geom_name=name)
    if material is None or uv_from_xy is None:
        raise ValueError("The official table model must supply a textured TableTop material")

    corners = np.asarray(((-half_x, -half_y), (half_x, -half_y),
                          (half_x, half_y), (-half_x, half_y)))
    floor_vertices = np.column_stack((corners, np.full(4, floor_z)))
    floor_uv = np.column_stack((corners, np.ones(4))) @ uv_from_xy
    floor = trimesh.Trimesh(vertices=floor_vertices, faces=((0, 1, 2), (0, 2, 3)),
                            process=False)
    floor.visual = trimesh.visual.texture.TextureVisuals(uv=floor_uv, material=material)
    floor.apply_transform(aperture_to_table)
    result.add_geometry(floor, node_name="pocket-floor", geom_name="pocket-floor")

    uv_scale = float(np.linalg.norm(uv_from_xy[:2], axis=1).mean())
    for index, first in enumerate(corners):
        following = corners[(index + 1) % 4]
        vertices = np.asarray(((*first, floor_z), (*following, floor_z),
                               (*following, 0.), (*first, 0.)))
        width = float(np.linalg.norm(following - first))
        wall = trimesh.Trimesh(vertices=vertices, faces=((0, 3, 2), (0, 2, 1)),
                               process=False)
        wall_uv = np.asarray(((0., floor_z), (width, floor_z), (width, 0.), (0., 0.)))
        wall.visual = trimesh.visual.texture.TextureVisuals(
            uv=wall_uv * uv_scale + floor_uv[index], material=material,
        )
        wall.apply_transform(aperture_to_table)
        name = f"pocket-wall-{index}"
        result.add_geometry(wall, node_name=name, geom_name=name)
    return result


def pocket_table_assets(root=None, *, cutout_center_xy, cutout_size_xy,
                        cutout_yaw, pocket_floor_z=-.04):
    """Cache finite-pocket assets and return parts, visual path, and metadata.

    Cache paths do not enter metadata, so relocation preserves geometry identity.
    """
    root = Path(root) if root is not None else asset_root()
    parts, opening_world, table_to_world, aperture_to_world = _pocket_geometry(
        cutout_center_xy, cutout_size_xy, cutout_yaw, pocket_floor_z,
    )
    source_path = Path(inspect.getfile(TableSceneBuilder)).parent / "assets" / "table.glb"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    specification = {
        "schema_version": 1,
        "kind": "pocket",
        "floor_z": float(pocket_floor_z),
        "pocket_floor_z_m": float(pocket_floor_z),
        "generator": {"source_sha256": _MODULE_HASH,
                      "aperture_source_sha256": _APERTURE_MODULE_HASH,
                      "asset_generator": GENERATOR_ID},
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
    folder = root / "fixtures" / f"pocket-table-{key}"
    visual_path = folder / "visual.glb"
    if not visual_path.is_file():
        visual = _pocket_visual(source_path, cutout_size_xy, pocket_floor_z,
                                table_to_world, aperture_to_world)
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


class PocketTableSceneBuilder(TableSceneBuilder):
    """Keep official table initialization while replacing its top with a pocket."""

    def __init__(self, env, robot_init_qpos_noise=0.02, *, cutout_center_xy,
                 cutout_size_xy, cutout_yaw, pocket_floor_z=-.04):
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.cutout_center_xy = tuple(cutout_center_xy)
        self.cutout_size_xy = tuple(cutout_size_xy)
        self.cutout_yaw = float(cutout_yaw)
        self.pocket_floor_z = float(pocket_floor_z)

    def build(self):
        root = getattr(self.env, "asset_dir", None)
        self.collision_parts, self.visual_path, self.aperture_metadata = pocket_table_assets(
            root, cutout_center_xy=self.cutout_center_xy, cutout_size_xy=self.cutout_size_xy,
            cutout_yaw=self.cutout_yaw, pocket_floor_z=self.pocket_floor_z,
        )
        builder = self.scene.create_actor_builder()
        for part in self.collision_parts:
            # The official table also uses the simulator's default contact material.
            builder.add_convex_collision_from_file(
                filename=str(primitive_mesh_path(part, root)), pose=pose(part.center, part.quat),
            )
        builder.add_visual_from_file(filename=str(self.visual_path))
        builder.initial_pose = sapien.Pose(p=_TABLE_POSITION, q=_TABLE_QUATERNION)
        self.table = builder.build_kinematic(name="table-workspace")
        self.table_length, self.table_width, self.table_height = (
            _TABLE_VISUAL_BOUNDS[1] - _TABLE_VISUAL_BOUNDS[0]
        )
        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        self.ground = build_ground(self.scene, floor_width=floor_width, altitude=-self.table_height)
        self.scene_objects = [self.table, self.ground]
