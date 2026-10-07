"""Deterministic procedural geometry with matching physical and rendered primitives.

All sizes are metres, quaternions WXYZ, cylinders/capsules point along local Z.
Concave objects consist of convex parts: holes stay open in collision geometry.
Generated OBJ meshes, GLBs, and specifications use the project's data directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat

from .paths import asset_root

# Both implementation and dependency changes create a fresh cache namespace.
GENERATOR_ID = {
    "version": "2.3.0",
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "trimesh_version": trimesh.__version__,
    "shapely_version": package_version("shapely"),
    "manifold_version": package_version("manifold3d"),
    "earcut_version": package_version("mapbox-earcut"),
}


@dataclass(frozen=True)
class Part:
    kind: str
    size: tuple
    center: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    color: tuple = (0.6, 0.6, 0.6, 1.0)
    collision: bool = True
    visual: bool = True
    friction: float | None = None
    face_friction: tuple = ()


def quat_euler(roll=0.0, pitch=0.0, yaw=0.0):
    return tuple(float(v) for v in euler2quat(roll, pitch, yaw))


def pose(position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    return sapien.Pose(p=np.asarray(position, dtype=float), q=np.asarray(quaternion, dtype=float))


def box(half_size, center=(0, 0, 0), color=(0.6, 0.6, 0.6, 1), quat=(1, 0, 0, 0), **kwargs):
    return Part("box", tuple(half_size), tuple(center), tuple(quat), tuple(color), **kwargs)


def cylinder(radius, half_length, center=(0, 0, 0), color=(0.6, 0.6, 0.6, 1), quat=(1, 0, 0, 0), **kwargs):
    return Part("cylinder", (radius, half_length), tuple(center), tuple(quat), tuple(color), **kwargs)


def sphere(radius, center=(0, 0, 0), color=(0.6, 0.6, 0.6, 1), **kwargs):
    return Part("sphere", (radius,), tuple(center), color=tuple(color), **kwargs)



def ellipsoid(semi_axes, center=(0, 0, 0), color=(0.6, 0.6, 0.6, 1),
              quat=(1, 0, 0, 0), **kwargs):
    """A smooth convex ellipsoid with the same mesh for rendering and collision."""
    axes = np.asarray(semi_axes, dtype=float)
    if axes.shape != (3,) or not np.all(np.isfinite(axes)) or np.any(axes <= 0):
        raise ValueError("ellipsoid requires three finite positive semi-axes")
    return Part("ellipsoid", tuple(axes), tuple(center), tuple(quat), tuple(color), **kwargs)


def convex_polyhedron(vertices, center=(0, 0, 0), color=(0.6, 0.6, 0.6, 1),
                      quat=(1, 0, 0, 0), **kwargs):
    """Convex hull of finite 3D vertices, shared by collision, visual and mass.

    Useful for tapered solids whose upper and lower profiles differ. Concave
    solids must still be decomposed into convex parts to preserve their notches.
    """
    points = np.asarray(vertices, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
            or not np.all(np.isfinite(points))
            or np.linalg.matrix_rank(points - points[0]) < 3):
        raise ValueError("convex_polyhedron requires at least four finite noncoplanar 3D vertices")
    return Part("convex_polyhedron", tuple(tuple(float(x) for x in row) for row in points),
                tuple(center), tuple(quat), tuple(color), **kwargs)


def capsule_between(start, end, radius, color=(0.6, 0.6, 0.6, 1), **kwargs):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return sphere(radius, start, color, **kwargs)
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction / length)
    quaternion = trimesh.transformations.quaternion_from_matrix(transform)
    return Part("capsule", (radius, length / 2), tuple((start + end) / 2),
                tuple(quaternion), tuple(color), **kwargs)


def part_mesh(part, *, transformed=True):
    if part.kind == "box":
        mesh = trimesh.creation.box(extents=np.asarray(part.size) * 2)
    elif part.kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=part.size[0], height=2 * part.size[1], sections=48)
    elif part.kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=part.size[0])
    elif part.kind == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(np.asarray(part.size, dtype=float))
    elif part.kind == "capsule":
        mesh = trimesh.creation.capsule(radius=part.size[0], height=2 * part.size[1], count=[12, 24])
    elif part.kind == "convex_polyhedron":
        mesh = trimesh.convex.convex_hull(np.asarray(part.size, dtype=float))
    elif part.kind == "prism":
        polygon, half_length = part.size
        xy = np.asarray(polygon)
        n = len(xy)
        vertices = np.vstack([np.column_stack([xy, np.full(n, z)])
                              for z in (-half_length, half_length)])
        faces = [(0, i + 1, i) for i in range(1, n - 1)]
        faces += [(n, n + i, n + i + 1) for i in range(1, n - 1)]
        for i in range(n):
            j = (i + 1) % n
            faces += [(i, j, n + j), (i, n + j, n + i)]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    else:
        raise ValueError(f"Unknown primitive {part.kind}")
    if transformed:
        transform = np.eye(4)
        transform[:3, :3] = quat2mat(part.quat)
        transform[:3, 3] = part.center
        mesh.apply_transform(transform)
    rgba = np.asarray(part.color if len(part.color) == 4 else (*part.color, 1.0))
    mesh.visual.face_colors = np.clip(rgba * 255, 0, 255).astype(np.uint8)
    return mesh


def _write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    if isinstance(data, str):
        tmp.write_text(data)
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


def primitive_mesh_path(part, root=None):
    root = Path(root) if root is not None else asset_root()
    key = hashlib.sha256(json.dumps([GENERATOR_ID, part.kind, list(part.size)],
                                    sort_keys=True).encode()).hexdigest()[:20]
    path = root / "primitives" / f"{part.kind}-{key}.obj"
    if not path.exists():
        _write_atomic(path, part_mesh(part, transformed=False).export(file_type="obj"))
    return path


def collision_parts(part):
    """Partition a convex solid to assign real materials to selected faces.

    SAPIEN uses one material per convex shape. Face-to-center pyramids retain
    the exact exterior and fill the original volume without overlapping skins
    or projecting lips. Face normals are expressed in the part's local frame.
    """
    if not part.face_friction:
        return (part,)
    if part.kind not in ("box", "prism", "convex_polyhedron"):
        raise ValueError("Face friction requires a flat-faced convex solid")
    mesh = part_mesh(part, transformed=False)
    groups = [np.asarray(group) for group in mesh.facets]
    grouped = {int(index) for group in groups for index in group}
    groups.extend(np.asarray([index]) for index in range(len(mesh.faces)) if index not in grouped)
    overrides = []
    for normal, coefficient in part.face_friction:
        normal = np.asarray(normal, dtype=float)
        if (normal.shape != (3,) or not np.all(np.isfinite(normal))
                or np.linalg.norm(normal) <= 1e-10 or not np.isfinite(coefficient) or coefficient < 0):
            raise ValueError("Face friction requires a finite normal and nonnegative coefficient")
        overrides.append((normal / np.linalg.norm(normal), float(coefficient)))
    matched = set()
    pieces = []
    for group in groups:
        vertices = mesh.vertices[np.unique(mesh.faces[group])]
        normal = mesh.face_normals[group[0]]
        coefficient = part.friction
        for index, (requested, value) in enumerate(overrides):
            if np.dot(normal, requested) > 1 - 1e-7:
                coefficient = value
                matched.add(index)
        pieces.append(convex_polyhedron(np.vstack((vertices, mesh.center_mass)),
                                       center=part.center, quat=part.quat, color=part.color,
                                       collision=part.collision, visual=False, friction=coefficient))
    if len(matched) != len(overrides):
        raise ValueError("A face friction normal does not match an exterior face")
    return tuple(pieces)


def add_parts(builder, parts, *, density=500, friction=1.0, asset_dir=None):
    materials = {}
    for original in parts:
        for part in collision_parts(original) if original.collision else ():
            coefficient = friction if part.friction is None else part.friction
            if not np.isfinite(coefficient) or coefficient < 0:
                raise ValueError("Friction must be finite and nonnegative")
            if coefficient not in materials:
                materials[coefficient] = physx.PhysxMaterial(
                    static_friction=coefficient, dynamic_friction=coefficient, restitution=0.0)
            material = materials[coefficient]
            _add_collision_part(builder, part, material=material, density=density, asset_dir=asset_dir)
        part = original
        local_pose = pose(part.center, part.quat)
        if part.visual:
            render_material = sapien.render.RenderMaterial()
            render_material.base_color = part.color if len(part.color) == 4 else (*part.color, 1)
            builder.add_visual_from_file(filename=str(primitive_mesh_path(part, asset_dir)),
                                         pose=local_pose, material=render_material)
    return builder


def _add_collision_part(builder, part, *, material, density, asset_dir):
    local_pose = pose(part.center, part.quat)
    if part.kind == "box":
        builder.add_box_collision(pose=local_pose, half_size=part.size, material=material, density=density)
    elif part.kind == "sphere":
        builder.add_sphere_collision(pose=local_pose, radius=part.size[0], material=material, density=density)
    elif part.kind == "capsule":
        # Native capsules point along X; our geometry convention is Z.
        native_pose = local_pose * pose(quaternion=quat_euler(0, -np.pi / 2, 0))
        builder.add_capsule_collision(pose=native_pose, radius=part.size[0], half_length=part.size[1],
                                     material=material, density=density)
    else:
        builder.add_convex_collision_from_file(filename=str(primitive_mesh_path(part, asset_dir)),
                                              pose=local_pose, material=material, density=density)


def export_actor(name, parts, *, density, friction, static, root=None, mass_properties=None):
    root = Path(root) if root is not None else asset_root()
    physical_parts = [piece for part in parts if part.collision for piece in collision_parts(part)]
    spec = {"schema_version": 3, "generator": GENERATOR_ID, "name": name, "units": "metres", "quaternion": "wxyz",
            "density_kg_m3": density, "friction": friction, "static": static,
            "parts": [asdict(part) for part in parts],
            "collision_parts": [{**asdict(part), "friction": friction if part.friction is None else part.friction}
                                for part in physical_parts]}
    if mass_properties is not None:
        spec["mass_properties"] = mass_properties
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    folder = root / "actors" / f"{name}-{key}"
    _write_atomic(folder / "spec.json", json.dumps(spec, indent=2))
    visual_parts = [part_mesh(p) for p in parts if p.visual]
    if visual_parts and not (folder / "visual.glb").exists():
        _write_atomic(folder / "visual.glb", trimesh.util.concatenate(visual_parts).export(file_type="glb"))
    for index, part in enumerate(physical_parts):
        _write_atomic(folder / f"collision-{index:03d}.obj", part_mesh(part).export(file_type="obj"))
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(folder.iterdir()) if path.is_file() and ".tmp" not in path.name}
    return {"name": name, "asset_id": key, "spec_sha256": files["spec.json"],
            "directory": str(folder), "parts": len(parts), "files_sha256": files,
            "mass_properties": mass_properties}


def build_actor(env, name, parts, position=(0, 0, 0), quaternion=(1, 0, 0, 0), *,
                static=False, kinematic=False, density=500, friction=None, mass=None):
    """Build a free body or a fixture, with resettable kinematic fixtures optional."""
    if static and kinematic:
        raise ValueError("Choose either a static or a kinematic fixture")
    fixed = static or kinematic
    parts = tuple(parts)
    if not parts:
        raise ValueError(f"Actor {name} has no geometry")
    builder = env.scene.create_actor_builder()
    root = getattr(env, "asset_dir", None)
    friction = (0.3 if fixed else 1.0) if friction is None else friction
    add_parts(builder, parts, density=density, friction=friction, asset_dir=root)
    mass_properties = None
    if mass is not None:
        if fixed:
            raise ValueError("Explicit mass is only supported for dynamic actors")
        mass_properties = union_mass_properties(parts, mass, root)
        builder.set_mass_and_inertia(
            mass, pose(mass_properties["center_of_mass_m"], mass_properties["principal_axes_quaternion_wxyz"]),
            mass_properties["principal_inertia_kg_m2"])
    builder.initial_pose = pose(position, quaternion)
    if static:
        actor = builder.build_static(name=name)
    elif kinematic:
        actor = builder.build_kinematic(name=name)
    else:
        actor = builder.build(name=name)
    if not fixed and mass_properties is None:
        body = actor._bodies[0]
        principal = np.asarray(body.inertia, dtype=float)
        principal_pose = body.cmass_local_pose
        axes = quat2mat(principal_pose.q)
        mass_properties = {
            "method": "physx-collision-density", "mass_kg": float(body.mass),
            "center_of_mass_m": np.asarray(principal_pose.p).tolist(),
            "principal_inertia_kg_m2": principal.tolist(),
            "principal_axes_quaternion_wxyz": np.asarray(principal_pose.q).tolist(),
            "inertia_kg_m2": (axes @ np.diag(principal) @ axes.T).tolist(),
        }
    record = export_actor(name, parts, density=density, friction=friction, static=fixed, root=root, mass_properties=mass_properties)
    if hasattr(env, "asset_records"):
        env.asset_records.append(record)
    return actor


def prism(vertices_xy, half_length, center=(0, 0, 0),
          quat=(1, 0, 0, 0), color=(0.6, 0.6, 0.6, 1), **kwargs):
    """A convex polygon extruded along local Z, in metres."""
    from shapely.geometry import Polygon
    vertices = np.asarray(vertices_xy, dtype=float)
    polygon = Polygon(vertices)
    if (not polygon.is_valid or polygon.area <= 0 or half_length <= 0
            or abs(polygon.convex_hull.area - polygon.area) > 1e-12):
        raise ValueError("prism requires a valid convex polygon and positive half_length")
    if not polygon.exterior.is_ccw:
        vertices = vertices[::-1]
    return Part("prism", (tuple(map(tuple, vertices)), float(half_length)),
                tuple(center), tuple(quat), tuple(color), **kwargs)


def extruded_polygon(vertices_xy, half_length, holes=(), **kwargs):
    """Convex decomposition retaining polygon openings in collision geometry."""
    from shapely.geometry import Polygon
    polygon = Polygon(vertices_xy, holes=holes)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Invalid polygon or holes")
    vertices, faces = trimesh.creation.triangulate_polygon(polygon, engine="earcut")
    pieces = [Polygon(vertices[face]) for face in faces]
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                combined = pieces[i].union(pieces[j])
                if (combined.geom_type == "Polygon" and not combined.interiors
                        and abs(combined.convex_hull.area - combined.area) < 1e-12):
                    pieces[i] = combined
                    pieces.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [prism(list(piece.exterior.coords)[:-1], half_length, **kwargs) for piece in pieces]


def ring_parts(inner_radius, outer_radius, half_length, segments=48,
               start_angle=0.0, end_angle=2 * np.pi, **kwargs):
    """Annular convex sectors, including open C/J shapes, about local Z."""
    if not 0 < inner_radius < outer_radius or not 0 < end_angle - start_angle <= 2 * np.pi + 1e-9:
        raise ValueError("Invalid annulus dimensions or angular interval")
    count = max(2, int(np.ceil(segments * (end_angle - start_angle) / (2 * np.pi))))
    angles = np.linspace(start_angle, end_angle, count + 1)
    return [prism([(r * np.cos(t), r * np.sin(t))
                   for r, t in ((inner_radius, a), (outer_radius, a),
                                (outer_radius, b), (inner_radius, b))], half_length, **kwargs)
            for a, b in zip(angles[:-1], angles[1:])]


def union_mass_properties(parts, mass, root=None):
    """Uniform-density solid-union inertia; overlaps are counted only once."""
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Dynamic actor mass must be positive and finite")
    root = Path(root) if root is not None else asset_root()
    key = hashlib.sha256(json.dumps([GENERATOR_ID, [asdict(p) for p in parts if p.collision], mass],
                                    sort_keys=True).encode()).hexdigest()[:24]
    path = root / "mass-properties" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    meshes = [part_mesh(p) for p in parts if p.collision]
    if not meshes:
        raise ValueError("Mass calculation requires collision geometry")
    solid = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
    if not solid.is_volume or solid.volume <= 0:
        raise ValueError("Geometry union is not a positive closed solid")
    inertia = np.asarray(solid.moment_inertia) * (mass / solid.volume)
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    record = {"method": "uniform-density-manifold-union", "mass_kg": float(mass),
              "volume_m3": float(solid.volume), "density_kg_m3": float(mass / solid.volume),
              "center_of_mass_m": solid.center_mass.tolist(), "inertia_kg_m2": inertia.tolist(),
              "principal_inertia_kg_m2": moments.tolist(),
              "principal_axes_quaternion_wxyz": mat2quat(axes).tolist(), "union_sha256": key}
    _write_atomic(path, json.dumps(record, indent=2))
    return record
