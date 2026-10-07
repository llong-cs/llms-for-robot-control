"""Measured LIBERO scene state for recording, never attached to policy observations.

Loaded by path in the Python 3.8 simulator. Reading this module has no dependency
on the policy process, task-specific predicates, or privileged observation mode.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "recording_state.py"
_SCHEMA_SPEC = importlib.util.spec_from_file_location("_libero_recording_state_schema", _SCHEMA_PATH)
_SCHEMA = importlib.util.module_from_spec(_SCHEMA_SPEC)
_SCHEMA_SPEC.loader.exec_module(_SCHEMA)


def _vector(value, width, label):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != width or not np.isfinite(array).all():
        raise ValueError("Invalid measured recording state: %s" % label)
    return array.tolist()


def _quaternion(value, label):
    value = _vector(value, 4, label)
    if not np.isclose(np.linalg.norm(value), 1.0, atol=1e-4):
        raise ValueError("Non-unit measured quaternion: %s" % label)
    return value[1:] + value[:1]  # MuJoCo WXYZ -> XYZW.


def _descendant(model, body_id, root):
    seen = set()
    while body_id != root and body_id != 0:
        if body_id in seen:
            raise ValueError("Invalid MuJoCo body parent cycle")
        seen.add(body_id)
        body_id = int(model.body_parentid[body_id])
    return body_id == root


def _joints(model, data, root, descendants=True):
    result = {}
    definitions = {0: ("free", 7, "m+unit_quaternion_xyzw"),
                   1: ("ball", 4, "unit_quaternion_xyzw"),
                   2: ("slide", 1, "m"), 3: ("hinge", 1, "rad")}
    for index in range(int(model.njnt)):
        owner = int(model.jnt_bodyid[index])
        if not (_descendant(model, owner, root) if descendants else owner == root):
            continue
        kind, width, unit = definitions[int(model.jnt_type[index])]
        name = _native_name(model, "joint", index) or "__mujoco_joint_%d" % index
        if name in result:
            raise ValueError("Recorded object joints require unique native names")
        start = int(model.jnt_qposadr[index])
        value = _vector(data.qpos[start:start + width], width, name)
        if kind == "free":
            value = value[:3] + _quaternion(value[3:], name)
        elif kind == "ball":
            value = _quaternion(value, name)
        result[name] = {"value": value, "type": kind, "unit": unit}
    return result


def _tokens(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, (tuple, list)):
        return set().union(*(_tokens(item) for item in value))
    return set()


def _native_name(model, kind, index):
    lookup = getattr(model, kind + "_id2name", None)
    if callable(lookup):
        name = lookup(index)
    elif callable(getattr(model, kind, None)):
        name = getattr(model, kind)(index).name
    else:
        names = getattr(model, kind + "_names", ())
        name = names[index] if index < len(names) else None
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name if isinstance(name, str) and name else None


def _robot_bodies(native, model):
    """Use native robot model identities; never classify by guessed name prefixes."""
    robots = getattr(native, "robots", None)
    robot_catalog_known = isinstance(robots, (list, tuple))
    robots = robots if robot_catalog_known else ()
    roots, explicit = set(), set()
    identified = 0
    for robot in robots:
        found = False
        robot_model = getattr(robot, "robot_model", None)
        for entity in (robot_model, getattr(robot, "gripper", None),
                       getattr(getattr(robot, "robot_model", None), "mount", None)):
            if entity is None:
                continue
            root = getattr(entity, "root_body", None)
            if isinstance(root, str):
                try:
                    root_id = int(model.body_name2id(root))
                except (KeyError, ValueError, TypeError):
                    root_id = -1
                if 0 < root_id < int(model.nbody):
                    roots.add(root_id)
                    found = found or entity is robot_model
            names = getattr(entity, "bodies", ())
            for name in names if isinstance(names, (tuple, list)) else ():
                try:
                    body_id = int(model.body_name2id(name))
                except (KeyError, ValueError, TypeError):
                    continue
                if 0 < body_id < int(model.nbody):
                    explicit.add(body_id)
                    found = found or entity is robot_model
        identified += int(found)
    body_ids = explicit | {index for index in range(1, int(model.nbody))
                           if any(_descendant(model, index, root) for root in roots)}
    available = robot_catalog_known and len(robots) == identified
    return body_ids, available, roots


def _body_type(model, body_id):
    joint_bodies = set(int(value) for value in model.jnt_bodyid)
    mocap = getattr(model, "body_mocapid", None)
    visited = set()
    while body_id:
        if body_id in visited:
            raise ValueError("Invalid MuJoCo body parent cycle")
        visited.add(body_id)
        if mocap is not None and int(mocap[body_id]) >= 0:
            return "kinematic"
        if body_id in joint_bodies:
            return "dynamic"
        body_id = int(model.body_parentid[body_id])
    return "static"


def read_recording_state(env, control_step, control_hz=20.0, task=None):
    """Read declared objects plus every native world body and cached task feedback.

    The object-name index remains stable. The independent bodies index
    includes robots, static scene bodies and bodies absent from task dictionaries.
    Task feedback must come from normal reset/step; reading never evaluates it.
    """
    native = getattr(env, "env", env)
    model, data = native.sim.model, native.sim.data
    declared_objects = getattr(native, "objects_dict", None)
    fixtures = getattr(native, "fixtures_dict", None)
    if not isinstance(declared_objects, Mapping) or not isinstance(fixtures, Mapping):
        raise ValueError("LIBERO recording state requires object and fixture dictionaries")
    if not declared_objects and not fixtures:
        raise ValueError("LIBERO recording state has no declared objects or fixtures")
    if set(declared_objects) & set(fixtures):
        raise ValueError("LIBERO object and fixture names collide")
    parsed = getattr(native, "parsed_problem", {})
    parsed = parsed if isinstance(parsed, Mapping) else {}
    goal_tokens = _tokens(parsed.get("goal_state"))
    sites = getattr(native, "object_sites_dict", {})
    sites = sites if isinstance(sites, Mapping) else {}
    goal_parents = {getattr(sites[token], "parent_name", None)
                    for token in goal_tokens if token in sites}
    interest = _tokens(parsed.get("obj_of_interest"))
    objects, roots = {}, {}
    positions = getattr(data, "body_xpos", None)
    rotations = getattr(data, "body_xquat", None)
    positions = data.xpos if positions is None else positions
    rotations = data.xquat if rotations is None else rotations
    names = {index: _native_name(model, "body", index) for index in range(int(model.nbody))}
    body_keys = {index: name or "__mujoco_body_%d" % index for index, name in names.items()}
    if len(set(body_keys.values())) != len(body_keys):
        raise ValueError("MuJoCo body identities collide")
    for kind, collection in (("object", declared_objects), ("fixture", fixtures)):
        for name, entity in sorted(collection.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("LIBERO recorded objects require nonempty stable names")
            root = getattr(entity, "root_body", None)
            if not isinstance(root, str) or not root:
                raise ValueError("Missing root body for recorded object %s" % name)
            body_id = int(model.body_name2id(root))
            if not 0 < body_id < int(model.nbody):
                raise ValueError("Missing MuJoCo body for recorded object %s" % name)
            if body_id in roots:
                raise ValueError("LIBERO task entity root bodies collide")
            roots[body_id] = name
            roles = []
            if name in goal_tokens:
                roles.append("goal_argument")
            if name in goal_parents:
                roles.append("goal_region_parent")
            if name in interest:
                roles.append("object_of_interest")
            objects[name] = {
                "position": _vector(positions[body_id], 3, name + " position"),
                "quaternion_xyzw": _quaternion(rotations[body_id], name + " quaternion"),
                "kind": kind, "roles": roles, "source_body": root,
                "joint_positions": _joints(model, data, body_id),
                "body_ids": [body_keys[index] for index in range(int(model.nbody))
                             if _descendant(model, index, body_id)],
            }
    robot_ids, robot_complete, robot_roots = _robot_bodies(native, model)
    bodies = {}
    for index, key in body_keys.items():
        owner, current, visited = None, index, set()
        while current:
            if current in visited:
                raise ValueError("Invalid MuJoCo body parent cycle")
            visited.add(current)
            if current in roots:
                owner = roots[current]
                break
            current = int(model.body_parentid[current])
        kind = ("robot" if index in robot_ids else objects[owner]["kind"] if owner
                else "world" if index == 0 else "scene")
        parent = None if index == 0 else body_keys[int(model.body_parentid[index])]
        bodies[key] = {
            "position": _vector(positions[index], 3, key + " position"),
            "quaternion_xyzw": _quaternion(rotations[index], key + " quaternion"),
            "kind": kind, "source_body": names[index], "native_id": index,
            "parent_body": parent, "object_name": owner,
            "is_robot": index in robot_ids if robot_complete or index in robot_ids else None,
            "body_type": _body_type(model, index),
            "joint_positions": _joints(model, data, index, descendants=False),
        }
    articulations = {}
    for name, entity in objects.items():
        if entity["joint_positions"]:
            articulations[name] = {
                "position": entity["position"], "quaternion_xyzw": entity["quaternion_xyzw"],
                "source_body": entity["source_body"], "kind": entity["kind"],
                "joints": entity["joint_positions"],
                "links": {key: bodies[key] for key in entity["body_ids"]},
            }
    stamp = float(data.time)
    if not np.isfinite(stamp):
        raise ValueError("Nonfinite simulator recording timestamp")
    return _SCHEMA.build_recording_state(
        adapter="libero", control_step=control_step, control_hz=control_hz,
        simulation_time_s=stamp, objects=objects, bodies=bodies, articulations=articulations,
        task=task if task is not None else {
            "availability": "unavailable", "source": None, "data": {},
            "reason": "native reset/step task feedback was not supplied; no evaluation was invoked",
        },
        coverage={
            "objects": "all LIBERO objects_dict and fixtures_dict task entities",
            "bodies": "every MuJoCo model body including world, static scene, robot and undeclared entities",
            "body_count": int(model.nbody), "body_ids": list(body_keys.values()),
            "object_count": len(objects), "robot_body_ids": [body_keys[index] for index in sorted(robot_ids)],
            "robot_identification": "complete" if robot_complete else "partial_or_unavailable",
            "robot_identity_source": "native robots robot_model/gripper/mount root_body and bodies",
            "robot_roots": [body_keys[index] for index in sorted(robot_roots)],
            "unnamed_body_ids": [body_keys[index] for index, name in names.items() if name is None],
            "non_task_body_ids": [key for key, body in bodies.items() if body["object_name"] is None],
            "time_source": "MuJoCo data.time; includes reset initialization/stabilization offset",
            "excluded": "geometry/site/camera frames are not independent rigid bodies; their parent bodies are recorded",
        },
        role_source="LIBERO parsed_problem.goal_state/obj_of_interest + object_sites_dict",
    )
