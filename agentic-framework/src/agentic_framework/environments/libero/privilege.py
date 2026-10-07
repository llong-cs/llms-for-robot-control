"""Generic MuJoCo scene truth plus optional LIBERO metadata (Python 3.8).

The standalone simulator worker loads this module by path. Base scene extraction
uses only MuJoCo model/data; no task name or object category selects behavior.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "privilege.py"
_SCHEMA_SPEC = importlib.util.spec_from_file_location("_libero_privilege_schema", _SCHEMA_PATH)
_SCHEMA = importlib.util.module_from_spec(_SCHEMA_SPEC)
_SCHEMA_SPEC.loader.exec_module(_SCHEMA)

# MuJoCo mjtJoint enum and its native qpos / qvel widths.
_JOINTS = {
    0: ("free", 7, 6),
    1: ("ball", 4, 3),
    2: ("slide", 1, 1),
    3: ("hinge", 1, 1),
}


def _name(model, kind, index):
    lookup = getattr(model, kind + "_id2name", None)
    if callable(lookup):
        name = lookup(index)
    else:
        accessor = getattr(model, kind, None)
        if callable(accessor):
            name = accessor(index).name
        else:
            names = getattr(model, kind + "_names", ())
            name = names[index] if index < len(names) else None
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name if isinstance(name, str) and name else None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _optional(value, name):
    try:
        return getattr(value, name, None)
    except NotImplementedError:
        return None


def _semantic_aliases(native, model):
    aliases = {}
    for name, body_id in _mapping(_optional(native, "obj_body_id")).items():
        if isinstance(name, str) and name.strip() and isinstance(body_id, (int, np.integer)):
            if 0 <= int(body_id) < int(model.nbody):
                aliases.setdefault(int(body_id), []).append(name)
    # Some wrappers expose entity metadata before creating obj_body_id.
    lookup = getattr(model, "body_name2id", None)
    if callable(lookup):
        for attr in ("objects_dict", "fixtures_dict"):
            for name, entity in _mapping(_optional(native, attr)).items():
                root_body = getattr(entity, "root_body", None)
                if not isinstance(name, str) or not name.strip() or not isinstance(root_body, str):
                    continue
                try:
                    body_id = int(lookup(root_body))
                except (KeyError, ValueError, IndexError, TypeError, NotImplementedError):
                    continue
                if 0 <= body_id < int(model.nbody):
                    aliases.setdefault(body_id, []).append(name)
    return {body_id: sorted(set(names)) for body_id, names in aliases.items()}


def _vector(value, width, label):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != width or not np.isfinite(array).all():
        raise ValueError("Invalid measured MuJoCo %s" % label)
    return array.tolist()


def _pose(data, body_id):
    positions = getattr(data, "body_xpos", None)
    quaternions = getattr(data, "body_xquat", None)
    # robosuite/mujoco-py use body_xpos/body_xquat; native MuJoCo uses xpos/xquat.
    if positions is None:
        positions = data.xpos
    if quaternions is None:
        quaternions = data.xquat
    quaternion = _vector(quaternions[body_id], 4, "body quaternion")
    return {
        "position_m": _vector(positions[body_id], 3, "body position"),
        "quaternion_xyzw": quaternion[1:] + quaternion[:1],
    }


def _velocity(data, name, kind):
    # Native getters report body velocity in world orientation. Do not reinterpret
    # MuJoCo cvel, whose origin is the subtree center of mass, as body velocity.
    getter = getattr(data, "get_body_xvel" + kind, None)
    if not callable(getter):
        return None
    try:
        measured = getter(name)
    except NotImplementedError:
        return None
    return _vector(measured, 3, "body velocity")


def _root(model, body_id):
    parents = model.body_parentid
    visited = set()
    while body_id != 0 and int(parents[body_id]) != 0:
        if body_id in visited:
            raise ValueError("Invalid MuJoCo body parent cycle")
        visited.add(body_id)
        body_id = int(parents[body_id])
    return body_id


def _body_type(model, body_id, joint_bodies):
    mocap_ids = getattr(model, "body_mocapid", None)
    visited = set()
    while body_id != 0:
        if body_id in visited:
            raise ValueError("Invalid MuJoCo body parent cycle")
        visited.add(body_id)
        if mocap_ids is not None and int(mocap_ids[body_id]) >= 0:
            return "kinematic"
        if body_id in joint_bodies:
            return "dynamic"
        body_id = int(model.body_parentid[body_id])
    return "static"


def _joint(model, data, joint_id):
    native_type = int(model.jnt_type[joint_id])
    if native_type not in _JOINTS:
        raise ValueError("Unsupported MuJoCo joint type %s" % native_type)
    kind, position_width, velocity_width = _JOINTS[native_type]
    start = int(model.jnt_qposadr[joint_id])
    position = _vector(data.qpos[start : start + position_width], position_width, "joint qpos")
    if kind == "free":
        position = position[:3] + position[4:] + position[3:4]
        position_semantics = "world xyz in m followed by world quaternion xyzw"
        velocity_semantics = "native qvel: world linear xyz in m/s, local angular xyz in rad/s"
    elif kind == "ball":
        position = position[1:] + position[:1]
        position_semantics = "relative joint quaternion xyzw"
        velocity_semantics = "native qvel: local angular xyz in rad/s"
    else:
        position_semantics = "joint translation in m" if kind == "slide" else "joint angle in rad"
        velocity_semantics = "joint speed in m/s" if kind == "slide" else "joint speed in rad/s"
    qvel = getattr(data, "qvel", None)
    dof_address = getattr(model, "jnt_dofadr", None)
    velocity = None
    if qvel is not None and dof_address is not None:
        start = int(dof_address[joint_id])
        velocity = _vector(qvel[start : start + velocity_width], velocity_width, "joint qvel")
    limits = None
    limited = getattr(model, "jnt_limited", None)
    ranges = getattr(model, "jnt_range", None)
    if limited is not None and ranges is not None and bool(limited[joint_id]):
        limits = _vector(ranges[joint_id], 2, "joint range")
    return {
        "joint_id": joint_id,
        "type": kind,
        "position": position,
        "velocity": velocity,
        "position_semantics": position_semantics,
        "velocity_semantics": velocity_semantics if velocity is not None else None,
        "limits": limits,
        "limit_semantics": (
            "native MuJoCo jnt_range: m for slide; rad for hinge/ball; ball lower bound unused"
            if limits is not None
            else None
        ),
    }


def _regions(native, model, data, mat2quat):
    sites = _optional(native, "object_sites_dict")
    if not isinstance(sites, Mapping):
        return _SCHEMA.unavailable("LIBERO declared region sites are unavailable")
    source = "LIBERO object_sites_dict + MuJoCo site_xpos/site_xmat/site_size"
    try:
        regions = {}
        for name, region in sorted(sites.items()):
            site_id = int(model.site_name2id(name))
            if site_id < 0:
                raise ValueError("unknown site %s" % name)
            regions[name] = {
                "site_id": site_id,
                "position_m": _vector(data.site_xpos[site_id], 3, "region position"),
                "quaternion_xyzw": _vector(
                    mat2quat(np.asarray(data.site_xmat[site_id]).reshape(3, 3)),
                    4,
                    "region quaternion",
                ),
                "size_m": _vector(model.site_size[site_id], 3, "region size"),
                "size_semantics": "native MuJoCo site_size: shape-dependent size parameters in m",
                "parent_name": getattr(region, "parent_name", None),
            }
            if hasattr(model, "site_type"):
                regions[name]["native_site_type"] = int(model.site_type[site_id])
        return _SCHEMA.available(regions, source)
    except (
        AttributeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        NotImplementedError,
    ) as exc:
        return _SCHEMA.unavailable("Declared regions cannot be read: %s" % exc, source)


def _task_sections(native):
    parsed = _optional(native, "parsed_problem")
    if not isinstance(parsed, Mapping) or "goal_state" not in parsed:
        return (
            _SCHEMA.unavailable("LIBERO parsed goal predicates are unavailable"),
            _SCHEMA.unavailable("Goal predicates are unavailable for evaluation"),
        )
    source = "LIBERO parsed_problem.goal_state"
    predicates = parsed["goal_state"]
    if not isinstance(predicates, (list, tuple)):
        return (
            _SCHEMA.unavailable("LIBERO goal predicates have no supported representation", source),
            _SCHEMA.unavailable("Goal predicates are unavailable for evaluation"),
        )
    try:
        goals = _SCHEMA.available({"predicates": _SCHEMA.plain(predicates)}, source)
    except (TypeError, ValueError) as exc:
        return (
            _SCHEMA.unavailable("Goal predicates cannot be read: %s" % exc, source),
            _SCHEMA.unavailable("Goal predicates are unavailable for evaluation"),
        )
    evaluator = _optional(native, "_eval_predicate")
    if not callable(evaluator):
        return goals, _SCHEMA.unavailable("LIBERO predicate evaluator is unavailable")
    source = "LIBERO _eval_predicate on current simulator state"
    try:
        measured = [evaluator(predicate) for predicate in predicates]
        if any(not isinstance(value, (bool, np.bool_)) for value in measured):
            raise ValueError("Predicate evaluator did not return Boolean truth values")
        evaluated = [bool(value) for value in measured]
        evaluation = _SCHEMA.available({"goal_satisfied": evaluated}, source)
    except Exception as exc:
        # Task plugins may lack a predicate implementation. Preserve valid scene
        # truth and mark that optional task-level result unavailable.
        evaluation = _SCHEMA.unavailable("Predicate evaluation unavailable: %s" % exc, source)
    return goals, evaluation


def _unique_key(candidates, fallback, used, reserved):
    for candidate in candidates:
        if candidate is not None and candidate not in used:
            return candidate
    key = fallback
    suffix = 1
    # Reserve every native name/semantic alias up front, including later bodies,
    # so an index-based identifier cannot steal or overwrite an actual name.
    while key in used or key in reserved:
        key = "%s:%s" % (fallback, suffix)
        suffix += 1
    return key


def read_privileged_state(env, mat2quat):
    """Read a new snapshot; task dictionaries enrich but never define scene truth."""
    native = getattr(env, "env", env)
    model, data = native.sim.model, native.sim.data
    aliases = _semantic_aliases(native, model)
    joint_bodies = {int(model.jnt_bodyid[i]) for i in range(int(model.njnt))}
    body_names = {i: _name(model, "body", i) for i in range(int(model.nbody))}
    names = {}
    used = set()
    reserved = {name for name in body_names.values() if name is not None}
    reserved.update(name for values in aliases.values() for name in values)
    for body_id, native_name in body_names.items():
        candidates = aliases.get(body_id, []) + ([native_name] if native_name else [])
        key = _unique_key(candidates, "body:%s" % body_id, used, reserved)
        names[body_id] = key
        used.add(key)
    objects = {}
    for body_id, native_name in body_names.items():
        objects[names[body_id]] = {
            **_pose(data, body_id),
            "body_type": _body_type(model, body_id, joint_bodies),
            "linear_velocity_m_s": _velocity(data, native_name, "p") if native_name else None,
            "angular_velocity_rad_s": _velocity(data, native_name, "r") if native_name else None,
            "native_name": native_name,
            "body_id": body_id,
            "aliases": aliases.get(body_id, []),
            "parent_name": names[int(model.body_parentid[body_id])] if body_id else None,
        }
    articulations = {}
    joint_names = {i: _name(model, "joint", i) for i in range(int(model.njnt))}
    reserved_joint_names = {name for name in joint_names.values() if name is not None}
    for joint_id in range(int(model.njnt)):
        body_id = int(model.jnt_bodyid[joint_id])
        root_id = _root(model, body_id)
        key = names[root_id]
        if key not in articulations:
            articulations[key] = {
                **_pose(data, root_id),
                "native_name": body_names[root_id],
                "root_body_id": root_id,
                "joints": {},
            }
        native_name = joint_names[joint_id]
        joint_name = _unique_key(
            [native_name],
            "joint:%s" % joint_id,
            articulations[key]["joints"],
            reserved_joint_names,
        )
        articulations[key]["joints"][joint_name] = {
            **_joint(model, data, joint_id),
            "native_name": native_name,
            "body_name": names[body_id],
        }
    goals, evaluation = _task_sections(native)
    return _SCHEMA.build_privilege(
        simulator="mujoco",
        adapter="libero",
        objects=objects,
        articulations=articulations,
        regions=_regions(native, model, data, mat2quat),
        goals=goals,
        evaluation=evaluation,
    )
