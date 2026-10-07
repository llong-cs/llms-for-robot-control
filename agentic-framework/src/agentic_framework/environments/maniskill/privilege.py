"""Task-independent SAPIEN scene truth for a single ManiSkill environment.

Only native scene registries, measured state, an explicit optional metadata
provider, and the environment's own evaluator are read. No task names, actor
name patterns, initial poses, or reconstructed goal predicates are used.
This module can be loaded by path in an isolated simulator interpreter.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_agentic_privilege_schema", Path(__file__).resolve().parents[1] / "privilege.py",
)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _vector(value, size, label):
    values = _array(value)
    if values.shape == (1, size):
        values = values[0]
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must have {size} finite values for one environment")
    return values.tolist()


def _row(value, label):
    values = _array(value)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite vector for one environment")
    return values.tolist()


def _pose(value, label):
    quaternion = _vector(value.q, 4, f"{label} quaternion wxyz")
    if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-5):
        raise ValueError(f"{label} quaternion is not unit length")
    return {"position_m": _vector(value.p, 3, f"{label} position"),
            "quaternion_xyzw": quaternion[1:] + quaternion[:1]}


def _optional_call(value, method):
    function = getattr(value, method, None)
    if not callable(function):
        return None
    try:
        return function()
    except (AttributeError, NotImplementedError):
        return None


def _velocity(value, method, label):
    measured = _optional_call(value, method)
    return None if measured is None else _vector(measured, 3, label)


def _actor(actor, name):
    body_type = getattr(actor, "px_body_type", None)
    # PhysxRigidStaticComponent has no velocity API. Do not fabricate zero or
    # call the inherited dynamic getter (which reads nonexistent static data).
    static = body_type == "static"
    return {
        **_pose(actor.pose, name), "body_type": body_type,
        "linear_velocity_m_s": None if static else _velocity(
            actor, "get_linear_velocity", f"{name} linear velocity"),
        "angular_velocity_rad_s": None if static else _velocity(
            actor, "get_angular_velocity", f"{name} angular velocity"),
    }


def _joint_type(joint):
    value = _optional_call(joint, "get_type")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if value is not None and not isinstance(value, str):
        raise ValueError("Joint type must describe one environment")
    return value


def _joint_dof(joint):
    value = _array(joint.get_dof())
    if (value.size != 1 or not np.isfinite(value).all()
            or value.item() < 0 or int(value.item()) != value.item()):
        raise ValueError("Joint DOF must be one nonnegative integer")
    return int(value.item())


def _joint_semantics(kind, active):
    if not active:
        return "No generalized coordinates; articulation root pose is reported separately", \
            "No generalized velocity coordinates"
    if kind == "prismatic":
        return "Native SAPIEN prismatic displacement in meters", "Meters per second"
    if kind in ("revolute", "revolute_unwrapped"):
        return "Native SAPIEN joint angle in radians", "Radians per second"
    return (f"Native SAPIEN generalized coordinates for joint type {kind!r}; "
            "no representation conversion inferred"), "Native SAPIEN generalized velocities"


def _articulation(articulation, name):
    result = {
        **_pose(articulation.pose, name),
        "linear_velocity_m_s": _velocity(
            articulation, "get_root_linear_velocity", f"{name} root linear velocity"),
        "angular_velocity_rad_s": _velocity(
            articulation, "get_root_angular_velocity", f"{name} root angular velocity"),
        "joints": {},
    }
    joints = _optional_call(articulation, "get_joints")
    active = _optional_call(articulation, "get_active_joints")
    if joints is None or active is None:
        result["joints_available"] = False
        result["joints_unavailable_reason"] = "Native articulation does not expose joint topology"
        return result
    joints, active = list(joints), list(active)
    if any(all(joint is not candidate for candidate in joints) for joint in active):
        raise ValueError("Active joint is missing from native articulation joint topology")
    qpos = _row(articulation.get_qpos(), f"{name} qpos")
    raw_qvel = _optional_call(articulation, "get_qvel")
    qvel = None if raw_qvel is None else _row(raw_qvel, f"{name} qvel")
    offsets, offset = {}, 0
    for joint in active:
        dof = _joint_dof(joint)
        offsets[id(joint)] = (offset, offset + dof)
        offset += dof
    if len(qpos) != offset or (qvel is not None and len(qvel) != offset):
        raise ValueError("Native generalized state size disagrees with active joint DOFs")
    raw_limits = _optional_call(articulation, "get_qlimits")
    limits = None
    if raw_limits is not None:
        limits = _array(raw_limits)
        if limits.shape == (1, offset, 2):
            limits = limits[0]
        if limits.shape != (offset, 2):
            raise ValueError("Native joint limits must have shape (1, dof, 2) or (dof, 2)")
    for index, joint in enumerate(joints):
        native_name = getattr(joint, "name", None)
        # Some native root joints are unnamed. Preserve that fact and identify
        # them by their measured topology index instead of inventing a name.
        key = native_name if isinstance(native_name, str) and native_name else f"@joint:{index}"
        if key in result["joints"]:
            raise ValueError(f"Duplicate articulation joint name {key!r}")
        is_active = id(joint) in offsets
        kind = _joint_type(joint)
        position_semantics, velocity_semantics = _joint_semantics(kind, is_active)
        start, end = offsets.get(id(joint), (0, 0))
        joint_limits = None
        if limits is not None and np.isfinite(limits[start:end]).all():
            joint_limits = limits[start:end].tolist()
        result["joints"][key] = {
            "name": native_name, "native_index": index, "type": kind,
            "position": qpos[start:end] if is_active else [],
            "velocity": (qvel[start:end] if is_active else []) if qvel is not None else None,
            "position_semantics": position_semantics,
            "velocity_semantics": velocity_semantics if qvel is not None else None,
            "limits": joint_limits,
        }
    result["joints_available"] = True
    return result


def _task_sections(native):
    provider = getattr(native, "get_privileged_task_metadata", None)
    source = "native.get_privileged_task_metadata()"
    if not callable(provider):
        return {name: _common.unavailable("Environment has no explicit task metadata provider")
                for name in ("regions", "goals")}
    try:
        metadata = provider()
    except NotImplementedError:
        metadata = None
    if metadata is None:
        return {name: _common.unavailable("Task metadata provider returned no metadata", source)
                for name in ("regions", "goals")}
    if not isinstance(metadata, Mapping):
        raise ValueError("Task metadata provider must return a mapping or None")
    return {
        name: (_common.available(metadata[name], source)
               if name in metadata and metadata[name] is not None
               else _common.unavailable(f"Task metadata provider does not expose {name}", source))
        for name in ("regions", "goals")
    }


def _evaluation(native):
    evaluator = getattr(native, "evaluate", None)
    source = "native.evaluate()"
    if not callable(evaluator):
        return _common.unavailable("Environment does not expose an evaluator")
    try:
        data = evaluator()
    except NotImplementedError:
        data = None
    if data is None:
        return _common.unavailable("Native evaluator returned no evaluation data", source)
    if not isinstance(data, Mapping):
        raise ValueError("Native evaluator must return a mapping or None")
    return _common.available(data, source)


def collect(native):
    """Collect registered actors/articulations and explicitly available task data.

    This function never steps/resets the simulator or consults the environment
    ID, task-specific attributes, initial states, or a state-dict registry that
    could omit static fixtures. Required malformed measurements fail visibly.
    """
    scene = native.scene
    actors, articulations = scene.actors, scene.articulations
    if not isinstance(actors, Mapping) or not isinstance(articulations, Mapping):
        raise ValueError("Native scene actors and articulations must be named mappings")
    sections = _task_sections(native)
    return _common.build_privilege(
        simulator="sapien", adapter="maniskill",
        objects={name: _actor(actor, name) for name, actor in sorted(actors.items())},
        articulations={name: _articulation(value, name)
                       for name, value in sorted(articulations.items())},
        regions=sections["regions"], goals=sections["goals"], evaluation=_evaluation(native),
    )
