"""Read-only scene recording contract shared by simulator workers (Python 3.8).

This module imports neither the framework nor a simulator. Workers may load it by
path. Recording truth is separate from the privileged model observation contract.
"""

import math
from collections.abc import Mapping

import numpy as np

RECORDING_STATE_SCHEMA_VERSION = 2


def recording_json(value, label="recording state"):
    """Detach finite JSON, including NumPy and optional tensor-like native values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("%s contains nonfinite numeric data" % label)
        return value
    if isinstance(value, np.generic):
        return recording_json(value.item(), label)
    if isinstance(value, np.ndarray):
        return recording_json(value.tolist(), label)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("%s requires string JSON keys" % label)
        return {key: recording_json(item, label + "." + key) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [recording_json(item, label + "[]") for item in value]
    # Torch remains optional. Never stringify opaque simulator objects.
    detach = getattr(value, "detach", None)
    if callable(detach):
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        if callable(cpu):
            detached = cpu()
        tolist = getattr(detached, "tolist", None)
        if callable(tolist):
            return recording_json(tolist(), label)
    raise ValueError("%s contains unsupported %s" % (label, type(value).__name__))


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % label)
    return value


def _text(value, label, nullable=False):
    if nullable and value is None:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty string" % label)
    return value


def _number(value, label, nonnegative=False, nullable=False):
    if nullable and value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("%s must be finite numeric data" % label)
    if nonnegative and value < 0:
        raise ValueError("%s must be nonnegative" % label)


def _vector(value, width, label):
    if not isinstance(value, list) or len(value) != width:
        raise ValueError("%s must contain %s numbers" % (label, width))
    for item in value:
        _number(item, label)


def _quaternion(value, label):
    _vector(value, 4, label)
    if not math.isclose(sum(item * item for item in value), 1.0, rel_tol=0.0, abs_tol=2e-4):
        raise ValueError("%s must be a unit XYZW quaternion" % label)


def _pose(value, label):
    _mapping(value, label)
    _vector(value.get("position"), 3, label + ".position")
    _quaternion(value.get("quaternion_xyzw"), label + ".quaternion_xyzw")


def _joints(value, label):
    joints = _mapping(value, label)
    widths = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}
    for name, joint in joints.items():
        _text(name, label + " identity")
        _mapping(joint, label + "." + name)
        kind = _text(joint.get("type"), label + "." + name + ".type")
        _text(joint.get("unit"), label + "." + name + ".unit")
        values = joint.get("value")
        if not isinstance(values, list) or not values:
            raise ValueError("%s.%s.value requires numeric values" % (label, name))
        for item in values:
            _number(item, label + "." + name + ".value")
        if kind in widths and len(values) != widths[kind]:
            raise ValueError("%s.%s has wrong native joint dimension" % (label, name))
        if kind in ("free", "ball"):
            _quaternion(values[-4:], label + "." + name + ".quaternion_xyzw")


def validate_recording_state(value, *, expected_step=None, control_hz=None):
    """Validate v1/v2 and return a detached finite JSON document.

    Unknown extension fields are retained; all their numeric values must be finite.
    Pose conventions and declared topology are strict. Missing task truth remains
    explicit and does not invalidate otherwise available measured scene poses.
    """
    result = _mapping(recording_json(value), "recording state")
    version = result.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("recording state schema_version must be 1 or 2")
    _text(result.get("adapter"), "adapter")
    for key, expected in (("frame", "world"), ("position_unit", "m"),
                          ("quaternion_order", "xyzw")):
        if result.get(key) != expected:
            raise ValueError("recording state %s must be %s" % (key, expected))
    if result.get("policy_visible") is not False:
        raise ValueError("recording state must declare policy_visible=false")
    step = result.get("control_step")
    if type(step) is not int or step < 0:
        raise ValueError("control_step must be a nonnegative integer")
    if expected_step is not None and step != expected_step:
        raise ValueError("recording state control_step does not match confirmed step")
    _number(result.get("policy_time_s"), "policy_time_s", nonnegative=True)
    _number(result.get("simulation_time_s"), "simulation_time_s", nonnegative=True,
            nullable=version == 2)
    if control_hz is not None:
        _number(control_hz, "control_hz", nonnegative=True)
        if control_hz <= 0 or not math.isclose(result["policy_time_s"], step / control_hz,
                                               rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("recording state policy_time_s does not match control_step/control_hz")
    objects = _mapping(result.get("objects"), "objects")
    if version == 1 and not objects:
        raise ValueError("legacy recording state objects cannot be empty")
    for name, entity in objects.items():
        label = "objects." + _text(name, "object identity")
        _pose(entity, label)
        _text(entity.get("kind"), label + ".kind")
        _text(entity.get("source_body"), label + ".source_body", nullable=version == 2)
        roles = entity.get("roles")
        if (not isinstance(roles, list) or any(not isinstance(role, str) for role in roles)
                or len(set(roles)) != len(roles)):
            raise ValueError("%s.roles must be unique strings" % label)
        for role in roles:
            _text(role, label + ".roles")
        _joints(entity.get("joint_positions"), label + ".joint_positions")
    if version == 1:
        return result
    bodies = _mapping(result.get("bodies"), "bodies")
    if not bodies:
        raise ValueError("recording state requires measured bodies")
    for name, entity in bodies.items():
        label = "bodies." + _text(name, "body identity")
        _pose(entity, label)
        _text(entity.get("kind"), label + ".kind")
        _text(entity.get("source_body"), label + ".source_body", nullable=True)
        if "is_robot" in entity and entity["is_robot"] is not None and type(entity["is_robot"]) is not bool:
            raise ValueError("%s.is_robot must be boolean or null" % label)
        parent = entity.get("parent_body")
        if parent is not None:
            _text(parent, label + ".parent_body")
            if parent not in bodies or parent == name:
                raise ValueError("%s has an invalid parent body identity" % label)
        if "joint_positions" in entity:
            _joints(entity["joint_positions"], label + ".joint_positions")
    for name in bodies:
        current, visited = name, set()
        while current is not None:
            if current in visited:
                raise ValueError("recording state body parent cycle")
            visited.add(current)
            current = bodies[current].get("parent_body")
    for name, entity in objects.items():
        if "body_ids" in entity:
            ids = entity["body_ids"]
            if not isinstance(ids, list) or any(not isinstance(item, str) or item not in bodies for item in ids):
                raise ValueError("objects.%s.body_ids must reference measured bodies" % name)
            if len(ids) != len(set(ids)):
                raise ValueError("objects.%s.body_ids contains duplicates" % name)
    articulations = _mapping(result.get("articulations", {}), "articulations")
    for name, articulation in articulations.items():
        label = "articulations." + _text(name, "articulation identity")
        _mapping(articulation, label)
        if "position" in articulation or "quaternion_xyzw" in articulation:
            _pose(articulation, label)
        for link, entity in _mapping(articulation.get("links", {}), label + ".links").items():
            _pose(entity, label + ".links." + _text(link, "link identity"))
        for joint, entity in _mapping(articulation.get("joints", {}), label + ".joints").items():
            _mapping(entity, label + ".joints." + _text(joint, "joint identity"))
    task = _mapping(result.get("task"), "task")
    if task.get("availability") not in ("available", "unavailable"):
        raise ValueError("task.availability must be available or unavailable")
    _text(task.get("source"), "task.source", nullable=task["availability"] == "unavailable")
    _mapping(task.get("data"), "task.data")
    if "reason" in task:
        _text(task["reason"], "task.reason")
    _mapping(result.get("coverage"), "coverage")
    return result


def build_recording_state(*, adapter, control_step, control_hz, simulation_time_s,
                          objects, bodies, task, coverage, articulations=None, role_source=None):
    """Build one synchronized scene snapshot; does not evaluate task predicates."""
    _number(control_hz, "control_hz", nonnegative=True)
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if type(control_step) is not int or control_step < 0:
        raise ValueError("control_step must be a nonnegative integer")
    document = {
        "schema_version": RECORDING_STATE_SCHEMA_VERSION, "adapter": adapter,
        "frame": "world", "position_unit": "m", "quaternion_order": "xyzw",
        "policy_visible": False, "control_step": control_step,
        "policy_time_s": control_step / control_hz, "simulation_time_s": simulation_time_s,
        "objects": objects, "bodies": bodies, "task": task, "coverage": coverage,
    }
    if articulations is not None:
        document["articulations"] = articulations
    if role_source is not None:
        document["role_source"] = role_source
    return validate_recording_state(document, expected_step=control_step, control_hz=control_hz)


def recording_state_catalog(value):
    """Stable entity, topology and joint identities, excluding measured state."""
    state = validate_recording_state(value)

    def joint_catalog(joints):
        return {name: {key: joint[key] for key in ("type", "unit", "parent", "child") if key in joint}
                for name, joint in joints.items()}

    identity_keys = ("kind", "source_body", "parent_body", "object_name", "is_robot", "native_id", "body_ids",
                     "parent_link", "root_body", "root_link", "link_id", "actor_id", "body_type")
    catalog = {"schema_version": state["schema_version"]}
    for section in ("objects", "bodies"):
        catalog[section] = {
            name: {**{key: entity[key] for key in identity_keys if key in entity},
                   "joint_positions": joint_catalog(entity.get("joint_positions", {}))}
            for name, entity in state.get(section, {}).items()
        }
    catalog["articulations"] = {
        name: {**{key: entity[key] for key in identity_keys if key in entity},
               "joints": joint_catalog(entity.get("joints", {})),
               "links": {link: {key: item[key] for key in identity_keys if key in item}
                         for link, item in entity.get("links", {}).items()}}
        for name, entity in state.get("articulations", {}).items()
    }
    return catalog
