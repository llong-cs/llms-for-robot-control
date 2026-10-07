"""Simulator-independent, JSON-only privilege contract (also usable on Python 3.8).

Workers load this module by path in their isolated interpreter. It imports no
simulator or policy packages. Optional task information never gets fabricated.
"""

import math
from collections.abc import Mapping

PRIVILEGE_KEY = "privileged"
LEGACY_PRIVILEGE_KEYS = ("privileged_libero", "privileged_maniskill")
SCHEMA_VERSION = 2


def plain(value):
    """Copy measured values into finite JSON data, without importing torch/numpy."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Privilege values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Privilege mapping keys must be strings")
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError("Unsupported privilege value: %s" % type(value).__name__)


def available(data, source):
    """An explicitly provided section; an empty collection is still provided."""
    if data is None:
        raise ValueError("Available privilege sections must contain data")
    return {"available": True, "data": plain(data), "source": source, "reason": None}


def unavailable(reason, source=None):
    """Distinguish an absent capability from a known empty result or false goal."""
    return {"available": False, "data": None, "source": source, "reason": reason}


def build_privilege(simulator, adapter, objects, articulations,
                    regions=None, goals=None, evaluation=None):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {"simulator": simulator, "adapter": adapter},
        "coordinate_frame": "world",
        "quaternion_convention": "xyzw",
        "objects": objects,
        "articulations": articulations,
        "regions": regions if regions is not None else unavailable("Task does not provide regions"),
        "goals": goals if goals is not None else unavailable("Task does not provide goal definitions"),
        "evaluation": evaluation if evaluation is not None else unavailable("Task does not provide evaluation metrics"),
    }
    return validate_privilege(payload)


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping" % label)
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty string" % label)


def _numbers(value, label, length=None):
    if (not isinstance(value, list) or (length is not None and len(value) != length)
            or any(type(item) not in (int, float) for item in value)):
        raise ValueError("%s must be a numeric list%s" % (
            label, " of length %d" % length if length is not None else ""))


def _pose(value, label):
    _mapping(value, label)
    _numbers(value.get("position_m"), label + ".position_m", 3)
    quat = value.get("quaternion_xyzw")
    _numbers(quat, label + ".quaternion_xyzw", 4)
    if not math.isclose(sum(item * item for item in quat), 1.0, abs_tol=2e-5):
        raise ValueError("%s.quaternion_xyzw must have unit length" % label)


def _section(value, label):
    _mapping(value, label)
    if set(value) != {"available", "data", "source", "reason"}:
        raise ValueError("%s must declare available, data, source and reason" % label)
    if type(value["available"]) is not bool:
        raise ValueError("%s.available must be boolean" % label)
    if value["source"] is not None:
        _text(value["source"], label + ".source")
    if value["available"]:
        _text(value["source"], label + ".source")
        if value["data"] is None or value["reason"] is not None:
            raise ValueError("%s: available requires data and no missing reason" % label)
    else:
        _text(value["reason"], label + ".reason")
        if value["data"] is not None:
            raise ValueError("%s: unavailable data must be null" % label)


def validate_privilege(payload, expected_adapter=None):
    """Return a validated independent copy of the v2 live-sensor payload."""
    result = _mapping(plain(payload), "privileged")
    if type(result.get("schema_version")) is not int or result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("privileged requires schema_version=2")
    source = _mapping(result.get("source"), "privileged.source")
    _text(source.get("simulator"), "privileged.source.simulator")
    _text(source.get("adapter"), "privileged.source.adapter")
    if expected_adapter is not None and source["adapter"] != expected_adapter:
        raise ValueError("privileged source adapter must be %s" % expected_adapter)
    if result.get("coordinate_frame") != "world" or result.get("quaternion_convention") != "xyzw":
        raise ValueError("privileged poses require world coordinates and xyzw quaternions")
    objects = _mapping(result.get("objects"), "privileged.objects")
    for name, entity in objects.items():
        _text(name, "object name")
        _pose(entity, "object " + name)
        if "body_type" not in entity:
            raise ValueError("object %s must declare body_type (null if unavailable)" % name)
        if entity["body_type"] is not None:
            _text(entity["body_type"], "object body_type")
        for field in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            if field not in entity:
                raise ValueError("object %s must declare %s (null if unavailable)" % (name, field))
            if entity[field] is not None:
                _numbers(entity[field], "object " + field, 3)
    articulations = _mapping(result.get("articulations"), "privileged.articulations")
    for name, entity in articulations.items():
        _text(name, "articulation name")
        _pose(entity, "articulation " + name)
        joints = _mapping(entity.get("joints"), "articulation joints")
        for joint_name, joint in joints.items():
            _text(joint_name, "joint name")
            _mapping(joint, "joint " + joint_name)
            if "type" not in joint or "velocity" not in joint or "velocity_semantics" not in joint:
                raise ValueError("joints must declare type, velocity and velocity_semantics")
            if joint["type"] is not None:
                _text(joint["type"], "joint type")
            _numbers(joint.get("position"), "joint position")
            _text(joint.get("position_semantics"), "joint position_semantics")
            if joint["velocity"] is not None:
                _numbers(joint["velocity"], "joint velocity")
                _text(joint["velocity_semantics"], "joint velocity_semantics")
            elif joint["velocity_semantics"] is not None:
                _text(joint["velocity_semantics"], "joint velocity_semantics")
    for field in ("regions", "goals", "evaluation"):
        _section(result.get(field), "privileged." + field)
    return result
