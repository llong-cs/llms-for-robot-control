"""Read-only recording of every registered ManiSkill actor/articulation/link.

This is a recorder RPC side channel. It never invokes a task evaluator or a
privileged observation collector and never resets, steps, or edits the scene.
The existing privilege module supplies only its pure numeric pose/joint readers.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

import numpy as np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_privilege = _load("_recording_maniskill_pose_readers", Path(__file__).with_name("privilege.py"))
_schema = _load("_recording_environment_schema", Path(__file__).resolve().parents[1] / "recording_state.py")


def native_info(value):
    """Detach all native task fields, retaining tensor shapes and nested keys."""
    if not isinstance(value, Mapping):
        raise ValueError("Native ManiSkill info must be a mapping")
    return _privilege._common.plain(value)


def _registry(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"Native scene {label} must be a named mapping")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"Native scene {label} requires nonempty registry names")
    return value


def _name(value):
    name = getattr(value, "name", None)
    return name if isinstance(name, str) and name else None


def _pose(value, label):
    result = _privilege._pose(value.pose, label)
    return {"position": result["position_m"], "quaternion_xyzw": result["quaternion_xyzw"]}


def _joint_positions(joints):
    result = {}
    for name, joint in joints.items():
        kind = joint["type"]
        # Legacy joint_positions describes generalized coordinates only. Fixed
        # and unknown-type topology remains complete in articulations.joints.
        if not joint["position"] or kind is None:
            continue
        result[name] = {
            "type": kind, "value": joint["position"], "velocity": joint["velocity"],
            "unit": ("m" if kind == "prismatic" else "rad"
                     if kind in ("revolute", "revolute_unwrapped") else "none"
                     if not joint["position"] else "native generalized coordinates"),
            "position_semantics": joint["position_semantics"],
            "velocity_semantics": joint["velocity_semantics"],
        }
    return result


def _links(articulation, name, prefix, *, is_robot):
    getter = getattr(articulation, "get_links", None)
    links = getter() if callable(getter) else getattr(articulation, "links", None)
    if links is None:
        raise ValueError(f"Native articulation {name!r} does not expose link topology")
    result = {}
    for index, link in enumerate(links):
        native_name = _name(link)
        key = native_name if native_name is not None else f"@link:{index}"
        if key in result:
            raise ValueError(f"Duplicate articulation link name {name}/{key}")
        result[key] = {
            **_pose(link, f"{name}/{key}"), "kind": "articulation_link",
            "source_body": native_name, "native_index": index,
            "articulation": name, "is_robot": is_robot,
            "body_key": prefix + "/link:" + quote(key, safe=""),
        }
    if not result:
        raise ValueError(f"Native articulation {name!r} exposes no links")
    return result


def read_recording_state(native, control_step, control_hz, *, task=None):
    """Enumerate scene registries; every pose is freshly measured in world space."""
    actors = _registry(native.scene.actors, "actors")
    articulations = _registry(native.scene.articulations, "articulations")
    robot = getattr(getattr(native, "agent", None), "robot", None)
    objects, bodies, articulation_state = {}, {}, {}
    for name, actor in sorted(actors.items()):
        measured = _privilege._actor(actor, name)
        source_body = _name(actor)
        kind = "fixture" if measured["body_type"] == "static" else "object"
        pose = {"position": measured.pop("position_m"),
                "quaternion_xyzw": measured.pop("quaternion_xyzw")}
        objects[name] = {
            **pose, **measured, "kind": kind, "roles": [],
            "source_body": source_body, "joint_positions": {},
            "registry": "scene.actors", "registry_name": name,
        }
        key = "actor:" + quote(name, safe="")
        bodies[key] = {
            **pose, "kind": kind, "source_body": source_body,
            "object_name": name, "is_robot": actor is robot,
            "registry": "scene.actors", "registry_name": name,
        }
    for name, articulation in sorted(articulations.items()):
        measured = _privilege._articulation(articulation, name)
        pose = {"position": measured.pop("position_m"),
                "quaternion_xyzw": measured.pop("quaternion_xyzw")}
        prefix = "articulation:" + quote(name, safe="")
        object_key = "@" + prefix
        if object_key in objects:
            raise ValueError(f"Actor/articulation recording keys collide: {object_key}")
        source_body = _name(articulation)
        is_robot = articulation is robot
        links = _links(articulation, name, prefix, is_robot=is_robot)
        objects[object_key] = {
            **pose, "kind": "articulation", "roles": [], "source_body": source_body,
            "joint_positions": _joint_positions(measured["joints"]),
            "registry": "scene.articulations", "registry_name": name, "is_robot": is_robot,
        }
        bodies[prefix + "/root"] = {
            **pose, "kind": "articulation_root", "source_body": source_body,
            "object_name": object_key, "articulation": name, "is_robot": is_robot,
        }
        for link in links.values():
            bodies[link["body_key"]] = {**link, "object_name": object_key}
        articulation_state[name] = {
            **pose, **measured, "source_body": source_body,
            "is_robot": is_robot, "links": links,
        }
    coverage = {
        "status": "complete", "source": "native.scene.actors and native.scene.articulations",
        "scope": "all registered actors, articulation roots and every native articulation link",
        "static_actors_included": True, "robot_links_included": True,
        "robot_identity_source": "identity with native.agent.robot; no name heuristics",
        "actor_count": len(actors), "articulation_count": len(articulations),
        "body_count": len(bodies),
        "simulation_time_availability": "unavailable",
        "time_source": "no independent physical simulation clock is exposed by this adapter",
        "task_evaluation_recomputed": False,
    }
    elapsed = getattr(native, "elapsed_steps", None)
    if elapsed is not None:
        values = _privilege._array(elapsed)
        if values.size != 1 or not np.isfinite(values).all() or values.item() < 0:
            raise ValueError("Native elapsed_steps must be one finite nonnegative value")
        coverage["native_elapsed_steps"] = values.item()
        coverage["derived_simulation_time_s"] = values.item() / control_hz
        coverage["derived_time_source"] = "native.elapsed_steps / control_hz; logical control time"
    if task is None:
        task = {"availability": "unavailable", "source": None, "data": {},
                "reason": "No cached native reset/step task receipt is available"}
    return _schema.build_recording_state(
        adapter="maniskill", control_step=control_step, control_hz=control_hz,
        simulation_time_s=None, objects=objects, bodies=bodies,
        articulations=articulation_state, task=task, coverage=coverage,
        role_source="No task roles inferred; scene registry identities only",
    )
