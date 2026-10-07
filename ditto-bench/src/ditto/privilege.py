"""Optional task-declared metadata; physical state belongs to the simulator adapter.

These fields are the suite's existing public task-spec contract. No positions,
regions, grasp points or predicates are inferred from geometry or task names.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


def task_privilege_metadata(task_spec):
    if not isinstance(task_spec, Mapping):
        return {"regions": None, "goals": None}
    # New tasks may provide explicit structured sections. Existing tasks already
    # declare these goal fields; keep only them, not authored grasp suggestions,
    # initial placements or privileged validation poses.
    if "goals" in task_spec:
        goals = task_spec["goals"]
    else:
        definition = {
            key: task_spec[key]
            for key in ("goal", "goal_position", "goal_quaternion", "success")
            if task_spec.get(key) is not None
        }
        goals = {"definition": definition} if definition else None
        if goals is not None and "goal_position" in definition:
            goals["goal_position_semantics"] = "Declared reference position in world meters; not a full success predicate"
        if goals is not None and "goal_quaternion" in definition:
            # assets.py's documented native convention; do not silently label
            # this task-definition array as a normalized live xyzw pose.
            goals["goal_quaternion_semantics"] = "Declared reference orientation in world coordinates, wxyz"
    return deepcopy({"regions": task_spec.get("regions"), "goals": goals})
