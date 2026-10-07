"""Recorded native task assessments; never calculate task-specific progress."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


class TaskFeedbackRecordingError(ValueError):
    """The physical sample was saved, but some native feedback was invalid."""


def sanitize_task_feedback(value, *, path="task_feedback"):
    """Preserve finite native values; replace invalid leaves with audited nulls.

    Only task/reset/step feedback uses this repair. Robot and object poses retain
    strict validation, so invalid measurements never become fabricated states.
    """
    issues, ancestors = [], set()

    def invalid(item, location, error):
        issues.append({"path": location, "error": error, "type": type(item).__name__})
        return None

    def visit(item, location, depth=0):
        if depth > 64:
            return invalid(item, location, "maximum_nesting_exceeded")
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else invalid(item, location, "nonfinite_number")
        if isinstance(item, np.generic):
            return visit(item.item(), location, depth + 1)
        if isinstance(item, Path):
            return str(item)
        if id(item) in ancestors:
            return invalid(item, location, "cyclic_reference")
        ancestors.add(id(item))
        try:
            if isinstance(item, np.ndarray):
                return visit(item.tolist(), location, depth + 1)
            if isinstance(item, Mapping):
                result = {}
                for key, child in item.items():
                    if not isinstance(key, str):
                        invalid(key, location + ".<invalid_key>", "nonstring_mapping_key_omitted")
                        continue
                    result[key] = visit(child, location + "." + key, depth + 1)
                return result
            if isinstance(item, (list, tuple)):
                return [visit(child, "%s[%d]" % (location, index), depth + 1)
                        for index, child in enumerate(item)]
            if isinstance(item, (set, frozenset)):
                return [visit(child, "%s[%d]" % (location, index), depth + 1)
                        for index, child in enumerate(sorted(item, key=repr))]
            if is_dataclass(item) and not isinstance(item, type):
                return {field.name: visit(getattr(item, field.name), location + "." + field.name, depth + 1)
                        for field in fields(item)}
            detach = getattr(item, "detach", None)
            if callable(detach):
                try:
                    detached = detach()
                    if callable(getattr(detached, "cpu", None)):
                        detached = detached.cpu()
                    if callable(getattr(detached, "tolist", None)):
                        return visit(detached.tolist(), location, depth + 1)
                except Exception as exc:
                    return invalid(item, location, "tensor_conversion_failed:" + type(exc).__name__)
            return invalid(item, location, "unsupported_value")
        finally:
            ancestors.remove(id(item))

    cleaned = visit(value, path)
    return cleaned, {"complete": not issues, "status": "partial" if issues else "complete", "issues": issues}


def sanitize_task_section(state):
    """Repair only a returned task receipt, leaving measured scene truth strict."""
    if not isinstance(state, Mapping) or not isinstance(state.get("task"), Mapping):
        return state
    task, integrity = sanitize_task_feedback(state["task"], path="environment_state.task")
    if not integrity["complete"]:
        prior = task.get("integrity")
        issues = list(prior.get("issues", [])) if isinstance(prior, Mapping) else []
        task["integrity"] = {**integrity, "issues": issues + integrity["issues"]}
    return {**state, "task": task}


def task_evaluation(state, feedback, *, step, sim_time_s, feedback_integrity=None):
    native = state.get("task") if isinstance(state, Mapping) else None
    cleaned_feedback, integrity = sanitize_task_feedback(feedback, path="environment_feedback")
    issues = list(integrity["issues"])
    if isinstance(feedback_integrity, Mapping):
        issues.extend(feedback_integrity.get("issues", []))
    if isinstance(native, Mapping):
        receipt, native_integrity = sanitize_task_feedback(native, path="environment_state.task")
        issues.extend(native_integrity["issues"])
        previous = receipt.get("integrity")
        if isinstance(previous, Mapping):
            issues.extend(previous.get("issues", []))
    elif cleaned_feedback:
        receipt = {"availability": "available", "source": "environment.reset/step feedback",
                   "data": cleaned_feedback}
    else:
        receipt = {"availability": "unavailable", "source": None, "data": {},
                   "reason": "No native task feedback was recorded"}
    receipt["environment_feedback"] = cleaned_feedback
    if issues:
        receipt["integrity"] = {"complete": False, "status": "partial", "issues": issues,
                                "semantics": "invalid native feedback leaves are null; other measured values retained"}
    receipt.update(control_step=step, sim_time_s=sim_time_s)
    return receipt


def task_progress(evaluation):
    """Expose an explicitly named scalar and its exact path, without rescaling.

    Full feedback remains in task_evaluation even for custom metric names or
    vectors. A count of completed goals is not a progress fraction.
    """
    candidates = []

    def visit(value, path, depth=0):
        if not isinstance(value, Mapping) or depth > 4:
            return
        for key in ("progress_score", "task_progress", "progress", "completion_rate"):
            if key not in value:
                continue
            item = value[key]
            # Single-environment evaluators may retain a singleton batch axis.
            if isinstance(item, list) and len(item) == 1:
                item = item[0]
            if type(item) in (int, float) and math.isfinite(item):
                candidates.append((key, path + "." + key, item))
        for key in ("info", "native_task_info", "evaluation", "metrics", "task", "task_metrics"):
            if key in value:
                visit(value[key], path + "." + key, depth + 1)

    if evaluation.get("availability") == "available":
        visit(evaluation.get("data"), "task_evaluation.data")
    visit(evaluation.get("environment_feedback"), "task_evaluation.environment_feedback")
    if candidates:
        order = {key: index for index, key in enumerate(
            ("progress_score", "task_progress", "progress", "completion_rate"))}
        candidates.sort(key=lambda item: (order[item[0]], item[1]))
        field, path, value = candidates[0]
        return {"availability": "available", "value": value, "field": field,
                "source": ("environment.reset/step feedback"
                           if path.startswith("task_evaluation.environment_feedback")
                           else evaluation.get("source")), "path": path,
                "control_step": evaluation.get("control_step"),
                "semantics": "native scalar as reported; range and normalization are not inferred",
                "reported_candidates": [{"field": key, "path": name, "value": item}
                                        for key, name, item in candidates]}
    return {"availability": "unavailable", "value": None, "field": None, "path": None,
            "source": evaluation.get("source"), "control_step": evaluation.get("control_step"),
            "reason": "No explicitly named finite scalar progress was provided; other completion information remains in task_evaluation"}
