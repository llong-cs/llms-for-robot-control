"""Deterministic LIBERO task selection and official initial-state expansion.

Simulator imports and initial-state loading belong to the embodiment's
isolated worker.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from inspect_robots.scene import Scene

# Pinned OpenPI examples/libero/main.py: longest-demonstration rollout limits.
# These count policy-controlled simulator steps, excluding settling steps.
HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

class TaskListingEmbodiment(Protocol):
    benchmark: str

    def list_tasks(self, suite: str, task_ids: list[int] | None = None) -> list[dict]: ...


def parse_task_ids(value: str) -> list[int] | None:
    """Parse zero-based IDs or inclusive ranges; ``all`` returns ``None``.

    Duplicates are errors so overlapping ranges cannot silently duplicate the
    evaluation denominator. The returned IDs use canonical ascending order.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task_ids must be 'all' or comma-separated zero-based IDs/ranges")
    if value.strip().lower() == "all":
        return None
    selected: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(r"\s*([0-9]+)(?:\s*-\s*([0-9]+))?\s*", part)
        if match is None:
            raise ValueError(f"invalid task selection component: {part!r}")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if first > last:
            raise ValueError(f"task range is reversed: {part!r}")
        # Bound range expansion before contacting the worker to reject
        # accidental enormous inputs.
        if last > 100_000:
            raise ValueError("task IDs above 100000 are outside supported LIBERO suites")
        for task_id in range(first, last + 1):
            if task_id in selected:
                raise ValueError(f"duplicate task ID: {task_id}")
            selected.add(task_id)
    return sorted(selected)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def select_tasks(
    embodiment: TaskListingEmbodiment,
    suite: str,
    task_ids: str = "0",
    category: str | None = None,
) -> list[dict]:
    """Select native task rows and require exact requested IDs and initial states."""
    if suite not in HORIZONS:
        raise ValueError(f"unsupported suite {suite!r}; choose from {', '.join(HORIZONS)}")
    if embodiment.benchmark != "libero":
        raise ValueError(f"unsupported benchmark: {embodiment.benchmark!r}")
    if category is not None:
        raise ValueError("Category filtering is not supported for LIBERO")
    selected = parse_task_ids(task_ids)

    rows = embodiment.list_tasks(suite, task_ids=selected)
    if not isinstance(rows, list) or not rows:
        raise ValueError("embodiment returned an empty or invalid task list")
    # An unfiltered worker listing may skip tensor loading. Ask explicitly for
    # the selected IDs to resolve counts before constructing any scenes.
    if selected is None and any(row.get("num_init_states") is None for row in rows):
        selected = [_integer(row.get("task_id"), "task_id") for row in rows]
        if len(selected) != len(set(selected)):
            raise ValueError("embodiment returned duplicate task IDs")
        rows = embodiment.list_tasks(suite, task_ids=selected)
        if not isinstance(rows, list) or not rows:
            raise ValueError("embodiment returned an empty or invalid task list")
    result: list[dict] = []
    seen: set[int] = set()
    for task in rows:
        if not isinstance(task, dict) or task.get("suite") != suite:
            raise ValueError("embodiment returned a task from a different or missing suite")
        task_id = _integer(task.get("task_id"), "task_id")
        _integer(task.get("num_init_states"), "num_init_states", 1)
        if task_id in seen:
            raise ValueError(f"embodiment returned duplicate task ID {task_id}")
        seen.add(task_id)
        task = deepcopy(task)
        task["category"] = None
        result.append(task)
    if selected is not None and seen != set(selected):
        raise ValueError("embodiment returned task IDs different from the requested selection")
    return sorted(result, key=lambda row: row["task_id"])


def build_scenes(
    task_rows: Sequence[Mapping[str, Any]],
    trials: int = 1,
    init_start: int = 0,
    seed: int = 0,
) -> list[Scene]:
    """Expand tasks into distinct official initial states, without wrapping.

    Every scene receives the same RNG seed. Variation comes from the explicitly
    indexed official initial state, so changing task selection does not change
    the seed of later tasks. Scene IDs exclude the run seed intentionally.
    """
    _integer(trials, "trials", 1)
    _integer(init_start, "init_start")
    _integer(seed, "seed")
    if not task_rows:
        raise ValueError("cannot build scenes from an empty task selection")
    scenes: list[Scene] = []
    seen: set[tuple[str, int]] = set()
    for task in task_rows:
        suite = task.get("suite")
        if suite not in HORIZONS:
            raise ValueError(f"unsupported task suite: {suite!r}")
        task_id = _integer(task.get("task_id"), "task_id")
        key = (suite, task_id)
        if key in seen:
            raise ValueError(f"duplicate task row: {suite}:{task_id}")
        seen.add(key)
        count = _integer(task.get("num_init_states"), "num_init_states", 1)
        if init_start + trials > count:
            raise ValueError(
                f"{suite} task {task_id} has {count} initial states; "
                f"requested indices {init_start}..{init_start + trials - 1}"
            )
        instruction = task.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{suite} task {task_id} has no instruction")
        for init_index in range(init_start, init_start + trials):
            metadata = deepcopy(dict(task))
            metadata.update(
                suite=suite,
                task_id=task_id,
                init_state_index=init_index,
                category=task.get("category"),
                reference_horizon=HORIZONS[suite],
            )
            scenes.append(
                Scene(
                    id=f"{suite}-task{task_id:04d}-init{init_index:03d}",
                    instruction=instruction,
                    init_seed=seed,
                    metadata=metadata,
                )
            )
    return scenes
