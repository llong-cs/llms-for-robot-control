"""Lightweight public task registry; importing it does not load the simulator."""
from __future__ import annotations

from dataclasses import dataclass

SUITE_ID = "ditto"
DEFAULT_HORIZON = 2000


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    env_id: str
    slug: str
    instruction: str
    max_episode_steps: int = DEFAULT_HORIZON


TASKS = (
    TaskSpec(0, "DittoSlantedBoard-v1", "slanted_board", "Pull the blue board completely out of the slot."),
    TaskSpec(1, "DittoOddGeometry-v1", "odd_geometry", "Pick up the blue object and put it in the basket."),
    TaskSpec(2, "DittoPrecisionInsert-v1", "precision_insert", "Insert the yellow object fully into the matching slot."),
    TaskSpec(3, "DittoStandObject-v1", "stand_object", "Keep the orange object upright with its gray base down, without the robot arm or gripper touching it."),
    TaskSpec(4, "DittoToolDrawer-v1", "tool_drawer", "Use the blue tool to open the drawer."),
    TaskSpec(5, "DittoUnlockRing-v1", "unlock_ring", "Take the blue open ring off the fixed bridge."),
)
TASK_BY_ID = {task.task_id: task for task in TASKS}
TASK_BY_ENV = {task.env_id: task for task in TASKS}


def get_task(value: int | str) -> TaskSpec:
    """Resolve a zero-based integer task ID, environment ID, or task slug."""
    if type(value) is int:
        try:
            return TASK_BY_ID[value]
        except KeyError:
            pass
    elif isinstance(value, str):
        if value in TASK_BY_ENV:
            return TASK_BY_ENV[value]
        for task in TASKS:
            if task.slug == value:
                return task
    raise ValueError(f"Unknown suite task: {value!r}; task IDs are 0..5")
