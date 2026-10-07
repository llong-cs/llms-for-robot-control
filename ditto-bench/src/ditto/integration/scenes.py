"""Deterministic task and seed expansion using the framework's Scene contract."""
from __future__ import annotations

from dataclasses import asdict

from agentic_framework.environments.libero.benchmarks import parse_task_ids
from inspect_robots.scene import Scene

from ditto.catalog import SUITE_ID, TASK_BY_ID, TASKS
from ditto.difficulties import expand_difficulties


def build_suite_scenes(
    *, suite=SUITE_ID, task_ids="all", trials=1, init_start=0, seed=42, category=None,
    difficulty="easy",
):
    if suite != SUITE_ID:
        raise ValueError(f"Expected suite {SUITE_ID!r}, received {suite!r}")
    if category is not None:
        raise ValueError("LIBERO categories do not apply to ditto")
    selected = parse_task_ids(task_ids)
    selected = list(TASK_BY_ID) if selected is None else selected
    if any(task_id not in TASK_BY_ID for task_id in selected):
        raise ValueError("ditto task IDs must be in 0..5")
    for value, minimum, label in ((trials, 1, "trials"), (init_start, 0, "init_start"), (seed, 0, "seed")):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if seed + init_start + trials - 1 >= 2**32:
        raise ValueError("ManiSkill reset seeds must be below 2**32")
    scenes = []
    for task_id in selected:
        task = TASK_BY_ID[task_id]
        for level in expand_difficulties(difficulty):
            for trial_index in range(init_start, init_start + trials):
                paired_seed = seed + trial_index
                scenes.append(Scene(
                    id=f"ditto-{task.task_id:02d}-{task.slug}-{level}-seed{paired_seed:06d}",
                    instruction=task.instruction,
                    init_seed=paired_seed,
                    metadata={
                        "benchmark": "maniskill", "suite": SUITE_ID, "env_id": task.env_id,
                        "task_id": task.task_id, "task_slug": task.slug, "difficulty": level,
                        "instruction": task.instruction, "init_state_index": trial_index,
                        "reset_seed": paired_seed, "paired_seed": paired_seed,
                        "policy_seed": paired_seed,
                        "reference_horizon": task.max_episode_steps,
                        "initialization": "paired suite seed; actual simulator reset seed recorded separately",
                        "category": None,
                    },
                ))
    return scenes


def list_suite_tasks():
    return [dict(asdict(task), suite=SUITE_ID, num_init_states=None,
                 initialization="env.reset(seed=seed)") for task in TASKS]
