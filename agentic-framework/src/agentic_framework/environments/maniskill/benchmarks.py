"""Official MolmoAct2 task registry and reproducible reset-seed expansion."""

from inspect_robots.scene import Scene

from agentic_framework.environments.libero.benchmarks import parse_task_ids

ENV_ID = "DroidPutEverythingInBox-v1"
MANISKILL_HORIZON = 2000  # Pinned sim_eval/run_eval.py, not stale README/register limit.


def build_maniskill_scenes(
    *, suite=ENV_ID, task_ids="0", trials=1, init_start=0, seed=42, category=None
):
    if suite != ENV_ID:
        raise ValueError(f"Unsupported official MolmoAct2 task: {suite}")
    if category is not None:
        raise ValueError("Category filtering is not supported for the official MolmoAct2 task")
    selected = parse_task_ids(task_ids)
    if selected not in (None, [0]):
        raise ValueError("The registered DROID task has task_id 0")
    for value, minimum, label in (
        (trials, 1, "trials"),
        (init_start, 0, "init_start"),
        (seed, 0, "seed"),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    scenes = []
    for index in range(init_start, init_start + trials):
        reset_seed = seed + index
        if reset_seed >= 2**32:
            raise ValueError("ManiSkill reset seed must be below 2**32")
        scenes.append(
            Scene(
                id=f"maniskill-{ENV_ID}-seed{reset_seed:06d}",
                instruction="put everything into the box",
                init_seed=reset_seed,
                metadata={
                    "benchmark": "maniskill",
                    "suite": ENV_ID,
                    "env_id": ENV_ID,
                    "task_id": 0,
                    "init_state_index": index,
                    "reset_seed": reset_seed,
                    "initialization": "seeded simulator reset; not a LIBERO initial-state file",
                    "reference_horizon": MANISKILL_HORIZON,
                    "category": None,
                },
            )
        )
    return scenes
