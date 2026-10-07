# Ditto Bench

Ditto Bench contains six physical-interaction tasks with four difficulties each, built on ManiSkill 3 / SAPIEN and the official MolmoAct2 Franka FR3 + Robotiq 2F-85 robot. The tasks cover constrained extraction, odd-object grasping, precision insertion, standing stability, tool-assisted drawer opening, and ring release.

Difficulty changes object geometry, clearance, friction, or access constraints while preserving the task goal. Each task reports binary success and continuous progress; full progress alone does not imply success. See the [task definitions and scoring](docs/TASKS.md).

[Installation](../docs/INSTALLATION.md) · [Tasks](docs/TASKS.md) · [API](docs/API.md) · [Assets](assets/README.md)

## Policy evaluation

Ditto uses the same policy, controller, demonstration, and recording interfaces as the other environments in this repository. Its integration supplies the task scenes and simulator worker, so LLM agents and DROID-native pi05 / MolmoAct2 policies share the benchmark implementation.

From the project root, use a configuration with `evaluation.bench` set to `ditto`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py examples/agent.json
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py examples/molmoact2.json
```

The configuration selects tasks, difficulties, reset seeds, and the policy budget. See the [configuration guide](../docs/USAGE.md) and [experiment configurations](../configs/experiments).

## Preview

After [installing the simulator and robot assets](../docs/INSTALLATION.md), render the task variants without running a policy:

```bash
envs/agentic-framework-maniskill/bin/python -m ditto.preview \
  --task-ids all --difficulty all --gpu 0 \
  --output-dir outputs/ditto-preview
```

The output includes external and wrist camera images, difficulty comparison sheets, and the initial task settings.

## Direct Gym example

Use `make_env` directly for a custom Gymnasium policy. Run this example from the project root in `envs/agentic-framework-maniskill`; it resets a task and takes one joint-position hold step.

```python
from ditto import make_env

env = make_env(
    "slanted_board", difficulty="easy", obs_mode="rgb",
    control_mode="pd_joint_pos", sim_backend="cpu", num_envs=1,
)
try:
    observation, info = env.reset(seed=42)
    action = env.unwrapped.agent.robot.get_qpos()[:, :8].clone()
    observation, reward, terminated, truncated, info = env.step(action)
    print(info["success"])
finally:
    env.close()
```

The [API guide](docs/API.md) describes action conventions, observations, timing, and task-state snapshots.

## Source guide

| Component | Responsibility |
| --- | --- |
| [catalog.py](src/ditto/catalog.py) and [envs.py](src/ditto/envs.py) | Task IDs, instructions, and Gymnasium registration. |
| [base.py](src/ditto/base.py) | Shared robot/camera setup, public observations, progress tracking, and task-state restoration. |
| [tasks/](src/ditto/tasks) | Task-specific geometry, initialization, contact logic, and success conditions. |
| [assets.py](src/ditto/assets.py) | Procedural visual and collision geometry; see [asset generation](assets/README.md). |
| [integration/](src/ditto/integration) | Framework scene expansion, embodiment adapter, and isolated simulator worker. |

For a task implementation, start with its definition in [TASKS.md](docs/TASKS.md), then read its class in `tasks/` together with `base.py`.

[MIT License](LICENSE) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
