# API and execution

## Tasks and reset seeds

`ditto.make_env(task, **kwargs)` accepts IDs `0..5`, slugs, or environment IDs from [TASKS.md](TASKS.md). Difficulty defaults to `easy`; preview and evaluation also accept `--difficulty all` or comma-separated levels.

`make_env` registers the tasks and forwards other keyword arguments to Gymnasium / ManiSkill. For direct use, choose `obs_mode="rgb"`, `control_mode="pd_joint_pos"`, and `num_envs=1` as in the [Gym example](../README.md#direct-gym-example). The registered Gym episode limit is 2,000 control steps; framework runs apply their configured policy budget separately. Rewards default to `none`; success and progress are reported through `info`.

Evaluation pairs reset seeds `seed + init_start` through `seed + init_start + trials - 1` across selected variants. Layout randomization defaults to off; `randomize=True` in Gym or `--task-randomize` in preview/evaluation enables seeded layout jitter.

## Observations, actions, and timing

Direct Gym RGB observations contain ManiSkill sensor data and robot state, including the TCP pose. Framework evaluation projects these into its default `control` observation profile: external/wrist RGB images and public robot state. Object poses, task geometry, scores, and contact diagnostics remain audit output. Privileged observations require explicit opt-in; teacher roles and demonstration injection do not enable them. A custom Gym policy should consume `observation`, rather than forwarding diagnostic `info` to the model.

Direct `pd_joint_pos` actions contain seven absolute arm joint positions followed by the gripper joint target. Framework agents instead request bounded TCP pose increments; native VLA profiles use their own action adapters.

Defaults are 150 Hz physics and 30 Hz control; native VLA profiles may select another control frequency. One model decision can cover multiple control steps. Simulation pauses during inference.

## Step results

`reset()` returns `(observation, info)` and `step(action)` returns `(observation, reward, terminated, truncated, info)`. ManiSkill observations and metrics retain their environment batch dimension, including when `num_envs=1`.

The main task metrics in `info` are:

| Field | Meaning |
| --- | --- |
| `success` / `fail` | Current task completion and failure conditions. |
| `progress_score` | Highest reset-relative progress reached during simulation steps. |
| `final_progress_score` | Reset-relative progress at the current state. |
| `hold_seconds` | Time for which the task's completion condition has remained satisfied. |

Additional task metrics describe insertion, standing, contact, or clearance as appropriate. They are evaluation diagnostics and do not enter the default framework policy observation.

## Termination and progress

Agent `done` defaults to a four-second verification hold, which may exceed the policy step budget, without further model calls. It preserves the end-effector pose and gripper target; it does not release the object or withdraw the arm. Success during this hold counts. Native VLAs do not enter `done` verification.

Progress starts at zero relative to reset. The primary score is peak improvement during real steps; final progress can decrease. Full progress does not imply success; see [task scoring](TASKS.md#success-and-progress). Read-only `evaluate()` calls do not advance timers or score history.

## Snapshots

`env.unwrapped.get_state_dict()` / `set_state_dict()` preserve physics and Ditto task history, including completion dwell, contact attribution, and progress baselines. Restore the complete dictionary into a compatible environment; flat physics-only states are unsupported. Restoration rejects incompatible tasks, difficulties, timing, or geometry. Policy memory is maintained by the framework and is outside the environment snapshot.
