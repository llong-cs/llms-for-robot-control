# API and execution contracts

## Task selection and scenes

`ditto.make_env(task, **kwargs)` accepts the zero-based task IDs `0..5`, public slugs, or environment IDs in [TASKS.md](TASKS.md). It forwards environment options to `gymnasium.make`. The default difficulty is `easy`; `medium`, `hard`, and `xhard` are supported. The evaluation and preview entrypoints support `--difficulty all` or comma-separated difficulties.

The framework launcher is `python -m ditto.integration.agentic`. It automatically resolves the sibling `agentic-framework` checkout. For another location use `--framework-root PATH` or `DITTO_FRAMEWORK_ROOT`. The benchmark adapts the framework only for the duration of that invocation, preserving its agent policy, native VLA adapters, logging, and action execution contracts.

Scene expansion evaluates each selected task/difficulty with reset seeds `seed + init_start` through `seed + init_start + trials - 1`. This pairs the same seed list across variants. All seeds must fit unsigned 32-bit values. Layout randomization is disabled by default. Enable it using `randomize=True` in the Gym API or `--task-randomize` in evaluation/preview. It jitters task layouts with the reset RNG; tasks 0, 4, and 5 also jitter fixture X/Y by up to 10 mm and yaw by up to 10 degrees. A jittered layout is a separate evaluation setting.

## Robot observations and actions

The standard `control` observation profile contains the official external/wrist RGB cameras and public robot state. Environment observation extras contain only the TCP pose. Object poses, task geometry, grasp sites, success predicates, scores, and contact measurements are reserved for audit output. The optional privileged observation profile is opt-in and must be explicitly requested for an experiment. Selecting a teacher or injecting a demonstration does not enable it.

`pd_joint_pos` actions contain seven absolute arm joint positions followed by the Robotiq gripper joint target. The framework agent interface uses bounded TCP pose increments; native DROID VLA profiles use their model-specific action and timing adapters. Preserve the official joint ordering, robot base pose, controllers, and camera calibration. These settings come from the pinned MolmoAct2 source.

The default benchmark runs 150 Hz physics and 30 Hz control, or five physics integration samples per control step. A model decision can cover several control steps. Model inference time is separate from simulated execution time. A native model profile may select a different control frequency; report that frequency and its step budget when comparing results.

## Termination and progress

The environment checks success after each real control step. Read-only calls to `evaluate()` never advance dwell timers or progress history. Time limits count policy-controlled steps. A budget exhausted before success is a failed trial.

Only an agent `done` call enters the common 4-second verification period. The framework holds the measured end-effector pose and the controller's gripper drive target using ordinary bounded actions, without calling a model. Success during verification counts; otherwise the attempt fails. Verification can extend past the policy step budget. It does not automatically release the object or move the arm away. `give_up` performs one neutral stop step and ends the attempt. Native VLAs do not call `done` and do not enter this verification.

Every trial begins with zero relative progress. Let `m0` be its reset measure and `m` its current measure. The common normalization is `clip((m - m0) / (1 - m0), 0, 1)`; an initially saturated measure reports zero improvement. Several tasks already compute a reset-relative measure, so their initial measure is zero and normalization is the identity. The primary progress score is the highest improvement reached during real steps. The final score can decrease. Score one and success are independent; task success predicates are specified in [TASKS.md](TASKS.md).

Logs record raw physical measurements, initial/final/peak scores, peak timing, reset poses, scoring version, geometry fingerprints, and source/asset hashes. Keep these with result tables and videos.

## Snapshots

Use `env.unwrapped.get_state_dict()` and `set_state_dict()` to save/restore both physics and complete task history. Flat states are rejected because they lose the custom multidimensional counters. Snapshot schema 3 records the task, difficulty, simulation/control frequencies, and geometry identity. Restoration rejects incompatible variants before modifying physical state. Save the full dictionary without dropping or rewriting task fields.
