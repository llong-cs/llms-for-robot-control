# Architecture

The repository has two Python packages: `agentic_framework` runs policies and records rollouts; `ditto` defines the benchmark and connects it to ManiSkill. The root evaluation scripts select experiments and manage their processes. Task physics and scoring stay in Ditto, while the same policy/controller loop also supports LIBERO and the official MolmoAct2-ManiSkill box task.

## From configuration to rollout

1. [scripts/eval.py](../scripts/eval.py) loads an experiment through [eval_config.py](../scripts/eval_config.py), applies command-line overrides, resolves the benchmark and native model profile, and validates a run plan. `--dry-run` stops here.
2. [eval_parallel.py](../scripts/eval_parallel.py) assigns independent trials to the selected CUDA devices and launches evaluators. For native policies, [eval_server.py](../scripts/eval_server.py) starts a local model service for each worker and checks that it loaded the expected checkpoint.
3. [harness/run.py](../agentic-framework/src/agentic_framework/harness/run.py) builds scenes and the execution schedule, creates the environment adapter, and selects a policy through [models/factory.py](../agentic-framework/src/agentic_framework/models/factory.py). Ditto enters this runner through [integration/agentic.py](../ditto-bench/src/ditto/integration/agentic.py), which supplies its scenes, embodiment and task completion hold.
4. Inspect Robots' rollout loop resets a scene and asks the controller for each action. The controller requests policy decisions as needed, translates or schedules their action chunks, and returns one action. The rollout applies it through `embodiment.step()` and supplies the next observation. Simulation advances only when the adapter steps it.
5. Each trial writes its outcome and optional trajectory records. The launcher combines trial results into `evaluation/results.json` and `evaluation/summary.json`.

Experiment JSON files describe research choices: tasks, initial states, model, reasoning, memory, demonstrations and execution budget. [Native model profiles](../agentic-framework/configs/models) bind checkpoints to their camera/action adapters, native horizon, control frequency and serving environment. [Motion profiles](../agentic-framework/configs/motion-profiles.json) describe embodiment-specific motion conventions. See [configuration](USAGE.md#configuration) for settings and path rules.

## Policy and controller boundaries

The shared contracts live in [harness/types.py](../agentic-framework/src/agentic_framework/harness/types.py). Observations carry camera images, public robot state and a language instruction. Policies return action chunks; controllers convert or schedule those chunks into individual actions for the rollout to execute. Model API payloads are handled inside the model backend rather than by the simulator.

For LLM agents, [policy.py](../agentic-framework/src/agentic_framework/harness/policy.py) builds context, calls the backend, validates motion tools and attaches a `MotionPlan` to the action chunk. [context.py](../agentic-framework/src/agentic_framework/harness/context.py) supplies the current observation and execution feedback; the backend manages interaction history. [controller.py](../agentic-framework/src/agentic_framework/harness/controller.py) translates relative targets or chunk waypoints into bounded end-effector actions using the live embodiment's `ActionSpec`. Its horizon and execution cap are defined in [motion and budgets](USAGE.md#motion-and-budgets).

Native pi05 and MolmoAct2 policies use [models/vla](../agentic-framework/src/agentic_framework/models/vla) for checkpoint-specific image/state preprocessing and action conversion. DROID policies execute joint targets through [native_controller.py](../agentic-framework/src/agentic_framework/harness/native_controller.py); pi05's normalized joint velocities are converted to position targets, while MolmoAct2 provides joint positions. LIBERO profiles use their native end-effector delta convention. Native chunks bypass LLM motion-target interpretation.

## Simulator and task boundaries

The [environment adapters](../agentic-framework/src/agentic_framework/environments) expose reset, observation and step operations to the framework. They communicate with simulator workers in separate Python environments: ManiSkill/SAPIEN for Ditto and the official box task, and LIBERO/robosuite for LIBERO. This keeps simulator dependencies out of the policy process.

Within Ditto, [catalog.py](../ditto-bench/src/ditto/catalog.py) identifies tasks, [tasks](../ditto-bench/src/ditto/tasks) defines their geometry and behavior, and [base.py](../ditto-bench/src/ditto/base.py) supplies shared scene setup, scoring and state handling. [integration](../ditto-bench/src/ditto/integration) adapts these environments to the framework without changing the policy implementations. Success and progress are evaluated by the task, independently of the policy's claim that it is finished; see [task definitions](../ditto-bench/docs/TASKS.md).

## Recording and demonstrations

[trajectory_recorder.py](../agentic-framework/src/agentic_framework/harness/trajectory_recorder.py) associates policy requests and responses with executed actions, camera frames and dense environment states. [observability](../agentic-framework/src/agentic_framework/observability) handles request records, timing, progress reporting and video export. Per-trial records retain the resolved schedule and model identity so outcomes can be interpreted alongside the actual execution.

Policy observations and evaluation diagnostics have separate boundaries. Scene states and task diagnostics recorded for analysis are not automatically exposed to the policy. Privileged observations require explicit selection, including when recording teachers.

[demonstration.py](../agentic-framework/src/agentic_framework/harness/demonstration.py) loads a recorded trajectory and projects the selected content into student context. Observation demonstrations accept LLM or native teachers; full interaction demonstrations require compatible LLM recordings. The projection preserves the public observation boundary. The collection and selection workflow is described under [demonstrations](USAGE.md#demonstrations).

## Reading and modifying the code

| Component | Main code |
| --- | --- |
| A minimal policy/controller loop | [examples/offline.py](../examples/offline.py) |
| Experiment selection and validation | [eval_config.py](../scripts/eval_config.py), [configuration](../agentic-framework/src/agentic_framework/configuration) |
| LLM context, tool definitions and inference | [context.py](../agentic-framework/src/agentic_framework/harness/context.py), [tools.py](../agentic-framework/src/agentic_framework/harness/tools.py), [models/llm](../agentic-framework/src/agentic_framework/models/llm) |
| Checkpoint observation/action conversion | [models/vla/adapters.py](../agentic-framework/src/agentic_framework/models/vla/adapters.py), [model profiles](../agentic-framework/configs/models) |
| Simulator process communication | [ManiSkill adapter](../agentic-framework/src/agentic_framework/environments/maniskill/embodiment.py), [LIBERO adapter](../agentic-framework/src/agentic_framework/environments/libero/embodiment.py) |
| Ditto task variants and scoring | [tasks](../ditto-bench/src/ditto/tasks), [base.py](../ditto-bench/src/ditto/base.py), [task and Gym API guide](../ditto-bench/docs/API.md) |
| Result tables across runs | [tools/summarize.py](../tools/summarize.py) |

Change an experiment JSON for a new research condition. Changes to checkpoint conventions belong in the model profile and VLA adapter; changes to task geometry or success belong in Ditto. Changes to LLM motion semantics involve the shared tool contract, policy validation and controller together.
