# Usage

## Run an evaluation

After [installation](INSTALLATION.md) and [API key setup](../README.md#api-key), use the same entry point for previews, LLM agents and native policies:

```bash
source envs/agentic-framework/bin/activate
python3 scripts/eval.py examples/agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/agent.json
```

`--dry-run` validates and prints the resolved configuration and launch plan without inference, simulation or credential access. `--list-benches` and `--list-models` list supported selections. The [README examples](../README.md#quick-start) provide starting points for each policy and benchmark.

`evaluation.mode` selects `preview` for initial-scene inspection, `agent` for LLM motion control, or `vla` for native pi05/MolmoAct2 inference. Native runs automatically start the selected model server in its model environment; simulator workers run in their own environment.

## Configuration

An experiment JSON combines the choices needed to define a run:

| Group | Main settings | Purpose |
| --- | --- | --- |
| `evaluation` | `bench`, `mode`, `task_randomize`, `repeats` | Select the benchmark, policy mode, Ditto scene randomization and repeated runs. |
| `environment` | `task_ids`, `difficulty`, `trials`, `init_start`, `seed` | Select tasks and initial conditions. |
| `execution` | `control_interface`, `h`, `k`, `motion_time_scale`, `max_steps` | Set LLM motion execution and the trial's control-step budget. |
| `policy` | `model_profile`, `seed`, `inference_timeout` | Select native weights/scheduling or set inference timeout; the policy seed is independent of the environment seed. |
| `llm` | `model`, `backend`, `env_file`, `max_output_tokens`, `max_attempts`, `augmentations` | Set the API model, request limits, online history and reasoning. |
| `observation` | `profile`, `max_images`, `max_context_chars`, `demo*` | Set the policy's observation/context and optional teacher demonstration. |
| `output` | `directory`, `record_trajectory` | Set the run directory and trajectory recording. |

Start from the closest [example](../examples) or [experiment](../configs/experiments), change the conditions and output name, and inspect the resolved configuration before running. For example, copy the agent example beside the original, edit `environment.task_ids`, `environment.difficulty` and the desired LLM/execution settings, then run it:

```bash
cp examples/agent.json examples/my-agent.json
python3 scripts/eval.py examples/my-agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/my-agent.json
```

File-valued experiment settings resolve relative to the JSON file, including a custom `policy.model_profile` JSON path. Paths inside model/resource profiles resolve from the repository root. `{timestamp}` in `output.directory` expands at launch.

For `agent`, set `llm.model` explicitly. `llm.augmentations.reasoning.effort` selects reasoning effort; `llm.augmentations.memory.history_length` sets the number of recent decisions retained in the Responses context. A nonpositive history length disables online memory. These controls are independent of a fixed demonstration.

For `vla`, choose a matching `policy.model_profile`: `pi05-droid` or `molmoact2-droid` for Ditto/the official MolmoAct2-ManiSkill box task, and `pi05-libero` or `molmoact2-libero` for LIBERO. The [model profiles](../agentic-framework/configs/models) supply checkpoint paths, action adapters and native timing. LIBERO selections include its spatial, object, goal, 10-task and 90-task suites; `libero_all` combines the first four.

Task IDs are zero-based and accept `all`, a single ID, or lists/ranges such as `0,2-4`. Ditto difficulty accepts `easy`, `medium`, `hard`, `xhard`, `all`, or a comma-separated selection. `environment.trials` chooses consecutive initial states starting at `init_start`. Ditto reset seeds are `seed + initial-state index`; LIBERO uses official saved initial states. `evaluation.repeats` reruns those same conditions without changing their environment or policy seeds.

## Supplied experiments

The JSON files under [configs/experiments](../configs/experiments) define individual conditions that run through `scripts/eval.py`:

| Directory | Conditions |
| --- | --- |
| [main](../configs/experiments/main) | Astra and MolmoAct2 across six tasks, four difficulties and 20 initial states: 480 trials per method. |
| [control](../configs/experiments/control) | Fixed-target versus chunked LLM control with H/K values of 5 or 10; chunk variants also include doubled execution time. |
| [reasoning](../configs/experiments/reasoning) | Low, medium, high and xhigh reasoning effort on constrained extraction at xhard difficulty. |
| [memory](../configs/experiments/memory) | History windows of 0, 2, 5 and 10 decisions on constrained extraction at xhard difficulty. |
| [oneshot](../configs/experiments/oneshot) | In-distribution and out-of-distribution LLM teacher demonstrations, with online memory disabled. |
| [collaboration](../configs/experiments/collaboration) | Astra using a MolmoAct2 observation demonstration on the standing task. |

For example, run the full native baseline with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py configs/experiments/main/main-molmoact2.json
```

On multiple selected GPUs, the launcher distributes trials across workers and starts a native model replica for each GPU group. The one-shot and collaboration configs require a recorded teacher; replace their `observation.demo_path` placeholders as described below.

## Motion and budgets

LLM motion tools use physical end-effector translations and rotations in the world frame. `move_by` tracks one fixed target, with per-axis motion bounds scaled by H. `move_by_chunk` converts H increments into cumulative waypoints from the pose at the start of the call. Positive K caps control steps before replanning; **K=-1 is supported only by `move_by`**. `execution.max_motion_steps`, when supplied, also bounds one fixed-target motion.

`execution.motion_time_scale` subdivides chunk segments without changing endpoints and scales positive K execution budgets. Native VLA H must match the checkpoint's output horizon, and native timing requires `motion_time_scale: 1`.

Control frequency is the number of executed action updates per simulated second. One policy inference can cover several control steps; the simulator advances its physics internally between these updates and pauses while awaiting inference. The main Ditto configs allow 15 seconds of policy-controlled simulation: 450 steps at 30 Hz for Astra, or 225 steps at 15 Hz for MolmoAct2. pi05-DROID also uses 15 Hz; LIBERO uses 20 Hz. Native control frequency is set by the model profile, rather than an experiment-level simulator override.

`execution.max_steps` counts executed policy control steps. A Ditto LLM `done` call can add four seconds of completion confirmation while holding the current pose and gripper target. Those steps are recorded separately from the policy budget.

## Demonstrations

To prepare a one-shot or collaboration run:

1. Run the teacher's agent or native configuration with `output.record_trajectory: true` and demonstrations disabled.
2. Select a finalized trial's `trajectory.json`, retaining its referenced image files. A directory is also accepted when it contains exactly one recorded trial.
3. Set the student's `observation.demo: true` and replace `observation.demo_path` with that recording. Choose `demo_mode` and `demo_content`, then validate and run the student configuration.

`demo_content: "observations"` accepts LLM or native VLA teachers and supplies images and public robot states from at least two distinct teacher inference points. `"full"` includes visible interactions from compatible LLM recordings, including an accepted motion call and its following observation; its observation profile and control interface must match the student's.

`demo_mode: "id"` supplies task-relevant guidance, while `"ood"` transfers control conventions across tasks. Demonstrations stay available independently of the online history window. Teacher success is not required, but the recording must be finalized and cannot be a discarded trial. Privileged observations remain disabled for both teacher and student unless explicitly selected; recording scene diagnostics does not expose them to the policy.

## Outputs and metrics

Each run contains the launch plan and resolved per-job configuration, with evaluation artifacts under `evaluation/`:

| Artifact | Contents |
| --- | --- |
| `evaluation/results.json` | One result per trial, including task success, progress, steps and model-call counts. |
| `evaluation/summary.json` | Aggregate outcomes and counts of completed, discarded and failed jobs. |
| `evaluation/<job>/evals/<scene>/trajectory.json` | Recorded decisions, observation references, execution metadata and outcome. |
| `trajectory-steps.jsonl`, `trajectory-events.jsonl` beside the trajectory | Dense measured control-step samples and inference/execution events. |
| `evaluation/<job>/videos/` | Composite videos of the registered cameras. |

Trajectory artifacts and videos require `output.record_trajectory: true`. Video recording uses the reset image and one image per executed control step; playback FPS comes from the recorded `control_hz`. Thus a 15 Hz rollout is exported at 15 FPS. Inference waiting time is excluded from playback, including in agent mode.

Task success is the explicit `oracle_success` value. A trial's execution status `"success"` means it completed normally and can still represent a task failure. Retry-exhausted trials and infrastructure errors are listed but excluded from aggregate metrics; ordinary task failures count. Ditto `progress_score` is the episode peak, while `final_progress_score` describes the terminal state. Logical decisions, model calls including retries, and control steps are distinct counts.

Summarize one or more runs offline with [tools/summarize.py](../tools/summarize.py):

```bash
python3 tools/summarize.py outputs/run-a outputs/run-b --output-dir outputs/summary
```

The utility writes JSON, CSV and a Markdown report grouped by run, task and difficulty, with success/progress and recorded timing, token and call statistics. Success rates pool evaluated trials; parallel trials' summed wall time describes cumulative work rather than elapsed run time.
