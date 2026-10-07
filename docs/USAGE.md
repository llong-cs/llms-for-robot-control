# Usage

## Evaluate a JSON configuration

From the project root, activate `envs/agentic-framework`, then run `python3 scripts/eval.py CONFIG.json`. Examples and experiment configs expose the research settings relevant to their mode and condition. Paths resolve relative to the config file; model/resource profile paths resolve from the project root. `{timestamp}` gives each run a fresh directory under `outputs/`. `--dry-run` prints the resolved plan without reading credentials, calling a provider, loading weights or starting a simulator.

```bash
source envs/agentic-framework/bin/activate
python3 scripts/eval.py examples/agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/agent.json
```

| Group         | Research settings                                                          |
| ------------- | -------------------------------------------------------------------------- |
| `evaluation`  | Benchmark, mode, layout randomization and condition-specific repeats       |
| `environment` | Task IDs, difficulty, trial count, initial-state offset and reset seed     |
| `execution`   | Motion interface, H, K, step budget and timing changes                     |
| `policy`      | Native model profile, policy seed and condition-specific inference timeout |
| `llm`         | Model, backend, reasoning, memory, request budgets and credential path     |
| `observation` | Demonstration path/type and condition-specific context limits              |
| `output`      | Recording and output directory                                             |

For LLM runs, edit `llm.model`, `llm.augmentations.reasoning.effort` and `llm.augmentations.memory.history_length`. For native runs, select `policy.model_profile`; its profile supplies the matching execution schedule. Each supplied config has one model selector. Preview has no LLM/API settings.

Optional implementation settings are omitted when the framework's defaults suffice. Add them only when needed, such as a different simulator interpreter, native server endpoint or request transport. Native model profiles supply checkpoint identity, embodiment, action timing and adapter defaults. Conditions that vary timing, repeats or demonstration context keep those changes in their JSON.

For example, an agent example's credential path is `../credentials.env`, while an experiment under `configs/experiments/GROUP/` uses `../../../credentials.env`. Keep that config-file-relative rule when moving a config or assigning a recorded teacher path.

Task IDs are zero-based; `all`, `0` and `0,2-4` are accepted. Ditto difficulties are `easy`, `medium`, `hard`, `xhard`, comma-separated lists or `all`. `environment.trials` schedules consecutive initial-state indices starting at `init_start`; Ditto reset seeds are `seed + initial-state index`. LIBERO uses official saved initial states. `evaluation.repeats` repeats those conditions and policy seeds; the launcher numbers repeated jobs automatically.

Supported benchmarks are `ditto`, `molmoact2-maniskill`, and the LIBERO suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_all`. `libero_all` combines the four 10-task evaluation suites; select `libero_90` separately. Ditto Bench's distribution, import package and suite identifier are `ditto`.

Supported native profiles are `pi05-droid`, `pi05-libero`, `molmoact2-droid` and `molmoact2-libero`. DROID profiles serve Ditto Bench and the official MolmoAct2-ManiSkill task; LIBERO profiles serve LIBERO. `--list-benches` and `--list-models` list the entries.

## Credentials and providers

Fill your personal OpenAI key in the root's local `credentials.env` and run `chmod 600 credentials.env`. Preserve other entries if the file exists. This file is ignored by Git; the blank `credentials.env.example` is tracked. See the brief [API key instructions](../README.md#api-key).

LLM configs select `llm.env_file`. Responses uses `OPENAI_API_KEY` and `https://api.openai.com/v1`; keep `OPENAI_BASE_URL` in the private file aligned with that endpoint. The LLM implementation is primarily adapted for Astra/OpenAI Responses; compatibility with other providers or models is not guaranteed. Private integrations are external packages; [architecture](ARCHITECTURE.md) describes the optional extension contract.

Each decision permits `llm.max_attempts` calls, including repair and transport retries. Exhaustion discards that trial. Discarded and infrastructure-error trials are excluded from success/progress/cost means; normal task failures remain evaluated trials. Model permissions are checked when inference starts. A missing/malformed key, unreadable credential file or endpoint mismatch fails locally before sending a request.

## Motion and budgets

`move_by` supplies one fixed target with per-axis bounds scaled by positive H. `move_by_chunk` supplies H increments, frozen into cumulative waypoints from the measured call-start pose. K limits execution before replanning; K=-1 consumes the complete available plan (only for `move_by` control mode). `execution.motion_time_scale: 2` divides each chunk segment into two controller intervals while retaining its endpoint. Native VLA H must match its checkpoint horizon.

Ditto uses 30 Hz control for Astra and 15 Hz control for MolmoAct2, and 150 Hz physics. LIBERO uses 20 Hz. Native profiles identify their action cadence. `execution.max_steps` counts actual policy-controlled steps. A Ditto LLM `done` call adds up to four seconds of physical completion confirmation, recorded separately; `give_up` stops with one neutral step.

## Demonstrations

A teacher is a finalized schema-v2 `trajectory.json` from a recorded run. Set `observation.demo: true` and edit `observation.demo_path` to the chosen file; a directory containing exactly one teacher is also accepted. Teacher recording does not enable privilege.

`demo_content: "observations"` accepts LLM/native VLA teachers and supplies public robot states and camera images. It excludes actions, reasoning, outcomes and object ground truth. `"full"` retains visible LLM interaction turns and requires compatible observation/action contracts.

`demo_mode: "id"` provides task-relevant context; `"ood"` transfers embodiment/control conventions to a different task. Neither changes the live reset. Image/context limits must fit the full projected demonstration; frames are never silently dropped. To collect a teacher, run your own configuration with `output.record_trajectory: true` and demonstrations disabled. Use compatible LLM teachers for full transcripts; observation-only context also accepts native VLA teachers.

## Processes, outputs and metrics

Select devices with `CUDA_VISIBLE_DEVICES`; multiple devices can run independent trial workers. Native VLA mode starts model servers automatically and closes them when evaluation ends. Agents and preview need no native server settings. Environments remain separate under `envs/`, and output paths point into `outputs/`; the launcher refuses nonempty run directories.

Each run saves its plan, resolved config, status, per-job logs, `evaluation/results.json` and `evaluation/summary.json`. Recorded trials also contain `trajectory.json`, step/event sidecars, camera frames, API telemetry and video.

Read `evaluation/summary.json` for the run's built-in summary and `evaluation/results.json` for per-trial results. Success rate uses explicit oracle outcomes from eligible trials. Ditto progress is the peak task score; final progress is separate. Logical decisions, model calls and control steps are recorded separately; model calls include retries. Missing provider usage remains explicit.

Use [tools/summarize.py](../tools/summarize.py) to summarize recorded runs offline using only the Python standard library:

```bash
python3 tools/summarize.py outputs/run-a outputs/run-b --output-dir outputs/summary
```

The output directory contains `summary.json`, `summary.csv`, `trials.csv`, `runs.csv` and `report.md`.
