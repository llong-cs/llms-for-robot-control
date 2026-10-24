# Usage

## Run an evaluation

After [installation](INSTALLATION.md) and [API key setup](../README.md#api-key), use the same entry point for previews, LLM agents and native policies:

```bash
source envs/agentic-framework/bin/activate
python3 scripts/eval.py examples/agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/agent.json
```

`--dry-run` validates and prints the resolved configuration and launch plan without inference, simulation or reading credentials. `--list-benches` and `--list-models` list supported selections. The [README examples](../README.md#quick-start) provide starting points for each policy and benchmark.

`evaluation.mode` selects `preview` for initial-scene inspection, `agent` for LLM motion control, or `vla` for native pi05/MolmoAct2 inference. Native runs automatically start the selected model server in its model environment; simulator workers run in their own environment.

## Configuration

An experiment JSON combines the choices needed to define a run:

| Group | Main settings | Purpose |
| --- | --- | --- |
| `evaluation` | `bench`, `mode`, `task_randomize`, `repeats` | Select the benchmark, policy mode, Ditto scene randomization and repeated runs. |
| `environment` | `task_ids`, `difficulty`, `trials`, `init_start`, `seed` | Select tasks and initial conditions. |
| `execution` | `control_interface`, `h`, `k`, `motion_time_scale`, `max_steps` | Set LLM motion execution and the trial's control-step budget. |
| `policy` | `model_profile`, `seed`, `inference_timeout` | Select native weights/scheduling or set inference timeout; the policy seed is independent of the environment seed. |
| `llm` | `model`, `max_output_tokens`, `max_attempts`, `augmentations` | Select the LLM provider and model, and set request limits, online history and reasoning. |
| `observation` | `profile`, `max_images`, `max_context_chars`, `demo*` | Set the policy's observation/context and optional teacher demonstration. |
| `output` | `directory`, `record_trajectory` | Set the run directory and trajectory recording. |
| `paths` | `framework_python`, `checkpoints`, `cache_root`, `outputs_root`, ... | Optional, read only by `scripts/eval.py`: place interpreters, weights, assets, caches and runs outside the checkout; see [resource locations](#resource-locations). |

Start from the closest [example](../examples) or [experiment](../configs/experiments), change the conditions and output name, and inspect the resolved configuration before running. For example, copy the agent example beside the original, edit `environment.task_ids`, `environment.difficulty` and the desired LLM/execution settings, then run it:

```bash
cp examples/agent.json examples/my-agent.json
python3 scripts/eval.py examples/my-agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/my-agent.json
```

File-valued experiment settings resolve relative to the JSON file, including a custom `policy.model_profile` JSON path. Paths inside model/resource profiles resolve from the repository root. `{timestamp}` in `output.directory` expands at launch.

For `agent`, set `llm.model` explicitly (see [model routing](#model-routing-and-api-keys)). `llm.augmentations.reasoning.effort` selects reasoning effort; `llm.augmentations.memory.history_length` sets the number of recent decisions retained in the request context (on Anthropic models see the [thinking caveat](#anthropic-messages)). A nonpositive history length disables online memory. These controls are independent of a fixed demonstration.

For `vla`, choose a matching `policy.model_profile`: `pi05-droid` or `molmoact2-droid` for Ditto/the official MolmoAct2-ManiSkill box task, and `pi05-libero` or `molmoact2-libero` for LIBERO. The [model profiles](../agentic-framework/configs/models) supply checkpoint paths, action adapters and native timing. LIBERO selections include its spatial, object, goal, 10-task and 90-task suites; `libero_all` combines the first four.

Task IDs are zero-based and accept `all`, a single ID, or lists/ranges such as `0,2-4`. Ditto difficulty accepts `easy`, `medium`, `hard`, `xhard`, `all`, or a comma-separated selection. `environment.trials` chooses consecutive initial states starting at `init_start`. Ditto reset seeds are `seed + initial-state index`; LIBERO uses official saved initial states. `evaluation.repeats` reruns those same conditions without changing their environment or policy seeds.

## Resource locations

By default every resource is inside the checkout: interpreters in `envs/`, weights in `models/`, assets in `data/`, upstream sources in `third_party/`, caches in `cache/` and runs in `outputs/`, so a fresh clone needs no setting. An optional top-level `paths` group, read only by `scripts/eval.py`, places resources elsewhere. Its value is an object, or the path of a JSON file holding one, so many experiments can share one site file:

```json
{"paths": "../site-paths.json", "evaluation": {"bench": "ditto", "mode": "vla"}}
```

```json
{"framework_python": "~/envs/agentic-framework/bin/python",
 "simulator_python": {"maniskill": "~/envs/agentic-framework-maniskill/bin/python",
                      "libero": "~/envs/agentic-framework-libero/bin/python"},
 "server_python": {"molmoact2": "~/envs/agentic-framework-molmoact2/bin/python",
                   "openpi": "~/envs/agentic-framework-openpi/bin/python"},
 "checkpoints": {"molmoact2-droid": "~/models/molmoact2-droid", "pi05-droid": "~/models/pi05-droid"},
 "sources": {"molmoact2": "~/src/molmoact2", "openpi": "~/src/openpi"},
 "droid_assets": "~/datasets/molmoact2-sim-eval-assets",
 "cache_root": "~/cache", "outputs_root": "~/runs/llms-for-robot-control"}
```

| Key | Sets | Default |
| --- | --- | --- |
| `framework_python` | The interpreter `scripts/eval.py` re-executes into, which also runs every evaluator | `envs/agentic-framework/bin/python` |
| `simulator_python` (`maniskill`, `libero`) | `environment.python` for the benchmark's simulator | `envs/agentic-framework-<simulator>/bin/python` |
| `server_python` (`molmoact2`, `openpi`) | The managed model server's interpreter, by model family | the profile's `server.environment` |
| `checkpoints` (`<profile ID>`) | The managed server's `--checkpoint` | the profile's `model.checkpoint` |
| `sources` (`molmoact2`, `openpi`) | The managed server's `--source-root` or `--openpi-root`; `molmoact2` also sets `environment.molmoact2_source` | `third_party/<family>` |
| `droid_assets` | `environment.maniskill_assets` | `data/molmoact2-sim-eval-assets` |
| `maniskill_asset_dir`, `ditto_asset_dir`, `libero_config`, `openpi_data_home` | `MS_ASSET_DIR`, `DITTO_ASSET_DIR`, `LIBERO_CONFIG_PATH`, `OPENPI_DATA_HOME` in evaluators, simulator workers and managed servers | `data/maniskill`, `data/generated/ditto-bench/v1`, `.config/libero`, `models/openpi` |
| `cache_root` | `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` in subdirectories below it | `cache/` |
| `outputs_root` | The parent of the default run directory, used when neither `output.directory` nor `--output-dir` is set | `outputs/` |

- Keys name resources explicitly; there is no prefix rewriting. An unknown key fails, and `checkpoints` keys must be model profile IDs.
- `~` expands, and relative values resolve against the file that contains them: the experiment, or the site file. Interpreter symlinks are kept, because resolving a venv's `bin/python` would select its base interpreter; other symlinks are resolved.
- The command line wins over `paths` (`--python`, `--sim-python`, `--server-python`, `--server-arg=--checkpoint=...`, and framework options after `--` such as `--maniskill-assets`), `paths` wins over the experiment's own setting (`environment.python` for the selected simulator, `environment.molmoact2_source`, `environment.maniskill_assets`, or `--checkpoint`, `--source-root` or `--openpi-root` in `evaluation.server_args`) and over the shell environment, and the shell wins over the defaults.
- A managed server still has to report the profile's expected metadata, so wrong weights fail the readiness check.

`--dry-run` prints `resolved.paths`: every location the run reads, with its `value`, its `source` (`cli`, `paths`, `config`, `environment` or `default`) and whether it `exists`, and `paths_file` names the site file. `eval-plan.json` records the same entries with their `exists` flags, and `jobs/<job>/paths.json` beside each job's `config.json` records them without the flags. `resolved-config.json` holds only the framework groups, so it stays a valid framework `--config`.

## Model routing and API keys

`llm.model` is a selector of the form `<provider>/<model ID>`. Its first segment always names the provider, which fixes the wire protocol, the endpoint and the environment variable holding the API key:

| Provider | `llm.model` | Protocol | Endpoint | Environment variables |
| --- | --- | --- | --- | --- |
| `openai` | `openai/<model ID>`, or the shorthand `gpt-*` | Responses | `https://api.openai.com/v1` | `OPENAI_API_KEY` (required); `OPENAI_BASE_URL`, if set, must name the same endpoint |
| `anthropic` | `anthropic/<model ID>`, or the shorthand `claude-*` (see [Anthropic Messages](#anthropic-messages)) | Messages | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` (required); `ANTHROPIC_BASE_URL`, if set, must name the same endpoint, with or without `/v1` |
| `local` | `local/<served model name>`, for a [server you run](#local-chat-completions-servers) | Chat Completions | `LOCAL_LLM_BASE_URL`, default `http://127.0.0.1:8000/v1` | `LOCAL_LLM_API_KEY` (optional); `LOCAL_LLM_BASE_URL` (optional) |

`agent`, `llm` or an unset model select the runtime default, `openai/gpt-5.6-sol`. A bare ID that is not a shorthand, or an unknown provider, is an error that lists the registered providers; nothing is guessed. A native VLA name with `evaluation.mode: "agent"`, or an LLM selector with `"vla"`, is also an error. `--list-models` prints the native profiles and every registered provider with its variables.

Keys are read from the environment only when a request is sent, so export them in the shell that runs `scripts/eval.py`, for example `export OPENAI_API_KEY=...`. A missing or malformed key ends the trial as a configuration error before any request is sent. `--dry-run` and `preview` runs need no key and no network. Run records name the provider, protocol, endpoint and variable names, never their values: `llm_route` in `eval-plan.json`, each job's `config.json`, trajectory metadata and every decision record.

A set `OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL` that names another endpoint than the provider's is a configuration error too. The request archive stores the URL and the exact body bytes sent, and shows the authentication header only by its variable name; keys are never written to run records. Simulator workers, native model servers and every evaluator start without `*_API_KEY`, `*_API_TOKEN`, `*_TOKEN`, `*_SECRET*`, `*_PASSWORD` and `*_BASE_URL` variables or any other variable a provider reads; a live `agent` evaluator gets back only the variables of its own route.

Each provider is one module under [models/llm/providers](../agentic-framework/src/agentic_framework/models/llm/providers) that declares `PROVIDER = ProviderSpec(...)`: its endpoint, key variable and authentication, model ID rule and capabilities. Modules in that directory are discovered automatically; the wire protocols are shared code. Modules named `private_*.py` are ignored by git, and each job's `config.json` records them under `private_sources` instead of `source_sha256`, so public source fingerprints stay comparable. The settings `llm.backend`, `llm.backend_options`, `llm.base_url`, `llm.api_key_env` and `llm.env_file` were removed; a configuration that still uses them fails as an unknown field.

### Anthropic Messages

`anthropic/<model ID>`, or its shorthand `claude-*`, sends Messages requests to `https://api.anthropic.com/v1/messages` with the key from `ANTHROPIC_API_KEY` in the `x-api-key` header and `anthropic-version: 2026-10-21`. Export the key like the OpenAI key, `export ANTHROPIC_API_KEY=...`; if `ANTHROPIC_BASE_URL` is set, it must name the same endpoint: `https://api.anthropic.com`, the form Anthropic's client libraries read, or `https://api.anthropic.com/v1`. [examples/agent-claude.json](../examples/agent-claude.json) runs the agent example on `anthropic/claude-opus-5-5`.

The model ID is sent as written. Requests use adaptive thinking and send the configured effort unchanged as `output_config.effort`; `default` omits it, so the model's own default applies. A model that rejects the effort, or adaptive thinking, fails with the provider's error, which discards the trials. Disabled reasoning (`reasoning.enabled: false`) selects the lowest effort in `llm.supported_efforts`; without that list it ends the trial as a configuration error before any request is sent. `llm.max_output_tokens` becomes the required `max_tokens`; unset, it is 16000. `llm.reasoning_mode` is a Responses option and is rejected. Prompt caching is on: the system prompt is sent as one text block with a `cache_control` breakpoint, which also caches the tools rendered before it, and a top-level `cache_control` caches the growing conversation; cache reads and writes are recorded in the usage records. Current Claude models reject forced tool use, so requests send `tool_choice: auto` with parallel calls disabled, and the framework still requires exactly one robot tool call: a reply without one is repaired like any other format failure. Format-repair rates are therefore not comparable with Responses, which forces a call. A refusal (its `stop_details` category is recorded), a `max_tokens` or `pause_turn` stop and every non-200 status, including a 529 overload, are resent like other transport failures, and nothing of such a response enters the history. Server-side refusal fallbacks are not used, so a refused request never moves to another model. Usage records report `input_tokens` as the whole prompt including cache reads and writes, as for Responses; the provider's own count, which excludes cache tokens, is recorded as `uncached_input_tokens`.

**Memory and thinking.** A thinking block's signature binds the conversation prefix that produced it: the system prompt, the tools and every earlier message. A request that replays the block after that prefix changed is rejected. The history window edits the prefix whenever it evicts the oldest decision, which with a teacher demonstration also moves the shared teacher copy into the next retained observation. The framework therefore records the prefix each turn was sent with and, once that prefix has changed, leaves the turn's `thinking` and `redacted_thinking` blocks out of later requests, keeping its text and tool calls. Each request records the kept and stripped block counts under `thinking_replay`. Until the window fills, earlier decisions keep their reasoning. Once it is full, every request evicts a decision, so earlier decisions are replayed without their reasoning, while a format repair keeps its own decision's reasoning. Memory ablations on Anthropic models are therefore not equivalent to Responses ablations, which replay the reasoning of every retained decision: compare history lengths within one provider.

### Local Chat Completions servers

`local/<served model name>` sends Chat Completions requests to an OpenAI-compatible server that you start yourself, such as vLLM. The served name is sent verbatim and may contain `/`. Requests go to `${LOCAL_LLM_BASE_URL}/chat/completions`; the variable includes the API prefix and defaults to `http://127.0.0.1:8000/v1`. `LOCAL_LLM_API_KEY` is optional: when it is set, it is sent as a Bearer token, and otherwise requests carry no authentication header. Previews never read the variable and record a placeholder origin instead. The launcher neither starts nor probes the server: if it is unreachable, every request fails as a transport error and the trials are discarded. [examples/agent-local.json](../examples/agent-local.json) runs the agent example on `local/robot-vlm`.

Run the server from its own environment; it is not a dependency of this repository. For example, with vLLM, on a GPU that is free and agreed on for this run, set `CUDA_VISIBLE_DEVICES` for the server process only:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=3 vllm serve models/<vision-language-model> \
  --served-model-name robot-vlm --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser <parser for the model>
python3 scripts/eval.py examples/agent-local.json --dry-run
```

**The served model must accept image inputs and tool calls.** Every request carries the camera images of the current observation and of each replayed decision, and the robot motions are function tools. Its chat template must render assistant tool calls and `tool` messages, and the server must accept `tool_choice: "required"`, which requests send with parallel calls disabled. If the server limits the images per prompt (vLLM: `--limit-mm-per-prompt`) or the context length (`--max-model-len`), allow at least the cameras times `history_length + 1`, plus any teacher frames, and `max_output_tokens`. The server must also return tool call IDs that differ between the attempts of one decision: a repair that reuses an earlier attempt's ID is resent as a transport failure.

No reasoning effort, `reasoning_mode` or sampling parameter is sent: `llm.reasoning_mode` is rejected, `reasoning.effort` is recorded but has no effect, and the server's generation defaults apply. `llm.max_output_tokens` becomes `max_tokens`; unset, it is omitted and the server's own limit applies. A reply with no call, several calls or an unknown tool is repaired like any other format failure. A `finish_reason` of `length`, `max_tokens` or `content_filter`, a refusal and every non-200 status are resent like other transport failures, and nothing of such a response enters the history; any other `finish_reason`, including none, is accepted and the reply's tool calls are checked as usual. Usage records map `prompt_tokens` and `completion_tokens` to `input_tokens` and `output_tokens`. Reasoning text a server returns, such as `reasoning_content`, stays in the decision log but is never replayed: the history window replays each retained decision's observation, its visible text and tool calls, and one `tool` message per call with the execution receipt. Memory ablations on local models therefore replay no reasoning and are comparable with neither Responses nor Messages runs.

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
