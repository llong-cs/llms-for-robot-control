# Agentic Framework

Inspired by [Inspect Robots](https://github.com/robocurve/inspect-robots), this framework simplifies the robot control loop for research analysis and adds detailed trajectory recording. The LLM path is primarily adapted for OpenAI-family models such as Astra through OpenAI Responses; compatibility with other models is not guaranteed.

LLM policies issue motion tools; pi05 and MolmoAct2 use their native action adapters. The framework supplies isolated simulator workers, memory, fixed teacher demonstrations, control scheduling and evaluation on LIBERO, the official MolmoAct2-ManiSkill box task, and Ditto Bench. The repository [README](../README.md) covers initial setup; [usage](../docs/USAGE.md) describes JSON configuration fields.

## Install

From the project root, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
bash scripts/setup.sh
envs/agentic-framework/bin/python -B examples/offline.py
```

The pinned Inspect Robots runtime is bundled with its MIT license. The core, simulators and model servers use separate environments under `envs/`. Installation does not download model weights or call a model. Source checkouts, data, models, caches and run output stay in the project's `third_party/`, `data/`, `models/`, `cache/` and `outputs/` directories.

For Ditto Bench and the official ManiSkill box task:

```bash
bash agentic-framework/scripts/setup/setup_maniskill.sh
envs/agentic-framework-maniskill/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py assets
envs/agentic-framework-maniskill/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py ycb
```

For LIBERO:

```bash
bash agentic-framework/scripts/setup/setup_libero.sh
```

LIBERO uses a separate Python 3.8 environment and the OpenPi-pinned source. Headless rendering requires OSMesa or supported EGL. The ManiSkill worker uses CPU physics and SAPIEN rendering; graphics/Vulkan support is required. See [installation and resource locations](../docs/INSTALLATION.md) for complete model and simulator setup.

## API key and execution

Create a personal project key in [OpenAI Platform](https://platform.openai.com/api-keys), fill it in the project root's local `credentials.env`, and run `chmod 600 credentials.env` from the root. Setup creates it from the tracked [credentials.env.example](../credentials.env.example) only if it is absent. The configs explicitly select the local credential file, which is ignored by Git.

The agent example exposes Astra, reasoning, memory, H/K, tasks, budgets, recording and output. Native and preview examples contain their relevant settings. Edit the JSON before running; optional runtime details use framework defaults.

```bash
source envs/agentic-framework/bin/activate
python3 scripts/eval.py examples/agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/agent.json
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/preview.json
```

Preview requires a simulator and makes no API calls. The offline example requires neither simulator nor API key. Choose a JSON under [configs/experiments](../configs/experiments) and pass it directly to `scripts/eval.py`.

## Motion, memory and demonstrations

`move_by` freezes one target and tracks it from measured feedback. `move_by_chunk` integrates H waypoint increments and executes each for one control period. Positive K limits execution before the next decision; K=-1 removes that periodic limit (only for `move_by` control mode). `execution.motion_time_scale` subdivides LLM chunk segments while keeping their endpoints. Native VLA actions retain their original adapters and timing.

For Responses, `llm.augmentations.memory.history_length` includes the last N logical decisions; N<=0 disables online history. Fixed demonstrations are separate from online memory. `observation.demo_content: "observations"` supplies public states and images; `"full"` supplies compatible LLM turns, including teacher calls and outputs. The `id`/`ood` demonstration modes do not enable privileged observations or imply a successful teacher. See [configuration fields and outputs](../docs/USAGE.md).

## Native model servers

| Profile            | Installer            | Server               | Native H / K / Hz |
| ------------------ | -------------------- | -------------------- | ----------------- |
| `pi05-libero`      | `setup_openpi.sh`    | `serve_openpi.py`    | 10 / 5 / 20       |
| `pi05-droid`       | `setup_openpi.sh`    | `serve_openpi.py`    | 15 / 8 / 15       |
| `molmoact2-libero` | `setup_molmoact2.sh` | `serve_molmoact2.py` | 10 / 10 / 20      |
| `molmoact2-droid`  | `setup_molmoact2.sh` | `serve_molmoact2.py` | 15 / 15 / 15      |

Installers live in `scripts/setup`; servers and downloaders live in `scripts/models`. Model profiles under `configs/models` identify official source, checkpoint, endpoint and action adapter. Their resource paths resolve from the project root. Servers check source/checkpoint manifests and record loaded-model metadata under `outputs/`. Native VLA mode starts and closes its model servers automatically.

## Recording and observations

`output.record_trajectory` saves finalized `trajectory.json`, frames, video and simulator-state records under `output.directory`. Recorded scene truth never enters ordinary model observations. Privileged observations are disabled by default in every environment, including teacher recording.

Results distinguish task success/failure, infrastructure errors and discarded decisions. A decision exhausting `llm.max_attempts` is discarded and excluded from metrics. Ditto Bench reports continuous task progress and performs a physical hold after an LLM `done` call to confirm task completion.

`configuration` resolves arguments and model profiles; `harness` implements policy, tools, controllers, demonstrations and recording; `models` contains LLM transports and VLA adapters; `environments` supplies simulator workers; `common` contains geometry/serialization; `observability` stores requests/results. [Architecture](../docs/ARCHITECTURE.md) describes the extension contracts.

Original framework code is licensed under the [MIT License](LICENSE). Bundled third-party components retain their original licenses; see [third-party notices](../THIRD_PARTY_NOTICES.md).
