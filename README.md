# Agentic Framework and Ditto Bench

[![Blog](https://img.shields.io/badge/Blog-LLMs%20for%20Robot%20Control-blue?style=flat-square)](https://llong-cs.github.io/llms-for-robot-control/)

Inspired by [Inspect Robots](https://github.com/robocurve/inspect-robots), this framework simplifies the robot control loop for research analysis and adds detailed trajectory recording. The agentic framework is primarily adapted for OpenAI-family models such as Astra through OpenAI Responses; compatibility with other models is not guaranteed.

The framework runs an observation–decision–motion loop. Ditto Bench supplies six physical interaction tasks with four difficulty levels each. Native model support covers **pi05** and **MolmoAct2**, with DROID and LIBERO profiles. Environments cover **LIBERO**, the official **MolmoAct2-ManiSkill** box task, and **Ditto Bench**.

```text
agentic-framework/  Policies, controllers, model adapters, simulator workers, recording
ditto-bench/        Task physics, procedural assets, success and progress scoring
examples/          Offline loop and focused agent, native VLA, and preview configs
configs/experiments/  Research settings for the experiment conditions
scripts/           Installation, evaluation and native model serving
tools/             Result summaries
docs/              Installation, usage and architecture
envs/              Separate framework, simulator and model environments
third_party/       Pinned upstream source checkouts
models/            Downloaded model checkpoints
data/              Simulator assets and generated geometry
cache/             Download, package and model caches
outputs/           Evaluation results, recordings and previews
.config/libero/    Project-local LIBERO configuration
credentials.env.example  Blank tracked credential template
credentials.env    Your local ignored API credentials
```

## Quick start

Requires Linux and Python 3.11. Install [uv](https://docs.astral.sh/uv/) first, then run:

```bash
cd /path/to/project
bash scripts/setup.sh
envs/agentic-framework/bin/python -B examples/offline.py
```

The offline example runs two scripted decisions through the real policy and motion controller. It requires no GPU, simulator or API key and makes no model calls.

## API key

Create a personal project key in [OpenAI Platform](https://platform.openai.com/api-keys), following the [official API quickstart](https://developers.openai.com/api/docs/quickstart). Fill `OPENAI_API_KEY` in the root `credentials.env`, keep `OPENAI_BASE_URL=https://api.openai.com/v1`, and run `chmod 600 credentials.env`. Setup creates this file from the tracked [credentials.env.example](credentials.env.example) only if it is absent; preserve other entries if your local credentials file already exists.

LLM configs select this local file through `llm.env_file`.

## Run a configuration

Edit the JSON's research settings: model, reasoning effort, history, motion interface and H/K, tasks, seeds, budgets, recording and output path. Agent, native VLA and preview configs contain the settings relevant to that mode. Runtime details use framework defaults unless a condition needs an override.

After [simulator installation](docs/INSTALLATION.md), activate the framework environment and select a GPU:

```bash
source envs/agentic-framework/bin/activate
python3 scripts/eval.py examples/agent.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/preview.json
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/agent.json
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/molmoact2.json
```

Experiment conditions are ordinary JSON files under [configs/experiments](configs/experiments); launch each with `python3 scripts/eval.py CONFIG.json`. See [usage](docs/USAGE.md), [installation](docs/INSTALLATION.md) and [architecture](docs/ARCHITECTURE.md).

Original project code is licensed under the [MIT License](LICENSE). Bundled third-party code and separately downloaded resources retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
