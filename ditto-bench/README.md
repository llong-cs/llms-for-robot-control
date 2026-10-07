# Ditto Bench

Six physical-interaction tasks with four difficulties each, using ManiSkill 3, SAPIEN, and the official MolmoAct2 Franka FR3 + Robotiq 2F-85 robot. All 24 variants and their physical scoring are included. The Python distribution, import package and suite identifier are `ditto`.

The benchmark includes procedural asset generation, preview rendering and integration with the sibling `agentic-framework`. Start with the [repository README](../README.md) for setup and example configs. See [task definitions](docs/TASKS.md), [API and execution](docs/API.md), and [assets](assets/README.md) for the benchmark contracts.

## Installation

Use Python 3.11 on Linux with an NVIDIA GPU and working Vulkan/CUDA drivers. Install `uv`, then run these commands from the project root:

```bash
bash ditto-bench/scripts/setup/environment.sh framework
bash ditto-bench/scripts/setup/environment.sh sim
```

The framework lives in `envs/agentic-framework`; simulation lives in `envs/agentic-framework-maniskill`. Model servers use their own environments. The framework simulator lock fixes ManiSkill 3.0.1, SAPIEN 3.0.3, and the compatible geometry and CUDA dependencies. The `sim` package extra is available for custom installations; use the sibling lock for the pinned simulator dependencies.

Setup prepares the upstream source under `third_party/molmoact2`. It must be an unmodified checkout of [allenai/molmoact2](https://github.com/allenai/molmoact2) at `66b87e64efd99dfd103241418113955cf64dfa9c`:

```bash
"$PWD/envs/agentic-framework-maniskill/bin/python" \
  agentic-framework/scripts/models/download_droid_resources.py assets
"$PWD/envs/agentic-framework-maniskill/bin/python" \
  agentic-framework/scripts/models/download_droid_resources.py ycb
```

Setup checks existing checkout revisions. The downloader fetches the pinned DROID asset snapshot into `data/molmoact2-sim-eval-assets` and writes the provenance manifest required by the framework worker. The direct Gym API and preview do not run those worker integrity checks; use the framework evaluation entrypoint when producing reproducible policy results. No model weights are needed for a preview.

The shared simulator adapter also verifies the official YCB inventory on startup, so its downloader is required for framework evaluation even though Ditto Bench task props are procedural.

## Preview and offline example

Select a GPU, replacing `0` below. Preview all tasks and difficulties without calling a model:

```bash
"$PWD/envs/agentic-framework-maniskill/bin/python" -B \
  -m ditto.preview --task-ids all --difficulty all --gpu 0 \
  --output-dir "$PWD/outputs/ditto-preview"
```

The output includes the external and wrist policy images, contact sheets, initial measurements, task definitions, and asset hashes. The displayed default is the actual external policy camera.

Run the offline framework example without a simulator, GPU or model call:

```bash
"$PWD/envs/agentic-framework/bin/python" -B examples/offline.py
```

For an agent or native VLA, edit its research settings in the JSON example and use the evaluation launcher from the project root:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/agent.json
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/molmoact2.json
```

The agent config selects Astra/OpenAI Responses, reasoning, memory, H/K, tasks, budgets, recording and output. The native config selects the MolmoAct2-DROID profile and schedule. Runtime details use framework defaults; experiment conditions are under `configs/experiments`.

## Direct Gym example

Run this from the project root with `envs/agentic-framework-maniskill` activated:

```python
from ditto import make_env

env = make_env(
    "slanted_board", difficulty="easy", obs_mode="rgb",
    control_mode="pd_joint_pos", sim_backend="cpu", num_envs=1,
)
try:
    observation, info = env.reset(seed=42)
    # Hold the current seven arm joints and gripper joint target for one step.
    action = env.unwrapped.agent.robot.get_qpos()[:, :8].clone()
    observation, reward, terminated, truncated, info = env.step(action)
    print(info["success"])
finally:
    env.close()
```

Use `CUDA_VISIBLE_DEVICES` on the invoking process to select a GPU. See [API.md](docs/API.md) for actions, snapshots, seed schedules, and observation boundaries. Task geometry is generated on first use under `data/generated/ditto-bench/v1`. Assets and run output remain in the project's ignored `data/` and `outputs/` directories; share tracked source without those generated files or local API credentials.

Original benchmark code and geometry generators are licensed under the [MIT License](LICENSE). Downloaded assets and resources derived from them retain their upstream terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
