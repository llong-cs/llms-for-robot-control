# Installation

Requires Linux, Python 3.11 and [uv](https://docs.astral.sh/uv/). Simulator and model dependencies use separate environments under `envs/`.

Choose the simulator for your benchmark and add a model runtime only for native VLA evaluation:

| Evaluation | Required components |
| --- | --- |
| Offline example | Framework |
| LLM on Ditto Bench / MolmoAct2-ManiSkill | Framework + ManiSkill + robot/YCB assets + API credentials |
| LLM on LIBERO | Framework + LIBERO + API credentials |
| Native DROID policy | Framework + ManiSkill + robot/YCB assets + MolmoAct2 or OpenPi + DROID checkpoint |
| Native LIBERO policy | Framework + LIBERO + MolmoAct2 or OpenPi + LIBERO checkpoint |

## Framework

```bash
bash scripts/setup.sh
source envs/agentic-framework/bin/activate
python examples/offline.py
```

[API key setup](../README.md#api-key).

## Ditto Bench and ManiSkill

Ditto Bench and the official MolmoAct2-ManiSkill box task share a simulator. Camera rendering requires NVIDIA graphics/CUDA and Vulkan support. Download both robot and YCB assets before running either environment:

```bash
bash ditto-bench/scripts/setup/environment.sh sim
envs/agentic-framework-maniskill/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py assets
envs/agentic-framework-maniskill/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py ycb
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/preview.json
```

## LIBERO

LIBERO uses a separate Python 3.8 simulator environment. Headless rendering requires the system OSMesa library, or `MUJOCO_GL=egl` on a supported GPU.

```bash
bash agentic-framework/scripts/setup/setup_libero.sh
```

## Native models

Evaluation launches model servers automatically.

### MolmoAct2

```bash
bash agentic-framework/scripts/setup/setup_molmoact2.sh
```

For the `molmoact2-droid` profile on Ditto Bench or the ManiSkill box task:

```bash
envs/agentic-framework-molmoact2/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py molmo
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/molmoact2.json
```

For `molmoact2-libero`, also install LIBERO and download its separate checkpoint:

```bash
envs/agentic-framework-molmoact2/bin/python \
  agentic-framework/scripts/models/download_molmoact2.py
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/molmoact2-libero.json
```

If the checkpoint requires Hugging Face authorization, add `HF_TOKEN` to `credentials.env` and append `--env-file credentials.env` to the download command.

### pi05

Both pi05 profiles use OpenPi and the same tokenizer:

```bash
bash agentic-framework/scripts/setup/setup_openpi.sh
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py tokenizer
```

For `pi05-libero`, install LIBERO and download:

```bash
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py pi05-libero
```

For pi05 evaluation, set `policy.model_profile` to `pi05-libero` in [the LIBERO example](../examples/molmoact2-libero.json).

For `pi05-droid`, install the ManiSkill simulator and download:

```bash
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py pi05
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/pi05-maniskill.json
```

For DROID policies on Ditto Bench, set `evaluation.bench` to `ditto`. Model profiles supply the native control schedule.
