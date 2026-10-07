# Installation and project resources

Use Linux with Python 3.11, Git, and uv. Run the commands below from the project root. The framework, simulators and model servers use separate environments under `envs/`; source checkouts, data, model weights, caches and outputs remain under this project.

Only the pi05 and MolmoAct2 native model families are included. LIBERO uses its own worker; the official MolmoAct2-ManiSkill task and Ditto Bench share the ManiSkill worker.

| Resource                  | Location relative to the project root                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| Framework                 | `envs/agentic-framework`                                                                  |
| ManiSkill/SAPIEN worker   | `envs/agentic-framework-maniskill`                                                        |
| LIBERO worker             | `envs/agentic-framework-libero` (Python 3.8)                                              |
| MolmoAct2 model server    | `envs/agentic-framework-molmoact2`                                                        |
| OpenPi model server       | `envs/agentic-framework-openpi`                                                           |
| Third-party source        | `third_party/molmoact2`, `third_party/openpi`                                             |
| DROID robot assets        | `data/molmoact2-sim-eval-assets`                                                          |
| YCB assets                | `data/maniskill`                                                                          |
| Procedural task meshes    | `data/generated/ditto-bench/v1`                                                           |
| LIBERO configuration      | `.config/libero`                                                                          |
| Models                    | `models/molmoact2-droid`, `models/molmoact2-libero`, `models/pi05-droid`, `models/openpi` |
| Package/model caches      | `cache/uv`, `cache/huggingface`, `cache/torch`, `cache/pip`, `cache/python`               |
| Logs and reports          | `outputs`                                                                                 |

## Framework and offline example

```bash
cd /path/to/project
bash scripts/setup.sh
envs/agentic-framework/bin/python -B examples/offline.py
source envs/agentic-framework/bin/activate
```

Setup installs the pinned core dependencies and the two local editable packages. No model or simulator is loaded. `AGENTIC_ENV` overrides the framework environment path for setup; pass `--python /that/environment/bin/python` to the evaluator when using that override.

## Ditto Bench and the box-task simulator

```bash
bash ditto-bench/scripts/setup/environment.sh sim
envs/agentic-framework-maniskill/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py assets
envs/agentic-framework-maniskill/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py ycb
```

This installs ManiSkill 3.0.1, SAPIEN 3.0.3, and the pinned compatible dependencies. It fetches the official MolmoAct2 checkout under `third_party/molmoact2` for DROID embodiment definitions; it does not load a model. Generated Ditto collision/visual meshes are built under `data/generated/ditto-bench/v1` when first needed.

The common worker verifies **both** the DROID asset manifest and YCB asset manifest, including for procedural Ditto tasks. Download both before preview. The DROID asset revision is `9332a64224ff0a813d9f77bd377b845270232513`; MolmoAct2 source is `66b87e64efd99dfd103241418113955cf64dfa9c`. NVIDIA graphics/CUDA support is needed for camera rendering. The setup script does not change the driver.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/preview.json
```

## MolmoAct2 inference

```bash
bash agentic-framework/scripts/setup/setup_molmoact2.sh
envs/agentic-framework-molmoact2/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py molmo
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/molmoact2.json
```

The model profile specifies `allenai/MolmoAct2-DROID` revision `d8c1abd8a27d8e859455bbe514df2bcc617db0fb`, the official two-camera simulator preprocessing, seven absolute joint targets plus knuckle gripper target, H=K=15, and 15 Hz. Download manifests record a complete file/tree digest. The server checks source, checkpoint, normalization, and readiness metadata before evaluation. Model loading needs GPU memory appropriate to the checkpoint.

## LIBERO and pi05

```bash
bash agentic-framework/scripts/setup/setup_libero.sh
bash agentic-framework/scripts/setup/setup_openpi.sh
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py pi05-libero
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py tokenizer
```

OpenPi is fixed at commit `215abfb217dbac7d5f1273282331b9b1866c0479`; its exact LIBERO submodule and native lockfile are used. LIBERO configuration is written to `.config/libero`, with paths to official task metadata, assets, and initial-state files. Headless rendering requires the system OSMesa library, or `MUJOCO_GL=egl` on a supported installation. Python 3.8 is confined to the simulator. OpenPi uses its own model environment and `OPENPI_DATA_HOME=models/openpi`.

The pi05 download records GCS object generations, sizes, hashes, and a tree manifest. The checkpoint URI is an upstream checkpoint alias rather than an immutable revision; retain the generated manifest with your run results. The tokenizer downloader records its resulting SHA-256. Source and resource checks are read-only during evaluation.

For pi05 evaluation, copy [the LIBERO example](../examples/molmoact2-libero.json) to your own configuration, set `policy.model_profile` to `pi05-libero`, and choose a new output directory. The profile supplies pi05's native action schedule. Run the copied file with `python3 scripts/eval.py CONFIG.json`.

## Other native profiles

To use MolmoAct2 on LIBERO, install the LIBERO and MolmoAct2 environments above, then download its separate pinned LIBERO checkpoint:

```bash
envs/agentic-framework-molmoact2/bin/python \
  agentic-framework/scripts/models/download_molmoact2.py
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/molmoact2-libero.json
```

The LIBERO checkpoint is `allenai/MolmoAct2-LIBERO` revision `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`. Its profile uses H=K=10 at 20 Hz. The downloader verifies authorized access and checkpoint file hashes before writing its provenance manifest.

To use pi05 with the DROID embodiment, install the ManiSkill and OpenPi environments above and download the DROID checkpoint plus tokenizer:

```bash
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py pi05
envs/agentic-framework/bin/python \
  agentic-framework/scripts/models/download_droid_resources.py tokenizer
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval.py examples/pi05-maniskill.json
```

To run the same DROID model on Ditto Bench, edit the JSON's `evaluation.bench`, `environment.task_ids` and `environment.difficulty`, and select the appropriate native profile in `policy.model_profile`. Add `--dry-run` to inspect the plan without loading a checkpoint or simulator. See [usage](USAGE.md) for configuration fields.

## Project paths and existing resources

Installers validate existing `third_party/` checkouts against the required commits and refuse incompatible source. Bundled JSON paths resolve relative to their config files, while model/resource profiles resolve relative to the project root. No fixed home-directory layout is required. If you relocate an installed project, rerun setup to refresh virtual environments and generated simulator configuration, whose internal files can contain absolute paths.

Responses transport is included. Fill the root's local ignored `credentials.env` using the [README instructions](../README.md#api-key); `credentials.env.example` is the tracked blank template. The optional `codex` backend requires a separate authenticated CLI. Private backend extensions remain outside the public source tree and can be installed separately into `envs/agentic-framework`.
