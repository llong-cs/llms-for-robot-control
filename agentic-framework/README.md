# Agentic Framework

Inspired by [Inspect Robots](https://github.com/robocurve/inspect-robots), this framework simplifies its rollout interfaces for research into LLM robot control. It connects LLM motion tools and native pi05/MolmoAct2 policies to Ditto Bench, LIBERO and the official MolmoAct2-ManiSkill box task.

The LLM path is primarily adapted for OpenAI-family models such as Astra through OpenAI Responses; compatibility with other models is not guaranteed.

## Policy and execution

An LLM decision receives camera images, public robot state and a task instruction. Optional context includes recent interactions and a recorded teacher demonstration. The model chooses a motion through `move_by` or `move_by_chunk`; the motion controller translates the command into bounded simulator actions, then returns observations for the next decision. Motion horizon, replanning interval, reasoning effort and memory length are experiment settings.

Native VLA policies use the same rollout and recording path, with model-specific camera/state preprocessing and action conversion instead of LLM motion tools. The supported profiles are:

| Profile | Benchmark | Action conversion |
| --- | --- | --- |
| `pi05-droid` | Ditto Bench, MolmoAct2-ManiSkill | DROID joint velocities to simulator joint-position targets |
| `molmoact2-droid` | Ditto Bench, MolmoAct2-ManiSkill | DROID joint-position targets |
| `pi05-libero` | LIBERO | End-effector delta pose and gripper commands |
| `molmoact2-libero` | LIBERO | End-effector delta pose and gripper commands |

Profiles in [configs/models](configs/models) provide checkpoint locations, transport settings and native action schedules. Embodiment profiles describe robot coordinates and motion limits. Ordinary experiments select these profiles and define their own conditions in one JSON configuration.

## Observations, demonstrations and records

Policy observations contain images and public proprioception. Benchmark success/progress diagnostics are evaluated separately; privileged observations are disabled by default, including teacher recording. Enabling demonstrations or interaction history does not enable privilege.

Trajectory recording collects camera frames, robot/scene states, actions, model interactions and video. These records support both inspecting a rollout and supplying demonstrations to a later LLM run. Demonstration observations can come from LLM or native policies; full interaction demonstrations require compatible LLM records. [Usage](../docs/USAGE.md) describes demonstration selection and result interpretation.

## Code map

| Module | Responsibility |
| --- | --- |
| [harness](src/agentic_framework/harness) | Policy lifecycle, context construction, motion execution, demonstrations and trajectory recording |
| [models](src/agentic_framework/models) | OpenAI Responses requests, native model clients and action adapters |
| [environments](src/agentic_framework/environments) | Isolated ManiSkill/LIBERO workers, embodiment contracts and observation projection |
| [configuration](src/agentic_framework/configuration) | Configuration parsing, model/embodiment profiles and motion settings |
| [observability](src/agentic_framework/observability) | Model-request records, progress reporting and videos |

Ditto task geometry and scoring live in the separate [ditto-bench](../ditto-bench) package. Its integration layer connects the tasks to the framework's ManiSkill worker.

## Entry points

Use [scripts/eval.py](../scripts/eval.py) with an [example](../examples) or [experiment configuration](../configs/experiments) to launch evaluations. It resolves the configuration, starts simulator workers and, for native policies, manages model servers. The lower-level `agentic-framework` command exposes the framework runner directly.

For the policy/controller API, start with [the offline example](../examples/offline.py), which uses `LLMMotionPolicy`, `FrameworkController` and Inspect Robots' `rollout` with scripted decisions. Then read [harness/policy.py](src/agentic_framework/harness/policy.py), [harness/controller.py](src/agentic_framework/harness/controller.py) and [harness/types.py](src/agentic_framework/harness/types.py). The [architecture guide](../docs/ARCHITECTURE.md) follows the complete path from configuration to simulator and results.

Original framework code is licensed under [MIT](LICENSE); bundled components retain their [upstream licenses](../THIRD_PARTY_NOTICES.md).

[Overview](../README.md) · [Installation](../docs/INSTALLATION.md) · [Usage](../docs/USAGE.md) · [Architecture](../docs/ARCHITECTURE.md)
