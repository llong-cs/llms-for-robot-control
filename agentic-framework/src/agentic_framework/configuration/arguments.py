"""Categorized runtime options, strict JSON validation and documented precedence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from agentic_framework.common.project import project_path
from agentic_framework.configuration.profiles import config_directory
from agentic_framework.harness.types import AugmentationConfig, AutoMotionConfig, ScheduleConfig


def horizon(value):
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("H must be a positive integer; select the tool with control_interface") from exc
    if str(value) != str(result) or result <= 0:
        raise argparse.ArgumentTypeError("H must be a positive integer; select the tool with control_interface")
    return result


def execution_horizon(value):
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("K must be a positive integer, or -1 for control_interface=move_by") from exc
    if str(value) != str(result) or result == 0 or result < -1:
        raise argparse.ArgumentTypeError("K must be a positive integer, or -1 for control_interface=move_by")
    return result


def augmentation_value(value):
    raw = json.loads(value) if isinstance(value, str) else value
    return asdict(AugmentationConfig.from_dict(raw))


def auto_motion_value(value):
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, dict):
        raise ValueError("auto_motion must be a JSON object")
    if "max_action_fraction" in raw:
        message = "auto_motion.max_action_fraction has been removed; delete this field from the configuration"
        error = argparse.ArgumentTypeError if isinstance(value, str) else ValueError
        raise error(message)
    return asdict(AutoMotionConfig(**raw))


def backend_options_value(value):
    """Accept a JSON object containing the selected extension's options."""
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("backend_options must be a JSON object") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        error = argparse.ArgumentTypeError if isinstance(value, str) else ValueError
        raise error("backend_options must be a JSON object")
    return dict(raw)


GROUP_TITLES = {
    "run": "Run mode",
    "environment": "Environment and tasks",
    "execution": "Action execution",
    "policy": "Model family and endpoint",
    "llm": "Language model",
    "observation": "Model observations",
    "output": "Output",
}

FIELD_GROUPS = {
    "mode": "run",
    "repeat_id": "run",
    "preview_context": "run",
    "preview_dx": "run",
    "benchmark": "environment",
    "suite": "environment",
    "task_ids": "environment",
    "difficulty": "environment",
    "trials": "environment",
    "init_start": "environment",
    "seed": "environment",
    "category": "environment",
    "sim_python": "environment",
    "control_hz": "environment",
    "sim_hz": "environment",
    "sim_backend": "environment",
    "sim_shader": "environment",
    "gpu": "environment",
    "molmoact2_source": "environment",
    "maniskill_assets": "environment",
    "h": "execution",
    "control_interface": "execution",
    "k": "execution",
    "motion_time_scale": "execution",
    "max_motion_steps": "execution",
    "auto_motion": "execution",
    "max_steps": "execution",
    "done_verification_seconds": "execution",
    "action_tools": "execution",
    "approver": "execution",
    "policy_family": "policy",
    "model_profile": "policy",
    "policy_profile": "policy",
    "policy_host": "policy",
    "policy_port": "policy",
    "policy_url": "policy",
    "policy_metadata": "policy",
    "policy_seed": "policy",
    "inference_timeout": "policy",
    "model": "llm",
    "backend": "llm",
    "base_url": "llm",
    "api_key_env": "llm",
    "env_file": "llm",
    "backend_options": "llm",
    "max_output_tokens": "llm",
    "supported_efforts": "llm",
    "reasoning_mode": "llm",
    "max_response_chars": "llm",
    "max_attempts": "llm",
    "augmentations": "llm",
    "demo": "observation",
    "demo_path": "observation",
    "demo_mode": "observation",
    "demo_content": "observation",
    "demo_image_max_side": "observation",
    "record_trajectory": "output",
    "max_images": "observation",
    "max_context_chars": "observation",
    "cameras": "observation",
    "observation_profile": "observation",
    "output_dir": "output",
}

PARAMETER_HELP = {
    "difficulty": "Custom suite difficulty: easy/medium/hard/xhard, all or comma list; standard aliases medium.",
    "repeat_id": "Repeat identifier for result metadata; does not change environment or model seeds.",
    "policy_seed": "Independent model sampling seed (0..2^32-1); requires an OpenPi/MolmoAct2 seed_protocol_v1 server.",
    "config": "Read grouped JSON configuration; explicit command-line values take precedence.",
    "mode": "Run mode: agent calls an LLM, vla calls a native action model, and preview exports configuration, tasks and observations without model calls.",
    "benchmark": "Simulation benchmark; must be compatible with the native model adapter.",
    "suite": "Task suite within the benchmark; comma-separated names select multiple suites.",
    "task_ids": "Zero-based task IDs: one ID, comma-separated IDs/ranges, or all.",
    "trials": "Number of independent trials per task.",
    "init_start": "Starting LIBERO initial-state index or ManiSkill seed offset.",
    "seed": "Random seed; unset uses the selected environment default.",
    "category": "Filter tasks by a benchmark-supported category.",
    "h": "Positive action horizon. For move_by, pose bounds are H times the single-step bounds; for move_by_chunk, each requested dimension has H values. K independently limits execution before motion_time_scale. Native VLA H must match its checkpoint.",
    "control_interface": "LLM motion tool: move_by tracks one fixed target within H-scaled pose bounds; move_by_chunk executes H target segments, each subdivided by motion_time_scale. Native VLA profiles use move_by_chunk and retain their native controller.",
    "k": "Maximum execution steps per decision before motion_time_scale; positive LLM K permits K times motion_time_scale physical steps. -1 removes this cap only with control_interface=move_by; move_by_chunk requires positive K. Completion or other execution limits may stop earlier. Not provided to the language model.",
    "motion_time_scale": "Positive integer execution time multiplier for LLM motion tools (default: 1). Subdivide each move_by_chunk segment into this many physical steps without changing model-facing H or displacement; multiply positive move_by K by the same factor. For move_by, K=-1 keeps the periodic execution cap disabled; physical episode/motion limits remain unchanged. Native VLA policies require 1.",
    "max_motion_steps": "Maximum physical control steps for one feedback-tracked motion.",
    "auto_motion": "JSON completion criteria for pose tolerance, stability, stall detection and gripper tracking.",
    "max_steps": (
        "Policy control-step limit per trial. The environment judges success after every step; "
        "reaching this limit without success is a failure. Only a model done call may be "
        "followed by done verification (see done_verification_seconds). JSON null uses the "
        "suite horizon."
    ),
    "done_verification_seconds": (
        "ManiSkill only: after a language model calls done, hold for "
        "ceil(seconds * control_hz) further control steps without model calls; the arm holds "
        "its measured pose and the gripper its driven target. Success within the hold counts; "
        "otherwise the trial fails (done_verification_timeout). give_up, max_steps, "
        "max_attempts and errors never verify; native VLA policies never call done. "
        "0 disables it; the custom task suite uses 4 (120 steps at 30 Hz)."
    ),
    "model": "Server-side LLM model ID; native VLA checkpoints are selected with model_profile.",
    "backend": "LLM transport: Responses API, Codex conversation, or an installed backend extension.",
    "base_url": "API endpoint; unset uses the selected backend's standard endpoint.",
    "api_key_env": "Name of the environment variable containing the API key.",
    "env_file": "Shared API credential dotenv path, read only when sending a request; null disables file loading for Responses.",
    "backend_options": "JSON object with options for an installed backend extension; built-in backends require an empty object.",
    "max_output_tokens": "Maximum output tokens per API request.",
    "preview_context": "Use scripted actions in LLM preview to export later context without model calls.",
    "preview_dx": "Scripted x displacement in meters for LLM preview-context.",
    "supported_efforts": "Comma-separated supported reasoning efforts for servers without capability discovery.",
    "reasoning_mode": "Explicit API reasoning mode: standard or pro; unset omits mode. Codex does not support this option.",
    "inference_timeout": "Request timeout in seconds for both LLM and native VLA models.",
    "max_response_chars": "Maximum response characters; exceeding the limit stops that response.",
    "max_attempts": (
        "Maximum attempts of one LLM decision (one model round); the count restarts after "
        "every accepted decision. A format failure, including a rejected motion preflight, "
        "is repaired with its validation error. A transport/provider failure (every non-200 "
        "HTTP status, including 400/401/403/404, timeout, connection or stream error, "
        "refusal, incomplete response) or an unexpected exception inside the backend call "
        "(building the wire request, sending it or parsing its response) resends the same "
        "request after an exponential backoff (or a longer 429/503 Retry-After); each "
        "consumes one attempt, and a resent request may already have been billed. Never "
        "retried: cancellation, local configuration errors (for example missing credentials "
        "or a missing Codex CLI) and a Codex host-tool isolation breach; any other exception "
        "outside the backend call ends the trial. A decision that fails max_attempts times "
        "ends the trial as max_attempts: it is discarded (status=discarded) and excluded "
        "from all results. A misconfigured provider (wrong key, model id or parameters) "
        "therefore discards every trial; the evaluation launcher then fails the run, "
        "while this entry point exits 0, warns on stderr and reports discarded_trials in "
        "summary.json."
    ),
    "augmentations": "Grouped memory/reasoning JSON. history_length<=0 disables memory; positive N sets the API decision window or enables persistent Codex dialogue. Reasoning controls effort, notes and hindsight.",
    "demo": "Include one fixed teacher demonstration with every current observation, independently of online history. demo_content selects observation/state sequences or complete LLM interactions.",
    "demo_path": "Finalized trajectory.json recorded with --record-trajectory, or a directory containing exactly one such trial. observations accepts agent and native VLA recordings; full requires a compatible LLM recording. Task success is not required; required when demo=true.",
    "demo_mode": "Teacher interpretation: ood learns embodiment behavior from observed motion and, when provided, actions; id also analyzes demonstrated successes, failures, and progress to improve task-specific actions. Does not enable privilege or assume teacher success.",
    "demo_content": "observations (default) includes only images and measured robot state at teacher inference points, for agent or VLA recordings; full retains compatible LLM tool calls, results, and visible text. Independent of demo_mode id/ood.",
    "demo_image_max_side": "Maximum teacher-image side in pixels; preserve aspect ratio and never upscale. 0 keeps original resolution. Live images and recorded files are unchanged.",
    "record_trajectory": "Save trajectories, camera frames, MP4 and simulator state at each control step under output_dir. Disabled by default; recorded simulator state is never added to model observations.",
    "max_images": "Maximum images in one model request, including every teacher-demo frame; frames are never dropped.",
    "max_context_chars": "Maximum model-facing context characters, including the complete teacher demo; exceeding it is an error.",
    "cameras": "Comma-separated camera names; an empty string uses every environment camera.",
    "observation_profile": "control (default) provides measured robot state; openpi_matched uses LIBERO-matched preprocessing; privileged explicitly exposes LIBERO/ManiSkill scene truth and is never enabled by a teacher or demo setting. Full-content teacher demos require the same observation profile; observations demos expose only measured robot state and images.",
    "action_tools": "Relative-motion tool family. control_interface selects the model-facing move_by or move_by_chunk tool independently of positive H.",
    "approver": "none executes unchanged; clamp clips control values to the declared action space.",
    "sim_python": "Python interpreter for the isolated simulator worker.",
    "control_hz": "Physical control frequency; native policies inherit their model profile and LIBERO uses 20 Hz.",
    "sim_hz": "ManiSkill physics frequency; must be an integer multiple of control frequency.",
    "sim_backend": "ManiSkill physics device: cpu or gpu.",
    "sim_shader": "ManiSkill rendering shader pack.",
    "gpu": "Single GPU index exposed to the simulator worker.",
    "molmoact2_source": "Official MolmoAct2 source containing the ManiSkill DROID robot implementation.",
    "maniskill_assets": "External ManiSkill DROID robot and scene asset directory.",
    "output_dir": "Result directory; defaults to a fresh run under project outputs/.",
    "policy_family": "Policy family; native models infer it from model_profile and language models use llm.",
    "model_profile": "Native model profile ID or JSON path declaring checkpoint identity, adapter and defaults.",
    "policy_profile": "native requires original H/K/frequency and provenance; custom may change K/frequency while H remains the checkpoint horizon.",
    "policy_host": "Override native WebSocket model host or ws/wss URL.",
    "policy_port": "Override native model server port; unset uses the model profile.",
    "policy_url": "Override the complete native HTTP action endpoint URL.",
    "policy_metadata": "Loaded-model provenance manifest generated by the server to verify checkpoint identity.",
}

CONFIG_KEYS = {
    "policy_family": "family",
    "policy_profile": "profile",
    "policy_host": "host",
    "policy_port": "port",
    "policy_url": "url",
    "policy_metadata": "metadata",
    "policy_seed": "seed",
    "observation_profile": "profile",
    "output_dir": "directory",
    "sim_python": "python",
}


# Nullability is a parameter contract, independent of the configured default value.
NULLABLE_FIELDS = frozenset(
    {
        "repeat_id",
        "policy_seed",
        "demo_path",
        "preview_dx",
        "suite",
        "seed",
        "category",
        "sim_python",
        "control_hz",
        "gpu",
        "max_steps",
        "policy_family",
        "model_profile",
        "policy_host",
        "policy_port",
        "policy_url",
        "policy_metadata",
        "base_url",
        "api_key_env",
        "env_file",
        "max_output_tokens",
        "supported_efforts",
        "reasoning_mode",
        "output_dir",
    }
)


def flatten_config(document):
    """Normalize one grouped document or a flat programmatic parameter mapping."""
    if not isinstance(document, dict):
        raise ValueError("config must be a JSON object")
    if not any(key in GROUP_TITLES for key in document):
        unknown = set(document) - FIELD_GROUPS.keys()
        if unknown:
            raise ValueError(f"unknown configuration parameters: {sorted(unknown)}")
        return dict(document)
    unknown = set(document) - GROUP_TITLES.keys()
    if unknown:
        raise ValueError(
            f"unknown configuration groups or mixed flat parameters: {sorted(unknown)}"
        )
    result = {}
    for group, fields in document.items():
        if not isinstance(fields, dict):
            raise ValueError(f"config {group} must be an object")
        names = {
            CONFIG_KEYS.get(name, name): name
            for name, owner in FIELD_GROUPS.items()
            if owner == group
        }
        for key, value in fields.items():
            if key not in names:
                raise ValueError(f"unknown or misplaced parameter: {group}.{key}")
            result[names[key]] = value
    return result


def group_config(values):
    result = {}
    for name, value in values.items():
        if name not in FIELD_GROUPS:
            raise ValueError(f"unknown configuration parameter: {name}")
        result.setdefault(FIELD_GROUPS[name], {})[CONFIG_KEYS.get(name, name)] = value
    return result


@dataclass(frozen=True)
class SettingsGroup:
    values: Mapping[str, Any]

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True)
class RunConfiguration:
    """Resolved groups; flat attribute access is restricted to declared parameters."""

    groups: Mapping[str, SettingsGroup]
    config: Path | None = None

    @classmethod
    def from_namespace(cls, args):
        fields = vars(args).copy()
        config = fields.pop("config", None)
        groups = {
            name: SettingsGroup(MappingProxyType(values))
            for name, values in group_config(fields).items()
        }
        return cls(MappingProxyType(groups), config)

    def __getattr__(self, name):
        if name in self.groups:
            return self.groups[name]
        if name in FIELD_GROUPS:
            return getattr(self.groups[FIELD_GROUPS[name]], CONFIG_KEYS.get(name, name))
        raise AttributeError(name)

    def flat(self):
        return flatten_config({name: dict(group.values) for name, group in self.groups.items()})

    def to_dict(self):
        return group_config(self.flat())


def parser():
    from agentic_framework.models.llm.extensions import backend_names

    defaults = flatten_config(
        json.loads((config_directory() / "runtime-defaults.json").read_text())
    )
    for field in ("output_dir", "policy_metadata", "env_file", "demo_path",
                  "molmoact2_source", "maniskill_assets", "sim_python"):
        if defaults.get(field) is not None:
            defaults[field] = str(project_path(defaults[field]))
    p = argparse.ArgumentParser(
        description="Evaluate robot policies with categorized configuration.", allow_abbrev=False
    )
    p.add_argument("--config", type=Path, help=PARAMETER_HELP["config"])
    groups = {key: p.add_argument_group(title) for key, title in GROUP_TITLES.items()}
    groups["run"].add_argument(
        "--repeat-id", type=int, default=defaults["repeat_id"], help=PARAMETER_HELP["repeat_id"]
    )
    groups["run"].add_argument(
        "--mode",
        choices=["agent", "vla", "preview"],
        default=defaults["mode"],
        help=PARAMETER_HELP["mode"],
    )
    groups["environment"].add_argument(
        "--benchmark",
        choices=["maniskill", "libero"],
        default=defaults["benchmark"],
        help=PARAMETER_HELP["benchmark"],
    )
    groups["environment"].add_argument(
        "--suite", default=defaults["suite"], help=PARAMETER_HELP["suite"]
    )
    groups["environment"].add_argument(
        "--task-ids", default=defaults["task_ids"], help=PARAMETER_HELP["task_ids"]
    )
    groups["environment"].add_argument(
        "--difficulty", default=defaults["difficulty"], help=PARAMETER_HELP["difficulty"]
    )
    groups["environment"].add_argument(
        "--trials", type=int, default=defaults["trials"], help=PARAMETER_HELP["trials"]
    )
    groups["environment"].add_argument(
        "--init-start", type=int, default=defaults["init_start"], help=PARAMETER_HELP["init_start"]
    )
    groups["environment"].add_argument(
        "--seed", type=int, default=defaults["seed"], help=PARAMETER_HELP["seed"]
    )
    groups["environment"].add_argument(
        "--category", default=defaults["category"], help=PARAMETER_HELP["category"]
    )
    groups["execution"].add_argument(
        "--h", type=horizon, default=defaults["h"], help=PARAMETER_HELP["h"]
    )
    groups["execution"].add_argument(
        "--control-interface", choices=["move_by", "move_by_chunk"],
        default=defaults["control_interface"], help=PARAMETER_HELP["control_interface"],
    )
    groups["execution"].add_argument(
        "--k", type=execution_horizon, default=defaults["k"], help=PARAMETER_HELP["k"]
    )
    groups["execution"].add_argument(
        "--motion-time-scale", type=int, default=defaults["motion_time_scale"],
        help=PARAMETER_HELP["motion_time_scale"],
    )
    groups["execution"].add_argument(
        "--max-motion-steps",
        type=int,
        default=defaults["max_motion_steps"],
        help=PARAMETER_HELP["max_motion_steps"],
    )
    groups["execution"].add_argument(
        "--auto-motion",
        type=auto_motion_value,
        default=defaults["auto_motion"],
        help=PARAMETER_HELP["auto_motion"],
    )
    groups["execution"].add_argument(
        "--max-steps", type=int, default=defaults["max_steps"], help=PARAMETER_HELP["max_steps"]
    )
    groups["execution"].add_argument(
        "--done-verification-seconds", type=float,
        default=defaults["done_verification_seconds"],
        help=PARAMETER_HELP["done_verification_seconds"],
    )
    groups["llm"].add_argument("--model", default=defaults["model"], help=PARAMETER_HELP["model"])
    groups["llm"].add_argument(
        "--backend",
        choices=backend_names(),
        default=defaults["backend"],
        help=PARAMETER_HELP["backend"],
    )
    groups["llm"].add_argument(
        "--base-url", default=defaults["base_url"], help=PARAMETER_HELP["base_url"]
    )
    groups["llm"].add_argument(
        "--api-key-env", default=defaults["api_key_env"], help=PARAMETER_HELP["api_key_env"]
    )
    groups["llm"].add_argument(
        "--env-file", type=Path, default=defaults["env_file"], help=PARAMETER_HELP["env_file"]
    )
    groups["llm"].add_argument(
        "--backend-options",
        type=backend_options_value,
        default=defaults["backend_options"],
        help=PARAMETER_HELP["backend_options"],
    )
    groups["llm"].add_argument(
        "--max-output-tokens",
        type=int,
        default=defaults["max_output_tokens"],
        help=PARAMETER_HELP["max_output_tokens"],
    )
    groups["run"].add_argument(
        "--preview-context",
        action="store_true",
        default=defaults["preview_context"],
        help=PARAMETER_HELP["preview_context"],
    )
    groups["run"].add_argument(
        "--preview-dx",
        type=float,
        default=defaults["preview_dx"],
        help=PARAMETER_HELP["preview_dx"],
    )
    groups["llm"].add_argument(
        "--supported-efforts",
        default=defaults["supported_efforts"],
        help=PARAMETER_HELP["supported_efforts"],
    )
    groups["llm"].add_argument(
        "--reasoning-mode",
        choices=["standard", "pro"],
        default=defaults["reasoning_mode"],
        help=PARAMETER_HELP["reasoning_mode"],
    )
    groups["policy"].add_argument(
        "--inference-timeout",
        type=float,
        default=defaults["inference_timeout"],
        help=PARAMETER_HELP["inference_timeout"],
    )
    groups["llm"].add_argument(
        "--max-response-chars",
        type=int,
        default=defaults["max_response_chars"],
        help=PARAMETER_HELP["max_response_chars"],
    )
    groups["llm"].add_argument(
        "--max-attempts",
        type=int,
        default=defaults["max_attempts"],
        help=PARAMETER_HELP["max_attempts"],
    )
    groups["llm"].add_argument(
        "--augmentations",
        type=augmentation_value,
        default=defaults["augmentations"],
        help=PARAMETER_HELP["augmentations"],
    )
    groups["observation"].add_argument(
        "--demo", action=argparse.BooleanOptionalAction, default=defaults["demo"], help=PARAMETER_HELP["demo"]
    )
    groups["observation"].add_argument(
        "--demo-path", type=Path, default=defaults["demo_path"], help=PARAMETER_HELP["demo_path"]
    )
    groups["observation"].add_argument(
        "--demo-mode", choices=("id", "ood"), default=defaults["demo_mode"],
        help=PARAMETER_HELP["demo_mode"],
    )
    groups["observation"].add_argument(
        "--demo-content", choices=("observations", "full"), default=defaults["demo_content"],
        help=PARAMETER_HELP["demo_content"],
    )
    groups["observation"].add_argument(
        "--demo-image-max-side", type=int, default=defaults["demo_image_max_side"],
        help=PARAMETER_HELP["demo_image_max_side"],
    )
    groups["output"].add_argument(
        "--record-trajectory", dest="record_trajectory",
        action=argparse.BooleanOptionalAction, default=defaults["record_trajectory"],
        help=PARAMETER_HELP["record_trajectory"]
    )
    groups["observation"].add_argument(
        "--max-images", type=int, default=defaults["max_images"], help=PARAMETER_HELP["max_images"]
    )
    groups["observation"].add_argument(
        "--max-context-chars",
        type=int,
        default=defaults["max_context_chars"],
        help=PARAMETER_HELP["max_context_chars"],
    )
    groups["observation"].add_argument(
        "--cameras", default=defaults["cameras"], help=PARAMETER_HELP["cameras"]
    )
    groups["observation"].add_argument(
        "--observation-profile",
        choices=["control", "openpi_matched", "privileged"],
        default=defaults["observation_profile"],
        help=PARAMETER_HELP["observation_profile"],
    )
    groups["execution"].add_argument(
        "--action-tools",
        choices=["move_by"],
        default=defaults["action_tools"],
        help=PARAMETER_HELP["action_tools"],
    )
    groups["execution"].add_argument(
        "--approver",
        choices=["none", "clamp"],
        default=defaults["approver"],
        help=PARAMETER_HELP["approver"],
    )
    groups["environment"].add_argument(
        "--sim-python", default=defaults["sim_python"], help=PARAMETER_HELP["sim_python"]
    )
    groups["environment"].add_argument(
        "--control-hz",
        type=float,
        default=defaults["control_hz"],
        help=PARAMETER_HELP["control_hz"],
    )
    groups["environment"].add_argument(
        "--sim-hz", type=int, default=defaults["sim_hz"], help=PARAMETER_HELP["sim_hz"]
    )
    groups["environment"].add_argument(
        "--sim-backend",
        choices=["cpu", "gpu"],
        default=defaults["sim_backend"],
        help=PARAMETER_HELP["sim_backend"],
    )
    groups["environment"].add_argument(
        "--sim-shader",
        choices=["rt-fast", "rt", "default"],
        default=defaults["sim_shader"],
        help=PARAMETER_HELP["sim_shader"],
    )
    groups["environment"].add_argument(
        "--gpu", type=int, default=defaults["gpu"], help=PARAMETER_HELP["gpu"]
    )
    groups["environment"].add_argument(
        "--molmoact2-source",
        default=defaults["molmoact2_source"],
        help=PARAMETER_HELP["molmoact2_source"],
    )
    groups["environment"].add_argument(
        "--maniskill-assets",
        default=defaults["maniskill_assets"],
        help=PARAMETER_HELP["maniskill_assets"],
    )
    groups["output"].add_argument(
        "--output-dir", type=Path, default=defaults["output_dir"], help=PARAMETER_HELP["output_dir"]
    )
    groups["policy"].add_argument(
        "--policy",
        type=str,
        choices=["llm", "openpi", "molmoact2"],
        dest="policy_family",
        default=defaults["policy_family"],
        help=PARAMETER_HELP["policy_family"],
    )
    groups["policy"].add_argument(
        "--model-profile",
        type=str,
        default=defaults["model_profile"],
        help=PARAMETER_HELP["model_profile"],
    )
    groups["policy"].add_argument(
        "--policy-profile",
        type=str,
        choices=["native", "custom"],
        default=defaults["policy_profile"],
        help=PARAMETER_HELP["policy_profile"],
    )
    groups["policy"].add_argument(
        "--policy-host",
        type=str,
        default=defaults["policy_host"],
        help=PARAMETER_HELP["policy_host"],
    )
    groups["policy"].add_argument(
        "--policy-port",
        type=int,
        default=defaults["policy_port"],
        help=PARAMETER_HELP["policy_port"],
    )
    groups["policy"].add_argument(
        "--policy-url", type=str, default=defaults["policy_url"], help=PARAMETER_HELP["policy_url"]
    )
    groups["policy"].add_argument(
        "--policy-metadata",
        type=Path,
        default=defaults["policy_metadata"],
        help=PARAMETER_HELP["policy_metadata"],
    )
    groups["policy"].add_argument(
        "--policy-seed", type=int, default=defaults["policy_seed"], help=PARAMETER_HELP["policy_seed"]
    )
    p.set_defaults(**_validated(defaults, p))
    return p


def _validated(values, parser):
    actions = {a.dest: a for a in parser._actions if a.dest in FIELD_GROUPS}
    output = {}
    for name, value in values.items():
        action = actions[name]
        if value is None:
            if name not in NULLABLE_FIELDS:
                raise ValueError(f"config {name} cannot be null")
            output[name] = None
            continue
        if action.choices and value not in action.choices:
            raise ValueError(f"invalid config {name}: {value!r}; choices={action.choices}")
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
            valid = type(value) is bool
        elif action.type is int:
            valid = type(value) is int
        elif action.type is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        elif action.type is horizon:
            if type(value) is not int or value <= 0:
                raise ValueError("config h must be a positive integer; select the tool with control_interface")
            valid = True
        elif action.type is execution_horizon:
            valid = type(value) is int and (value == -1 or value > 0)
        elif action.type in (augmentation_value, auto_motion_value, backend_options_value):
            value = action.type(value)
            valid = True
        else:
            valid = (
                isinstance(value, (str, Path)) if action.type is Path else isinstance(value, str)
            )
        if not valid:
            raise ValueError(f"invalid type for config {name}")
        output[name] = value
    return output


def parse_configuration(
    argv=None, *, parser_factory=parser, maniskill_suite="DroidPutEverythingInBox-v1"
):
    """Resolve common defaults < model profile < experiment file < explicit CLI."""
    from agentic_framework.configuration.profiles import load_model_profile, validate_native_horizon

    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser_factory()
    preliminary, _ = p.parse_known_args(argv)
    file_values = {}
    if preliminary.config:
        file_values = _validated(flatten_config(json.loads(preliminary.config.read_text())), p)
        config_base = preliminary.config.expanduser().absolute().parent
        for field in ("output_dir", "policy_metadata", "env_file", "demo_path",
                      "molmoact2_source", "maniskill_assets", "sim_python"):
            if file_values.get(field) is not None:
                file_values[field] = str(project_path(file_values[field], base=config_base))
        selected_profile = file_values.get("model_profile")
        if selected_profile and (str(selected_profile).endswith(".json") or "/" in str(selected_profile)):
            file_values["model_profile"] = str(project_path(selected_profile, base=config_base))
    # Suppressed defaults distinguish an explicit CLI selector from a file/default.
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--model-profile", default=argparse.SUPPRESS)
    selector, _ = probe.parse_known_args(argv)
    selected = getattr(
        selector, "model_profile", file_values.get("model_profile", preliminary.model_profile)
    )
    model = load_model_profile(selected) if selected else None
    model_defaults = {}
    if model:
        model_defaults = {
            **model["defaults"],
            "benchmark": model["environment"]["benchmark"],
            "observation_profile": model["environment"].get("observation_profile", "control"),
            "policy_family": model["family"],
            "control_interface": "move_by_chunk",
            "policy_host": model["transport"].get("host"),
            "policy_port": model["transport"].get("port"),
            "policy_url": model["transport"].get("url"),
        }
    model_defaults = _validated(model_defaults, p)
    p.set_defaults(**(model_defaults | file_values))
    args = p.parse_args(argv)
    for field in ("output_dir", "policy_metadata", "env_file", "demo_path"):
        if getattr(args, field) is not None:
            setattr(args, field, project_path(getattr(args, field), base=Path.cwd()))
    if args.policy_family is None:
        args.policy_family = "llm"
    if model is None and (args.mode == "vla" or args.policy_family != "llm"):
        raise ValueError(
            "native policies require --model-profile (configuration policy.model_profile)"
        )
    if model is not None:
        if args.policy_family != model["family"]:
            raise ValueError("policy family and model profile disagree")
        expected_benchmark = model["environment"]["benchmark"]
        if args.benchmark != expected_benchmark:
            raise ValueError(f"model profile requires benchmark {expected_benchmark}")
        if args.mode not in ("vla", "preview"):
            raise ValueError("a native model profile requires mode vla or preview")
        if args.control_interface != "move_by_chunk":
            raise ValueError("native model profiles require control_interface=move_by_chunk and retain their native controller")
        if args.motion_time_scale != 1:
            raise ValueError("native model profiles require motion_time_scale=1 and retain their native action timing")
        validate_native_horizon(model, args.h)
    if args.reasoning_mode is not None and args.backend == "codex":
        raise ValueError("reasoning_mode is not supported by Codex")
    if args.backend in ("responses", "codex"):
        if args.backend_options:
            raise ValueError("backend_options requires an installed backend extension")
        args.base_url = args.base_url or "https://api.openai.com/v1"
        args.api_key_env = args.api_key_env or "OPENAI_API_KEY"
    else:
        from agentic_framework.models.llm.extensions import load_backend_extension

        load_backend_extension(args.backend).configure(args)
    if args.demo and args.demo_path is None:
        raise ValueError("demo=true requires --demo-path")
    if args.demo_image_max_side < 0:
        raise ValueError("demo_image_max_side must be nonnegative")
    if args.demo and model is not None:
        raise ValueError("demonstration injection requires an LLM policy")
    if args.repeat_id is not None and args.repeat_id < 0:
        raise ValueError("repeat_id must be nonnegative")
    if args.policy_seed is not None:
        if not 0 <= args.policy_seed < 2**32:
            raise ValueError("policy_seed must be within 0..2^32-1")
        if model is None or args.policy_family not in ("openpi", "molmoact2"):
            raise ValueError("policy_seed requires an OpenPi/MolmoAct2 seed-capable policy")
    if args.preview_context:
        if args.mode != "preview":
            raise ValueError("preview_context is only valid in mode preview")
        if model is not None:
            raise ValueError(
                "preview_context requires an LLM policy; native model previews export initial inputs only"
            )
    if args.preview_dx is not None:
        if not args.preview_context or args.mode != "preview":
            raise ValueError("preview_dx requires a scripted context preview")
        if not math.isfinite(args.preview_dx):
            raise ValueError("preview_dx must be finite")
    args.augmentations = augmentation_value(args.augmentations)
    args.auto_motion = auto_motion_value(args.auto_motion)
    if args.suite is None:
        args.suite = maniskill_suite if args.benchmark == "maniskill" else "libero_spatial"
    if args.seed is None:
        args.seed = 42 if args.benchmark == "maniskill" else 0
    if args.benchmark == "maniskill" and args.observation_profile == "openpi_matched":
        raise ValueError("openpi_matched is the LIBERO observation profile, not DROID")
    if args.benchmark == "libero" and args.control_hz not in (None, 20.0):
        raise ValueError("LIBERO retains its official 20 Hz controller")
    if args.control_hz is None:
        args.control_hz = 30.0 if args.benchmark == "maniskill" else 20.0
    for field in (
        "trials",
        "max_images",
        "max_context_chars",
        "max_attempts",
        "max_response_chars",
        "sim_hz",
        "inference_timeout",
        "control_hz",
        "max_output_tokens",
        "max_steps",
    ):
        value = getattr(args, field)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{field} must be positive and finite")
    if args.policy_port is not None and not 1 <= args.policy_port <= 65535:
        raise ValueError("policy_port must be within 1..65535")
    if not math.isfinite(args.done_verification_seconds) or args.done_verification_seconds < 0:
        raise ValueError("done_verification_seconds must be finite and nonnegative")
    if args.done_verification_seconds and args.benchmark != "maniskill":
        raise ValueError("done verification requires the ManiSkill benchmark")
    if args.benchmark == "maniskill":
        ratio = args.sim_hz / args.control_hz
        if not math.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError("sim_hz must be an integer multiple of control_hz")
    if model is not None:
        explicit = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
        port_override = "--policy-port" in explicit or file_values.get("policy_port") is not None
        url_override = "--policy-url" in explicit or file_values.get("policy_url") is not None
        host_override = "--policy-host" in explicit or file_values.get("policy_host") is not None
        if model["transport"]["kind"] == "http":
            if host_override:
                raise ValueError("HTTP policies use --policy-url, not --policy-host")
            if port_override:
                if url_override:
                    raise ValueError("set either policy_url or policy_port for HTTP policies")
                parts = urlsplit(args.policy_url)
                host = parts.hostname
                if not host or parts.scheme not in ("http", "https") or parts.username:
                    raise ValueError("HTTP policy_url must be an HTTP(S) URL without credentials")
                host = f"[{host}]" if ":" in host else host
                args.policy_url = urlunsplit(parts._replace(netloc=f"{host}:{args.policy_port}"))
        elif url_override:
            raise ValueError("WebSocket policies use --policy-host/--policy-port, not --policy-url")
    if model is not None and args.policy_profile == "native":
        expected = model["defaults"]
        native_cadence = (expected["h"], expected["k"], expected["control_hz"])
        if (args.h, args.k, args.control_hz) != native_cadence:
            raise ValueError(
                f"native {model['id']} requires H/K/Hz={native_cadence}; "
                "use --policy-profile custom to change K or control_hz while retaining native H"
            )
    ScheduleConfig(
        h=args.h,
        k=args.k,
        control_interface=args.control_interface,
        motion_time_scale=args.motion_time_scale,
        max_motion_steps=args.max_motion_steps,
        max_steps=220 if args.max_steps is None else args.max_steps,
        auto_motion=AutoMotionConfig(**args.auto_motion),
    )
    for field in ("molmoact2_source", "maniskill_assets", "sim_python"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, str(project_path(value, base=Path.cwd())))

    return RunConfiguration.from_namespace(args)
