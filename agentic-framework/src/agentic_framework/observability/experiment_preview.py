"""Local experiment and observation exports; never invoke a model or simulator."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_framework.common.io import plain, write_json


def write_observation_preview(directory, observation, *, scene, reset_info):
    """Keep preview observations separate from the serialized model input."""
    directory = Path(directory) / "observation"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "state.json", observation.state)
    # Only the declared proprioceptive adapter inputs, never arbitrary extra data.
    extra = {
        key: plain(observation.extra[key])
        for key in (
            "raw_libero",
            "raw_droid",
            "robot_base_quat",
            "robot_base_position",
            "gripper_target",
            "env_step",
        )
        if key in observation.extra
    }
    write_json(directory / "adapter-state.json", extra)
    images = []
    for index, (camera, value) in enumerate(observation.images.items()):
        pixels = np.asarray(value)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
            raise ValueError(f"preview camera {camera} must be HWC uint8 RGB/RGBA")
        # Camera labels are data, never filenames or filesystem paths.
        path = directory / f"camera-{index:03d}.png"
        Image.fromarray(pixels).save(path)
        images.append(
            {
                "camera": camera,
                "path": str(path.resolve()),
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    report = {
        "scene": asdict(scene),
        "instruction": observation.instruction,
        "state_path": str((directory / "state.json").resolve()),
        "adapter_state_path": str((directory / "adapter-state.json").resolve()),
        "simulator_reset": plain(reset_info),
        "images": images,
        "scope": "Initial environment observation for inspection; actual model inputs are archived separately after preprocessing.",
    }
    write_json(directory / "manifest.json", report)
    return {"manifest_path": str((directory / "manifest.json").resolve()), **report}


def _resources(config):
    resources = []
    profile = config.get("model_profile_definition")
    paths = {
        "simulator_python": config.get("sim_python"),
    }
    if config.get("benchmark") == "maniskill":
        paths.update(
            simulator_source=config.get("molmoact2_source"),
            simulator_assets=config.get("maniskill_assets"),
        )
    if profile:
        paths.update(
            checkpoint=profile["model"]["checkpoint"],
            model_environment=profile["server"]["environment"],
            model_source=profile["server"]["source"],
        )
        for name, resource in profile["server"].get("resources", {}).items():
            paths[name] = resource["path"]
    for name, value in paths.items():
        if value is not None:
            path = Path(value).expanduser()
            resources.append({"name": name, "path": str(path), "exists": path.exists()})
    return resources


def write_experiment_preview(output, config, scenes, results=(), *, trial_budgets=None):
    """Write/update a browsable index, including partial exports after a failure."""
    output = Path(output)
    by_id = {row["scene_id"]: row for row in results}
    native = config["policy_family"] != "llm"
    hz = config["control_hz"]
    verification = config["done_verification_steps"]
    items = []
    for scene in scenes:
        row = by_id.get(scene.id)
        steps = (trial_budgets or {}).get(scene.id, config["schedule"]["max_steps"])
        directory = output / "evals" / scene.id
        items.append(
            {
                "scene": plain(asdict(scene)),
                "status": row["status"] if row else "pending",
                "error": row.get("error") if row else None,
                "physical_step_limit": steps,
                "simulation_time_limit_s": steps / hz,
                # Formal runs add these holds only after a model done call.
                "done_verification_steps": verification,
                # Attempts of one LLM decision, not a trial total.
                "max_attempts": None if native else config["max_attempts"],
                "scripted_steps": row.get("steps", 0) if row else 0,
                "preview_path": str(directory / "preview.json"),
                "result_path": str(directory / "result.json"),
                "preview": row.get("preview") if row else None,
            }
        )
    report = {
        "mode": "preview",
        "preview_kind": "scripted_context" if config["preview_context"] else "initial_observation",
        "model_calls": 0,
        "api_calls": 0,
        "task_success_evaluated": False,
        "planned_trials": len(items),
        "exported_trials": sum(item["status"] == "success" for item in items),
        "failed_trials": sum(item["status"] in {"error", "cancelled"} for item in items),
        "discarded_trials": sum(item["status"] == "discarded" for item in items),
        "config_path": str(output / "config.json"),
        "manifest_path": str(output / "manifest.json"),
        "configuration": config["configuration"],
        "execution": {"schedule": config["schedule"], **config["effective_control"]},
        "model": config.get("model_profile_definition")
        or {"family": "llm", "backend": config["backend"], "model": config["model"]},
        "provenance": {
            "source_file_count": len(config["source_sha256"]),
            "dependency_source": config["dependency_source"],
            "python": config["python"],
            "package_count": len(config["packages"]),
            "policy_provenance_sha256": config["policy_provenance_sha256"],
        },
        "simulator": config["simulator"],
        "resources": _resources(config),
        "resource_check_scope": "Path existence only; no model load, server connection or weight integrity verification.",
        "budget_scope": (
            "Physical time is a configured ceiling, not wall time. max_attempts bounds the "
            "attempts of one LLM decision, format repairs and resends after transport/provider "
            "or internal failures included; a trial with a decision that fails all of them is "
            "discarded. Token usage, cost and task success are not predicted."
            + (f" In a formal run, a model done call starts {verification} done-verification "
               "hold steps without model calls; success within them counts, otherwise the "
               "trial fails. This preview never executes them."
               if verification else "")
        ),
        "total_physical_step_limit": sum(item["physical_step_limit"] for item in items),
        "trials": items,
    }
    report = plain(report)
    write_json(output / "preview.json", report)
    lines = [
        "# Experiment preview",
        "",
        "No real model/API calls. This export describes the configured experiment and its inputs; it is not an evaluation result.",
        "",
        "[Complete configuration and source/environment fingerprints](config.json) · "
        "[Task manifest](manifest.json) · [Structured preview](preview.json)",
        "",
        f"Policy: `{config['policy_family']}`; input mode: `{report['preview_kind']}`.",
        f"Control: H={config['h']}, K={config['k']}, {hz:g} Hz; interface `{config['effective_control']['interface']}`.",
        f"Trials exported: {report['exported_trials']}/{len(items)}; failed: {report['failed_trials']}; "
        f"discarded: {report['discarded_trials']}.",
        "",
        "## Tasks and budgets",
        "",
        "| Trial | Status | Physical step limit | Simulated seconds | max_attempts per decision | Export |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in items:
        scene_id = item["scene"]["id"]
        directory = Path("evals") / scene_id
        lines.append(
            f"| {scene_id} | {item['status']} | {item['physical_step_limit']} | "
            f"{item['simulation_time_limit_s']:.2f} | {item['max_attempts'] if not native else 'N/A'} | "
            f"[Trial preview]({directory}/preview.json) |"
        )
    lines += ["", "## Trial inputs", ""]
    for item in items:
        preview = item["preview"] or {}
        lines += [f"### {item['scene']['id']}", "", item["scene"]["instruction"], ""]
        observation = preview.get("observation") or {}
        if observation.get("manifest_path"):
            path = Path(observation["manifest_path"]).relative_to(output)
            lines.append(f"[Initial state and camera manifest]({path})")
        requests = preview.get("requests", [])
        if preview.get("request"):
            requests = [preview["request"], *requests]
        for index, request in enumerate(requests, 1):
            if request.get("request_readable_path"):
                path = Path(request["request_readable_path"]).relative_to(output)
                lines.append(f"[Model input {index}]({path})")
        lines.append("")
    lines += [
        "",
        report["budget_scope"],
        "",
        "## Available exports",
        "",
        "- `config.json`: resolved settings, complete model profile, source hashes, simulator and package versions.",
        "- `manifest.json`: task instructions, seeds, initial-state indices and scene metadata.",
        "- `evals/<trial>/observation/`: environment state and original camera images for inspection.",
        "- `evals/<trial>/model/`: serialized model inputs, processed images, tool schemas and readable request views.",
        "- With scripted context enabled: per-step logs, execution feedback and successive request archives.",
        "",
        "## Local resources",
        "",
        report["resource_check_scope"],
        "",
    ]
    lines += [
        f"- {item['name']}: `{item['path']}` — {'present' if item['exists'] else 'missing'}"
        for item in report["resources"]
    ]
    (output / "preview.md").write_text("\n".join(lines) + "\n")
    return report
