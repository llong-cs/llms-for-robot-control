"""Zero-model-call request previews over the production policy and serializers."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from inspect_robots.rollout import derive_seed

from agentic_framework.common.io import plain
from agentic_framework.models.llm.backend import BackendConfigurationError, scripted_reply


def preview_first_request(policy, embodiment, scene, *, eval_seed, epoch=0, directory=None):
    """Use the same reset and context as eval, without a policy or HTTP step."""
    policy.bind(embodiment.info)
    policy.reset(scene)
    trial_seed = derive_seed(eval_seed, scene.init_seed, epoch)
    observation = embodiment.reset(scene, seed=trial_seed)
    observation = replace(observation, extra={**observation.extra, "env_step": 0})
    report = {
        "kind": "first_request_preview",
        "api_calls": 0,
        "policy_steps": 0,
        "model_calls": 0,
        "trial_seed": trial_seed,
        "eval_seed": eval_seed,
        "scene_seed": scene.init_seed,
        "epoch": epoch,
        "simulator_reset": plain(embodiment.last_reset_info),
        "initially_solved": bool(embodiment.last_reset_info.get("success", False)),
    }
    if directory is not None:
        from agentic_framework.observability.experiment_preview import write_observation_preview

        report["observation"] = write_observation_preview(
            directory, observation, scene=scene, reset_info=embodiment.last_reset_info
        )
        info = embodiment.info
        report["embodiment"] = {
            "name": info.name,
            "control_hz": info.control_hz,
            "action_space": plain(info.action_space),
            "observation_space": plain(info.observation_space),
            "capabilities": sorted(info.capabilities),
            "is_simulated": info.is_simulated,
        }
    if report["initially_solved"]:
        report["request"] = None
        return report
    if hasattr(policy, "preview_request"):
        report["request"] = policy.preview_request(observation)
    else:
        request = policy.prepare_request(observation)
        report["request"] = record_preview_request(policy.backend, request)
        assert policy.calls == 0 and not policy.decisions and not policy.actions
    return report


class PreviewEmbodiment:
    """Record the real initial observation at the rollout's single reset."""

    def __init__(self, embodiment, directory):
        self.embodiment = embodiment
        self.directory = directory
        self.observation_artifact = None

    def __getattr__(self, name):
        return getattr(self.embodiment, name)

    def reset(self, scene, *, seed=None):
        from agentic_framework.observability.experiment_preview import write_observation_preview

        observation = self.embodiment.reset(scene, seed=seed)
        self.observation_artifact = write_observation_preview(
            self.directory, observation, scene=scene, reset_info=self.embodiment.last_reset_info
        )
        return observation


class OfflinePreviewBackend:
    """Exercise real context/servo scheduling using explicitly scripted replies.

    This is an engineering example of later contexts, not a prediction of future
    paid-model outputs. The wrapped backend's generate/authentication never runs.
    """

    is_offline_preview = True

    def __init__(self, api_backend, *, motion_delta=0.0):
        if getattr(api_backend, "request_recorder", None) is None:
            raise BackendConfigurationError("Context preview requires a request recorder")
        self.api_backend = api_backend
        self.model = api_backend.model
        self.motion_delta = motion_delta
        self.calls = 0

    def generate(self, request):
        artifact = record_preview_request(self.api_backend, request)
        self.calls += 1
        if self.calls <= 2:
            name = (
                "move_by_chunk"
                if any(t["name"] == "move_by_chunk" for t in request.tools)
                else "move_by"
            )
            arguments = {
                "deltas": {"dx": self.motion_delta},
                "note": "Scripted request preview: request this displacement from the observed pose.",
            }
        else:
            name = "done"
            arguments = {
                "summary": "Offline context preview finished; this is not a model task-success claim.",
                "hindsight": "These replies were scripted only to show request construction without an API call.",
            }
        tool = next(t for t in request.tools if t["name"] == name)
        properties = tool["parameters"]["properties"]
        # Derive the fixed horizon from the exact tool schema being previewed.
        if name == "move_by_chunk":
            horizon = properties["deltas"]["properties"]["dx"]["maxItems"]
            arguments = {
                "deltas": {
                    dimension: [value] * horizon for dimension, value in arguments["deltas"].items()
                },
                **({"note": arguments["note"]} if "note" in properties else {}),
            }
        elif "note" not in properties:
            arguments.pop("note", None)
        if "arm" in properties:
            arguments["arm"] = properties["arm"]["enum"][0]
        if "hindsight" not in properties:
            arguments.pop("hindsight", None)
        native = scripted_reply(
            request,
            {
                "tool_calls": [
                    {
                        "type": "function_call",
                        "call_id": f"preview-{self.calls}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                    }
                ]
            },
        )
        reply = replace(
            native,
            metadata={
                "backend": "offline_request_preview",
                "model_calls": 0,
                "requested_model": self.model,
                "api_calls": 0,
                "scripted_preview_response": True,
                "request_artifact": artifact,
            },
        )
        self.api_backend.accept_reply(request, reply)
        return reply

    def reset(self):
        self.calls = 0
        self.api_backend.reset()

    def record_tool_results(self, results):
        self.api_backend.record_tool_results(results)

    def end_rollout(self):
        end = getattr(self.api_backend, "end_rollout", None)
        if end is not None:
            return end()
        return None

    def close(self):
        self.api_backend.close()


def record_preview_request(backend, request):
    """Record through the backend's production serializer without calling it."""
    recorder = getattr(backend, "request_recorder", None)
    if recorder is None:
        raise BackendConfigurationError("Request preview requires a request recorder")
    if hasattr(backend, "prepare_http_request"):
        prepared = backend.prepare_http_request(request)
    elif hasattr(backend, "prepare_request"):
        prepared = backend.prepare_request(request)
    else:
        raise BackendConfigurationError("Backend does not support a pure request preview")
    return recorder(request, prepared)


class CodexRequestRecorder:
    """Archive initial native tools or callback continuation without starting Codex."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def __call__(self, request, prepared):
        from agentic_framework.observability.api_audit import (
            _append,
            _exclusive_root,
            _fence,
            _images,
            _json,
            _link,
            _slug,
            _write_new,
        )

        metadata = prepared["metadata"]
        continuation = "tool_response" in prepared
        if continuation:
            if "thread_start" in prepared or "turn_start" in prepared:
                raise ValueError("A Codex preview cannot start and continue the same request")
            response = prepared["tool_response"]
            source = response["result"]["contentItems"]
            heading = "Complete native tool result and current observation (in actual order)"
        else:
            thread, turn = prepared["thread_start"], prepared["turn_start"]
            source = turn["input"]
            heading = "Complete first observation in turn/start (in actual order)"
        # Normalize only the image preview view; archive the original RPC bytes.
        parts = []
        for item in source:
            if item["type"] in {"text", "inputText"}:
                parts.append({"type": "input_text", "text": item["text"]})
            elif item["type"] in {"image", "inputImage"}:
                parts.append(
                    {"type": "input_image", "image_url": item.get("url", item.get("imageUrl"))}
                )
            else:
                raise ValueError("Unexpected Codex preview input type")
        raw = json.dumps(prepared, ensure_ascii=False, allow_nan=False).encode()
        with _exclusive_root(self.directory):
            numbers = [
                int(p.name.split("-", 1)[0])
                for p in self.directory.iterdir()
                if p.is_dir() and p.name.split("-", 1)[0].isdigit()
            ]
            sequence = max(numbers, default=0) + 1
            directory = self.directory / f"{sequence:06d}-{_slug(request.request_id)}"
            directory.mkdir(mode=0o700)
            path = directory / "request.rpc.json"
            _write_new(path, raw)
            images = _images({"input": [{"content": parts}]}, directory)
            manifest = {
                "mode": "preview",
                "backend": "codex_app_server",
                "model_calls": 0,
                "sequence": sequence,
                "request_id": request.request_id,
                "model": metadata["model"],
                "requested_effort": request.reasoning_effort,
                "preview_effort": metadata["prepared_effort"],
                "operation": metadata["operation"],
                "context_lifetime": metadata["context_lifetime"],
                "history_length": metadata["history_length"],
                "memory_enabled": metadata["memory_enabled"],
                "history_length_applies": metadata["history_length_applies"],
                "history_length_semantics": metadata["history_length_semantics"],
                "capabilities_validated": False,
                "rpc_ids_are_placeholders": True,
                "rpc_bytes": len(raw),
                "rpc_sha256": hashlib.sha256(raw).hexdigest(),
                "images": images,
                "rpc_path": str(path),
                "scope": "Framework-supplied session start or native tool callback response; not the Codex provider HTTP wire or hidden model context.",
            }
            _write_new(directory / "manifest.json", _json(manifest).encode())
            readable = [
                f"# Codex request {sequence:06d}",
                "**Zero model calls.** Framework-supplied Codex App Server parameters, generated by the production request builder. No CLI process, credentials or model discovery was used.",
                "RPC IDs are placeholders. This preview shows the requested effort; live model capability validation still happens before inference. Codex-internal instructions, retained reasoning and provider HTTP serialization are outside this preview boundary.",
                _link(path, "Complete original RPC parameter bundle (including image base64)"),
                "## Session operation",
                _fence(_json(metadata), "json"),
            ]
            if continuation:
                readable += [
                    "This response continues the rollout's pending native robot tool call. The current observation follows the execution result. Robot tools and instructions were registered at rollout start; session history remains owned by Codex.",
                    "## Native callback envelope",
                    _fence(
                        _json(
                            {
                                "id": response["id"],
                                "result": {
                                    k: v
                                    for k, v in response["result"].items()
                                    if k != "contentItems"
                                },
                            }
                        ),
                        "json",
                    ),
                ]
            else:
                readable += [
                    "## Full baseInstructions",
                    _fence(thread["baseInstructions"]),
                    "## Full developerInstructions",
                    _fence(thread["developerInstructions"]),
                    "## Complete native dynamicTools",
                    _fence(_json(thread["dynamicTools"]), "json"),
                    "## Other thread/start parameters",
                    _fence(
                        _json(
                            {
                                k: v
                                for k, v in thread.items()
                                if k
                                not in ("baseInstructions", "developerInstructions", "dynamicTools")
                            }
                        ),
                        "json",
                    ),
                    "## Other turn/start parameters",
                    _fence(_json({k: v for k, v in turn.items() if k != "input"}), "json"),
                ]
            readable += ["## " + heading]
            image_by_part = {x["part_index"]: x for x in images}
            for index, part in enumerate(parts):
                if part["type"] == "input_text":
                    readable += [f"Text part {index}:", _fence(part["text"])]
                else:
                    record = image_by_part[index]
                    readable += [
                        f"Image part {index}: {record['camera_label']} ({record['width']} × {record['height']})",
                        f"![Image {record['order']}](<{record['path']}>)",
                    ]
            view = directory / "request.md"
            _write_new(view, "\n\n".join(readable).encode())
            entry = {
                "sequence": sequence,
                "request_id": request.request_id,
                "readable_path": str(view),
                "rpc_path": str(path),
            }
            _append(self.directory / "requests.jsonl", (json.dumps(entry) + "\n").encode())
            _append(
                self.directory / "index.md",
                (f"- {_link(view, str(sequence) + ': ' + request.request_id)}\n").encode(),
                initial=b"# Codex request previews\n\n",
            )
            return {
                "request_record_dir": str(directory),
                "request_body_path": str(path),
                "request_readable_path": str(view),
                "request_manifest_path": str(directory / "manifest.json"),
                "request_body_sha256": manifest["rpc_sha256"],
                "request_sequence": sequence,
                "request_mode": "preview",
                "request_index_path": str(self.directory / "index.md"),
            }
