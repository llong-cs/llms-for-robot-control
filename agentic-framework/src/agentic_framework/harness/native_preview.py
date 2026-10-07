"""Offline exports of native VLA inputs through the production request builders."""

from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_framework.configuration.profiles import load_model_profile, validate_native_horizon
from agentic_framework.models.vla.common import _NativePolicy
from agentic_framework.observability.api_audit import (
    _append,
    _exclusive_root,
    _fence,
    _json,
    _link,
    _slug,
    _write_new,
)


class NativePreviewPolicy:
    """Adapt and archive observations without constructing a model client.

    Family request methods are pure: their only dependencies are the selected
    adapter, cadence and measured gripper bounds. Reuse those methods directly,
    without invoking the live policy constructor, reset, inference or provenance
    discovery. The archive boundary is the payload passed to client.infer();
    transport envelopes and server-side preprocessing remain outside it.
    """

    def __init__(self, args, directory):
        from agentic_framework.models.factory import VLA_FAMILIES

        selected = load_model_profile(args.model_profile)
        validate_native_horizon(selected, args.h)
        module_name, class_name = VLA_FAMILIES[selected["family"]]
        self._policy_class = getattr(
            importlib.import_module(f"agentic_framework.models.vla.{module_name}"), class_name
        )
        if selected["adapter"] not in self._policy_class.supported_adapters:
            raise ValueError(f"Unsupported {selected['family']} adapter: {selected['adapter']}")
        self.model_profile = selected
        self.model_profile_id = selected["id"]
        self.h, self.k = args.h, args.k
        self.profile, self.control_hz = args.policy_profile, float(args.control_hz)
        self.adapter = self._policy_class._create_adapter(self, selected)
        self.raw_action_semantics = getattr(
            self._policy_class, "raw_action_semantics", self.adapter.raw_action_semantics
        )
        self.info = self.adapter.initial_info(self.model_profile_id, self.control_hz)
        self.directory = (Path(directory) / "model/native-requests").resolve()
        self._closed, self._bound = False, False
        self._instruction = None
        self._previews = 0
        self._metadata = {
            "mode": "preview",
            "model_profile": self.model_profile_id,
            "family": selected["family"],
            "adapter": selected["adapter"],
            "profile": self.profile,
            "h": self.h,
            "k": self.k,
            "control_hz": self.control_hz,
            "raw_action_semantics": self.raw_action_semantics,
            "transport": _NativePolicy._transport(
                selected["transport"],
                host=args.policy_host,
                port=args.policy_port,
                url=args.policy_url,
            ),
            "expected_server_metadata": copy.deepcopy(selected["model"]["expected_metadata"]),
            "server_metadata_validated": False,
            "provenance_verified": False,
            "model_weights_loaded": False,
            "request_builder": (
                f"{self._policy_class._request.__module__}."
                f"{self._policy_class._request.__qualname__}"
            ),
            "adapter_implementation": f"{type(self.adapter).__module__}.{type(self.adapter).__name__}",
            **self.adapter.metadata(selected),
        }

    def bind(self, embodiment_info):
        if self._closed:
            raise RuntimeError("Native preview is closed")
        self.adapter.bind(self, embodiment_info)
        self._bound = True

    def reset(self, scene):
        if self._closed:
            raise RuntimeError("Native preview is closed")
        self._instruction = scene.instruction
        self._previews = 0

    def preview_request(self, observation):
        if self._closed or not self._bound:
            raise RuntimeError("Native preview must be bound and open")
        step = observation.extra.get("env_step", 0)
        if type(step) is not int or step < 0:
            raise ValueError("Native policy env_step must be a nonnegative integer")
        instruction = observation.instruction or self._instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Native policy requires a nonempty instruction")
        request = self._policy_class._request(self, observation, instruction)
        state = self._policy_class._request_state(self, request)
        artifact = self._record(
            request,
            {
                "env_step": step,
                "request_sim_time_s": self.adapter.sim_time(observation, step, self.control_hz),
                "instruction": instruction,
                "state": state.tolist(),
            },
        )
        self._previews += 1
        return artifact

    def _record(self, request, observation_metadata):
        with _exclusive_root(self.directory):
            numbers = [
                int(p.name.split("-", 1)[0])
                for p in self.directory.iterdir()
                if p.is_dir() and p.name.split("-", 1)[0].isdigit()
            ]
            sequence = max(numbers, default=0) + 1
            request_id = f"{self.model_profile_id}-{sequence}"
            directory = self.directory / f"{sequence:06d}-{_slug(request_id)}"
            directory.mkdir(mode=0o700)
            arrays = []

            def archive(value, location):
                if isinstance(value, np.ndarray):
                    record = {
                        "type": "ndarray",
                        "key": location,
                        "dtype": str(value.dtype),
                        "dtype_str": value.dtype.str,
                        "shape": list(value.shape),
                        "data_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    }
                    stem = f"{len(arrays):03d}-{_slug(location)}"
                    data = io.BytesIO()
                    np.save(data, value, allow_pickle=False)
                    array_path = directory / f"{stem}.npy"
                    _write_new(array_path, data.getvalue())
                    record["npy_path"] = str(array_path)
                    record["npy_sha256"] = hashlib.sha256(data.getvalue()).hexdigest()
                    if value.ndim <= 1:
                        record["values"] = value.tolist()
                    if value.ndim == 3 and value.shape[-1] == 3 and value.dtype == np.uint8:
                        png = io.BytesIO()
                        Image.fromarray(value).save(png, format="PNG")
                        image_path = directory / f"{stem}.png"
                        _write_new(image_path, png.getvalue())
                        record["png_path"] = str(image_path)
                    arrays.append(record)
                    return record
                if isinstance(value, dict):
                    return {key: archive(item, f"{location}/{key}") for key, item in value.items()}
                if isinstance(value, (tuple, list)):
                    return [
                        archive(item, f"{location}/{index}") for index, item in enumerate(value)
                    ]
                if isinstance(value, np.generic):
                    return value.item()
                return value

            body = {key: archive(value, key) for key, value in request.items()}
            raw = _json(body).encode()
            path = directory / "request.json"
            _write_new(path, raw)
            scope = (
                "Exact framework-to-client inference payload from the production request builder. "
                "Arrays retain dtype, shape and values in .npy files; PNGs show the same input pixels. "
                "No server connection, inference, credentials or weight loading. Transport envelopes "
                "and server-side transforms and model-internal prompts "
                "are outside this boundary. Expected server metadata is a configuration requirement, "
                "not evidence of an available or verified server."
            )
            manifest = {
                **self.metadata(),
                **observation_metadata,
                "sequence": sequence,
                "request_id": request_id,
                "model_calls": 0,
                "api_calls": 0,
                "scope": scope,
                "arrays": arrays,
                "request_path": str(path),
                "request_sha256": hashlib.sha256(raw).hexdigest(),
            }
            manifest_path = directory / "manifest.json"
            _write_new(manifest_path, _json(manifest).encode())
            readable = [
                f"# Native VLA request {sequence:06d}: {self.model_profile_id}",
                "**Zero model calls.** " + scope,
                _link(path, "Request fields and recoverable array descriptors"),
                _link(manifest_path, "Input contract, configured provenance and array checksums"),
                "## Instruction",
                _fence(observation_metadata["instruction"]),
                "## State supplied to the model client",
                _fence(_json(observation_metadata["state"]), "json"),
                "## Complete request",
                _fence(_json(body), "json"),
                "Each ndarray descriptor's npy_path can be loaded with "
                "`numpy.load(path, allow_pickle=False)` to recover its exact dtype, shape and values.",
            ]
            for record in arrays:
                if "png_path" in record:
                    readable.extend(
                        [
                            f"## Image: {record['key']} ({record['shape']}, {record['dtype']})",
                            f"![{record['key']}](<{record['png_path']}>)",
                        ]
                    )
            view = directory / "request.md"
            _write_new(view, "\n\n".join(readable).encode())
            entry = {
                "sequence": sequence,
                "request_id": request_id,
                "readable_path": str(view),
                "request_path": str(path),
            }
            _append(self.directory / "requests.jsonl", (json.dumps(entry) + "\n").encode())
            _append(
                self.directory / "index.md",
                (f"- {_link(view, str(sequence) + ': ' + request_id)}\n").encode(),
                initial=b"# Native VLA request previews\n\n",
            )
            return {
                "request_record_dir": str(directory),
                "request_body_path": str(path),
                "request_readable_path": str(view),
                "request_manifest_path": str(manifest_path),
                "request_body_sha256": manifest["request_sha256"],
                "request_sequence": sequence,
                "request_mode": "preview",
                "request_index_path": str(self.directory / "index.md"),
            }

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def metrics(self):
        return {
            "model": self.model_profile_id,
            "model_calls": 0,
            "api_calls": 0,
            "planning_calls": 0,
            "polling_calls": 0,
            "format_repair_calls": 0,
            "format_errors": 0,
            "inference_wall_s": 0.0,
            "executed_actions": 0,
            "preview_requests": self._previews,
            "inference_control_steps": [],
            "decision_intervals": [],
            "usage": {},
        }

    def act(self, observation):
        raise RuntimeError("Native preview exports inputs only; it cannot infer or execute actions")

    def close(self):
        self._closed = True
