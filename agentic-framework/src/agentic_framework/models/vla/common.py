"""Shared native inference lifecycle, provenance, and DROID observation validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots.policy import PolicyBase, PolicyConfig
from inspect_robots.types import Action, ActionChunk, Observation

from agentic_framework._vendor.openpi_client import image_tools
from agentic_framework.configuration.profiles import load_model_profile, validate_native_horizon
from agentic_framework.models.seed_protocol import SEED_KEY, SEED_PROTOCOL, sampling_request
from agentic_framework.observability.progress import request_progress

DROID_COMMIT = "33ae6a67274f36d2e29525b86f23a56616ef43a7"


def _numeric(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name} must have shape {shape} and numeric values")
    if np.iscomplexobj(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be real and finite")
    return result


def droid_state(observation: Observation) -> np.ndarray:
    """Exact official MolmoAct2 sim qpos13 → float32 state8; no privileged state."""
    raw = observation.extra.get("raw_droid")
    if not isinstance(raw, Mapping):
        raise ValueError("DROID requires observation.extra['raw_droid']")
    qpos = _numeric(raw.get("qpos"), (13,), "raw DROID qpos").astype(np.float32)
    if not np.isfinite(qpos).all():
        raise ValueError("DROID qpos cannot be represented in float32")
    return qpos[:8].copy()


def droid_images(
    observation: Observation, *, resize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for key in ("external_cam", "wrist_cam"):
        image = np.asarray(observation.images.get(key))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"DROID {key} must be HWC uint8 RGB")
        if resize:
            image = image_tools.resize_with_pad(image, 224, 224)
        result.append(np.ascontiguousarray(image).copy())
    return tuple(result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _NativePolicy(PolicyBase):
    """One inference/recording path, with physical contracts supplied by adapters."""

    def __init__(
        self,
        *,
        model_profile,
        family,
        client=None,
        h=None,
        k=None,
        control_hz=None,
        profile="native",
        host=None,
        port=None,
        url=None,
        metadata_path=None,
        verified_metadata=None,
        timeout_s=None,
        policy_seed=None,
        strict_provenance=False,
    ):
        self._client, self._closed = client, False
        try:
            self._initialize(
                model_profile=model_profile,
                family=family,
                h=h,
                k=k,
                control_hz=control_hz,
                profile=profile,
                host=host,
                port=port,
                url=url,
                metadata_path=metadata_path,
                verified_metadata=verified_metadata,
                timeout_s=timeout_s,
                policy_seed=policy_seed,
                strict_provenance=strict_provenance,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        *,
        model_profile,
        family,
        h,
        k,
        control_hz,
        profile,
        host,
        port,
        url,
        metadata_path,
        verified_metadata,
        timeout_s,
        policy_seed,
        strict_provenance,
    ):
        selected = load_model_profile(model_profile)
        if selected["family"] != family:
            raise ValueError(f"{type(self).__name__} requires a {family} model profile")
        if selected["adapter"] not in self.supported_adapters:
            raise ValueError(f"Unsupported {family} adapter: {selected['adapter']}")
        defaults = selected["defaults"]
        h, k = defaults["h"] if h is None else h, defaults["k"] if k is None else k
        control_hz = defaults["control_hz"] if control_hz is None else control_hz
        timeout_s = defaults["inference_timeout"] if timeout_s is None else timeout_s
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if policy_seed is not None:
            sampling_request(policy_seed, 0)
        self.policy_seed = policy_seed
        self.trajectory_recorder = None
        self.timeout_s = float(timeout_s)
        validate_native_horizon(selected, h)
        if type(k) is not int or k < 1:
            raise ValueError("Native policies require positive K; K=-1 is supported only by move_by")
        if profile not in ("native", "custom"):
            raise ValueError("policy profile must be native or custom")
        if type(control_hz) not in (int, float) or not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        self.native_settings = (defaults["h"], defaults["k"], defaults["control_hz"])
        if profile == "native" and (h, k, control_hz) != self.native_settings:
            raise ValueError(
                f"native {selected['id']} requires H/K/Hz={self.native_settings}; "
                "use profile='custom' to change K or control_hz while retaining native H"
            )
        if metadata_path is not None and verified_metadata is not None:
            raise ValueError("use metadata_path or verified_metadata, not both")
        if metadata_path is not None:
            verified_metadata = json.loads(Path(metadata_path).expanduser().read_text())
        if verified_metadata is not None and not isinstance(verified_metadata, Mapping):
            raise ValueError("Native policy provenance must be an object")
        self.model_profile = selected
        self.model_profile_id = selected["id"]
        self.h, self.k, self.profile, self.control_hz = h, k, profile, float(control_hz)
        self.config = PolicyConfig(action_horizon=h, replan_interval=k)
        self.adapter = self._create_adapter(selected)
        self.raw_action_semantics = getattr(
            self, "raw_action_semantics", self.adapter.raw_action_semantics
        )
        self.requires_droid_controller = getattr(
            self, "requires_droid_controller", self.adapter.requires_droid_controller
        )
        self.info = self.adapter.initial_info(self.model_profile_id, self.control_hz)
        self._bound = not self.adapter.requires_bind
        self._native_controller_active = False
        self._events, self._executed = [], []
        self._delta_cursor = 0
        self._trial_started = time.perf_counter()
        self._instruction = None
        transport = self._transport(selected["transport"], host=host, port=port, url=url)
        if self._client is None:
            self._client = self._create_client(transport, timeout_s)
        server = (
            self._client.get_server_metadata()
            if hasattr(self._client, "get_server_metadata")
            else {}
        )
        if not isinstance(server, Mapping):
            raise ValueError("Native policy server metadata must be an object")
        provenance = dict(verified_metadata if verified_metadata is not None else server)
        expected = selected["model"]["expected_metadata"]
        optional = selected["model"].get("optional_metadata", {})
        integrity_keys = (
            "checkpoint_tree_sha256",
            "checkpoint_source_manifest_sha256",
            "norm_stats_sha256",
            "model_code_sha256",
        )
        for key in (*expected, *optional, *integrity_keys):
            if key in server and key in provenance and server[key] != provenance[key]:
                raise ValueError(f"server and provenance disagree on {key}")
        if profile == "native" or strict_provenance or policy_seed is not None:
            for key, value in expected.items():
                if provenance.get(key) != value:
                    raise ValueError(
                        f"native {self.model_profile_id} requires provenance {key}={value!r}"
                    )
            for key, value in optional.items():
                if key in provenance and provenance[key] != value:
                    raise ValueError(
                        f"native {self.model_profile_id} requires provenance {key}={value!r}"
                    )
        if policy_seed is not None and server.get("seed_protocol") != SEED_PROTOCOL:
            raise ValueError(f"Seeded inference requires server support for {SEED_PROTOCOL}")
        self._metadata = {
            "policy_seed": policy_seed,
            "seed_protocol": server.get("seed_protocol"),
            "sampling_controlled": policy_seed is not None,
            "strict_provenance": profile == "native"
            or strict_provenance
            or policy_seed is not None,
            "model_profile": self.model_profile_id,
            "family": family,
            "adapter": selected["adapter"],
            "profile": profile,
            "h": h,
            "k": k,
            "control_hz": self.control_hz,
            "inference_timeout": self.timeout_s,
            "provenance": _json_safe(provenance),
            "server_metadata": _json_safe(server),
            "provenance_source": "launch_manifest" if verified_metadata is not None else "server",
            "raw_action_semantics": self.raw_action_semantics,
            "transport": transport,
            **self.adapter.metadata(selected),
        }

    @staticmethod
    def _transport(value, *, host, port, url):
        from urllib.parse import urlsplit, urlunsplit

        result = copy.deepcopy(value)
        if host is not None:
            if not isinstance(host, str) or not host.strip():
                raise ValueError("policy host must be nonempty")
            result["host"] = host
        if port is not None:
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("policy port must be an integer in [1, 65535]")
            result["port"] = port
        if url is not None:
            if not isinstance(url, str) or not url.strip():
                raise ValueError("policy url must be nonempty")
            result["url"] = url
        elif result["kind"] == "http" and (host is not None or port is not None):
            parsed = urlsplit(result["url"])
            address = result.get("host", parsed.hostname)
            if ":" in address and not address.startswith("["):
                address = f"[{address}]"
            result["url"] = urlunsplit(
                (
                    parsed.scheme,
                    f"{address}:{result.get('port', parsed.port)}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
        return result

    def _create_adapter(self, model_profile):
        from agentic_framework.models.vla.adapters import DroidJointAdapter

        return DroidJointAdapter()

    def bind(self, embodiment_info):
        self.adapter.bind(self, embodiment_info)
        self._bound = True

    def reset(self, scene):
        if self._closed:
            raise RuntimeError("Native policy is closed")
        self._events.clear()
        self._executed.clear()
        self._delta_cursor = 0
        self._trial_started = time.perf_counter()
        self._instruction = scene.instruction
        if hasattr(self._client, "reset"):
            self._client.reset()

    def _request(self, observation, instruction):
        return self.adapter.request(self, observation, instruction)

    def _request_state(self, request):
        return self.adapter.request_state(self, request)

    def act(self, observation):
        if self._closed or not self._bound:
            raise RuntimeError("Native policy must be bound and open")
        if self.requires_droid_controller and not self._native_controller_active:
            raise RuntimeError(
                f"{type(self).__name__} requires NativeDroidController for native action conversion"
            )
        step = observation.extra.get("env_step", 0)
        if type(step) is not int or step < 0:
            raise ValueError("Native policy env_step must be a nonnegative integer")
        instruction = observation.instruction or self._instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Native policy requires a nonempty instruction")
        # Client stages use one monotonic clock. UTC stamps are correlation aids;
        # they must never be subtracted to estimate latency across machines.
        prepare_started = time.perf_counter()
        timing = {
            "preprocessing_started_at": datetime.now(timezone.utc).isoformat(),
            "request_started_at": None,
            "request_finished_at": None,
            "client_elapsed_s": None,
            "preprocessing_s": None,
            "postprocessing_s": None,
            "clock": "perf_counter",
            "relative_time_origin": "policy initialization or most recent reset",
            "client_elapsed_scope": "client.infer including serialization, transport and decoding",
            "preprocessing_scope": "observation adaptation, seed and request audit preparation",
            "postprocessing_scope": "response validation, metadata audit and action conversion",
            "total_elapsed_scope": "preprocessing through action conversion; excludes recorder I/O",
        }
        telemetry = {
            "schema_version": 1,
            "status": "started",
            "timing": timing,
            "request": {
                "model_profile": self.model_profile_id,
                "family": self.model_profile["family"],
                "adapter": self.model_profile["adapter"],
                "h": self.h, "k": self.k, "control_hz": self.control_hz,
            },
            "usage": {
                "availability": "not_applicable",
                "reason": "continuous action output; token usage is not exposed by this protocol",
            },
            "cost": {
                "status": "unavailable", "amount": None, "currency": None,
                "reason": "local compute pricing and resource accounting are not configured",
            },
            "server_timing": {},
            "server_timing_semantics": {
                "units": "*_ms is milliseconds; *_s is seconds; raw provider fields retained",
                "clock": "server-local durations; not GPU kernel time or client clock offsets",
                "openpi_policy_timing.infer_ms": (
                    "upstream action-sampling dispatch before CPU conversion; "
                    "may omit asynchronous device work"
                ),
                "openpi_server_timing.infer_ms": (
                    "upstream policy.infer including transforms and CPU output conversion"
                ),
                "openpi_server_timing.prev_total_ms": (
                    "previous receive-wait/infer/send interval, including client idle time; "
                    "not latency of this request"
                ),
                "molmoact2_dt_ms": "handler body-read through response-serialization start",
            },
        }
        request = {}
        event = {
            "request_id": f"{self.model_profile_id}-{len(self._events)}",
            "kind": "plan",
            "env_step": step,
            "observation_step": step,
            "request_sim_time_s": self.adapter.sim_time(observation, step, self.control_hz),
            "instruction": instruction,
            "h": self.h,
            "k": self.k,
            "telemetry": telemetry,
            "request_wall_s": prepare_started - self._trial_started,
        }
        started = post_started = None
        try:
            on_started = getattr(self.trajectory_recorder, "inference_started", None)
            if callable(on_started):
                # Persist intent before request preparation or any blocking RPC,
                # so an abruptly killed process leaves an unfinished request.
                on_started(
                    request_id=event["request_id"],
                    logical_decision_id=event["request_id"],
                    observation_step=step, kind="native", attempt=1,
                    request_summary=telemetry["request"],
                )
            prepare_started = time.perf_counter()
            timing["preprocessing_started_at"] = datetime.now(timezone.utc).isoformat()
            request = self._request(observation, instruction)
            if self.policy_seed is not None:
                request[SEED_KEY] = sampling_request(self.policy_seed, len(self._events))
            images = {
                key: value for key, value in request.items()
                if isinstance(value, np.ndarray) and value.ndim == 3
            }
            event.update({
                "state": self._request_state(request).tolist(),
                "image_sha256": {
                    key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in images.items()
                },
                "image_shapes": {key: list(value.shape) for key, value in images.items()},
                "sampling": request.get(SEED_KEY),
            })
            telemetry["request"].update({
                "image_count": len(images),
                "image_uncompressed_bytes": sum(value.nbytes for value in images.values()),
                "array_uncompressed_bytes": sum(
                    value.nbytes for value in request.values() if isinstance(value, np.ndarray)
                ),
                "byte_count_scope": "raw ndarray storage, not encoded network payload bytes",
                "payload_keys": sorted(request),
            })
            started = time.perf_counter()
            timing["preprocessing_s"] = started - prepare_started
            timing["request_started_at"] = datetime.now(timezone.utc).isoformat()
            event["request_wall_s"] = started - self._trial_started
            try:
                with request_progress(getattr(self, "runtime_progress", None)):
                    response = self._client.infer(request)
            finally:
                timing["client_elapsed_s"] = time.perf_counter() - started
                timing["request_finished_at"] = datetime.now(timezone.utc).isoformat()
            post_started = time.perf_counter()
            # Preserve returned timing/error metadata even if actions are invalid.
            if isinstance(response, Mapping):
                event["response_metadata"] = _json_safe(
                    {key: value for key, value in response.items() if key != "actions"}
                )
                telemetry["server_timing"] = {
                    key: _json_safe(response[key])
                    for key in ("policy_timing", "server_timing", "dt_ms") if key in response
                }
                telemetry["response"] = {"metadata": event["response_metadata"]}
            if not isinstance(response, Mapping) or "actions" not in response:
                raise ValueError("Native policy server response requires actions")
            if self.policy_seed is not None and response.get(SEED_KEY) != request[SEED_KEY]:
                raise ValueError("Server sampling echo does not match this request")
            raw = _numeric(
                response["actions"], (self.h, self.adapter.action_dim), "native action chunk"
            ).copy()
            event["raw_actions"] = raw.tolist()
            if self.adapter.execution_kind == "libero":
                event["actions"] = raw.tolist()
            chunk = ActionChunk(
                tuple(
                    Action(
                        row.copy(),
                        {
                            "source": self.model_profile_id,
                            "raw_action_semantics": self.raw_action_semantics,
                            "request_id": event["request_id"],
                            "chunk_index": index,
                            **(
                                {"droid_raw_action_semantics": self.raw_action_semantics}
                                if self.adapter.execution_kind == "droid"
                                else {}
                            ),
                        },
                    )
                    for index, row in enumerate(raw)
                ),
                control_hz=self.control_hz,
                inference_latency_s=time.perf_counter() - started,
                meta={
                    "source": self.model_profile_id,
                    "h": self.h,
                    "k": self.k,
                    "profile": self.profile,
                    "raw_action_semantics": self.raw_action_semantics,
                },
            )
            telemetry["status"] = "completed"
            return chunk
        except BaseException as exc:
            telemetry["status"] = (
                "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
            )
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            # HTTP backends may return server-side timing alongside an error.
            # Preserve that audit data without replacing the transport exception.
            http_response = getattr(exc, "response", None)
            if http_response is not None:
                telemetry["http_status"] = getattr(http_response, "status_code", None)
                try:
                    error_body = http_response.json()
                    if isinstance(error_body, Mapping):
                        telemetry["server_timing"] = {
                            key: _json_safe(error_body[key])
                            for key in ("policy_timing", "server_timing", "dt_ms")
                            if key in error_body
                        }
                except Exception:
                    pass
            raise
        finally:
            finished = time.perf_counter()
            if started is None:
                timing["preprocessing_s"] = finished - prepare_started
            if post_started is not None:
                timing["postprocessing_s"] = finished - post_started
            timing["total_elapsed_s"] = finished - prepare_started
            telemetry["transport"] = {
                "send_attempted": started is not None,
                "round_trip_s": timing["client_elapsed_s"],
                "http_status": telemetry.get("http_status"),
                "scope": "native client.infer; includes serialization and decoding; transport may be HTTP or websocket",
            }
            timing["finished_at"] = datetime.now(timezone.utc).isoformat()
            event["elapsed_s"] = finished - (started if started is not None else prepare_started)
            self._events.append(event)
            if self.trajectory_recorder is not None:
                try:
                    self.trajectory_recorder.record_native_inference(
                        event, request=request, observation=observation
                    )
                except Exception as exc:
                    if "error" not in event:
                        raise
                    # A storage error must not replace the original model failure.
                    event["recording_error"] = {"type": type(exc).__name__, "message": str(exc)}

    def native_action(self, raw, observation):
        return self.adapter.native_action(self, raw, observation)

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def record_action(self, t, action, observation):
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record_native_action(t, action, observation)
        self._executed.append(
            {
                "env_step": int(t),
                "data": np.asarray(action.data).tolist(),
                "meta": _json_safe(action.meta),
            }
        )

    def metrics(self):
        intervals = [
            {
                "from_request": a["request_id"],
                "to_request": b["request_id"],
                "control_steps": b["observation_step"] - a["observation_step"],
                "sim_time_s": b["request_sim_time_s"] - a["request_sim_time_s"],
                "wall_time_s": b["request_wall_s"] - a["request_wall_s"],
            }
            for a, b in zip(self._events, self._events[1:])
        ]
        return {
            "model": self.model_profile_id,
            "model_calls": len(self._events),
            "planning_calls": len(self._events),
            "polling_calls": 0,
            "format_repair_calls": 0,
            "format_errors": sum(
                e.get("error", {}).get("type") == "ValueError" for e in self._events
            ),
            "inference_wall_s": sum(e["elapsed_s"] for e in self._events),
            "decision_intervals": intervals,
            "inference_control_steps": [e["observation_step"] for e in self._events],
            "simulation_paused_during_inference": True,
            "usage": {},
            "schedule": {"h": self.h, "k": self.k},
            "executed_actions": len(self._executed),
            "outcomes": {
                "plan": sum("error" not in e for e in self._events),
                "error": sum("error" in e for e in self._events),
            },
        }

    def on_trial_end(self, record, log_dir, run_id):
        record.metadata["agentic"] = self.metrics()

    def transcript(self):
        return {
            "metadata": self.metadata(),
            "messages": copy.deepcopy(self._events),
            "executed_actions": copy.deepcopy(self._executed),
            "metrics": self.metrics(),
        }

    def transcript_delta(self):
        result = copy.deepcopy(self._events[self._delta_cursor :])
        self._delta_cursor = len(self._events)
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self._client, "close"):
                self._client.close()
