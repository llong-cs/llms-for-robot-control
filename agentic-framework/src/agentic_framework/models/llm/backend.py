"""Native tool backends with rollout-scoped conversation ownership."""

from __future__ import annotations

import base64
import copy
import email.utils
import hashlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import warnings
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from agentic_framework.harness.types import ModelReply, ModelRequest
from agentic_framework.models.llm.demo_transport import deduplicate_demonstrations
from agentic_framework.observability.request_telemetry import (
    attach_exception_telemetry,
    new_telemetry,
    observe_http_response,
    observe_response_json,
)

_DISABLED_FEATURES = (
    "shell_tool",
    "view_image",
    "apps",
    "plugins",
    "browser_use",
    "computer_use",
    "image_generation",
    "goals",
    "hooks",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "skill_search",
    "sleep_tool",
    "tool_suggest",
    "code_mode",
    "code_mode_host",
)
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_USAGE_KEYS = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cachedInputTokens": "cached_input_tokens",
    "reasoningOutputTokens": "reasoning_tokens",
    "totalTokens": "total_tokens",
}
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "secret",
        "credential",
        "credentials",
    }
)


class BackendError(RuntimeError):
    """A backend protocol, configuration, or inference failure."""

    category = "backend"

    def __init__(self, message="", *, usage=None, metadata=None):
        super().__init__(message)
        self.usage = dict(usage or {})
        self.metadata = dict(metadata or {})


class BackendConfigurationError(BackendError):
    """A local setup problem found before any model request was sent; never retried.

    For example missing credentials, an invalid local backend configuration, a
    closed backend, concurrent use, an exhausted or invalid test fixture, or a
    failed request-audit write. A provider response is never classified as one:
    every HTTP status is a transport/provider failure (``http_status_error``).
    The only exception is a Codex app-server that shows its local isolation
    failed (a disabled host tool ran), because a resend could repeat the breach.
    """

    category = "configuration"


class BackendFormatError(BackendError):
    """Invalid model output; a clean, fresh request may repair it."""

    category = "format"

    def __init__(self, message, *, response_text="", usage=None, metadata=None):
        super().__init__(message)
        self.response_text = response_text
        self.usage = dict(usage or {})
        self.metadata = dict(metadata or {})


class BackendOutputLimitError(BackendFormatError):
    """Model output exceeded a configured bound; partial output is never decoded."""

    category = "output_limit"


# Failure kinds the policy resends, unchanged, as the next attempt of the same
# decision after a backoff. "format" is repaired with feedback instead, and
# "configuration" and "cancelled" end the trial at once.
RESENT_FAILURE_KINDS = frozenset({"transport", "internal"})


def failure_kind(exc: BaseException) -> str:
    """Classify a failed model call for the policy's per-decision attempt budget.

    - ``cancelled``: KeyboardInterrupt, SystemExit, any other BaseException that
      is not an Exception, or a CancelledError. Never retried.
    - ``configuration``: BackendConfigurationError. Never retried.
    - ``format``: BackendFormatError (invalid model output), repaired with feedback.
    - ``transport``: a transport or provider failure, i.e. every non-200 HTTP
      status (400, 401, 403, 404, 413, 422, 429, 5xx, ...), a timeout, a
      connection, proxy or stream error, a provider refusal, an incomplete or
      output-token-limited response, a malformed envelope or app-server message,
      or any other BackendError.
    - ``internal``: any other Exception raised while building or sending the
      request or parsing its response, e.g. a framework bug.

    The policy resends ``transport`` and ``internal`` failures (see
    RESENT_FAILURE_KINDS); the provider may already have billed the failed request.
    """
    if not isinstance(exc, Exception) or type(exc).__name__ == "CancelledError":
        return "cancelled"
    if isinstance(exc, BackendConfigurationError):
        return "configuration"
    if isinstance(exc, BackendFormatError):
        return "format"
    if isinstance(exc, (BackendError, TimeoutError, ConnectionError, EOFError)):
        return "transport"
    # An httpx exception can only exist once httpx has been imported. A local
    # protocol violation or an unsupported URL scheme is our own request's fault.
    httpx = sys.modules.get("httpx")
    if (
        httpx is not None
        and isinstance(exc, (httpx.TransportError, httpx.DecodingError))
        and not isinstance(exc, (httpx.LocalProtocolError, httpx.UnsupportedProtocol))
    ):
        return "transport"
    return "internal"


# Statuses whose Retry-After header the policy honours before resending.
RETRY_AFTER_STATUSES = frozenset({429, 503})


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Return the wait a provider requested (retry-after-ms or Retry-After), or None."""
    lowered = {str(key).lower(): value for key, value in headers.items()}
    candidates = []
    if "retry-after-ms" in lowered:
        candidates.append((lowered["retry-after-ms"], 0.001))
    if "retry-after" in lowered:
        candidates.append((lowered["retry-after"], 1.0))
    for value, unit in candidates:
        text = str(value).strip()
        try:
            seconds = float(text) * unit
        except ValueError:
            if unit != 1.0:
                continue
            try:
                when = email.utils.parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return None


def http_status_error(endpoint: str, response: Any) -> BackendError:
    """Return the transport/provider failure for a non-200 HTTP response.

    Every status is resent by the policy after a backoff, including a
    deterministic rejection (400, 401, 403, 404, 413, 422, ...): a misconfigured
    provider (wrong key, model id or parameters) therefore fails every attempt,
    the trial is discarded after max_attempts, and a run whose every trial is
    discarded fails. Retry-After on 429 and 503 is recorded as retry_after_s.
    The response body is never included, because it may echo credentials.
    """
    status = response.status_code
    error = BackendError(f"{endpoint} endpoint rejected request with HTTP {status}")
    if status in RETRY_AFTER_STATUSES:
        delay = retry_after_seconds(response.headers)
        if delay is not None:
            error.metadata["retry_after_s"] = delay
    return error


def _validate_response_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError("max_response_chars must be a positive integer")
    return value


def _reject_json_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def image_data_url(value: np.ndarray) -> str:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Model images must be HWC uint8 RGB arrays")
    data = io.BytesIO()
    Image.fromarray(image).save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def _message_content(request: ModelRequest) -> list[dict]:
    """Encode local ordered messages once for both transports."""
    if not request.messages:
        # Compatibility for callers constructing a plain-text ModelRequest.
        parts = [{"type": "text", "text": request.text}]
        parts.extend({"type": "image", "name": name} for name in request.images)
        messages = ({"role": "user", "content": parts},)
    else:
        messages = request.messages
    encoded = []
    seen_images = set()
    for message in messages:
        if message["role"] not in ("user", "assistant"):
            raise ValueError("Context messages must have user or assistant roles")
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image":
                name = part["name"]
                if message["role"] != "user" or name in seen_images:
                    raise ValueError("Context images must appear once in user observations")
                seen_images.add(name)
                content.extend(
                    [
                        {"type": "input_text", "text": f"Camera: {name}"},
                        {"type": "input_image", "image_url": image_data_url(request.images[name])},
                    ]
                )
            else:
                raise ValueError("Unsupported context content type")
        encoded.append({"role": message["role"], "content": content})
    if seen_images != set(request.images):
        raise ValueError("Every request image must appear in the ordered context")
    return encoded


def _redacted(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): "[redacted]" if str(k).lower() in _SECRET_KEYS else _redacted(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redacted(v) for v in value]
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return "[image sha256=" + hashlib.sha256(value.encode()).hexdigest() + "]"
        value = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [redacted]", value)
        return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[redacted]", value)
    return value


def _terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def scripted_reply(request, payload):
    """Adapt deterministic internal fixtures into native replies, never used by a model backend."""
    if "tool_calls" in payload:
        calls = copy.deepcopy(payload["tool_calls"])
    else:
        if payload.get("decision") in ("done", "give_up"):
            name = payload["decision"]
            key = "summary" if name == "done" else "reason"
            terminal = {
                "name": name,
                "arguments": {
                    key: payload.get(key, ""),
                    "hindsight": payload.get("hindsight", "none"),
                },
            }
            payload = {"commands": [terminal, *payload.get("commands", [])]}
        entries = payload.get("calls", payload.get("commands", [payload]))
        calls = []
        for i, entry in enumerate(entries):
            name = entry.get("tool_name", entry.get("name", "invalid"))
            arguments = entry.get("arguments_json")
            if arguments is None:
                value = entry.get("arguments", {})
                arguments = value if isinstance(value, str) else json.dumps(value)
            calls.append(
                {
                    "type": "function_call",
                    "call_id": f"{request.request_id}-call-{i}",
                    "name": name,
                    "arguments": arguments,
                }
            )
        schemas = {tool["name"]: tool for tool in request.tools}
        if "move_by_chunk" in schemas and calls and all(c["name"] == "move_by" for c in calls):
            try:
                steps = [json.loads(c["arguments"]) for c in calls]
            except (ValueError, TypeError):
                pass
            else:
                if all(
                    isinstance(step, dict)
                    and set(step) <= {"deltas", "note"}
                    and isinstance(step.get("deltas"), dict)
                    and ("note" not in step or isinstance(step["note"], str))
                    for step in steps
                ):
                    dimensions = dict.fromkeys(name for step in steps for name in step["deltas"])
                    arguments = {
                        "deltas": {
                            name: [step["deltas"].get(name, None if name == "gripper" else 0)
                                   for step in steps]
                            for name in dimensions
                        }
                    }
                    # A missing gripper item remains omission, including for
                    # absolute grippers where an invented zero would open them.
                    if any("note" in step for step in steps):
                        arguments["note"] = " ".join(dict.fromkeys(
                            step["note"].strip() for step in steps if "note" in step
                        ))
                    calls = [{**calls[0], "name": "move_by_chunk",
                              "arguments": json.dumps(arguments)}]
    return ModelReply(
        {"tool_calls": calls},
        metadata={"backend": "scripted", "model": "scripted"},
        tool_calls=tuple(calls),
        output_items=tuple(copy.deepcopy(calls)),
    )


class ScriptedBackend:
    """Deterministic local backend for controller and context contract tests."""

    def __init__(self, responses: Iterable[Any]):
        self._responses = iter(responses)
        self.requests: list[ModelRequest] = []
        self.closed = False

    def generate(self, request: ModelRequest) -> ModelReply:
        if self.closed:
            raise BackendConfigurationError("Scripted backend is closed")
        self.requests.append(copy.deepcopy(request))
        try:
            response = next(self._responses)
        except StopIteration as exc:
            # A too-short script is a fixture error, never a failure to resend.
            raise BackendConfigurationError("Scripted responses exhausted") from exc
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(request)
        if isinstance(response, ModelReply):
            if response.tool_calls or response.output_items:
                return copy.deepcopy(response)
            from dataclasses import replace

            native = scripted_reply(request, response.payload)
            return replace(
                native,
                usage=response.usage,
                metadata=response.metadata,
                elapsed_s=response.elapsed_s,
            )
        if not isinstance(response, Mapping):
            raise BackendConfigurationError("Scripted responses must be ModelReply or mappings")
        return scripted_reply(request, response)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ResponsesBackend:
    """Native Responses tools with a window of complete decision exchanges.

    The official OpenAI endpoint accepts an optional ``openai/`` model prefix;
    requests use the bare model ID. Other endpoints retain their model IDs.
    HTTP is stateless (store=False); this object owns rollout history. The window
    counts decisions, including their format repairs, rather than HTTP requests.
    Authentication is resolved only at send time, from the process environment
    and an optional private dotenv file. Each generate() sends exactly
    one HTTP request and never retries it. History is committed only after a
    completed, well-formed response, so a failed request leaves no trace in it:
    the policy may resend the identical request as the next attempt. Every
    non-200 status, including a deterministic rejection (400, 401, 403, 404,
    ...), is a transport/provider failure (``http_status_error``) that the
    policy resends after a backoff.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str | None = "OPENAI_API_KEY",
        env_file: str | Path | None = None,
        timeout_s: float = 240,
        max_response_chars: int = 32768,
        max_output_tokens: int | None = None,
        supported_efforts: tuple[str, ...] | None = None,
        default_effort: str | None = None,
        reasoning_mode: str | None = None,
        log_dir: str | Path | None = None,
        transport: Any = None,
        request_recorder: Callable[[ModelRequest, Any], dict | None] | None = None,
    ):
        import httpx

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        endpoint = httpx.URL(base_url.rstrip("/") + "/")
        if endpoint == httpx.URL("https://api.openai.com/v1/") and model.startswith("openai/"):
            model = model.removeprefix("openai/")
            if not model or "/" in model or any(c.isspace() for c in model):
                raise ValueError("OpenAI model prefix must be followed by one nonempty model ID")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if default_effort is not None and supported_efforts is not None:
            if default_effort not in supported_efforts:
                raise ValueError("default_effort is not in supported_efforts")
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        if reasoning_mode is not None and reasoning_mode not in ("standard", "pro"):
            raise ValueError("reasoning_mode must be standard, pro, or None")
        if request_recorder is not None and not callable(request_recorder):
            raise ValueError("request_recorder must be callable or None")
        self.model = model
        self.api_key_env = api_key_env
        self.credential_env_file = Path(env_file).expanduser() if env_file is not None else None
        self.max_output_tokens = max_output_tokens
        self.request_recorder = request_recorder
        self.max_response_chars = _validate_response_limit(max_response_chars)
        self.supported_efforts = supported_efforts
        self.default_effort = default_effort
        self.reasoning_mode = reasoning_mode
        self.log_dir = Path(log_dir).resolve() if log_dir is not None else None
        self._closed = False
        self._history: list[dict] = []
        self._client = httpx.Client(
            base_url=endpoint,
            timeout=timeout_s,
            transport=transport,
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": "responses",
            "input_modalities": ["text", "image"],
            "supported_reasoning_efforts": list(self.supported_efforts)
            if self.supported_efforts is not None
            else None,
            "default_reasoning_effort": self.default_effort,
            "configured_reasoning_mode": self.reasoning_mode,
            "native_tool_calling": True,
            "context_lifetime": "rollout_window",
            "history_unit": "complete_decision_turn",
            "history_length_semantics": "window_size",
            "encrypted_reasoning_replay": True,
            "reasoning_replay": "selected_decision_records_and_current_repair",
            "history_record_format": "raw_provider_exchange",
            "previous_decision_reasoning_replay": True,
            "zero_history_native_exchange_replay": False,
            "capability_source": "configuration",
        }

    def reset(self) -> None:
        """Start a new rollout; neither replay items nor pending results survive."""
        self._history.clear()

    def _effort(self, asked: str) -> str | None:
        if asked == "default":
            return self.default_effort
        if asked in {"lowest", "lowest_supported"}:
            if not self.supported_efforts:
                raise BackendConfigurationError(
                    "lowest effort requires declared supported_efforts for an HTTP backend"
                )
            value = next((item for item in _EFFORT_ORDER if item in self.supported_efforts), None)
            if value is None:
                raise BackendConfigurationError(
                    "No known effort ordering for declared supported_efforts"
                )
            return value
        if self.supported_efforts is not None and asked not in self.supported_efforts:
            raise BackendConfigurationError(
                f"Reasoning effort {asked!r} is unsupported by the backend configuration"
            )
        return asked

    @staticmethod
    def _append_results(history: list[dict], results) -> None:
        """Attach results to their originating decision, never as orphan messages."""
        if not results:
            return
        if not history:
            raise BackendConfigurationError("Tool results have no originating model call")
        items = history[-1]["items"]
        calls = {item["call_id"] for item in items if item.get("type") == "function_call"}
        answered = {item["call_id"] for item in items if item.get("type") == "function_call_output"}
        pending = calls - answered
        received = set()
        for result in results:
            if (
                not isinstance(result, Mapping)
                or result.get("type") != "function_call_output"
                or not isinstance(result.get("call_id"), str)
                or not isinstance(result.get("output"), str)
            ):
                raise BackendConfigurationError("Tool results require call_id and string output")
            call_id = result["call_id"]
            if call_id not in pending or call_id in received:
                raise BackendConfigurationError("Unknown or duplicate tool result call_id")
            received.add(call_id)
        if received != pending:
            raise BackendConfigurationError("Every pending tool call must receive a result")
        items.extend(copy.deepcopy(dict(item)) for item in results)

    def record_tool_results(self, results) -> None:
        """Archive terminal execution results without another model request."""
        history = copy.deepcopy(self._history)
        self._append_results(history, results)
        self._history = history

    def _prepared_history(self, request: ModelRequest) -> tuple[list[dict], list[dict]]:
        if type(request.history_length) is not int:
            raise BackendConfigurationError("history_length must be an integer")
        window = max(0, request.history_length)
        history = copy.deepcopy(self._history)
        self._append_results(history, request.tool_results)
        if history:
            last = history[-1]["items"]
            calls = {i["call_id"] for i in last if i.get("type") == "function_call"}
            results = {i["call_id"] for i in last if i.get("type") == "function_call_output"}
            if calls != results:
                raise BackendConfigurationError("Pending tool calls require execution results")
        decision_id = request.decision_id or request.request_id
        repairing = bool(history and history[-1]["decision_id"] == decision_id)
        previous = history[:-1] if repairing else history
        selected = previous[-window:] if window else []
        # The window bounds complete records, not the information encoded in
        # their reasoning. Preserve selected native exchanges without changes.
        messages = [copy.deepcopy(item) for group in selected for item in group["items"]]
        if repairing:
            current_group = history[-1]
            messages.extend(copy.deepcopy(current_group["items"]))
        current = _message_content(request)
        messages.extend(current)
        if repairing:
            current_group["items"].extend(copy.deepcopy(current))
        else:
            current_group = {"decision_id": decision_id, "items": copy.deepcopy(current)}
        return [*selected, current_group], deduplicate_demonstrations(messages)

    def accept_reply(self, request: ModelRequest, reply: ModelReply) -> None:
        """Commit an offline scripted reply through the same history boundary.

        Used only by request previews: this method performs no HTTP or auth I/O.
        It requires raw native output, never fabricates encrypted reasoning.
        """
        history, _ = self._prepared_history(request)
        output = copy.deepcopy(list(reply.output_items or reply.tool_calls))
        calls, _ = _native_response_calls(output)
        _reject_reused_call_ids(history[-1]["items"], calls)
        history[-1]["items"].extend(output)
        self._history = history

    def prepare_request(self, request: ModelRequest) -> dict:
        """Preview actual native input without credentials, I/O or state changes."""
        effort = self._effort(request.reasoning_effort)
        _, messages = self._prepared_history(request)
        if not request.tools:
            raise BackendConfigurationError("Native robot decisions require function tools")
        body = {
            "model": self.model,
            "instructions": request.instructions,
            "input": messages,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "tools": copy.deepcopy(list(request.tools)),
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        if effort is not None:
            body["reasoning"] = {"effort": effort}
        if self.reasoning_mode is not None:
            body.setdefault("reasoning", {})["mode"] = self.reasoning_mode
        if self.max_output_tokens is not None:
            body["max_output_tokens"] = self.max_output_tokens
        return body

    def _build_http_request(self, body: dict):
        return self._client.build_request("POST", "responses", json=body)

    def prepare_http_request(self, request: ModelRequest):
        prepared = self._build_http_request(self.prepare_request(request))
        for name in ("authorization", "proxy-authorization", "api-key", "x-api-key"):
            prepared.headers.pop(name, None)
        return prepared

    def _before_send(self) -> None:
        self._client.headers.pop("Authorization", None)
        if self.api_key_env is None:
            return
        values = {}
        if self.credential_env_file is not None:
            from agentic_framework.models.llm.credentials import read_credentials_file

            try:
                values = read_credentials_file(self.credential_env_file)
            except (OSError, ValueError, TypeError):
                raise BackendConfigurationError(
                    "Unable to load Responses credentials; check llm.env_file, "
                    "file ownership and mode 600"
                ) from None
        if self.api_key_env == "OPENAI_API_KEY":
            configured_url = os.environ.get("OPENAI_BASE_URL") or values.get("OPENAI_BASE_URL")
            if configured_url:
                import httpx

                try:
                    configured_url = httpx.URL(configured_url.rstrip("/") + "/")
                except (ValueError, httpx.InvalidURL):
                    raise BackendConfigurationError(
                        "OPENAI_BASE_URL is invalid; check the credential settings"
                    ) from None
                if configured_url != self._client.base_url:
                    raise BackendConfigurationError(
                        "OPENAI_BASE_URL disagrees with the reviewed Responses endpoint; "
                        "set llm.base_url to the intended address"
                    )
        key = os.environ.get(self.api_key_env) or values.get(self.api_key_env) or ""
        if not key or not key.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in key):
            raise BackendConfigurationError(
                "Configured API key environment variable is missing or malformed in "
                "the process environment or credential file"
            )
        self._client.headers["Authorization"] = f"Bearer {key}"

    def _unwrap_response(self, data: Any) -> Any:
        return data

    def _backend_metadata(self) -> dict[str, Any]:
        return {}

    def _send_with_telemetry(self, http_request, metadata):
        telemetry = metadata["telemetry"]
        transport = telemetry["transport"]
        transport.update(request_bytes=len(http_request.content), outcome="in_flight", send_attempted=True,
                         started_at_unix_s=time.time())
        started = time.perf_counter()
        try:
            response = self._client.send(http_request)
        finally:
            transport["round_trip_s"] = time.perf_counter() - started
            transport["ended_at_unix_s"] = time.time()
            metadata["http_latency_s"] = transport["round_trip_s"]
        observe_http_response(telemetry, response)
        metadata["http_status"] = response.status_code
        metadata["request_body_bytes"] = transport["request_bytes"]
        metadata["response_body_bytes"] = transport["response_bytes"]
        ids = {name: transport["response_headers"][name]
               for name in ("x-request-id", "request-id", "apim-request-id", "x-ms-request-id")
               if name in transport["response_headers"]}
        if ids:
            metadata["request_ids"] = ids
        if response.status_code != 200:
            # Failed HTTP requests can still report usage/cost. Never log their body.
            try:
                observe_response_json(telemetry, response.json())
            except (ValueError, TypeError):
                pass
        return response

    @staticmethod
    def _response_usage(data: dict) -> dict[str, int]:
        raw = data.get("usage") or {}
        if not isinstance(raw, Mapping):
            raise BackendError("Responses usage envelope must be an object")
        try:
            usage = {
                key: int(raw[key])
                for key in ("input_tokens", "output_tokens", "total_tokens")
                if key in raw
            }
            for source, source_key, target in (
                ("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ):
                details = raw.get(source) or {}
                if not isinstance(details, Mapping):
                    raise TypeError("token details must be an object")
                if source_key in details:
                    usage[target] = int(details[source_key])
        except (ValueError, TypeError, OverflowError) as exc:
            raise BackendError("Responses usage envelope contains invalid token counts") from exc
        return usage

    def generate(self, request: ModelRequest) -> ModelReply:
        if self._closed:
            raise BackendConfigurationError("Responses backend is closed")
        started = time.monotonic()
        usage: dict[str, int] = {}
        metadata: dict[str, Any] = {
            "backend": "responses",
            "model": self.model,
            "requested_model": self.model,
            "requested_effort": request.reasoning_effort,
            "requested_reasoning_mode": self.reasoning_mode,
            "reasoning_mode_verified": False,
            "context_lifetime": "rollout_window",
            "history_length": request.history_length,
            "history_length_semantics": "window_size",
            "history_unit": "complete_decision_turn",
            "reasoning_replay": "selected_decision_records_and_current_repair",
            "history_record_format": "raw_provider_exchange",
            "native_tool_calling": True,
            "max_response_chars": self.max_response_chars,
            "max_output_tokens": self.max_output_tokens,
            **self._backend_metadata(),
            "telemetry": new_telemetry("responses"),
        }
        row = {
            "request_id": request.request_id,
            "model": self.model,
            "requested_effort": request.reasoning_effort,
            "requested_reasoning_mode": self.reasoning_mode,
            "status": "started",
        }
        try:
            body = self.prepare_request(request)
            next_history, _ = self._prepared_history(request)
            metadata.update(
                effective_history_length=max(0, request.history_length),
                previous_decision_reasoning_replay=request.history_length > 0,
                previous_decision_ids=[group["decision_id"] for group in next_history[:-1]],
                previous_decision_count=len(next_history) - 1,
            )
            metadata["reasoning_effort"] = body.get("reasoning", {}).get("effort")
            metadata["reasoning_mode"] = body.get("reasoning", {}).get("mode")
            self._before_send()
            http_request = self._build_http_request(body)
            metadata["telemetry"]["transport"]["request_bytes"] = len(http_request.content)
            if self.request_recorder is not None:
                try:
                    artifact = self.request_recorder(request, http_request)
                    if artifact is not None:
                        if not isinstance(artifact, dict):
                            raise TypeError("request recorder must return a dict or None")
                        json.dumps(artifact, allow_nan=False)
                        metadata["request_artifact"] = copy.deepcopy(artifact)
                except Exception:
                    raise BackendConfigurationError(
                        "Request recording failed; no HTTP request was sent"
                    ) from None
            response = self._send_with_telemetry(http_request, metadata)
            if response.status_code != 200:
                raise http_status_error("Responses", response)
            try:
                raw_response = response.json()
                observe_response_json(metadata["telemetry"], raw_response)
                data = self._unwrap_response(raw_response)
            except (ValueError, TypeError) as exc:
                raise BackendError(
                    "Responses endpoint returned a malformed HTTP JSON envelope"
                ) from exc
            if not isinstance(data, dict):
                raise BackendError("Responses endpoint returned an invalid HTTP response envelope")
            usage = self._response_usage(data)
            metadata.update(
                model=data.get("model", self.model),
                # Keep provider evidence separate from the compatibility fallback above.
                response_model=data.get("model"),
                response_id=data.get("id"),
                provider_response_id=data.get("id"),
                response_status=data.get("status", "completed"),
            )
            response_reasoning = data.get("reasoning")
            if isinstance(response_reasoning, Mapping) and "mode" in response_reasoning:
                returned_mode = response_reasoning["mode"]
                metadata["response_reasoning_mode"] = returned_mode
                if self.reasoning_mode is not None:
                    if returned_mode != self.reasoning_mode:
                        # A provider response, resent like any other provider failure.
                        raise BackendError(
                            "Responses endpoint returned a different reasoning mode than requested"
                        )
                    metadata["reasoning_mode_verified"] = True
            incomplete = data.get("incomplete_details")
            if isinstance(incomplete, dict) and "reason" in incomplete:
                metadata["incomplete_details"] = {"reason": incomplete["reason"]}
            if data.get("status", "completed") != "completed":
                # Partial calls cannot be executed or safely replayed. No state
                # is committed on incomplete, transport, auth or envelope errors.
                raise BackendError(f"Responses request ended with status {data.get('status')!r}")
            output = data.get("output")
            if not isinstance(output, list) or any(not isinstance(i, dict) for i in output):
                raise BackendError("Responses endpoint returned an invalid output array")
            calls, output_text = _native_response_calls(output)
            _reject_reused_call_ids(next_history[-1]["items"], calls)
            # Completed raw output is committed before decision-format checking,
            # so bounded repair calls retain exactly what the model produced.
            next_history[-1]["items"].extend(copy.deepcopy(output))
            self._history = next_history
            metadata["tool_calls"] = copy.deepcopy(calls)
            output_chars = len(output_text) + sum(len(c["arguments"]) for c in calls)
            if output_chars > self.max_response_chars:
                raise BackendOutputLimitError(
                    f"Model response exceeded max_response_chars={self.max_response_chars}; "
                    "use compact function arguments",
                    usage=usage,
                    metadata={**metadata, "output_chars_observed": output_chars},
                )
            if len(calls) != 1 or calls[0]["name"] not in {t["name"] for t in request.tools}:
                raise BackendFormatError(
                    "Call exactly one of the provided robot function tools",
                    response_text=output_text,
                    usage=usage,
                    metadata=metadata,
                )
            elapsed = time.monotonic() - started
            payload = {"tool_calls": copy.deepcopy(calls)}
            row.update(
                status="completed",
                payload=payload,
                output_items=output,
                usage=usage,
                metadata=metadata,
                elapsed_s=elapsed,
            )
            return ModelReply(
                payload=payload,
                usage=usage,
                metadata=metadata,
                elapsed_s=elapsed,
                tool_calls=tuple(copy.deepcopy(calls)),
                output_items=tuple(copy.deepcopy(output)),
                response_text=output_text,
            )
        except BackendFormatError as exc:
            attach_exception_telemetry(exc, metadata, usage)
            exc.usage = {**usage, **exc.usage}
            exc.metadata = {**metadata, **exc.metadata}
            row.update(
                status=exc.category if isinstance(exc, BackendOutputLimitError) else "format_error",
                error_type=type(exc).__name__,
                response_text=exc.response_text,
                usage=exc.usage,
                metadata=exc.metadata,
                elapsed_s=time.monotonic() - started,
            )
            raise
        except BaseException as exc:
            attach_exception_telemetry(exc, metadata, usage)
            if isinstance(exc, BackendError):
                exc.usage = {**usage, **exc.usage}
                exc.metadata = {**metadata, **exc.metadata}
            row.update(
                status="error",
                error_type=type(exc).__name__,
                usage=usage,
                metadata=metadata,
                elapsed_s=time.monotonic() - started,
            )
            raise
        finally:
            if self.log_dir is not None:
                try:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    with (self.log_dir / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(_redacted(row), allow_nan=False) + "\n")
                except (OSError, ValueError, TypeError) as exc:
                    warnings.warn(
                        f"Backend audit logging degraded: {type(exc).__name__}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _reject_reused_call_ids(group_items: list[dict], calls: list[dict]) -> None:
    """Reject a call ID that an earlier attempt of the same decision already returned.

    Committing it would make the next tool result ambiguous. It is a provider
    failure raised before anything is committed, so the request is resent.
    """
    earlier = {item.get("call_id") for item in group_items if item.get("type") == "function_call"}
    if any(call["call_id"] in earlier for call in calls):
        raise BackendError("Model reused a function call ID from an earlier attempt of this decision")


def _native_response_calls(output: list[dict]) -> tuple[list[dict], str]:
    """Validate the wire envelope, keeping its original items untouched for replay."""
    calls, text_parts, seen = [], [], set()
    for item in output:
        if item.get("type") == "function_call":
            if any(not isinstance(item.get(k), str) or not item[k] for k in ("call_id", "name")):
                raise BackendError("Responses function_call requires call_id and name")
            if not isinstance(item.get("arguments"), str) or item["call_id"] in seen:
                raise BackendError("Responses function_call has invalid arguments or duplicate ID")
            if item.get("status", "completed") != "completed":
                raise BackendError("Responses returned an incomplete function call")
            seen.add(item["call_id"])
            calls.append(
                {
                    "type": "function_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
        elif item.get("type") == "message":
            if not isinstance(item.get("content"), list):
                raise BackendError("Responses message envelope lacks a content array")
            for part in item["content"]:
                if not isinstance(part, dict):
                    raise BackendError("Responses message content must contain objects")
                if part.get("type") == "refusal":
                    raise BackendError("Model refused the robot decision request")
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise BackendError("Responses output_text must contain string text")
                    text_parts.append(part["text"])
    return calls, "".join(text_parts)


def __getattr__(name):
    # Keep the public import while allowing codex_backend to reuse transport helpers.
    if name == "CodexBackend":
        from agentic_framework.models.llm.codex_backend import CodexBackend

        return CodexBackend
    raise AttributeError(name)
