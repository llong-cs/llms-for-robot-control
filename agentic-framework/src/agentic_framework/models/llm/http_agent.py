"""The shared HTTP agent: one generate() template for every wire protocol.

A wire subclass (ResponsesAgent, ...) supplies only its protocol hooks: request encoding,
history replay, tool-result attachment, response decoding and usage. This class owns the
resolved provider route, the endpoint and key read from the environment at send time,
request archiving, transport telemetry, the decisions.jsonl audit log and the failure
classification of one call.
"""

from __future__ import annotations

import copy
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

from agentic_framework.common.redaction import redacted
from agentic_framework.harness.types import AgentContext, AgentOutput
from agentic_framework.models.llm.errors import (
    BackendConfigurationError,
    BackendError,
    BackendFormatError,
    BackendOutputLimitError,
    http_status_error,
)
from agentic_framework.models.llm.providers import Route, resolve
from agentic_framework.observability.request_telemetry import (
    attach_exception_telemetry,
    new_telemetry,
    observe_http_response,
    observe_response_json,
)

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
# Headers a request preview never carries.
AUTH_HEADERS = ("authorization", "proxy-authorization", "api-key", "x-api-key")
# A request preview, built without reading the environment, targets this origin when the
# provider reads its endpoint from an environment variable.
PLACEHOLDER_ORIGIN = "https://endpoint.invalid"


class HttpAgentModel:
    """An AgentModel over one HTTP endpoint, with a window of complete decision exchanges.

    The model selector (or its resolved Route) names the provider, whose declaration
    supplies the endpoint, the key variable and the wire model ID. HTTP is stateless; this
    object owns rollout history as ``{decision_id, items}`` groups, and the window counts
    decisions, including their format repairs, rather than HTTP requests. The key, and an
    endpoint the provider reads from the environment, are resolved only at send time.
    Each generate() sends exactly one HTTP request and never retries it. History is
    committed only after a completed, well-formed response, so a failed request leaves no
    trace in it and the policy may resend the identical request as its next attempt. Every
    non-200 status, including a deterministic rejection (400, 401, 403, 404, ...), is a
    transport/provider failure (``http_status_error``) that the policy resends after a
    backoff.
    """

    # Wire identity: recorded label and telemetry protocol, error-message name, request path.
    protocol = ""
    label = ""
    path = ""
    reasoning_replay = ""

    def __init__(
        self,
        model: str | Route,
        *,
        timeout_s: float = 300,
        max_response_chars: int = 32768,
        max_output_tokens: int | None = None,
        supported_efforts: tuple[str, ...] | None = None,
        default_effort: str | None = None,
        reasoning_mode: str | None = None,
        log_dir: str | Path | None = None,
        transport: Any = None,
        request_recorder: Callable[[AgentContext, Any], dict | None] | None = None,
    ):
        import httpx

        route = model if isinstance(model, Route) else resolve(model)
        # An environment endpoint is read at send time; until then requests name a placeholder.
        base = route.spec.base_url or PLACEHOLDER_ORIGIN + route.spec.base_url_path
        self.model = route.wire_model
        # Names and labels only; the provider declaration stays private to this object.
        self.route = route.record()
        self._provider = route.spec
        self._capabilities = route.capabilities
        # Added to every decision record: names and labels, never values.
        self._route_metadata = {
            "record_version": 2,
            "protocol": self.protocol,
            "provider": route.provider,
            "model_selector": route.selector,
            "wire_model": route.wire_model,
            "endpoint": route.endpoint,
            "api_key_env": route.api_key_env,
        }
        self.max_output_tokens = max_output_tokens
        self.request_recorder = request_recorder
        self.max_response_chars = max_response_chars
        self.supported_efforts = supported_efforts
        self.default_effort = default_effort
        self.reasoning_mode = reasoning_mode
        self.log_dir = Path(log_dir).resolve() if log_dir is not None else None
        self._closed = False
        self._history: list[dict] = []
        self._client = httpx.Client(
            base_url=httpx.URL(base.rstrip("/") + "/"),
            timeout=timeout_s,
            transport=transport,
        )

    # ------------------------------------------------------------------ wire hooks

    def prepare_request(self, context: AgentContext) -> dict:
        """Return the request body for context, without credentials, I/O or state changes."""
        raise NotImplementedError

    def _encode_context(self, context: AgentContext) -> list[dict]:
        """Encode the context's ordered messages as wire input items."""
        raise NotImplementedError

    def _assemble(self, items: list[dict]) -> list[dict]:
        """Turn replayed and current items into the request's input."""
        return items

    def _append_results(self, history: list[dict], results) -> None:
        """Attach tool results to the latest decision group, or raise a configuration error."""
        raise NotImplementedError

    def _answered(self, items: list[dict]) -> bool:
        """Whether every call in a decision group has its tool result."""
        raise NotImplementedError

    def _call_ids(self, items: list[dict]) -> set[str]:
        """Call IDs a decision group already holds."""
        raise NotImplementedError

    def _request_settings(self, body: dict) -> dict[str, Any]:
        """Effective request settings recorded before sending."""
        return {}

    def _history_settings(self, history: list[dict]) -> dict[str, Any]:
        """Replay facts of the history this request would commit, recorded before sending."""
        return {}

    def _usage(self, data: dict) -> dict[str, int]:
        """Token usage of a decoded response."""
        raise NotImplementedError

    def _decode(self, data: dict, metadata: dict) -> tuple[list[dict], list[dict], str]:
        """Validate a response; return (raw output items, normalized calls, visible text).

        Raising BackendError rejects the response before anything is committed.
        """
        raise NotImplementedError

    def _output_calls(self, output: list[dict]) -> list[dict]:
        """Normalized calls of raw output items (used to commit an offline preview reply)."""
        raise NotImplementedError

    # --------------------------------------------------------------- shared behavior

    def reset(self) -> None:
        """Start a new rollout; neither replay items nor pending results survive."""
        self._history.clear()

    def _effort(self, asked: str) -> str | None:
        if asked == "default":
            return self.default_effort
        if asked in {"lowest", "lowest_supported"}:
            if not self.supported_efforts:
                raise BackendConfigurationError("lowest effort requires llm.supported_efforts")
            value = next((item for item in EFFORT_ORDER if item in self.supported_efforts), None)
            if value is None:
                raise BackendConfigurationError(
                    "No known effort ordering for declared supported_efforts"
                )
            return value
        if self.supported_efforts is not None and asked not in self.supported_efforts:
            raise BackendConfigurationError(
                f"Reasoning effort {asked!r} is not among llm.supported_efforts"
            )
        return asked

    def record_tool_results(self, results) -> None:
        """Archive terminal execution results without another model request."""
        history = copy.deepcopy(self._history)
        self._append_results(history, results)
        self._history = history

    def _prepared_history(self, context: AgentContext) -> tuple[list[dict], list[dict]]:
        """Return (history after this request commits, the request's input items)."""
        window = max(0, context.history_length)
        history = copy.deepcopy(self._history)
        self._append_results(history, context.tool_results)
        if history and not self._answered(history[-1]["items"]):
            raise BackendConfigurationError("Pending tool calls require execution results")
        decision_id = context.decision_id or context.request_id
        repairing = bool(history and history[-1]["decision_id"] == decision_id)
        previous = history[:-1] if repairing else history
        selected = previous[-window:] if window else []
        # The window bounds complete records, not the information encoded in
        # their reasoning. Preserve selected native exchanges without changes.
        items = [copy.deepcopy(item) for group in selected for item in group["items"]]
        if repairing:
            current_group = history[-1]
            items.extend(copy.deepcopy(current_group["items"]))
        current = self._encode_context(context)
        items.extend(current)
        if repairing:
            current_group["items"].extend(copy.deepcopy(current))
        else:
            current_group = {"decision_id": decision_id, "items": copy.deepcopy(current)}
        return [*selected, current_group], self._assemble(items)

    def _commit(self, history: list[dict], output: list[dict]) -> None:
        history[-1]["items"].extend(copy.deepcopy(output))
        self._history = history

    def _reject_reused_call_ids(self, group_items: list[dict], calls: list[dict]) -> None:
        """Reject a call ID that an earlier attempt of the same decision already returned.

        Committing it would make the next tool result ambiguous. It is a provider
        failure raised before anything is committed, so the request is resent.
        """
        earlier = self._call_ids(group_items)
        if any(call["call_id"] in earlier for call in calls):
            raise BackendError(
                "Model reused a function call ID from an earlier attempt of this decision"
            )

    def accept_reply(self, context: AgentContext, reply: AgentOutput) -> None:
        """Commit an offline scripted reply through the same history boundary.

        Used only by request previews: this method performs no HTTP or auth I/O.
        It requires raw native output, never fabricates encrypted reasoning.
        """
        history, _ = self._prepared_history(context)
        output = copy.deepcopy(list(reply.output_items or reply.tool_calls))
        self._reject_reused_call_ids(history[-1]["items"], self._output_calls(output))
        self._commit(history, output)

    def _build_http_request(self, body: dict, endpoint=None):
        url = self.path if endpoint is None else endpoint.join(self.path)
        return self._client.build_request("POST", url, json=body)

    def prepare_http_request(self, context: AgentContext):
        """Build the exact request without auth headers or reading the environment."""
        prepared = self._build_http_request(self.prepare_request(context))
        for name in AUTH_HEADERS:
            prepared.headers.pop(name, None)
        return prepared

    def _resolve_environment(self):
        """Read the key, and an environment-supplied endpoint, without sending anything.

        Returns the endpoint when the provider reads it from the environment, otherwise
        None. Errors name only the variable, never its value.
        """
        import httpx

        provider = self._provider
        for name in ("Authorization", "x-api-key"):
            self._client.headers.pop(name, None)
        endpoint = None
        if provider.base_url is not None:
            variable = provider.base_url_check_env
            configured = os.environ.get(variable) if variable is not None else None
            if configured and not provider.names_fixed_endpoint(configured):
                raise BackendConfigurationError(
                    f"{variable} names a different endpoint; the {provider.name} provider "
                    f"sends only to {provider.base_url}. Unset it, or select the provider "
                    "of that endpoint in the model string"
                )
        else:
            variable = provider.base_url_env
            value = os.environ.get(variable) or provider.base_url_default
            if not value:
                raise BackendConfigurationError(
                    f"{variable} is not set; export the {provider.name} endpoint in the shell "
                    "that runs the evaluation"
                )
            try:
                endpoint = httpx.URL(value.rstrip("/") + provider.base_url_path + "/")
            except (ValueError, httpx.InvalidURL):
                raise BackendConfigurationError(f"{variable} is not a valid URL") from None
        if provider.key_env is None:
            return endpoint
        key = os.environ.get(provider.key_env) or ""
        if not key:
            if not provider.key_required:
                return endpoint
            raise BackendConfigurationError(
                f"{provider.key_env} is not set; export the API key in the shell that runs "
                "the evaluation"
            )
        if not key.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in key):
            raise BackendConfigurationError(
                f"{provider.key_env} is malformed; an API key is printable ASCII without spaces"
            )
        if provider.auth_scheme == "bearer":
            self._client.headers["Authorization"] = f"Bearer {key}"
        else:
            self._client.headers["x-api-key"] = key
        return endpoint

    def _send_with_telemetry(self, http_request, metadata):
        telemetry = metadata["telemetry"]
        transport = telemetry["transport"]
        transport.update(request_bytes=len(http_request.content), outcome="in_flight",
                         send_attempted=True, started_at_unix_s=time.time())
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

    def _write_decision(self, row: dict) -> None:
        if self.log_dir is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redacted(row), allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError) as exc:
            warnings.warn(
                f"Agent decision logging degraded: {type(exc).__name__}",
                RuntimeWarning,
                stacklevel=3,
            )

    def generate(self, context: AgentContext) -> AgentOutput:
        if self._closed:
            raise BackendConfigurationError(f"{self.label} agent is closed")
        started = time.monotonic()
        usage: dict[str, int] = {}
        metadata: dict[str, Any] = {
            "backend": self.protocol,
            "model": self.model,
            "requested_model": self.model,
            "requested_effort": context.reasoning_effort,
            "requested_reasoning_mode": self.reasoning_mode,
            "reasoning_mode_verified": False,
            "context_lifetime": "rollout_window",
            "history_length": context.history_length,
            "history_length_semantics": "window_size",
            "history_unit": "complete_decision_turn",
            "reasoning_replay": self.reasoning_replay,
            "history_record_format": "raw_provider_exchange",
            "native_tool_calling": True,
            "max_response_chars": self.max_response_chars,
            "max_output_tokens": self.max_output_tokens,
            **self._route_metadata,
            "telemetry": new_telemetry(self.protocol),
        }
        row = {
            "request_id": context.request_id,
            "model": self.model,
            "requested_effort": context.reasoning_effort,
            "requested_reasoning_mode": self.reasoning_mode,
            "status": "started",
        }
        try:
            # An unsupported effort or invalid history is a configuration error here.
            body = self.prepare_request(context)
            next_history, _ = self._prepared_history(context)
            metadata.update(
                effective_history_length=max(0, context.history_length),
                previous_decision_reasoning_replay=context.history_length > 0,
                previous_decision_ids=[group["decision_id"] for group in next_history[:-1]],
                previous_decision_count=len(next_history) - 1,
            )
            metadata.update(self._request_settings(body))
            metadata.update(self._history_settings(next_history))
            endpoint = self._resolve_environment()
            http_request = self._build_http_request(body, endpoint)
            metadata["telemetry"]["transport"]["request_bytes"] = len(http_request.content)
            if self.request_recorder is not None:
                try:
                    artifact = self.request_recorder(context, http_request)
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
                raise http_status_error(self.label, response)
            try:
                raw_response = response.json()
                observe_response_json(metadata["telemetry"], raw_response)
                data = self._provider.unwrap(raw_response)
            except (ValueError, TypeError) as exc:
                raise BackendError(
                    f"{self.label} endpoint returned a malformed HTTP JSON envelope"
                ) from exc
            if not isinstance(data, dict):
                raise BackendError(
                    f"{self.label} endpoint returned an invalid HTTP response envelope"
                )
            usage = self._usage(data)
            output, calls, output_text = self._decode(data, metadata)
            self._reject_reused_call_ids(next_history[-1]["items"], calls)
            # Completed raw output is committed before decision-format checking,
            # so bounded repair calls retain exactly what the model produced.
            self._commit(next_history, output)
            metadata["tool_calls"] = copy.deepcopy(calls)
            output_chars = len(output_text) + sum(len(c["arguments"]) for c in calls)
            if output_chars > self.max_response_chars:
                raise BackendOutputLimitError(
                    f"Model response exceeded max_response_chars={self.max_response_chars}; "
                    "use compact function arguments",
                    usage=usage,
                    metadata={**metadata, "output_chars_observed": output_chars},
                )
            if len(calls) != 1 or calls[0]["name"] not in {t["name"] for t in context.tools}:
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
            return AgentOutput(
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
            if isinstance(exc, BackendError):
                row.update(metadata=exc.metadata)
            raise
        finally:
            self._write_decision(row)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
