"""Codex robot tool sessions scoped by the configured memory switch."""

from __future__ import annotations

import copy
import json
import math
import queue
import subprocess
import threading
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

from agentic_framework.harness.types import ModelReply, ModelRequest
from agentic_framework.models.llm.backend import (
    _DISABLED_FEATURES,
    _EFFORT_ORDER,
    _USAGE_KEYS,
    BackendConfigurationError,
    BackendError,
    BackendFormatError,
    BackendOutputLimitError,
    _message_content,
    _redacted,
    _terminate_group,
    _validate_response_limit,
)
from agentic_framework.models.llm.demo_transport import (
    deduplicate_demonstrations,
    demonstration_signature,
)


def _protocol_field(value, *path):
    """Read a nested app-server message field; a malformed message is a stream error."""
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise BackendError(
                "Codex App Server sent a malformed message without " + ".".join(path)
            )
        value = value[key]
    return value


def _params(message: dict) -> dict:
    params = message.get("params", {})
    if not isinstance(params, dict):
        raise BackendError("Codex App Server sent a message with non-object params")
    return params


class CodexBackend:
    """Use a rollout thread with memory, or a fresh thread for each request.

    generate() pauses on a robot function call. Its matching execution result
    and the next observation resume that exact callback when memory is enabled.
    With memory disabled, cancel the old callback before opening a fresh thread;
    neither its result nor its conversation enters the new inference context.
    Codex owns the ordered conversation and its reasoning context. Host tools,
    files, skills, external apps, and cross-rollout memories remain disabled.

    A transport or provider failure (timeout, closed or malformed app-server
    stream or message, failed RPC, inference error, a turn that did not
    complete, or an extra, duplicate, misaddressed or undeclared robot callback,
    which is declined) or an internal exception stops the app-server process,
    so the native thread and its conversation are lost; they cannot be rebuilt.
    The policy then resends the same request as the next attempt of that
    decision. With memory enabled that attempt starts a new native thread from
    the request text alone, exactly like a memory-disabled request: the current
    observation plus the repair feedback the resent attempt carries when it
    repairs a rejected call. The lost thread's pending tool result and earlier
    turns are never delivered. Later decisions continue on the new thread. The
    attempt's metadata records conversation_restarted=true, and the policy counts
    it in conversation_restarts.

    Configuration errors are never resent: the local checks before the turn is
    sent (a missing or non-executable Codex CLI, model catalog, effort, images,
    session contract, the new thread's model and instruction sources, one active
    decision, a closed backend) and a completed disabled host tool item, which
    means the local isolation failed.
    """

    def __init__(
        self,
        model: str | None = None,
        log_dir: str | Path = "logs/codex",
        timeout_s: float = 240,
        max_response_chars: int = 32768,
    ):
        if model is None:
            from agentic_framework.configuration.profiles import config_directory

            defaults = json.loads((config_directory() / "runtime-defaults.json").read_text())
            model = defaults["llm"]["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.model = model
        self.log_dir = Path(log_dir).resolve()
        self.timeout_s = float(timeout_s)
        self.max_response_chars = _validate_response_limit(max_response_chars)
        self.calls = 0
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()
        self._pending: deque[dict] = deque(maxlen=2048)
        self._events = None
        self._rpc_id = 0
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._models: list[dict] | None = None
        self._usage: dict[str, int] = {}
        self._final_text = ""
        self._lock = threading.Lock()
        self.log_errors = []
        self._native_retry_notifications = 0
        self._message_phases: dict[str, str | None] = {}
        self._message_chars: dict[str, int] = {}
        self._discard_output = False
        self._turn_completed = False
        self._thread_info = None
        self._session_contract = None
        self._tool_request = None
        self._terminal_results = ()
        self._allowed_tools = set()
        self._seen_call_ids = set()
        self._reported_usage = {}
        self._preview_thread_id = None
        self._preview_call_id = None
        # Set when a failure stopped the server mid-rollout; the next memory
        # request starts a new native thread instead of resuming the lost one.
        self._rollout_lost = False
        self._conversation_restarted = False

    def _server_command(self) -> list[str]:
        command = ["codex", "app-server", "--listen", "stdio://"]
        for feature in _DISABLED_FEATURES:
            if feature != "code_mode_host":
                command.extend(["--disable", feature])
        # Astra's authoritative model catalog uses tool_mode=code_mode_only.
        # Its native function bridge requires the bundled JS tool host even
        # when optional Code Mode is disabled. Shell/files/apps remain disabled
        # independently above; the bridge exposes only registered robot tools.
        command.extend(["--enable", "code_mode_host", "--enable", "skip_host_skill_discovery"])
        for override in (
            'web_search="disabled"',
            "project_doc_max_bytes=0",
            "agents.enabled=false",
            "mcp_servers={}",
            'history.persistence="none"',
        ):
            command.extend(["-c", override])
        return command

    def _log_failure(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        if message not in self.log_errors:
            self.log_errors.append(message)
            warnings.warn(
                f"Backend audit logging degraded: {message}", RuntimeWarning, stacklevel=2
            )

    def _log(self, direction: str, message: dict) -> None:
        if self._events is not None:
            try:
                self._events.write(
                    json.dumps(
                        {
                            "direction": direction,
                            "time": time.time(),
                            "message": _redacted(message),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                self._events.flush()
            except (OSError, ValueError, TypeError) as exc:
                self._log_failure(exc)

    def _send(self, message: dict) -> None:
        self._log("send", message)
        if self._process is None or self._process.stdin is None:
            raise BackendError("Codex server is not running")
        self._process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self._process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self._process
        try:
            assert process is not None and process.stdout is not None
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    raise BackendError("Codex App Server emitted malformed JSON-RPC output") from None
                self._queue.put(message)
        except BackendError as exc:
            self._queue.put(exc)
        except Exception as exc:
            # Undecodable bytes or a broken pipe: a failed stream, retried like a closed one.
            error = BackendError(f"Codex App Server output stream failed: {type(exc).__name__}")
            error.__cause__ = exc
            self._queue.put(error)
        finally:
            self._queue.put(EOFError("Codex App Server closed its output"))

    def _receive(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for Codex App Server")
        try:
            message = self._queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for Codex App Server") from exc
        if isinstance(message, Exception):
            raise message
        if not isinstance(message, dict):
            raise BackendError("Codex emitted a non-object RPC message")
        self._log("receive", message)
        return message

    def _start(self, deadline: float) -> None:
        if self._process is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        working_dir = self.log_dir / "workdir"
        working_dir.mkdir(exist_ok=True)
        self._events = (self.log_dir / "server-rpc.jsonl").open("a", encoding="utf-8")
        # Do not persist arbitrary server stderr; auth material must not leak to logs.
        try:
            self._process = subprocess.Popen(
                self._server_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=working_dir,
                start_new_session=True,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError) as exc:
            # A local setup problem found before any request is sent; never resent.
            raise BackendConfigurationError(
                "Codex CLI is not installed or not executable"
            ) from exc
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._rpc(
            "initialize",
            {
                "clientInfo": {"name": "agentic-framework", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
            deadline,
        )
        self._send({"method": "initialized", "params": {}})

    def _model(self, deadline: float) -> dict:
        if self._models is None:
            self._models = []
            cursor = None
            seen_cursors = set()
            while True:
                params: dict[str, Any] = {"includeHidden": True, "limit": 100}
                if cursor is not None:
                    params["cursor"] = cursor
                page = self._rpc("model/list", params, deadline)
                data = page.get("data", []) if isinstance(page, dict) else None
                if not isinstance(data, list) or any(not isinstance(m, dict) for m in data):
                    raise BackendError("Codex model/list returned a malformed page")
                self._models.extend(data)
                cursor = page.get("nextCursor")
                if not cursor:
                    break
                if cursor in seen_cursors:
                    raise BackendError("Codex model/list repeated a pagination cursor")
                seen_cursors.add(cursor)
        for model in self._models:
            if model.get("model") == self.model or model.get("id") == self.model:
                return model
        raise BackendConfigurationError(
            f"Requested model {self.model!r} is not available in Codex model/list"
        )

    def _effort(self, request: ModelRequest, model: dict) -> str:
        options = [_protocol_field(entry, "reasoningEffort")
                   for entry in model.get("supportedReasoningEfforts", [])]
        asked = request.reasoning_effort
        if asked == "default":
            selected = model.get("defaultReasoningEffort")
        elif asked in {"lowest", "lowest_supported"}:
            selected = next((value for value in _EFFORT_ORDER if value in options), None)
        else:
            selected = asked
        if selected not in options:
            raise BackendConfigurationError(
                f"Reasoning effort {asked!r} is unsupported by {self.model!r}; "
                f"supported values: {options}"
            )
        modalities = model.get("inputModalities", ["text", "image"])
        if request.images and "image" not in modalities:
            raise BackendConfigurationError(f"Model {self.model!r} does not accept images")
        return selected

    def _matches(self, message: dict) -> bool:
        params = _params(message)
        if params.get("threadId") != self._thread_id:
            return False
        turn = params.get("turnId")
        if not turn:
            turn = params.get("turn", {})
            turn = turn.get("id") if isinstance(turn, dict) else None
        return turn == self._turn_id

    def _count_output(self, item_id, text, *, delta=False):
        if self._discard_output:
            return
        if not isinstance(text, str):
            raise BackendError("Codex emitted a non-string agent message")
        key = item_id or "legacy-message"
        previous = self._message_chars.get(key, 0)
        self._message_chars[key] = previous + len(text) if delta else max(previous, len(text))
        count = sum(self._message_chars.values())
        if count > self.max_response_chars:
            raise BackendOutputLimitError(
                f"Model response exceeded max_response_chars={self.max_response_chars}; "
                "use compact robot tool arguments",
                metadata={"output_chars_observed": count, "partial_output_discarded": True},
            )

    def _write_decision(self, row: dict) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_redacted(row), allow_nan=False) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self._log_failure(exc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _server_request(self, message: dict) -> None:
        self._send(
            {
                "id": message["id"],
                "error": {"code": -32601, "message": "Unexpected or disabled robot backend tool"},
            }
        )
        # Declined, so nothing ran: an extra, duplicate, misaddressed or undeclared
        # callback (or a host tool approval request) is a provider failure that the
        # policy resends on a new native thread.
        raise BackendError(f"Unexpected host tool request: {message.get('method')}")

    def _rpc(self, method: str, params: dict, deadline: float) -> dict:
        self._rpc_id += 1
        request_id = f"agentic-{self._rpc_id}"
        self._send({"id": request_id, "method": method, "params": params})
        while True:
            message = self._receive(deadline)
            if message.get("id") == request_id:
                if "error" in message:
                    raise BackendError(f"Codex RPC {method} failed; see server-rpc.jsonl")
                return _protocol_field(message, "result")
            # Fast native callbacks may precede turn/start's reply. Scope them
            # after the returned thread and turn ids are known.
            if "id" not in message or "method" in message:
                self._pending.append(message)

    def capabilities(self) -> dict[str, Any]:
        if self._closed:
            raise BackendError("Codex backend is closed")
        if not self._lock.acquire(blocking=False):
            raise BackendError("Cannot read capabilities during an active decision")
        try:
            deadline = time.monotonic() + self.timeout_s
            self._start(deadline)
            model = self._model(deadline)
            return {
                "model": model["model"],
                "backend": "codex_app_server",
                "input_modalities": list(model.get("inputModalities", ["text", "image"])),
                "supported_reasoning_efforts": [
                    entry["reasoningEffort"] for entry in model.get("supportedReasoningEfforts", [])
                ],
                "default_reasoning_effort": model.get("defaultReasoningEffort"),
                "native_tool_calling": True,
                "context_lifetime": "configurable",
                "context_lifetimes": ["request", "rollout"],
                "history_length_applies": True,
                "history_length_semantics": "memory_toggle",
            }
        except BaseException:
            self._stop_process()
            raise
        finally:
            self._lock.release()

    @staticmethod
    def _memory_metadata(request: ModelRequest) -> dict[str, Any]:
        if type(request.history_length) is not int:
            raise BackendConfigurationError("history_length must be an integer")
        enabled = request.history_length > 0
        return {
            "history_length": request.history_length,
            "memory_enabled": enabled,
            "history_length_applies": True,
            "history_length_semantics": "memory_toggle",
            "context_lifetime": "rollout" if enabled else "request",
        }

    @staticmethod
    def _observation_items(messages: list[dict]) -> list[dict]:
        """Convert the encoded current observation to native content items."""
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise BackendConfigurationError(
                "Codex requests must contain only the current observation; native session owns history"
            )
        return [
            {"type": "inputText", "text": part["text"]}
            if part["type"] == "input_text"
            else {"type": "inputImage", "imageUrl": part["image_url"]}
            for part in messages[0]["content"]
        ]

    @staticmethod
    def _dynamic_tools(request: ModelRequest) -> list[dict]:
        tools, names = [], set()
        for tool in request.tools:
            if (
                tool.get("type") != "function"
                or not isinstance(tool.get("name"), str)
                or not tool["name"]
                or tool["name"] in names
            ):
                raise BackendConfigurationError(
                    "Robot tools need unique named function definitions"
                )
            names.add(tool["name"])
            tools.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": copy.deepcopy(tool["parameters"]),
                    "deferLoading": False,
                }
            )
        if not tools:
            raise BackendConfigurationError("Codex robot backend requires native function tools")
        return tools

    def prepare_request(
        self,
        request: ModelRequest,
        *,
        effort: str | None = None,
        thread_id: str = "<new-thread-id>",
        restart: bool = False,
        include_demo: bool | None = None,
    ) -> dict[str, Any]:
        """Pure preview: build native parameters without inspecting a live session.

        Tool results describe callback continuation. The callback RPC id is
        represented by a placeholder; generate replaces it with the server id.
        With memory disabled, every request describes a fresh session containing
        only the current request text (the observation, plus repair feedback on a
        format repair), regardless of previous tool results. So does ``restart``,
        used after a failure lost the rollout's native thread. No past
        observations, reasoning, or model outputs are flattened into user text.

        Continuations reuse the fixed teacher already present in the thread.
        ``include_demo=False`` also describes a format-repair turn on an existing
        thread when there is no pending tool result. Live generation supplies
        that fact explicitly; a pure preview never inspects native session state.
        Fresh memory-disabled requests and restarts always include the teacher.
        """
        selected = request.reasoning_effort if effort is None else effort
        memory = self._memory_metadata(request)
        continuation = bool(request.tool_results) and memory["memory_enabled"] and not restart
        if include_demo is None:
            include_demo = not continuation
        if not memory["memory_enabled"] or restart:
            include_demo = True
        messages = _message_content(request)
        demo_signature = demonstration_signature(messages)
        observation = self._observation_items(
            deduplicate_demonstrations(messages, omit_all=not include_demo)
        )
        dynamic_tools = self._dynamic_tools(request)
        metadata = {
            "backend": "codex_app_server",
            "model": self.model,
            "requested_effort": request.reasoning_effort,
            "prepared_effort": selected,
            "effort_source": "request" if effort is None else "override",
            "effective_effort_verified": False,
            "model_capabilities_verified": False,
            **memory,
            "context_layout": "native_tool_loop",
            "native_message_roles": True,
            "native_tool_calling": True,
            "ephemeral": True,
            "demonstration_signature": demo_signature,
        }
        if request.tool_results:
            if len(request.tool_results) != 1:
                raise BackendConfigurationError("Exactly one pending robot tool result is required")
            result = request.tool_results[0]
            if (
                result.get("type") != "function_call_output"
                or not isinstance(result.get("call_id"), str)
                or not isinstance(result.get("output"), str)
            ):
                raise BackendConfigurationError("Malformed native robot tool result")
        if request.tool_results and memory["memory_enabled"] and not restart:
            result = request.tool_results[0]
            return {
                "tool_response": {
                    "id": "<pending-tool-rpc-id>",
                    "result": {
                        "success": True,
                        "contentItems": [
                            {"type": "inputText", "text": result["output"]},
                            *observation,
                        ],
                    },
                },
                "metadata": {
                    **metadata,
                    "operation": "resume_native_tool",
                    "call_id": result["call_id"],
                },
            }
        items = [
            {"type": "text", "text": part["text"]}
            if part["type"] == "inputText"
            else {"type": "image", "url": part["imageUrl"]}
            for part in observation
        ]
        return {
            "thread_start": {
                "model": self.model,
                "modelProvider": "openai",
                "allowProviderModelFallback": False,
                "cwd": str(self.log_dir / "workdir"),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": True,
                "environments": [],
                "selectedCapabilityRoots": [],
                "dynamicTools": dynamic_tools,
                "baseInstructions": request.instructions,
                "developerInstructions": "Control the robot exclusively through the provided native robot tools. Issue one robot tool call, then observe its execution result and new observation before deciding again. If Codex exposes the robot tools through functions.exec, each exec cell must invoke exactly ONE robot function and await its result. Store that return value in result. In this CLI a robot result is a STRING containing newline-separated text and image data URLs. Forward ALL returned text items and ALL returned image items, preserving order, using this exact pattern: for (const part of result.split('\\n')) { if (part.startsWith('data:image/')) image(part); else text(part); } Then finish that exec cell. Do not discard the result, print image data as text, or make a second robot call within the same exec cell. Inspect the surfaced new observation and reason about the next action before starting another exec cell. Continue this loop; do not end the assistant turn after an ordinary motion call returns. Finish only by invoking done or give_up. Never replace a robot tool call with a final text answer or JSON response. Do not use shell, files, apps or host tools.",
                "config": {
                    "model_reasoning_effort": selected,
                    "personality": "none",
                    "project_doc_max_bytes": 0,
                    "web_search": "disabled",
                },
            },
            "turn_start": {
                "threadId": thread_id,
                "input": items,
                "model": self.model,
                "effort": selected,
            },
            "metadata": {
                **metadata,
                "operation": (
                    "restart_rollout" if memory["memory_enabled"] and restart
                    else "start_rollout" if memory["memory_enabled"] else "start_request"
                ),
            },
        }

    def accept_reply(self, request: ModelRequest, reply: ModelReply) -> None:
        """Record offline preview progress without creating or modifying native state."""
        calls = reply.tool_calls or tuple(reply.payload.get("tool_calls", ()))
        self._preview_thread_id = (
            self._preview_thread_id or "<preview-rollout-thread>"
            if request.history_length > 0
            else f"<preview-request-{request.request_id}>"
        )
        self._preview_call_id = calls[0].get("call_id") if len(calls) == 1 else None

    def _notification(self, message: dict) -> None:
        method, params = message.get("method"), _params(message)
        if method == "thread/tokenUsage/updated":
            # Usage is cumulative per native thread, and may arrive after the
            # corresponding callback or turn completion. Do not lose final
            # inference usage just because _turn_id has already been cleared.
            if self._thread_id is not None and params.get("threadId") == self._thread_id:
                total = _protocol_field(params, "tokenUsage", "total")
                if not isinstance(total, dict):
                    raise BackendError("Codex App Server sent non-object token usage")
                for source, target in _USAGE_KEYS.items():
                    if source in total:
                        try:
                            value = int(total[source])
                        except (TypeError, ValueError, OverflowError):
                            raise BackendError(
                                "Codex App Server sent an invalid token count"
                            ) from None
                        self._usage[target] = max(self._usage.get(target, 0), value)
            return
        if not self._matches(message):
            return
        if method == "turn/completed":
            self._turn_completed = True
        elif method == "item/started":
            item = params.get("item", {})
            if not isinstance(item, dict):
                raise BackendError("Codex App Server sent a non-object item")
            if item.get("type") == "agentMessage":
                self._message_phases[item.get("id")] = item.get("phase")
        elif method == "item/agentMessage/delta":
            self._count_output(params.get("itemId"), params.get("delta", ""), delta=True)
        elif method == "item/completed":
            item = _protocol_field(params, "item")
            if not isinstance(item, dict):
                raise BackendError("Codex App Server sent a non-object item")
            if item.get("type") == "agentMessage":
                self._count_output(item.get("id"), item.get("text", ""))
                self._final_text = item.get("text", "")
            elif item.get("type") in {"commandExecution", "mcpToolCall", "webSearch", "fileChange"}:
                # A completed host tool item means the local isolation failed; a
                # resend could repeat the breach, so this is never retried.
                raise BackendConfigurationError("Codex attempted a disabled host tool")
            elif (
                item.get("type") == "dynamicToolCall"
                and item.get("tool") not in self._allowed_tools
            ):
                raise BackendError("Codex attempted an undeclared robot tool")
        elif method == "error":
            if params.get("willRetry", False):
                self._native_retry_notifications += 1
            else:
                raise BackendError("Codex reported an inference error; see server-rpc.jsonl")

    def _next_tool(self, deadline: float) -> dict:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a native robot tool call")
            message = self._pending.popleft() if self._pending else self._receive(deadline)
            if "id" in message:
                if "method" not in message:
                    continue
                params = _params(message)
                if (
                    message["method"] != "item/tool/call"
                    or not self._matches(message)
                    or params.get("tool") not in self._allowed_tools
                    or params.get("namespace") not in (None, "")
                    or not isinstance(params.get("callId"), str)
                    or not params["callId"]
                    or params["callId"] in self._seen_call_ids
                ):
                    self._server_request(message)
                self._tool_request = message
                self._seen_call_ids.add(params["callId"])
                try:
                    arguments = json.dumps(
                        params.get("arguments"), allow_nan=False, separators=(",", ":")
                    )
                except (TypeError, ValueError) as exc:
                    raise BackendFormatError("Codex emitted invalid robot tool arguments") from exc
                self._count_output(params["callId"], arguments)
                return {
                    "type": "function_call",
                    "call_id": params["callId"],
                    "name": params["tool"],
                    "arguments": arguments,
                }
            self._notification(message)
            if not self._matches(message):
                continue
            if message.get("method") == "turn/completed":
                turn = _protocol_field(message, "params", "turn")
                status = turn.get("status") if isinstance(turn, dict) else None
                self._turn_id = None
                if status != "completed":
                    raise BackendError(f"Codex turn ended with status {status!r}")
                raise BackendFormatError(
                    "Codex ended its turn without a robot tool call; use a native robot function",
                    response_text=self._final_text,
                )

    def _drain_waiting_events(self) -> None:
        """Reject extra requests emitted before the previous result was delivered.

        The CLI has no exposed parallel_tool_calls switch. Its prompt demands
        one call at a time; queued extra callbacks must never become sequential
        robot actions without the intervening observation they require.
        """
        while True:
            if self._pending:
                message = self._pending.popleft()
            else:
                try:
                    message = self._queue.get_nowait()
                except queue.Empty:
                    return
                if isinstance(message, Exception):
                    raise message
                self._log("receive", message)
            if "id" in message:
                if "method" in message:
                    self._server_request(message)
                continue
            self._notification(message)
            if self._matches(message) and message.get("method") == "turn/completed":
                raise BackendError("Codex ended its turn while a robot call was awaiting execution")

    def _usage_delta(self, previous_session: dict[str, int] | None = None) -> dict[str, int]:
        usage = {
            key: max(0, value - self._reported_usage.get(key, 0))
            for key, value in self._usage.items()
        }
        for key, value in (previous_session or {}).items():
            usage[key] = usage.get(key, 0) + value
        return usage

    def _reply_metadata(self, request: ModelRequest, previous_session_usage=None) -> dict:
        return {
            "backend": "codex_app_server",
            "model": self.model,
            "reasoning_effort": (self._thread_info or {}).get("reasoningEffort"),
            "requested_effort": request.reasoning_effort,
            "thread_id": self._thread_id,
            "turn_id": self._turn_id,
            **self._memory_metadata(request),
            "context_layout": "native_tool_loop",
            "native_tool_calling": True,
            "native_tool_waiting": self._tool_request is not None,
            "ephemeral": True,
            "usage_available": bool(self._usage or previous_session_usage),
            "usage_scope": "since_previous_decision",
            "session_total_usage": dict(self._usage),
            "previous_session_usage": dict(previous_session_usage or {}),
            "conversation_restarted": self._conversation_restarted,
            "native_retry_notifications": self._native_retry_notifications,
            "log_errors": list(self.log_errors),
            "max_response_chars": self.max_response_chars,
        }

    def generate(self, request: ModelRequest) -> ModelReply:
        memory = self._memory_metadata(request)
        if not self._lock.acquire(blocking=False):
            # A framework invariant violation, never a transient failure to resend.
            raise BackendConfigurationError("CodexBackend supports one active decision at a time")
        started = time.monotonic()
        row = {"request_id": request.request_id, "model": self.model, "status": "started"}
        previous_session_usage = {}
        # A memory request after a lost native thread starts a new one.
        self._conversation_restarted = restart = memory["memory_enabled"] and self._rollout_lost
        try:
            if self._closed:
                raise BackendConfigurationError("Codex backend is closed")
            if not memory["memory_enabled"]:
                # Cancel before acknowledging the old call so it cannot trigger
                # another inference. Keep the server process, but never its thread.
                if self._tool_request is not None and request.tool_results:
                    self.record_tool_results(request.tool_results)
                previous_session_usage = self._reset_session()
            deadline = started + self.timeout_s
            self._start(deadline)
            effort = self._effort(request, self._model(deadline))
            prepared = self.prepare_request(
                request, effort=effort, restart=restart,
                include_demo=not (memory["memory_enabled"] and self._thread_id is not None),
            )
            contract = (
                request.instructions,
                effort,
                json.dumps(list(request.tools), sort_keys=True),
                memory["memory_enabled"],
                prepared["metadata"]["demonstration_signature"],
            )
            if self._session_contract is not None and contract != self._session_contract:
                raise BackendConfigurationError(
                    "Robot instructions, tools, effort, memory mode and teacher demo must remain fixed "
                    "until session reset"
                )
            self._native_retry_notifications = 0
            self._message_chars.clear()
            self._message_phases.clear()
            self._final_text = ""
            self._discard_output = False
            if self._tool_request is not None:
                if (
                    not request.tool_results
                    or request.tool_results[0]["call_id"] != self._tool_request["params"]["callId"]
                ):
                    raise BackendConfigurationError(
                        "The pending robot call requires its matching execution result"
                    )
                self._drain_waiting_events()
                self._send({**prepared["tool_response"], "id": self._tool_request["id"]})
                self._tool_request = None
            else:
                if request.tool_results and memory["memory_enabled"] and not restart:
                    raise BackendConfigurationError(
                        "Received a robot result without a pending native call"
                    )
                if self._thread_id is None:
                    thread = self._rpc("thread/start", prepared["thread_start"], deadline)
                    self._thread_id = _protocol_field(thread, "thread", "id")
                    self._thread_info = thread
                    if thread.get("model") != self.model or thread.get("reasoningEffort") != effort:
                        raise BackendConfigurationError(
                            "Codex did not honor the requested model and reasoning effort"
                        )
                    if thread.get("instructionSources"):
                        raise BackendConfigurationError(
                            "Codex loaded host instruction files despite isolated configuration"
                        )
                    self._session_contract = contract
                    self._allowed_tools = {tool["name"] for tool in request.tools}
                    self._rollout_lost = False
                turn = self._rpc(
                    "turn/start", {**prepared["turn_start"], "threadId": self._thread_id}, deadline
                )
                self._turn_id = _protocol_field(turn, "turn", "id")
                self._turn_completed = False
            self.calls += 1
            call = self._next_tool(deadline)
            usage = self._usage_delta(previous_session_usage)
            self._reported_usage = dict(self._usage)
            metadata = self._reply_metadata(request, previous_session_usage)
            elapsed = time.monotonic() - started
            payload = {"tool_calls": [call]}
            row.update(
                status="tool_waiting",
                payload=payload,
                usage=usage,
                metadata=metadata,
                elapsed_s=elapsed,
            )
            return ModelReply(
                payload, usage, metadata, elapsed, tool_calls=(call,), output_items=(call,),
                response_text=self._final_text,
            )
        except BackendFormatError as exc:
            exc.usage = self._usage_delta(previous_session_usage)
            self._reported_usage = dict(self._usage)
            exc.metadata = {**self._reply_metadata(request, previous_session_usage), **exc.metadata}
            row.update(
                status=exc.category,
                error_type=type(exc).__name__,
                usage=exc.usage,
                metadata=exc.metadata,
                elapsed_s=time.monotonic() - started,
            )
            if self._tool_request is not None:
                # The actual call and reasoning remain in the native session.
                # Only its identity is needed to return the bounded policy error;
                # do not copy oversized/unserializable arguments into error logs.
                params = self._tool_request["params"]
                exc.metadata["tool_calls"] = [
                    {
                        "type": "function_call",
                        "call_id": params["callId"],
                        "name": params["tool"],
                        "arguments": "",
                    }
                ]
                row["metadata"] = exc.metadata
            elif self._turn_id is not None:
                try:
                    self._cancel_turn(time.monotonic() + 2)
                except BaseException:
                    self._stop_process()
                    self._rollout_lost = True
                    raise
            raise
        except BaseException as exc:
            usage = self._usage_delta(previous_session_usage)
            metadata = self._reply_metadata(request, previous_session_usage)
            # Every failure is recorded (and internal ones resent), so any exception
            # carries this attempt's usage and restart metadata, as attach_exception_telemetry
            # does for the HTTP backends; an exception refusing attributes keeps none.
            try:
                exc.metadata = {**metadata, **getattr(exc, "metadata", {})}
                exc.usage = {**usage, **getattr(exc, "usage", {})}
            except (AttributeError, TypeError):
                pass
            row.update(
                status="error",
                error_type=type(exc).__name__,
                usage=usage,
                metadata=metadata,
                elapsed_s=time.monotonic() - started,
            )
            # The native thread cannot survive a stopped server; a retry starts a new one.
            self._stop_process()
            self._rollout_lost = True
            raise
        finally:
            try:
                self._write_decision(row)
            finally:
                self._lock.release()

    def record_tool_results(self, results) -> None:
        """Hold terminal receipts until cancellation, avoiding an extra model call."""
        results = tuple(copy.deepcopy(results))
        if self._tool_request is not None:
            if (
                len(results) != 1
                or results[0].get("call_id") != self._tool_request["params"]["callId"]
            ):
                raise BackendConfigurationError("Terminal result must match the pending robot call")
        self._terminal_results = results

    def _cancel_turn(self, deadline: float) -> None:
        if self._turn_id is None:
            return
        self._discard_output = True
        # Interrupt first, while the model is waiting. Resolving the callback
        # first could trigger unwanted inference after the episode has ended.
        try:
            self._rpc(
                "turn/interrupt", {"threadId": self._thread_id, "turnId": self._turn_id}, deadline
            )
        except BackendError:
            if not any(
                self._matches(row) and row.get("method") == "turn/completed"
                for row in self._pending
            ):
                raise
        while not self._turn_completed:
            message = self._pending.popleft() if self._pending else self._receive(deadline)
            if "id" in message:
                if "method" in message:
                    self._send(
                        {
                            "id": message["id"],
                            "error": {"code": -32000, "message": "Rollout cancelled"},
                        }
                    )
                continue
            self._notification(message)
        if self._tool_request is not None:
            text = (
                self._terminal_results[0]["output"]
                if self._terminal_results
                else "Rollout ended; pending robot tool loop cancelled."
            )
            self._send(
                {
                    "id": self._tool_request["id"],
                    "result": {
                        "success": bool(self._terminal_results),
                        "contentItems": [{"type": "inputText", "text": text}],
                    },
                }
            )
            self._tool_request = None
        self._turn_id = None
        self._terminal_results = ()

    def _clear_rollout(self) -> None:
        self._thread_id = self._turn_id = None
        self._thread_info = self._session_contract = self._tool_request = None
        self._terminal_results = ()
        self._allowed_tools.clear()
        self._seen_call_ids.clear()
        self._pending.clear()
        self._usage.clear()
        self._reported_usage.clear()
        self._turn_completed = False
        self._preview_thread_id = self._preview_call_id = None

    def _stop_process(self) -> None:
        process = self._process
        if process is not None:
            _terminate_group(process)
            if self._reader is not None:
                self._reader.join(timeout=1)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        self._process = self._reader = None
        if self._events is not None:
            self._events.close()
            self._events = None
        self._queue = queue.Queue()
        self._models = None
        self._clear_rollout()

    def _reset_session(self) -> dict[str, int]:
        """Release native context and collect late usage while the caller holds the lock."""
        remaining_usage = {}
        if self._thread_id is not None:
            try:
                deadline = time.monotonic() + 2
                self._cancel_turn(deadline)
                self._rpc("thread/unsubscribe", {"threadId": self._thread_id}, deadline)
                while self._pending:
                    self._notification(self._pending.popleft())
            except Exception:
                # Termination guarantees no context contamination even if
                # cancellation/unsubscribe fails or the server stalls.
                remaining_usage = self._usage_delta()
                self._stop_process()
        remaining_usage = remaining_usage or self._usage_delta()
        self._clear_rollout()
        self._rollout_lost = False
        return remaining_usage

    def reset(self) -> dict[str, int]:
        """End the rollout and return any usage received after its last reply."""
        if not self._lock.acquire(blocking=False):
            raise BackendError("Cannot reset Codex during an active decision")
        try:
            return self._reset_session()
        finally:
            self._lock.release()

    end_rollout = reset

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_process()
