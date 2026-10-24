"""The Anthropic Messages wire: request encoding, prefix-bound thinking replay and decoding.

A request carries the static instructions as ``system``, the function tools with their
parameters as ``input_schema``, ``tool_choice: auto`` with parallel calls disabled (forced
tool use is rejected by current models), adaptive thinking with the configured
``output_config.effort`` (omitted for ``default``, so the model's own default applies) and
the required ``max_tokens``. A provider that declares ``prompt_caching`` sends the system
prompt as one text block with a ``cache_control`` breakpoint and enables automatic caching
with a top-level ``cache_control``. Images are base64 ``image`` blocks, each preceded by the
same camera label the Responses wire sends.

A thinking block's signature binds the conversation prefix that produced it: ``system``, the
tools and every earlier message. Window eviction and the move of the deduplicated teacher copy
edit that prefix, so replaying the block unchanged would be rejected. A local, deterministic
tracker therefore stores, with every committed assistant turn, a digest of the prefix it was
sent with. When a request is assembled, a turn keeps its thinking blocks only if the prefix
assembled before it (after the earlier turns' keep or strip decisions) has that digest;
otherwise its ``thinking`` and ``redacted_thinking`` blocks are left out and its text and tool
calls stay. The decision is a pure function of the stored history, so a resend after a strip
is byte-identical, and the archived request is exactly what the model received. Each request
records ``thinking_replay`` with the kept and stripped block counts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agentic_framework.harness.types import AgentContext, AgentOutput
from agentic_framework.models.llm.demo_transport import deduplicate_demonstrations
from agentic_framework.models.llm.errors import BackendConfigurationError, BackendError
from agentic_framework.models.llm.http_agent import HttpAgentModel
from agentic_framework.models.llm.responses import _message_content

# Sent when the provider declares no API version.
DEFAULT_API_VERSION = "2026-10-21"
# A response with another stop_reason (refusal, max_tokens, pause_turn, ...) is unusable.
COMPLETED_STOP_REASONS = frozenset({"tool_use", "end_turn"})
THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})
CONTENT_TYPES = frozenset({"text", "tool_use", *THINKING_TYPES})
# Local state of a stored assistant turn and of a request's pending turn; never sent.
BINDING = "prefix_binding"
PENDING = "pending_turn"


def _encode(context: AgentContext) -> list[dict]:
    """Encode the context's ordered messages as Messages text and base64 image blocks."""
    converted = []
    for message in _message_content(context):
        content = []
        for part in message["content"]:
            if part["type"] == "input_text":
                content.append({"type": "text", "text": part["text"]})
            else:
                prefix, data = part["image_url"].split(",", 1)
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": prefix[5:].split(";", 1)[0], "data": data,
                }})
        converted.append({"role": message["role"], "content": content})
    return converted


def _tools(context: AgentContext) -> list[dict]:
    if not context.tools:
        raise BackendConfigurationError("Native robot decisions require function tools")
    return [{"name": tool["name"], "description": tool.get("description", ""),
             "input_schema": copy.deepcopy(tool["parameters"])} for tool in context.tools]


def _calls(content: list) -> tuple[list[dict], str]:
    """Validate response content; return the canonical calls and the visible text.

    Each ``tool_use`` input is re-encoded as a canonical ``function_call`` record. The
    content itself, including signed and redacted thinking, is kept unchanged for replay.
    """
    calls, text, seen = [], [], set()
    for block in content:
        if not isinstance(block, dict):
            raise BackendError("Messages content blocks must be objects")
        kind = block.get("type")
        if kind == "tool_use":
            identifier, name, arguments = block.get("id"), block.get("name"), block.get("input")
            if (not isinstance(identifier, str) or not identifier or identifier in seen
                    or not isinstance(name, str) or not name or not isinstance(arguments, dict)):
                raise BackendError("Messages tool_use requires a unique ID, name and input object")
            seen.add(identifier)
            try:
                encoded = json.dumps(arguments, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise BackendError("Messages tool input must be finite JSON") from exc
            calls.append({"type": "function_call", "call_id": identifier,
                          "name": name, "arguments": encoded})
        elif kind == "text":
            if not isinstance(block.get("text"), str):
                raise BackendError("Messages text block requires string text")
            text.append(block["text"])
        elif kind == "thinking":
            if not isinstance(block.get("thinking"), str) or not isinstance(
                block.get("signature"), str
            ):
                raise BackendError("Messages thinking requires original text and signature")
        elif kind == "redacted_thinking":
            if not isinstance(block.get("data"), str):
                raise BackendError("Messages redacted thinking requires original data")
        else:
            raise BackendError("Unsupported native Messages response content")
    return calls, "".join(text)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bind(system: str, tools: list[dict], items: list[dict],
          earlier: int = 0) -> tuple[list[dict], str, dict]:
    """Apply the thinking binding to assembled items, before same-role messages coalesce.

    Returns the messages to send, the digest of their whole prefix (stored with the turn this
    request produces) and the request's ``thinking_replay`` record. A turn's prefix digest
    chains the digests of system and tools and of every earlier message as sent. The first
    ``earlier`` items belong to earlier decisions; ``earlier_kept`` counts their kept blocks.
    """
    head = _sha256({"system": system, "tools": tools})
    digests: list[str] = []
    sent: list[dict] = []
    kept = stripped = earlier_kept = 0
    first = None

    def prefix() -> str:
        return _sha256([head, digests])

    for number, item in enumerate(items):
        message = {"role": item["role"], "content": item["content"]}
        if item["role"] == "assistant":
            positions = [index for index, block in enumerate(message["content"])
                         if block.get("type") in THINKING_TYPES]
            if positions and item.get(BINDING) == prefix():
                kept += len(positions)
                if number < earlier:
                    earlier_kept += len(positions)
            elif positions:
                stripped += len(positions)
                if first is None:
                    first = (len(sent), positions[0])
                message["content"] = [block for block in message["content"]
                                      if block.get("type") not in THINKING_TYPES]
        sent.append(message)
        digests.append(_sha256(message))
    return sent, prefix(), {"kept_blocks": kept, "stripped_blocks": stripped, "first": first,
                            "earlier_kept": earlier_kept}


def _coalesce(messages: list[dict]) -> tuple[list[dict], list[int]]:
    """Place tool results before the next observation in the same user message.

    Messages left without content (a turn of thinking only, after a strip) are dropped.
    Returns the merged messages and, for each input message, its index among them.
    """
    merged, positions = [], []
    for message in messages:
        if not message["content"]:
            positions.append(len(merged))
            continue
        if merged and merged[-1]["role"] == message["role"] == "user":
            merged[-1]["content"].extend(copy.deepcopy(message["content"]))
        else:
            merged.append(copy.deepcopy(message))
        positions.append(len(merged) - 1)
    return merged, positions


class MessagesAgent(HttpAgentModel):
    """Native Messages tools with adaptive thinking and prefix-bound thinking replay.

    ``anthropic/claude-x`` and ``claude-x`` both send ``claude-x``. The provider declares
    the endpoint, the authentication and the ``anthropic-version`` header. Once the history
    window is full, every request evicts a decision, so earlier turns replay their text and
    tool calls without reasoning; a format repair keeps its own decision's thinking.
    """

    protocol = "messages"
    label = "Messages"
    path = "messages"
    reasoning_replay = "selected_decision_records_with_prefix_bound_thinking"

    def __init__(self, model, **options):
        super().__init__(model, **options)
        self.api_version = self._provider.api_version or DEFAULT_API_VERSION

    def _encode_context(self, context: AgentContext) -> list[dict]:
        return _encode(context)

    def _assemble(self, items: list[dict]) -> list[dict]:
        # Bound and coalesced in _prepared_history, which also knows system and tools.
        return deduplicate_demonstrations(items)

    def _prepared_history(self, context: AgentContext) -> tuple[list[dict], list[dict]]:
        history, items = super()._prepared_history(context)
        # Items of the retained earlier decisions come first; the current decision follows.
        earlier = sum(len(group["items"]) for group in history[:-1])
        bound, digest, replay = _bind(context.instructions, _tools(context), items, earlier)
        messages, positions = _coalesce(bound)
        first = replay.pop("first")
        earlier_kept = replay.pop("earlier_kept")
        # messages.N.content.M: where the first stripped block would be had it been kept (the
        # prefix before it is unchanged). A turn of thinking only is dropped once stripped, so
        # N may then name a later message, or none.
        replay["first_stripped_path"] = (
            None if first is None else f"messages.{positions[first[0]]}.content.{first[1]}"
        )
        # Committed with the turn this request produces; dropped if nothing is committed.
        history[-1][PENDING] = {"binding": digest, "thinking_replay": replay,
                                "earlier_kept": earlier_kept}
        return history, messages

    def _commit(self, history: list[dict], output: list[dict]) -> None:
        pending = history[-1].pop(PENDING, {})
        turn = {"role": "assistant", "content": copy.deepcopy(output),
                BINDING: pending.get("binding")}
        super()._commit(history, [turn])

    def _append_results(self, history: list[dict], results) -> None:
        """Attach results to their originating decision as one user message."""
        if not results:
            return
        if not history:
            raise BackendConfigurationError("Tool results have no originating model call")
        items = history[-1]["items"]
        pending = self._call_ids(items) - self._result_ids(items)
        received, converted = set(), []
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
            converted.append({"type": "tool_result", "tool_use_id": call_id,
                              "content": result["output"]})
        if received != pending:
            raise BackendConfigurationError("Every pending tool call must receive a result")
        items.append({"role": "user", "content": converted})

    def _call_ids(self, items: list[dict]) -> set[str]:
        return {block.get("id") for item in items if item.get("role") == "assistant"
                for block in item["content"] if block.get("type") == "tool_use"}

    @staticmethod
    def _result_ids(items: list[dict]) -> set[str]:
        return {block.get("tool_use_id") for item in items if item.get("role") == "user"
                for block in item["content"] if block.get("type") == "tool_result"}

    def _answered(self, items: list[dict]) -> bool:
        return self._call_ids(items) == self._result_ids(items)

    def accept_reply(self, context: AgentContext, reply: AgentOutput) -> None:
        """Commit an offline scripted reply; canonical calls become tool_use blocks.

        Thinking is never fabricated. Used only by request previews, without HTTP or auth.
        """
        history, _ = self._prepared_history(context)
        output = copy.deepcopy(list(reply.output_items))
        if not output or not all(
            isinstance(block, Mapping) and block.get("type") in CONTENT_TYPES for block in output
        ):
            calls = output or copy.deepcopy(list(reply.tool_calls))
            if not all(isinstance(c, Mapping) and c.get("type") == "function_call" for c in calls):
                raise BackendError("Unsupported or mixed native Messages response content")
            output = [{"type": "tool_use", "id": call["call_id"], "name": call["name"],
                       "input": json.loads(call["arguments"])} for call in calls]
        self._reject_reused_call_ids(history[-1]["items"], _calls(output)[0])
        self._commit(history, output)

    def prepare_request(self, context: AgentContext) -> dict:
        """Preview actual native input without credentials, I/O or state changes."""
        effort = self._effort(context.reasoning_effort)
        _, messages = self._prepared_history(context)
        system = context.instructions
        if self._provider.prompt_caching:
            # One breakpoint on the static system prompt (tools render before it, so both are
            # cached) and automatic caching of the growing conversation tail.
            system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        body = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "tools": _tools(context),
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "thinking": {"type": "adaptive"},
            "max_tokens": self.max_output_tokens,
        }
        # Without an effort (``default`` and none declared) the model's own default applies.
        if effort is not None:
            body["output_config"] = {"effort": effort}
        if self._provider.prompt_caching:
            body["cache_control"] = {"type": "ephemeral"}
        return body

    def _build_http_request(self, body: dict, endpoint=None):
        url = self.path if endpoint is None else endpoint.join(self.path)
        return self._client.build_request(
            "POST", url, json=body, headers={"anthropic-version": self.api_version}
        )

    def _request_settings(self, body: dict) -> dict[str, Any]:
        return {
            "reasoning_effort": body.get("output_config", {}).get("effort"),
            "thinking_type": body["thinking"]["type"],
            "api_version": self.api_version,
            "prompt_caching": "cache_control" in body,
        }

    def _history_settings(self, history: list[dict]) -> dict[str, Any]:
        pending = history[-1][PENDING]
        return {
            "thinking_replay": copy.deepcopy(pending["thinking_replay"]),
            # Whether this request replays thinking of an earlier decision; a repair's own
            # thinking does not count, and stripped turns replay none.
            "previous_decision_reasoning_replay": pending["earlier_kept"] > 0,
        }

    def _usage(self, data: dict) -> dict[str, int]:
        """Canonical usage: ``input_tokens`` is the whole prompt, including cache tokens.

        The Messages ``input_tokens`` counts only uncached input; it is recorded as
        ``uncached_input_tokens``, and the total is reported only when every component is.
        """
        raw = data.get("usage") or {}
        if not isinstance(raw, Mapping):
            raise BackendError("Messages usage envelope must be an object")
        try:
            counts = {key: int(raw[key]) for key in (
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens",
            ) if raw.get(key) is not None}
        except (ValueError, TypeError, OverflowError) as exc:
            raise BackendError("Messages usage envelope contains invalid token counts") from exc
        usage = {target: counts[source] for source, target in (
            ("input_tokens", "uncached_input_tokens"),
            ("cache_read_input_tokens", "cached_input_tokens"),
            ("cache_creation_input_tokens", "cache_creation_input_tokens"),
            ("output_tokens", "output_tokens"),
        ) if source in counts}
        components = ("uncached_input_tokens", "cached_input_tokens", "cache_creation_input_tokens")
        if all(key in usage for key in components):
            usage["input_tokens"] = sum(usage[key] for key in components)
            if "output_tokens" in usage:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return usage

    def _decode(self, data: dict, metadata: dict) -> tuple[list[dict], list[dict], str]:
        metadata.update(
            model=data.get("model", self.model),
            # Keep provider evidence separate from the compatibility fallback above.
            response_model=data.get("model"),
            response_id=data.get("id"),
            provider_response_id=data.get("id"),
            stop_reason=data.get("stop_reason"),
        )
        details = data.get("stop_details")
        if isinstance(details, Mapping):
            metadata["stop_details"] = {key: details[key] for key in ("type", "category")
                                        if isinstance(details.get(key), str)}
        if data.get("type", "message") != "message" or data.get("role", "assistant") != (
            "assistant"
        ):
            raise BackendError("Messages endpoint returned an invalid response envelope")
        # Nothing is committed for an unusable response, so the resend is identical.
        if data.get("stop_reason") == "refusal":
            metadata["provider_refusal"] = True
            raise BackendError("Messages provider refused the request; see the recorded stop_details")
        if data.get("stop_reason") not in COMPLETED_STOP_REASONS:
            raise BackendError(
                f"Messages request ended with stop_reason {data.get('stop_reason')!r}"
            )
        content = data.get("content")
        if not isinstance(content, list):
            raise BackendError("Messages endpoint returned an invalid content array")
        calls, output_text = _calls(content)
        return content, calls, output_text

    def _output_calls(self, output: list[dict]) -> list[dict]:
        return _calls(output)[0]
