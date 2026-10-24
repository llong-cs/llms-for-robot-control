"""The OpenAI-compatible Chat Completions wire: request encoding, history replay and decoding.

A request carries the static instructions as the first, ``system`` message, then the replayed
decisions and the current observation. Observations are ``user`` messages of ``text`` and
``image_url`` parts (inline PNG data URLs), each image preceded by the same camera label the
Responses wire sends. A replayed decision is its observation, the assistant turn with its
``tool_calls`` and one ``tool`` message per call carrying the execution result under
``tool_call_id``. Function tools are sent as ``{"type": "function", "function": {...}}`` with
the declared ``tool_choice`` (``required`` unless the provider declares ``auto``) and parallel
calls disabled.

Neither a reasoning effort nor a reasoning mode is sent. Reasoning text a server returns (for
example ``reasoning_content``) stays in the recorded output, but the history keeps only the
assistant turn's visible text and tool calls, so no reasoning is ever replayed.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from agentic_framework.harness.types import AgentContext, AgentOutput
from agentic_framework.models.llm.demo_transport import deduplicate_demonstrations
from agentic_framework.models.llm.errors import BackendConfigurationError, BackendError
from agentic_framework.models.llm.http_agent import HttpAgentModel
from agentic_framework.models.llm.responses import _message_content

DEFAULT_TOOL_CHOICE = "required"
# A response cut short by an output limit or a content filter is unusable. Every other
# finish_reason, including null or a server-specific value, is left to the tool-call checks.
UNUSABLE_FINISH_REASONS = frozenset({"length", "max_tokens", "content_filter"})


def _encode(context: AgentContext) -> list[dict]:
    """Encode the context's ordered messages as Chat text and image_url parts."""
    converted = []
    for message in _message_content(context):
        content = []
        for part in message["content"]:
            if part["type"] == "input_text":
                content.append({"type": "text", "text": part["text"]})
            else:
                content.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
        converted.append({"role": message["role"], "content": content})
    return converted


def _tools(context: AgentContext) -> list[dict]:
    if not context.tools:
        raise BackendConfigurationError("Native robot decisions require function tools")
    return [{"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""),
        "parameters": copy.deepcopy(tool["parameters"]),
    }} for tool in context.tools]


def _calls(message: Any) -> tuple[list[dict], str]:
    """Validate an assistant message; return the canonical calls and the visible text."""
    if not isinstance(message, Mapping) or message.get("role", "assistant") != "assistant":
        raise BackendError("Chat Completions choice requires an assistant message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise BackendError("Chat Completions message content must be string text or null")
    if message.get("function_call") is not None:
        raise BackendError("Chat Completions returned an unsupported legacy function_call")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise BackendError("Chat Completions tool_calls must be an array")
    calls, seen = [], set()
    for call in tool_calls or ():
        function = call.get("function") if isinstance(call, Mapping) else None
        identifier = call.get("id") if isinstance(call, Mapping) else None
        if (not isinstance(function, Mapping) or call.get("type", "function") != "function"
                or not isinstance(identifier, str) or not identifier or identifier in seen
                or not isinstance(function.get("name"), str) or not function["name"]
                or not isinstance(function.get("arguments"), str)):
            raise BackendError(
                "Chat Completions tool call requires a unique ID, a function name and string "
                "arguments"
            )
        seen.add(identifier)
        calls.append({"type": "function_call", "call_id": identifier,
                      "name": function["name"], "arguments": function["arguments"]})
    return calls, content or ""


def _replayed(message: Mapping[str, Any]) -> dict:
    """The assistant turn later requests replay: its visible text and tool calls only."""
    turn = {"role": "assistant", "content": message.get("content")}
    calls = message.get("tool_calls") or []
    if calls:
        turn["tool_calls"] = [{"id": call["id"], "type": "function", "function": {
            "name": call["function"]["name"], "arguments": call["function"]["arguments"],
        }} for call in calls]
    elif turn["content"] is None:
        # An assistant message needs text or tool calls.
        turn["content"] = ""
    return turn


class ChatAgent(HttpAgentModel):
    """Native Chat Completions tools for OpenAI-compatible servers; no reasoning is replayed.

    ``local/<served name>`` sends the served name verbatim. The provider declares the endpoint,
    the optional key and the ``tool_choice`` its server accepts.
    """

    protocol = "chat_completions"
    label = "Chat Completions"
    path = "chat/completions"
    reasoning_replay = "none"

    def __init__(self, model, *, tool_choice=None, **options):
        super().__init__(model, **options)
        self.tool_choice = tool_choice or self._capabilities.tool_choice or DEFAULT_TOOL_CHOICE

    def _encode_context(self, context: AgentContext) -> list[dict]:
        return _encode(context)

    def _assemble(self, items: list[dict]) -> list[dict]:
        return deduplicate_demonstrations(items)

    def _append_results(self, history: list[dict], results) -> None:
        """Attach results to their originating decision as one tool message per call."""
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
            converted.append({"role": "tool", "tool_call_id": call_id,
                              "content": result["output"]})
        if received != pending:
            raise BackendConfigurationError("Every pending tool call must receive a result")
        items.extend(converted)

    def _call_ids(self, items: list[dict]) -> set[str]:
        return {call.get("id") for item in items if item.get("role") == "assistant"
                for call in item.get("tool_calls") or ()}

    @staticmethod
    def _result_ids(items: list[dict]) -> set[str]:
        return {item.get("tool_call_id") for item in items if item.get("role") == "tool"}

    def _answered(self, items: list[dict]) -> bool:
        return self._call_ids(items) == self._result_ids(items)

    def _commit(self, history: list[dict], output: list[dict]) -> None:
        super()._commit(history, [_replayed(message) for message in output])

    def accept_reply(self, context: AgentContext, reply: AgentOutput) -> None:
        """Commit an offline scripted reply; canonical calls become an assistant turn.

        Used only by request previews, without HTTP or auth.
        """
        history, _ = self._prepared_history(context)
        output = copy.deepcopy(list(reply.output_items))
        if not (len(output) == 1 and isinstance(output[0], Mapping)
                and output[0].get("role") == "assistant"):
            calls = output or copy.deepcopy(list(reply.tool_calls))
            if not calls or not all(
                isinstance(c, Mapping) and c.get("type") == "function_call" for c in calls
            ):
                raise BackendError("Unsupported or mixed native Chat Completions response content")
            output = [{"role": "assistant", "content": None, "tool_calls": [
                {"id": call["call_id"], "type": "function",
                 "function": {"name": call["name"], "arguments": call["arguments"]}}
                for call in calls
            ]}]
        self._reject_reused_call_ids(history[-1]["items"], self._output_calls(output))
        self._commit(history, output)

    def prepare_request(self, context: AgentContext) -> dict:
        """Preview actual native input without credentials, I/O or state changes."""
        _, items = self._prepared_history(context)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": context.instructions}, *items],
            "tools": _tools(context),
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": False,
        }
        if self.max_output_tokens is not None:
            body["max_tokens"] = self.max_output_tokens
        return body

    def _request_settings(self, body: dict) -> dict[str, Any]:
        # The configured effort is recorded as requested_effort but never sent.
        return {
            "reasoning_effort": None,
            "tool_choice": body["tool_choice"],
            "previous_decision_reasoning_replay": False,
            "history_record_format": "assistant_text_and_tool_calls",
        }

    def _usage(self, data: dict) -> dict[str, int]:
        """Canonical usage: ``prompt_tokens`` is the whole prompt, including cached tokens."""
        raw = data.get("usage") or {}
        if not isinstance(raw, Mapping):
            raise BackendError("Chat Completions usage envelope must be an object")
        try:
            usage = {target: int(raw[source]) for source, target in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ) if raw.get(source) is not None}
            for source, source_key, target in (
                ("prompt_tokens_details", "cached_tokens", "cached_input_tokens"),
                ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ):
                details = raw.get(source) or {}
                if not isinstance(details, Mapping):
                    raise TypeError("token details must be an object")
                if details.get(source_key) is not None:
                    usage[target] = int(details[source_key])
        except (ValueError, TypeError, OverflowError) as exc:
            raise BackendError(
                "Chat Completions usage envelope contains invalid token counts"
            ) from exc
        return usage

    def _decode(self, data: dict, metadata: dict) -> tuple[list[dict], list[dict], str]:
        metadata.update(
            model=data.get("model", self.model),
            # Keep provider evidence separate from the compatibility fallback above.
            response_model=data.get("model"),
            response_id=data.get("id"),
            provider_response_id=data.get("id"),
        )
        if data.get("object", "chat.completion") != "chat.completion":
            raise BackendError("Chat Completions endpoint returned an invalid response envelope")
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(
            choices[0], Mapping
        ):
            raise BackendError("Chat Completions endpoint returned an invalid choices array")
        choice = choices[0]
        reason = choice.get("finish_reason")
        metadata["finish_reason"] = reason if isinstance(reason, str) else None
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise BackendError("Chat Completions choice lacks a message object")
        refusal = message.get("refusal")
        if refusal is not None and not isinstance(refusal, str):
            raise BackendError("Chat Completions refusal must be string text or null")
        # Nothing is committed for an unusable response, so the resend is identical.
        if refusal:
            metadata["provider_refusal"] = True
            raise BackendError("Model refused the robot decision request")
        if metadata["finish_reason"] in UNUSABLE_FINISH_REASONS:
            raise BackendError(f"Chat Completions request ended with finish_reason {reason!r}")
        calls, output_text = _calls(message)
        return [copy.deepcopy(dict(message))], calls, output_text

    def _output_calls(self, output: list[dict]) -> list[dict]:
        return [call for message in output for call in _calls(message)[0]]
