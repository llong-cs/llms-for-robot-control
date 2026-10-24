"""The OpenAI Responses wire: request encoding, history items and response decoding."""

from __future__ import annotations

import base64
import copy
import io
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from agentic_framework.harness.types import AgentContext
from agentic_framework.models.llm.demo_transport import deduplicate_demonstrations
from agentic_framework.models.llm.errors import BackendConfigurationError, BackendError
from agentic_framework.models.llm.http_agent import HttpAgentModel


def image_data_url(value: np.ndarray) -> str:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Model images must be HWC uint8 RGB arrays")
    data = io.BytesIO()
    Image.fromarray(image).save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def _message_content(context: AgentContext) -> list[dict]:
    """Encode local ordered messages as Responses input items."""
    if not context.messages:
        # Compatibility for callers constructing a plain-text AgentContext.
        parts = [{"type": "text", "text": context.text}]
        parts.extend({"type": "image", "name": name} for name in context.images)
        messages = ({"role": "user", "content": parts},)
    else:
        messages = context.messages
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
                        {"type": "input_image", "image_url": image_data_url(context.images[name])},
                    ]
                )
            else:
                raise ValueError("Unsupported context content type")
        encoded.append({"role": message["role"], "content": content})
    if seen_images != set(context.images):
        raise ValueError("Every request image must appear in the ordered context")
    return encoded


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


class ResponsesAgent(HttpAgentModel):
    """Native Responses tools; ``openai/gpt-x`` and ``gpt-x`` both send ``gpt-x``.

    Requests are stateless (``store: false``) with encrypted reasoning included, so the
    selected decision records, including their reasoning items, are replayed verbatim.
    """

    protocol = "responses"
    label = "Responses"
    path = "responses"
    reasoning_replay = "selected_decision_records_and_current_repair"

    def _encode_context(self, context: AgentContext) -> list[dict]:
        return _message_content(context)

    def _assemble(self, items: list[dict]) -> list[dict]:
        return deduplicate_demonstrations(items)

    def _append_results(self, history: list[dict], results) -> None:
        """Attach results to their originating decision, never as orphan messages."""
        if not results:
            return
        if not history:
            raise BackendConfigurationError("Tool results have no originating model call")
        items = history[-1]["items"]
        pending = self._call_ids(items) - {
            item["call_id"] for item in items if item.get("type") == "function_call_output"
        }
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

    def _call_ids(self, items: list[dict]) -> set[str]:
        return {item.get("call_id") for item in items if item.get("type") == "function_call"}

    def _answered(self, items: list[dict]) -> bool:
        results = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
        return self._call_ids(items) == results

    def prepare_request(self, context: AgentContext) -> dict:
        """Preview actual native input without credentials, I/O or state changes."""
        effort = self._effort(context.reasoning_effort)
        _, items = self._prepared_history(context)
        if not context.tools:
            raise BackendConfigurationError("Native robot decisions require function tools")
        body = {
            "model": self.model,
            "instructions": context.instructions,
            "input": items,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "tools": copy.deepcopy(list(context.tools)),
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

    def _request_settings(self, body: dict) -> dict[str, Any]:
        return {
            "reasoning_effort": body.get("reasoning", {}).get("effort"),
            "reasoning_mode": body.get("reasoning", {}).get("mode"),
        }

    def _usage(self, data: dict) -> dict[str, int]:
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

    def _decode(self, data: dict, metadata: dict) -> tuple[list[dict], list[dict], str]:
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
        return output, calls, output_text

    def _output_calls(self, output: list[dict]) -> list[dict]:
        return _native_response_calls(output)[0]
