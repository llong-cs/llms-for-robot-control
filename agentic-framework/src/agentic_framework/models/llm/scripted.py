"""Scripted agents and replies for offline tests, examples and request previews."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from agentic_framework.harness.types import AgentContext, AgentOutput
from agentic_framework.models.llm.errors import BackendConfigurationError


def scripted_reply(request, payload):
    """Adapt deterministic internal fixtures into native replies, never used by a model agent."""
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
    return AgentOutput(
        {"tool_calls": calls},
        metadata={"backend": "scripted", "model": "scripted"},
        tool_calls=tuple(calls),
        output_items=tuple(copy.deepcopy(calls)),
    )


class ScriptedAgent:
    """Deterministic local agent for controller and context contract tests."""

    def __init__(self, responses: Iterable[Any]):
        self._responses = iter(responses)
        self.requests: list[AgentContext] = []
        self.closed = False

    def generate(self, request: AgentContext) -> AgentOutput:
        if self.closed:
            raise BackendConfigurationError("Scripted agent is closed")
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
        if isinstance(response, AgentOutput):
            if response.tool_calls or response.output_items:
                return copy.deepcopy(response)
            native = scripted_reply(request, response.payload)
            return replace(
                native,
                usage=response.usage,
                metadata=response.metadata,
                elapsed_s=response.elapsed_s,
            )
        if not isinstance(response, Mapping):
            raise BackendConfigurationError("Scripted responses must be AgentOutput or mappings")
        return scripted_reply(request, response)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
