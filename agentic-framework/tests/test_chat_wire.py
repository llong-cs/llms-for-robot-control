"""Characterize the Chat Completions wire of ChatAgent and the ``local`` provider, offline.

Part A pins the request: the body of a small context, the static system and tool bytes across
K and max_steps (derived from the Responses goldens and pinned), the declared tool choice, the
absence of any reasoning setting, and the provider's routing. Part B drives ``generate()``
through ``httpx.MockTransport``: acceptance and commit, the failure matrix, the optional key,
loopback and environment endpoints, and the policy's resends. Part C covers the history
window and tool-result replay. Part D covers previews with an empty environment, the request
archive and the telemetry of Chat responses.

Every exchange is mocked; no server is started and no model is loaded.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from collections.abc import MutableMapping
from pathlib import Path

import httpx
import numpy as np
import pytest
from inspect_robots.errors import PolicyStop
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from PIL import Image

from agentic_framework.configuration.arguments import parse_configuration
from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness import policy as policy_module
from agentic_framework.harness.policy import LLMMotionPolicy
from agentic_framework.harness.preview import OfflinePreviewAgent, record_preview_request
from agentic_framework.harness.types import (
    ActionSpec,
    AgentContext,
    AgentOutput,
    AugmentationConfig,
    ExecutionFeedback,
    MemoryConfig,
    ScheduleConfig,
    unpack_plan,
)
from agentic_framework.models import factory
from agentic_framework.models.llm import providers
from agentic_framework.models.llm.chat import ChatAgent
from agentic_framework.models.llm.errors import (
    BackendConfigurationError,
    BackendError,
    BackendFormatError,
    BackendOutputLimitError,
    failure_kind,
)
from agentic_framework.models.llm.http_agent import PLACEHOLDER_ORIGIN
from agentic_framework.models.llm.providers import ProviderError, local, resolve
from agentic_framework.models.llm.responses import ResponsesAgent, image_data_url
from agentic_framework.observability.api_audit import APIRequestRecorder
from agentic_framework.observability.request_telemetry import (
    canonical_usage,
    new_telemetry,
    observe_response_json,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "responses"
# The provider's variables; the shared conftest removes them from every test.
KEY_ENV = "LOCAL_LLM_API_KEY"
BASE_URL_ENV = "LOCAL_LLM_BASE_URL"
KEY = "local-test-key-0123456789"
SERVED = "robot-vlm"
SELECTOR = f"local/{SERVED}"
LOOPBACK = "http://127.0.0.1:8000/v1"
LOCAL_ROUTE = {
    "selector": SELECTOR, "provider": "local", "protocol": "chat_completions",
    "wire_model": SERVED, "endpoint": "${LOCAL_LLM_BASE_URL}", "api_key_env": KEY_ENV,
    "base_url_env": BASE_URL_ENV,
}
INSTRUCTION = "Lift the red cube."
INSTRUCTIONS = "Choose one robot tool call."
AUTH_HEADERS = frozenset({"authorization", "proxy-authorization", "api-key", "x-api-key"})
MOVE = {"deltas": {"dz": 0.001}, "note": "Raise the gripper slightly."}
TOOL = {"type": "function", "name": "move_by", "description": "Move.",
        "parameters": {"type": "object"}, "strict": False}
WIRE_TOOL = {"type": "function", "function": {"name": "move_by", "description": "Move.",
                                              "parameters": {"type": "object"}}}
USAGE = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
         "prompt_tokens_details": {"cached_tokens": 100},
         "completion_tokens_details": {"reasoning_tokens": 10}}
# Static Chat bytes (system and tools, compact JSON) for the LIBERO embodiment with H=5: the
# request-side experimental condition. Change only on purpose.
PINNED_STATIC_SHA256 = {
    "move_by": "05624c0206675d23e42b07dd9c526a43739b7f777224e0366e0115c467c40fd2",
    "move_by_chunk": "feda0e4a8bef3f8b54101750bb63502f3b8ae00f95b52ee568cbe9d48e0e5b04",
}


# ---------------------------------------------------------------------------- helpers


def _image(seed):
    """Return a distinct deterministic 4x4 RGB frame."""
    return ((np.arange(48).reshape(4, 4, 3) * (seed + 1) + seed) % 256).astype(np.uint8)


def _pixels(url):
    """Decoded RGB pixel bytes of a PNG data URL (independent of the PNG encoder)."""
    data = url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
        assert image.format == "PNG"
        return np.asarray(image.convert("RGB")).tobytes()


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _tool_call(call_id, name="move_by", arguments=None):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(MOVE if arguments is None else arguments)}}


def _native(call_id, name="move_by", arguments=None):
    """The canonical call a reply exposes for a _tool_call."""
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": json.dumps(MOVE if arguments is None else arguments)}


def _replayed(content, *calls):
    """The assistant turn a later request replays for a reply."""
    turn = {"role": "assistant", "content": content}
    if calls:
        turn["tool_calls"] = list(calls)
    return turn


def _reply(content=None, *calls, finish_reason="tool_calls", usage=USAGE, **message):
    body = {"role": "assistant", "content": content, **message}
    if calls:
        body["tool_calls"] = list(calls)
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": SERVED,
            "choices": [{"index": 0, "message": body, "finish_reason": finish_reason,
                         "logprobs": None}],
            "usage": usage}


def _decision_reply(tag):
    return _reply(None, _tool_call(f"call_{tag}"), reasoning_content=f"Thinking {tag}.")


def _result(call_id, status="accepted"):
    return {"type": "function_call_output", "call_id": call_id,
            "output": json.dumps({"status": status})}


def _tool_message(call_id, status="accepted"):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"status": status})}


def _request(index, *, decision=None, history_length=2, tool_results=(), effort="low",
             tools=(TOOL,), instructions=INSTRUCTIONS):
    text = json.dumps({"observation": index})
    return AgentContext(
        request_id=f"req-{index}",
        instructions=instructions,
        text=text,
        images={"cam": _image(index)},
        tools=tools,
        reasoning_effort=effort,
        messages=({"role": "user", "content": [{"type": "text", "text": text},
                                               {"type": "image", "name": "cam"}]},),
        history_length=history_length,
        tool_results=tuple(tool_results),
        decision_id=decision or f"decision-{index}",
    )


def _user(index):
    return {"role": "user", "content": [
        {"type": "text", "text": json.dumps({"observation": index})},
        {"type": "text", "text": "Camera: cam"},
        {"type": "image_url", "image_url": {"url": image_data_url(_image(index))}},
    ]}


class Wire:
    """A ChatAgent with the capabilities the factory applies, over scripted exchanges."""

    def __init__(self, *replies, model=SELECTOR, **options):
        self.replies = list(replies)
        self.sent: list[httpx.Request] = []
        defaults = {"max_output_tokens": resolve(model).capabilities.max_output_tokens}
        self.agent = ChatAgent(
            model, transport=httpx.MockTransport(self._handle), **{**defaults, **options}
        )

    def _handle(self, request):
        self.sent.append(request)
        if not self.replies:
            raise AssertionError("unexpected HTTP request")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    def body(self, index=-1):
        return json.loads(self.sent[index].content)


def _unexpected_request(request):
    raise AssertionError("unexpected HTTP request")


UNUSED_TRANSPORT = httpx.MockTransport(_unexpected_request)


class ForbiddenEnviron(MutableMapping):
    """Fail on any access to the process environment."""

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("the process environment was read")

    __getitem__ = __setitem__ = __delitem__ = __iter__ = __len__ = _forbidden
    get = __contains__ = copy = _forbidden


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Fail on any backoff; a test that expects backoff records it with its own patch."""

    def forbidden(seconds):
        raise AssertionError(f"unexpected backoff of {seconds} s")

    monkeypatch.setattr(policy_module, "_sleep", forbidden)


@pytest.fixture
def provider_directory(tmp_path, monkeypatch):
    """Append a directory to the provider package path and rebuild the registry around it."""
    import importlib
    import sys

    directory = tmp_path / "providers"
    directory.mkdir()
    monkeypatch.setattr(providers, "__path__", [*providers.__path__, str(directory)])
    providers.registry.cache_clear()
    loaded = set(sys.modules)
    importlib.invalidate_caches()
    yield directory
    providers.registry.cache_clear()
    for name in set(sys.modules) - loaded:
        if name.startswith(f"{providers.__name__}."):
            del sys.modules[name]


def _spec_module(directory: Path, module: str, source: str) -> None:
    import importlib

    (directory / f"{module}.py").write_text(source, encoding="utf-8")
    importlib.invalidate_caches()
    providers.registry.cache_clear()


def _observation(step):
    return Observation(
        images={"agentview": _image(step), "robot0_eye_in_hand": _image(step + 100)},
        state={"eef_pos": np.array([0.1, -0.2, 0.3 + step / 1000]),
               "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
               "gripper_width": np.array([0.04])},
        instruction=INSTRUCTION,
        extra={"env_step": step},
    )


def _policy(wire, *, schedule=None, history_length=2, **options):
    policy = LLMMotionPolicy(
        wire.agent,
        schedule or ScheduleConfig(h=5, k=5, max_steps=40),
        action_spec=ActionSpec(),
        augmentation=AugmentationConfig(memory=MemoryConfig(history_length=history_length)),
        **options,
    )
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("wire", INSTRUCTION))
    return policy


def _act(policy, step):
    """Run one accepted decision and deliver its execution receipt."""
    plan = unpack_plan(policy.act(_observation(step)))
    policy.record_execution(ExecutionFeedback(plan.plan_id, "completed", 5))


def _write_teacher(directory):
    """Write a finalized two-observation teacher recording; return its trajectory.json."""
    (directory / "images").mkdir(parents=True)
    turns = []
    for index, eef_z in enumerate((0.2, 0.25)):
        relative = f"images/agentview-{index}.png"
        Image.fromarray(_image(50 + index)).save(directory / relative)
        payload = {"instruction": "Touch the blue block.", "current_observation": {
            "state": {"eef_pos": [0.0, 0.1, eef_z], "eef_quat": [0.0, 0.0, 0.0, 1.0],
                      "gripper_width": [0.04]},
            "step_budget": {"steps_remaining": 40 - 5 * index, "max_steps": 40},
            "images": ["current/agentview"],
        }}
        turns.append({
            "kind": "llm", "observation_step": 5 * index,
            "request": {"text": json.dumps(payload)},
            "observation": {"instruction": "Touch the blue block.",
                            "images": {"agentview": relative}},
            "model_output": {"backend": {"backend": "chat_completions"}},
        })
    path = directory / "trajectory.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "metadata": {"policy_family": "llm", "observation_profile": "control"},
        "outcome": {"complete": True, "status": "failed"},
        "turns": turns,
    }))
    return path


# ------------------------------------------------------------- Part A: request shape


def test_small_context_body_and_headers():
    wire = Wire(_decision_reply("0"))
    preview = wire.agent.prepare_http_request(_request(0))
    wire.agent.generate(_request(0))
    sent = wire.sent[0]
    # Without LOCAL_LLM_BASE_URL the loopback default is used, and without a key no header.
    assert (sent.method, str(sent.url)) == ("POST", f"{LOOPBACK}/chat/completions")
    assert sent.headers["content-type"] == "application/json"
    assert not AUTH_HEADERS & set(sent.headers) and not AUTH_HEADERS & set(preview.headers)
    # A preview never reads the environment, so it names the placeholder origin.
    assert str(preview.url) == f"{PLACEHOLDER_ORIGIN}/chat/completions"
    assert preview.content == sent.content
    expected = {
        "model": SERVED,
        "messages": [{"role": "system", "content": INSTRUCTIONS}, _user(0)],
        "tools": [WIRE_TOOL],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    # Key order and bytes: compact UTF-8 JSON with no reasoning or sampling parameters.
    assert sent.content == _compact(expected).encode("utf-8")


def test_context_encoding_matches_the_responses_wire():
    """The same text, camera labels and PNG data URLs as Responses, in Chat part types."""
    context = _request(3)
    chat = ChatAgent(SELECTOR, transport=UNUSED_TRANSPORT)
    responses = ResponsesAgent("gpt-wire-test", transport=UNUSED_TRANSPORT)
    parts = chat.prepare_request(context)["messages"][1]["content"]
    native = responses.prepare_request(context)["input"][0]["content"]
    assert [part["type"] for part in parts] == ["text", "text", "image_url"]
    assert [p["text"] for p in parts if p["type"] == "text"] == [
        p["text"] for p in native if p["type"] == "input_text"
    ]
    assert parts[2]["image_url"] == {"url": native[2]["image_url"]}


class _StubAgent:
    model = "offline-scripted"

    def close(self):
        pass


def _policy_context(interface, k, max_steps):
    """The first context of a policy configured like the one the Responses goldens record."""
    schedule = ScheduleConfig(h=5, k=k, max_steps=max_steps, control_interface=interface)
    policy = LLMMotionPolicy(
        _StubAgent(), schedule, action_spec=ActionSpec(),
        augmentation=AugmentationConfig(memory=MemoryConfig(history_length=2)),
    )
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("wire", INSTRUCTION))
    return policy.prepare_request(_observation(0))


def _static(body):
    return _compact([body["messages"][0], body["tools"]]).encode("utf-8")


@pytest.mark.parametrize(
    "interface, ks", [("move_by", (1, 5, -1)), ("move_by_chunk", (1, 5))]
)
def test_static_system_and_tools_are_byte_identical_across_k_and_max_steps(interface, ks):
    agent = ChatAgent(SELECTOR, transport=UNUSED_TRANSPORT)
    bodies = [agent.prepare_request(_policy_context(interface, k, max_steps))
              for k in ks for max_steps in (6, 40, 1800)]
    assert len({_static(body) for body in bodies}) == 1
    body = bodies[0]
    # Derived from the Responses golden of the same interface: instructions and parameters.
    golden = json.loads((GOLDEN_DIR / f"{interface}_h5.json").read_text(encoding="utf-8"))
    assert body["messages"][0] == {"role": "system", "content": golden["body"]["instructions"]}
    assert body["tools"] == [
        {"type": "function", "function": {"name": tool["name"],
                                          "description": tool["description"],
                                          "parameters": tool["parameters"]}}
        for tool in golden["body"]["tools"]
    ]
    assert hashlib.sha256(_static(body)).hexdigest() == PINNED_STATIC_SHA256[interface], (
        "static Chat bytes changed; review the diff deliberately"
    )


@pytest.mark.parametrize("effort", ["low", "xhigh", "default", "lowest", "minimal", "none"])
def test_no_reasoning_effort_or_mode_is_ever_sent(effort):
    wire = Wire(_decision_reply("a"))
    body = wire.agent.prepare_request(_request(1, effort=effort))
    assert set(body) == {"model", "messages", "tools", "tool_choice", "parallel_tool_calls"}
    reply = wire.agent.generate(_request(1, effort=effort))
    metadata = reply.metadata
    assert (metadata["requested_effort"], metadata["reasoning_effort"]) == (effort, None)
    assert (metadata["reasoning_replay"], metadata["previous_decision_reasoning_replay"]) == (
        "none", False,
    )
    with pytest.raises(ValueError, match="the local provider does not support reasoning_mode"):
        parse_configuration(["--mode", "agent", "--model", SELECTOR, "--reasoning-mode", "pro"])


def test_max_output_tokens_is_sent_as_max_tokens_only_when_set():
    assert "max_tokens" not in ChatAgent(SELECTOR, transport=UNUSED_TRANSPORT).prepare_request(
        _request(1))
    body = ChatAgent(SELECTOR, max_output_tokens=512,
                     transport=UNUSED_TRANSPORT).prepare_request(_request(1))
    assert body["max_tokens"] == 512 and "max_completion_tokens" not in body


def test_tool_choice_is_a_declared_capability(provider_directory):
    assert ChatAgent(SELECTOR, transport=UNUSED_TRANSPORT).tool_choice == "required"
    assert resolve(SELECTOR).capabilities.tool_choice == "required"
    _spec_module(provider_directory, "private_auto", (
        "from agentic_framework.models.llm.providers import ProviderSpec\n"
        "PROVIDER = ProviderSpec(name='auto', protocol='chat_completions',\n"
        "                        base_url='https://auto.example.invalid/v1', tool_choice='auto')\n"
    ))
    agent = ChatAgent("auto/served", transport=UNUSED_TRANSPORT)
    assert agent.prepare_request(_request(1))["tool_choice"] == "auto"
    # An explicit constructor choice wins.
    assert ChatAgent(SELECTOR, tool_choice="auto", transport=UNUSED_TRANSPORT).tool_choice == "auto"


def test_local_provider_declaration_and_route():
    spec = providers.registry()["local"]
    assert spec is local.PROVIDER
    assert (spec.protocol, spec.base_url, spec.base_url_env, spec.base_url_default) == (
        "chat_completions", None, BASE_URL_ENV, LOOPBACK,
    )
    assert (spec.key_env, spec.auth_scheme, spec.key_required, spec.reasoning_mode) == (
        KEY_ENV, "bearer", False, False,
    )
    assert resolve(SELECTOR).record() == LOCAL_ROUTE
    # A served name is sent verbatim, including a repository-style name with slashes.
    route = resolve("Local/Org/Vision-Model-8B")
    assert (route.selector, route.wire_model) == ("local/Org/Vision-Model-8B",
                                                  "Org/Vision-Model-8B")
    with pytest.raises(ProviderError, match="names no model after local/"):
        resolve("local/")


def test_parser_and_factory_build_the_chat_agent(tmp_path):
    args = parse_configuration(["--mode", "agent", "--model", SELECTOR])
    assert (args.model, dict(args.llm_route)) == (SELECTOR, LOCAL_ROUTE)
    agent = factory.create_policy(
        args, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "a",
        ActionSpec(),
    ).agent
    assert isinstance(agent, ChatAgent)
    assert (agent.model, agent.tool_choice, agent.max_output_tokens) == (SERVED, "required", None)
    recorder = agent.request_recorder
    assert (recorder.protocol, recorder.auth_env, recorder.auth_scheme, recorder.auth_required) == (
        "chat_completions", KEY_ENV, "bearer", False)
    explicit = parse_configuration(["--mode", "agent", "--model", SELECTOR,
                                    "--max-output-tokens", "512"])
    agent = factory.create_policy(
        explicit, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "b",
        ActionSpec(),
    ).agent
    assert agent.max_output_tokens == 512


# ------------------------------------------------------- Part B: generate() over MockTransport


def test_accepted_call_is_returned_and_committed(tmp_path):
    raw = _reply("Raising first.", _tool_call("call_a"),
                 reasoning_content="The cube is below the gripper.")
    wire = Wire(raw, _decision_reply("b"), log_dir=tmp_path / "model")
    reply = wire.agent.generate(_request(1))
    assert reply.tool_calls == (_native("call_a"),)
    assert reply.payload == {"tool_calls": [_native("call_a")]}
    # The recorded output is the server's message, reasoning text included.
    assert reply.output_items == (raw["choices"][0]["message"],)
    assert reply.response_text == "Raising first."
    assert reply.usage == {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                           "cached_input_tokens": 100, "reasoning_tokens": 10}
    metadata = reply.metadata
    assert (metadata["backend"], metadata["protocol"], metadata["provider"]) == (
        "chat_completions", "chat_completions", "local",
    )
    assert (metadata["finish_reason"], metadata["response_model"], metadata["response_id"]) == (
        "tool_calls", SERVED, "chatcmpl-test",
    )
    assert (metadata["tool_choice"], metadata["endpoint"], metadata["api_key_env"]) == (
        "required", "${LOCAL_LLM_BASE_URL}", KEY_ENV,
    )
    assert metadata["history_record_format"] == "assistant_text_and_tool_calls"
    telemetry = metadata["telemetry"]["usage"]
    assert (telemetry["input_tokens"], telemetry["uncached_input_tokens"],
            telemetry["availability"]) == (120, 20, "reported")

    second = wire.agent.generate(_request(2, tool_results=[_result("call_a")]))
    assert second.metadata["previous_decision_ids"] == ["decision-1"]
    # The replay is the observation, the visible text and calls, then the tool result.
    assert wire.body(1)["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        _user(1),
        _replayed("Raising first.", _tool_call("call_a")),
        _tool_message("call_a"),
        _user(2),
    ]
    assert b"The cube is below the gripper" not in wire.sent[1].content
    records = (tmp_path / "model" / "decisions.jsonl").read_text()
    assert '"finish_reason": "tool_calls"' in records


@pytest.mark.parametrize(
    "status", [400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504]
)
def test_every_non_200_status_is_a_resent_transport_failure(status):
    secret = "sk-local-echoedsecret0123"
    echoed = {"object": "error", "message": f"invalid key {secret}", "type": "BadRequestError",
              "code": status}
    rejection = httpx.Response(status, headers={"Retry-After": "7"}, json=echoed)
    wire = Wire(rejection, _decision_reply("ok"))
    request = _request(1)
    with pytest.raises(BackendError) as caught:
        wire.agent.generate(request)
    error = caught.value
    assert type(error) is BackendError and failure_kind(error) == "transport"
    assert str(error) == f"Chat Completions endpoint rejected request with HTTP {status}"
    assert error.metadata["http_status"] == status
    assert error.metadata.get("retry_after_s", "absent") == (
        7.0 if status in (429, 503) else "absent"
    )
    # The rejected body is never kept: an echoed credential cannot reach the decision log.
    assert secret not in repr((str(error), error.metadata, error.usage))
    assert wire.agent.generate(request).tool_calls == (_native("call_ok"),)
    assert wire.sent[0].content == wire.sent[1].content


def _choice(**fields):
    reply = _reply(None, _tool_call("call_x"))
    reply["choices"][0].update(fields)
    return reply


@pytest.mark.parametrize("reply, message", [
    (httpx.Response(200, content=b"{not json", headers={"content-type": "application/json"}),
     "malformed HTTP JSON envelope"),
    (httpx.Response(200, json=[_tool_call("call_x")]), "invalid HTTP response envelope"),
    (_reply(None, _tool_call("call_x"), finish_reason="length"), "finish_reason 'length'"),
    (_reply(None, _tool_call("call_x"), finish_reason="max_tokens"),
     "finish_reason 'max_tokens'"),
    (_reply("Filtered.", finish_reason="content_filter"), "finish_reason 'content_filter'"),
    (_reply("I can't help with that.", finish_reason="stop", refusal="I can't help with that."),
     "Model refused the robot decision request"),
    (_reply(None, _tool_call("call_x"), refusal=["no"]), "refusal must be string text or null"),
    ({**_reply(None, _tool_call("call_x")), "object": "chat.completion.chunk"},
     "invalid response envelope"),
    ({**_reply(), "choices": []}, "invalid choices array"),
    ({**_reply(), "choices": None}, "invalid choices array"),
    ({**_reply(), "choices": ["x"]}, "invalid choices array"),
    ({**_reply(), "choices": _reply()["choices"] * 2}, "invalid choices array"),
    (_choice(message=None), "lacks a message object"),
    (_reply(None, _tool_call("call_x"), role="user"), "requires an assistant message"),
    (_reply([{"type": "text", "text": "x"}], _tool_call("call_x")),
     "content must be string text or null"),
    (_reply(None, finish_reason="tool_calls", tool_calls={"id": "call_x"}),
     "tool_calls must be an array"),
    (_reply(None, {**_tool_call("call_x"), "id": ""}), "unique ID, a function name"),
    (_reply(None, {"type": "function", "function": _tool_call("x")["function"]}),
     "unique ID, a function name"),
    (_reply(None, _tool_call("call_x"), _tool_call("call_x")), "unique ID, a function name"),
    (_reply(None, {**_tool_call("call_x"), "type": "custom"}), "unique ID, a function name"),
    (_reply(None, {"id": "call_x", "type": "function",
                   "function": {"name": "move_by", "arguments": MOVE}}),
     "string arguments"),
    (_reply(None, {"id": "call_x", "type": "function",
                   "function": {"name": "", "arguments": "{}"}}), "a function name"),
    (_reply(None, finish_reason="stop",
            function_call={"name": "move_by", "arguments": "{}"}), "legacy function_call"),
    (_reply(None, _tool_call("call_x"), usage={"prompt_tokens": "many"}),
     "invalid token counts"),
    (_reply(None, _tool_call("call_x"), usage={"prompt_tokens_details": [1]}),
     "invalid token counts"),
    (_reply(None, _tool_call("call_x"), usage=["tokens"]), "usage envelope must be an object"),
], ids=["malformed_json", "array_envelope", "length", "max_tokens", "content_filter",
        "refusal", "non_string_refusal", "chunk_object",
        "no_choices", "null_choices", "non_object_choice", "two_choices", "no_message",
        "user_role", "content_parts", "tool_calls_object", "empty_id", "missing_id",
        "duplicate_ids", "custom_tool_type", "object_arguments", "empty_name",
        "legacy_function_call", "invalid_usage", "non_object_details", "non_object_usage"])
def test_unusable_response_is_a_transport_failure_that_commits_nothing(reply, message):
    wire = Wire(reply, _decision_reply("ok"), _decision_reply("next"))
    request = _request(1)
    with pytest.raises(BackendError, match=re.escape(message)) as caught:
        wire.agent.generate(request)
    assert type(caught.value) is BackendError and failure_kind(caught.value) == "transport"
    assert wire.agent._history == []
    # Nothing was committed: the resend is byte-identical and the next decision is clean.
    wire.agent.generate(request)
    assert wire.sent[0].content == wire.sent[1].content
    wire.agent.generate(_request(2, tool_results=[_result("call_ok")]))
    assert [message["role"] for message in wire.body()["messages"]] == [
        "system", "user", "assistant", "tool", "user",
    ]


@pytest.mark.parametrize("reason", ["stop", "function_call", "eos_token", "abort", None])
def test_any_other_finish_reason_is_accepted_and_recorded(reason):
    wire = Wire(_reply(None, _tool_call("call_x"), finish_reason=reason))
    reply = wire.agent.generate(_request(1))
    assert reply.tool_calls == (_native("call_x"),)
    assert reply.metadata["finish_reason"] == reason


def test_refusal_and_finish_reason_are_recorded_without_the_text(tmp_path):
    refusal = _reply("I can't help with that.", finish_reason="stop",
                     refusal="Declined by a policy.")
    wire = Wire(refusal, _reply("Truncated", finish_reason="length"), log_dir=tmp_path / "model")
    with pytest.raises(BackendError) as caught:
        wire.agent.generate(_request(1))
    metadata = caught.value.metadata
    assert metadata["provider_refusal"] is True and metadata["finish_reason"] == "stop"
    assert metadata["telemetry"]["response"]["finish_reason"] == "refusal"
    with pytest.raises(BackendError) as caught:
        wire.agent.generate(_request(1))
    assert caught.value.metadata["finish_reason"] == "length"
    assert caught.value.metadata["telemetry"]["response"]["finish_reason"] == "length"
    record = (tmp_path / "model" / "decisions.jsonl").read_text()
    assert "Declined by a policy" not in record and "I can't help" not in record


@pytest.mark.parametrize("content, calls, text", [
    ("Let me think about it.", [], "Let me think about it."),
    (None, [_tool_call("call_x"), _tool_call("call_y")], ""),
    ("Using a shell.", [_tool_call("call_x", name="shell")], "Using a shell."),
], ids=["no_call", "two_calls", "unknown_tool"])
def test_wrong_call_count_is_a_format_error_after_commit(content, calls, text):
    finish = "tool_calls" if calls else "stop"
    wire = Wire(_reply(content, *calls, finish_reason=finish))
    first = _request(1, decision="d1")
    with pytest.raises(BackendFormatError) as caught:
        wire.agent.generate(first)
    error = caught.value
    assert type(error) is BackendFormatError and failure_kind(error) == "format"
    assert str(error) == "Call exactly one of the provided robot function tools"
    natives = [{"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                "arguments": call["function"]["arguments"]} for call in calls]
    assert (error.metadata["tool_calls"], error.response_text) == (natives, text)
    # The completed output was committed: a repair replays it and must answer its calls.
    if calls:
        with pytest.raises(BackendConfigurationError, match="Pending tool calls"):
            wire.agent.prepare_request(_request(2, decision="d1"))
    rejections = [_result(call["id"], "rejected") for call in calls]
    repair = wire.agent.prepare_request(_request(2, decision="d1", tool_results=rejections))
    assert repair["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        _user(1),
        _replayed(content, *calls),
        *[_tool_message(call["id"], "rejected") for call in calls],
        _user(2),
    ]


def test_an_empty_reply_is_replayed_as_empty_text():
    wire = Wire(_reply(None, finish_reason="stop"))
    with pytest.raises(BackendFormatError):
        wire.agent.generate(_request(1, decision="d1"))
    repair = wire.agent.prepare_request(_request(2, decision="d1"))
    assert repair["messages"][2] == {"role": "assistant", "content": ""}


@pytest.mark.parametrize("excess", [0, 1])
def test_output_limit_counts_visible_text_and_arguments(excess):
    text = "Raising now."
    observed = len(text) + len(json.dumps(MOVE))
    wire = Wire(_reply(text, _tool_call("call_a"), reasoning_content="x" * 1000),
                max_response_chars=observed - excess)
    if not excess:
        assert wire.agent.generate(_request(1)).tool_calls == (_native("call_a"),)
        return
    with pytest.raises(BackendOutputLimitError) as caught:
        wire.agent.generate(_request(1))
    error = caught.value
    assert (error.category, failure_kind(error)) == ("output_limit", "format")
    assert error.metadata["output_chars_observed"] == observed
    assert error.metadata["tool_calls"] == [_native("call_a")]
    with pytest.raises(BackendConfigurationError, match="Pending tool calls"):
        wire.agent.prepare_request(_request(2, decision="decision-1"))


def test_call_id_reused_by_a_later_attempt_is_rejected_without_commit():
    wire = Wire(
        _reply(None, _tool_call("call_x"), _tool_call("call_y")),
        _reply(None, _tool_call("call_x")),
        _reply(None, _tool_call("call_z")),
    )
    with pytest.raises(BackendFormatError):
        wire.agent.generate(_request(1, decision="d1"))
    rejections = [_result("call_x", "rejected"), _result("call_y", "rejected")]
    repair = _request(2, decision="d1", tool_results=rejections)
    with pytest.raises(BackendError, match="reused a function call ID") as caught:
        wire.agent.generate(repair)
    assert type(caught.value) is BackendError and failure_kind(caught.value) == "transport"
    assert wire.agent.generate(repair).tool_calls == (_native("call_z"),)
    assert wire.sent[1].content == wire.sent[2].content


def _failing_recorder(request, http_request):
    raise OSError("audit disk full")


@pytest.mark.parametrize("options, fields, key, closed", [
    ({}, {}, "two words", False),
    ({}, {}, "kéy-0123456789", False),
    ({}, {}, None, True),
    ({}, {"tools": ()}, None, False),
    ({}, {"tool_results": [_result("call_unknown")]}, None, False),
    ({"request_recorder": _failing_recorder}, {}, None, False),
], ids=["key_with_space", "non_ascii_key", "closed", "no_tools",
        "orphan_tool_result", "failing_recorder"])
def test_local_configuration_errors_send_nothing(monkeypatch, options, fields, key, closed):
    if key is not None:
        monkeypatch.setenv(KEY_ENV, key)
    wire = Wire(_decision_reply("unused"), **options)
    if closed:
        wire.agent.close()
    with pytest.raises(BackendConfigurationError) as caught:
        wire.agent.generate(_request(1, **fields))
    assert failure_kind(caught.value) == "configuration"
    assert wire.sent == []
    if key is not None:
        assert str(caught.value).startswith(f"{KEY_ENV} is malformed") and key not in str(
            caught.value)


def test_optional_key_is_sent_as_a_bearer_token_only_when_set(monkeypatch, tmp_path):
    wire = Wire(_decision_reply("a"), _decision_reply("b"), log_dir=tmp_path / "model")
    wire.agent.generate(_request(1))
    assert not AUTH_HEADERS & set(wire.sent[0].headers)
    monkeypatch.setenv(KEY_ENV, KEY)
    wire.agent.generate(_request(2, tool_results=[_result("call_a")]))
    assert wire.sent[1].headers["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in wire.sent[1].headers
    # The key is never logged.
    assert KEY not in (tmp_path / "model" / "decisions.jsonl").read_text()
    # Unset again: the next request carries no stale header.
    monkeypatch.delenv(KEY_ENV)
    wire.replies.append(_decision_reply("c"))
    wire.agent.generate(_request(3, tool_results=[_result("call_b")]))
    assert not AUTH_HEADERS & set(wire.sent[2].headers)


@pytest.mark.parametrize("configured, sent_url", [
    (None, f"{LOOPBACK}/chat/completions"),
    ("http://localhost:9000/v1/", "http://localhost:9000/v1/chat/completions"),
    ("https://vlm.example.invalid/serve/v1", "https://vlm.example.invalid/serve/v1/chat/completions"),
], ids=["default", "localhost", "remote"])
def test_endpoints_are_sent_and_archived_as_configured(monkeypatch, tmp_path, configured, sent_url):
    if configured is not None:
        monkeypatch.setenv(BASE_URL_ENV, configured)
    archive = tmp_path / "api-requests"
    recorder = APIRequestRecorder(archive, protocol="chat_completions", auth_env=KEY_ENV,
                                  auth_required=False)
    wire = Wire(_decision_reply("a"), request_recorder=recorder, log_dir=tmp_path / "model")
    reply = wire.agent.generate(_request(1))
    assert str(wire.sent[0].url) == sent_url
    manifest = json.loads(next(archive.rglob("manifest.json")).read_text())
    row = json.loads((archive / "requests.jsonl").read_text())
    assert manifest["url"] == row["url"] == sent_url
    assert (manifest["protocol"], manifest["auth_required"]) == ("chat_completions", False)
    assert reply.metadata["endpoint"] == "${LOCAL_LLM_BASE_URL}"


@pytest.mark.parametrize("failures, sleeps, sources", [
    ([(503, {"Retry-After": "30"}), (500, {})], [30.0, 2.25], ["retry_after", "backoff"]),
    ([(429, {})], [1.125], ["backoff"]),
], ids=["unavailable_then_server_error", "rate_limited_without_retry_after"])
def test_policy_resends_http_failures_unchanged_after_backoff(
    monkeypatch, failures, sleeps, sources
):
    slept = []
    monkeypatch.setattr(policy_module, "_sleep", slept.append)
    monkeypatch.setattr(policy_module, "_jitter", lambda: 0.5)
    rejections = [httpx.Response(status, headers=headers, json={"object": "error"})
                  for status, headers in failures]
    wire = Wire(_decision_reply("1"), *rejections, _decision_reply("2"))
    policy = _policy(wire, max_attempts=3)
    _act(policy, 0)
    _act(policy, 5)
    assert slept == sleeps
    attempts = policy.decisions[1:]
    assert [d["retry_wait_source"] for d in attempts] == [None, *sources]
    # Every attempt of the decision resent the same bytes, carrying the pending result once.
    assert len({request.content for request in wire.sent[1:]}) == 1
    assert [m["tool_call_id"] for m in wire.body()["messages"] if m["role"] == "tool"] == [
        "call_1"
    ]


def test_exhausted_attempts_discard_the_decision(monkeypatch):
    monkeypatch.setattr(policy_module, "_sleep", lambda seconds: None)
    wire = Wire(_decision_reply("1"), *[_reply("No call.", finish_reason="length")] * 3)
    policy = _policy(wire, max_attempts=3)
    _act(policy, 0)
    with pytest.raises(PolicyStop) as caught:
        policy.act(_observation(5))
    assert caught.value.name == "max_attempts"
    assert [d.get("error_kind") for d in policy.decisions[1:]] == ["transport"] * 3


# ------------------------------------------------------- Part C: history window and replay


def _run_decisions(wire, count, *, history_length, first=0):
    for index in range(first, first + count):
        results = [_result(f"call_{index - 1}")] if index else []
        wire.agent.generate(_request(index, history_length=history_length,
                                     tool_results=results))


def test_window_replays_complete_decisions_with_their_tool_results():
    wire = Wire(*(_decision_reply(str(n)) for n in range(4)))
    _run_decisions(wire, 4, history_length=2)
    body = wire.body()
    # Decision 0 left the window; decisions 1 and 2 replay whole, with their results.
    assert body["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        _user(1), _replayed(None, _tool_call("call_1")), _tool_message("call_1"),
        _user(2), _replayed(None, _tool_call("call_2")), _tool_message("call_2"),
        _user(3),
    ]
    assert all("reasoning_content" not in message for message in body["messages"])


def _responses_reply(*calls):
    return {"id": "resp_1", "object": "response", "model": "gpt-wire-test", "status": "completed",
            "output": [{"type": "function_call", "id": f"fc_{c}", "call_id": c, "name": "move_by",
                        "arguments": json.dumps(MOVE), "status": "completed"} for c in calls],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


@pytest.mark.parametrize("history_length", [-1, 0, 1, 2, 5])
def test_window_selects_the_same_decisions_as_the_responses_wire(monkeypatch, history_length):
    """The window counts complete decisions, and a repair replays its own attempt, on both."""
    monkeypatch.setenv("OPENAI_API_KEY", "responses-test-key-0123456789")
    script = [("d0", ["c0"], []), ("d1", ["c1a", "c1b"], ["c0"]),
              ("d1", ["c1c"], ["c1a", "c1b"]), ("d2", ["c2"], ["c1c"]), ("d3", ["c3"], ["c2"])]
    chat = Wire(*(_reply(None, *(_tool_call(c) for c in calls)) for _, calls, _ in script))
    replies = [httpx.Response(200, json=_responses_reply(*calls)) for _, calls, _ in script]
    sent = []

    def handle(request):
        sent.append(request)
        return replies.pop(0)

    responses = ResponsesAgent("gpt-wire-test", transport=httpx.MockTransport(handle))
    observed = {"chat": [], "responses": []}
    for index, (decision, calls, owed) in enumerate(script):
        status = "rejected" if decision == "d1" and index == 2 else "accepted"
        request = _request(index, decision=decision, history_length=history_length,
                           tool_results=[_result(c, status) for c in owed])
        for name, agent in (("chat", chat.agent), ("responses", responses)):
            try:
                reply = agent.generate(request)
            except BackendFormatError as exc:
                metadata = exc.metadata
            else:
                metadata = reply.metadata
            observed[name].append(metadata["previous_decision_ids"])
    assert observed["chat"] == observed["responses"]
    # Both replay the same observations, in the same order, in every request.
    for chat_request, responses_request in zip(chat.sent, sent, strict=True):
        chat_users = [json.loads(m["content"][0]["text"])["observation"]
                      for m in json.loads(chat_request.content)["messages"]
                      if m["role"] == "user"]
        responses_users = [json.loads(item["content"][0]["text"])["observation"]
                           for item in json.loads(responses_request.content)["input"]
                           if item.get("role") == "user"]
        assert chat_users == responses_users


def test_history_zero_replays_nothing_but_a_repair_replays_its_attempt():
    wire = Wire(
        _decision_reply("0"),
        _reply("No call yet.", finish_reason="stop"),
        _decision_reply("1b"),
    )
    wire.agent.generate(_request(0, history_length=0))
    with pytest.raises(BackendFormatError):
        wire.agent.generate(_request(1, decision="d1", history_length=0,
                                     tool_results=[_result("call_0")]))
    assert [m["role"] for m in wire.body()["messages"]] == ["system", "user"]
    wire.agent.generate(_request(2, decision="d1", history_length=0))
    assert wire.body()["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        _user(1), _replayed("No call yet."), _user(2),
    ]


def test_policy_receipts_are_replayed_as_tool_messages():
    wire = Wire(*(_decision_reply(str(n)) for n in range(3)))
    policy = _policy(wire)
    for step in (0, 5, 10):
        _act(policy, step)
    messages = wire.body()["messages"]
    tools = [message for message in messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tools] == ["call_0", "call_1"]
    for message in tools:
        # Only the plan's acceptance reaches the model, never executor diagnostics.
        receipt = json.loads(message["content"])
        assert set(receipt) == {"plan_id", "status"} and receipt["status"] == "accepted"
    # Each tool message directly follows the assistant turn that made the call.
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[index - 1]["tool_calls"][0]["id"] == message["tool_call_id"]


def test_teacher_is_sent_once_and_moves_into_the_next_retained_observation(tmp_path):
    teacher = _write_teacher(tmp_path / "teacher")
    wire = Wire(*(_decision_reply(str(n)) for n in range(3)))
    policy = _policy(wire, history_length=1, demo=True, demo_path=teacher)
    for step in (0, 5):
        _act(policy, step)
    preview = wire.agent.prepare_http_request(policy.prepare_request(_observation(10)))
    body = json.loads(preview.content)
    # Decision 0 is evicted: the teacher moves into the next retained observation.
    users = [message for message in body["messages"] if message["role"] == "user"]
    assert [sum('"demonstration"' in part.get("text", "") for part in user["content"])
            for user in users] == [1, 0]
    frames = [_pixels(part["image_url"]["url"]) for user in users for part in user["content"]
              if part["type"] == "image_url"]
    for seed in (50, 51):
        assert frames.count(_image(seed).tobytes()) == 1
    _act(policy, 10)
    assert wire.sent[-1].content == preview.content


# ------------------------------------------- Part D: previews, archives and response telemetry


def test_request_preview_is_auth_free_and_reads_no_environment(monkeypatch):
    wire = Wire(_decision_reply("a"), _decision_reply("b"))
    wire.agent.generate(_request(1))
    following = _request(2, tool_results=[_result("call_a")])
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", ForbiddenEnviron())
        preview = wire.agent.prepare_http_request(following)
        again = wire.agent.prepare_http_request(following)
    assert not AUTH_HEADERS & set(preview.headers)
    assert str(preview.url) == f"{PLACEHOLDER_ORIGIN}/chat/completions"
    assert preview.content == again.content
    monkeypatch.setenv(KEY_ENV, KEY)
    wire.agent.generate(following)
    assert wire.sent[-1].content == preview.content


def test_preview_policy_archives_a_request_with_an_empty_environment(tmp_path, monkeypatch):
    for name in list(os.environ):
        monkeypatch.delenv(name)
    args = parse_configuration(["--mode", "preview", "--model", SELECTOR])
    policy = factory.create_policy(
        args, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "trial",
        ActionSpec(),
    )
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("wire", INSTRUCTION))
    request = policy.prepare_request(_observation(0))
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", ForbiddenEnviron())
        artifact = record_preview_request(policy.agent, request)
    directory = Path(artifact["request_record_dir"])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert (manifest["mode"], manifest["protocol"], manifest["url"]) == (
        "preview", "chat_completions", f"{PLACEHOLDER_ORIGIN}/chat/completions",
    )
    assert (manifest["auth_env"], manifest["auth_scheme"], manifest["auth_required"]) == (
        KEY_ENV, "bearer", False,
    )
    assert [image["camera_label"] for image in manifest["images"]] == [
        "current/agentview", "current/robot0_eye_in_hand",
    ]
    assert [image["message_index"] for image in manifest["images"]] == [1, 1]
    readable = (directory / "request.md").read_text()
    assert "Authorization: Bearer ${LOCAL_LLM_API_KEY}" in readable
    assert "sent only when LOCAL_LLM_API_KEY is set" in readable
    assert "### Message 0 (system)" in readable and "### Message 1 (user)" in readable
    body = json.loads((directory / "request.body.json").read_bytes())
    assert body["model"] == SERVED and body["messages"][0]["role"] == "system"
    data = body["messages"][1]["content"][-1]["image_url"]["url"].split(",", 1)[1]
    assert data not in readable
    policy.close()


def test_scripted_preview_replies_commit_assistant_tool_calls(tmp_path):
    recorder = APIRequestRecorder(tmp_path / "archive", mode="preview",
                                  protocol="chat_completions", auth_env=KEY_ENV,
                                  auth_required=False)
    agent = ChatAgent(SELECTOR, request_recorder=recorder, transport=UNUSED_TRANSPORT)
    preview = OfflinePreviewAgent(agent)
    policy = LLMMotionPolicy(preview, ScheduleConfig(h=5, k=5, max_steps=40),
                             action_spec=ActionSpec(),
                             augmentation=AugmentationConfig(memory=MemoryConfig(history_length=2)))
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("wire", INSTRUCTION))
    _act(policy, 0)
    _act(policy, 5)
    bodies = [json.loads(path.read_bytes())
              for path in sorted((tmp_path / "archive").glob("*/request.body.json"))]
    assert len(bodies) == 2
    assistant, result = bodies[1]["messages"][2:4]
    assert (assistant["role"], assistant["content"]) == ("assistant", None)
    assert [call["id"] for call in assistant["tool_calls"]] == ["preview-1"]
    assert (result["role"], result["tool_call_id"]) == ("tool", "preview-1")


def test_preview_commit_rejects_a_reused_call_id_without_changing_history():
    agent = ChatAgent(SELECTOR, transport=UNUSED_TRANSPORT)

    def output(*items):
        return AgentOutput(payload={}, usage={}, metadata={}, elapsed_s=0.0, tool_calls=(),
                           output_items=tuple(items), response_text="")

    agent.accept_reply(_request(1, decision="d1"),
                       output(_native("call_x"), _native("call_y")))
    rejections = [_result("call_x", "rejected"), _result("call_y", "rejected")]
    repair = _request(2, decision="d1", tool_results=rejections)
    before = _compact(agent.prepare_request(repair))
    with pytest.raises(BackendError, match="reused a function call ID"):
        agent.accept_reply(repair, output(_replayed(None, _tool_call("call_y"))))
    assert _compact(agent.prepare_request(repair)) == before
    with pytest.raises(BackendError, match="Unsupported or mixed"):
        agent.accept_reply(repair, output({"type": "reasoning"}))


def _archive(directory, body, protocol="chat_completions"):
    recorder = APIRequestRecorder(directory, protocol=protocol, auth_env=KEY_ENV)
    request = httpx.Request("POST", f"{LOOPBACK}/chat/completions", json=body)
    return recorder(_request(1), request)


def test_archive_validates_chat_bodies_by_protocol(tmp_path):
    wire = Wire(_reply("Raising.", _tool_call("call_a")), _decision_reply("b"))
    wire.agent.generate(_request(1))
    wire.agent.generate(_request(2, tool_results=[_result("call_a")]))
    body = wire.body()
    assert _archive(tmp_path / "valid", body)["request_sequence"] == 1
    manifest = json.loads(next((tmp_path / "valid").rglob("manifest.json")).read_text())
    assert [(image["message_index"], image["part_index"]) for image in manifest["images"]] == [
        (1, 2), (4, 2),
    ]
    system = body["messages"][0]
    invalid = [
        {**body, "messages": [system, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://images.example.invalid/a.png"}}]}]},
        {**body, "messages": [system, {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {}}]}]},
        {key: value for key, value in body.items() if key != "messages"},
    ]
    for index, candidate in enumerate(invalid):
        with pytest.raises(ValueError):
            _archive(tmp_path / f"invalid-{index}", candidate)
    # Recognized by its keys, a Chat body would be validated as a Messages body and refused.
    for protocol in ("messages", None):
        with pytest.raises(ValueError, match="Messages entries require user or assistant role"):
            _archive(tmp_path / f"as-{protocol}", body, protocol)


@pytest.mark.parametrize("raw, expected, issues", [
    (USAGE, {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
             "cached_input_tokens": 100, "uncached_input_tokens": 20, "reasoning_tokens": 10,
             "availability": "reported"}, []),
    ({"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55,
      "prompt_tokens_details": None, "completion_tokens_details": None},
     {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55, "cached_input_tokens": None,
      "uncached_input_tokens": None, "reasoning_tokens": None, "availability": "reported"}, []),
    ({"prompt_tokens": 50, "completion_tokens": 5},
     {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55, "availability": "reported"}, []),
    ({"prompt_tokens": 50, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 60}},
     {"input_tokens": 50, "cached_input_tokens": None, "uncached_input_tokens": None,
      "availability": "partial"}, ["cached_input_tokens exceeds input_tokens"]),
], ids=["complete", "null_details", "no_total", "cache_exceeds_prompt"])
def test_canonical_usage_maps_prompt_and_completion_tokens(raw, expected, issues):
    usage = canonical_usage(raw, "chat_completions")
    assert {key: usage[key] for key in expected} == expected
    assert usage["issues"] == issues


def test_null_details_stay_an_issue_on_the_other_protocols():
    usage = canonical_usage({"input_tokens": 5, "output_tokens": 1,
                             "input_tokens_details": None}, "responses")
    assert usage["issues"] == [
        "input_tokens_details.cached_tokens: token details are not an object",
        "input_tokens_details.cache_write_tokens: token details are not an object",
    ]


def test_response_telemetry_reads_the_first_choice():
    telemetry = new_telemetry("chat_completions")
    observe_response_json(telemetry, _reply(None, _tool_call("call_x"), finish_reason="length"))
    assert telemetry["response"]["finish_reason"] == "length"
    assert (telemetry["response"]["id"], telemetry["response"]["model"]) == (
        "chatcmpl-test", SERVED,
    )
    assert telemetry["usage"]["input_tokens"] == 120
    refused = new_telemetry("chat_completions")
    observe_response_json(refused, _reply("No.", finish_reason="stop", refusal="No."))
    assert refused["response"]["finish_reason"] == "refusal"
