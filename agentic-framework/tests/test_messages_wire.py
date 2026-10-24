"""Characterize the Anthropic Messages wire of MessagesAgent, offline.

Part A pins the request: the body of a small context with its prompt-caching breakpoints, the
static system and tool bytes across K and max_steps (derived from the Responses goldens and
pinned), efforts passed through unchanged and the provider declaration. Part B drives ``generate()`` through ``httpx.MockTransport``: acceptance and
commit, the failure matrix, key and endpoint resolution, and the policy's resends. Part C
covers the prefix-bound thinking replay. Part D covers previews with an empty environment,
the request archive and the telemetry of Messages responses.

Every exchange is mocked; the official endpoint is only named, never contacted.
"""

from __future__ import annotations

import base64
import dataclasses
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
from agentic_framework.harness.preview import record_preview_request
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
from agentic_framework.models.llm import messages as messages_module
from agentic_framework.models.llm.errors import (
    BackendConfigurationError,
    BackendError,
    BackendFormatError,
    BackendOutputLimitError,
    failure_kind,
)
from agentic_framework.models.llm.messages import MessagesAgent
from agentic_framework.models.llm.providers import Capabilities, resolve
from agentic_framework.models.llm.responses import ResponsesAgent, image_data_url
from agentic_framework.observability.api_audit import APIRequestRecorder
from agentic_framework.observability.request_telemetry import (
    canonical_usage,
    new_telemetry,
    observe_response_json,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "responses"
# The official provider's variables; the shared conftest removes them from every test.
KEY_ENV = "ANTHROPIC_API_KEY"
BASE_URL_ENV = "ANTHROPIC_BASE_URL"
KEY = "messages-test-key-0123456789"
MODEL = "claude-opus-5-5"
SELECTORS = (MODEL, f"anthropic/{MODEL}")
OFFICIAL = "https://api.anthropic.com/v1"
EPHEMERAL = {"type": "ephemeral"}
ANTHROPIC_ROUTE = {
    "selector": f"anthropic/{MODEL}", "provider": "anthropic", "protocol": "messages",
    "wire_model": MODEL, "endpoint": OFFICIAL, "api_key_env": KEY_ENV,
    "base_url_env": BASE_URL_ENV,
}
INSTRUCTION = "Lift the red cube."
INSTRUCTIONS = "Choose one robot tool call."
AUTH_HEADERS = frozenset({"authorization", "proxy-authorization", "api-key", "x-api-key"})
MOVE ={"deltas": {"dz": 0.001}, "note": "Raise the gripper slightly."}
TOOL = {"type": "function", "name": "move_by", "description": "Move.",
        "parameters": {"type": "object"}, "strict": False}
WIRE_TOOL = {"name": "move_by", "description": "Move.", "input_schema": {"type": "object"}}
USAGE = {"input_tokens": 40, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20,
         "output_tokens": 30}
# Static Messages bytes (system and tools, compact JSON) for the LIBERO embodiment with H=5:
# the request-side experimental condition. Change only on purpose.
PINNED_STATIC_SHA256 = {
    "move_by": "0539486ce06965f4627ccdccecdbc99145ba290b022638b09de5ab78353c3b5c",
    "move_by_chunk": "6d93794113d5ce1e31694a9b0b134ef3214b175840657d4acb4d1d94374b8150",
}


# ---------------------------------------------------------------------------- helpers


def _image(seed):
    """Return a distinct deterministic 4x4 RGB frame."""
    return ((np.arange(48).reshape(4, 4, 3) * (seed + 1) + seed) % 256).astype(np.uint8)


def _png(seed):
    return image_data_url(_image(seed)).split(",", 1)[1]


def _pixels(data):
    """Decoded RGB pixel bytes of a base64 PNG block (independent of the PNG encoder)."""
    with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
        assert image.format == "PNG"
        return np.asarray(image.convert("RGB")).tobytes()


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _thinking(tag):
    return {"type": "thinking", "thinking": "", "signature": f"signature-{tag}"}


def _redacted(tag):
    return {"type": "redacted_thinking", "data": f"opaque-thinking-{tag}"}


def _text(text):
    return {"type": "text", "text": text}


def _tool_use(call_id, name="move_by", arguments=None):
    return {"type": "tool_use", "id": call_id, "name": name,
            "input": MOVE if arguments is None else arguments}


def _native(call_id, name="move_by", arguments=None):
    """The canonical call a reply exposes for a _tool_use block."""
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": json.dumps(MOVE if arguments is None else arguments)}


def _reply(*content, stop_reason="tool_use", **fields):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": MODEL,
            "content": list(content), "stop_reason": stop_reason, "stop_sequence": None,
            "stop_details": None, "usage": USAGE, **fields}


def _decision_reply(tag):
    return _reply(_thinking(tag), _text(f"Plan {tag}."), _tool_use(f"toolu_{tag}"))


def _result(call_id, status="accepted"):
    return {"type": "function_call_output", "call_id": call_id,
            "output": json.dumps({"status": status})}


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


class Wire:
    """A MessagesAgent with the capabilities the factory applies, over scripted exchanges."""

    def __init__(self, *replies, model=MODEL, **options):
        self.replies = list(replies)
        self.sent: list[httpx.Request] = []
        defaults = {"max_output_tokens": resolve(model).capabilities.max_output_tokens}
        self.agent = MessagesAgent(
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


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)


@pytest.fixture(params=SELECTORS, ids=["shorthand", "canonical"])
def selector(request):
    return request.param


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Fail on any backoff; a test that expects backoff records it with its own patch."""

    def forbidden(seconds):
        raise AssertionError(f"unexpected backoff of {seconds} s")

    monkeypatch.setattr(policy_module, "_sleep", forbidden)


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
            "model_output": {"backend": {"backend": "messages"}},
        })
    path = directory / "trajectory.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "metadata": {"policy_family": "llm", "observation_profile": "control"},
        "outcome": {"complete": True, "status": "failed"},
        "turns": turns,
    }))
    return path


def _thinking_blocks(body):
    return [block for message in body["messages"] if message["role"] == "assistant"
            for block in message["content"] if block["type"] in ("thinking", "redacted_thinking")]


# ------------------------------------------------------------- Part A: request shape


def test_small_context_body_and_headers(api_key, selector):
    wire = Wire(_decision_reply("0"), model=selector)
    preview = wire.agent.prepare_http_request(_request(0))
    wire.agent.generate(_request(0))
    sent = wire.sent[0]
    assert (sent.method, str(sent.url)) == ("POST", f"{OFFICIAL}/messages")
    assert (sent.headers["x-api-key"], sent.headers["anthropic-version"]) == (KEY, "2026-10-21")
    assert sent.headers["content-type"] == "application/json"
    # No OAuth bearer header and no beta header; the preview carries no key at all.
    assert "authorization" not in sent.headers and "anthropic-beta" not in sent.headers
    assert not AUTH_HEADERS & set(preview.headers)
    assert (str(preview.url), preview.content) == (str(sent.url), sent.content)
    expected = {
        "model": MODEL,
        # Prompt caching: a breakpoint on the static system prompt (tools render before it)
        # and top-level automatic caching of the conversation tail.
        "system": [{"type": "text", "text": INSTRUCTIONS, "cache_control": EPHEMERAL}],
        "messages": [{"role": "user", "content": [
            _text(json.dumps({"observation": 0})),
            _text("Camera: cam"),
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": _png(0)}},
        ]}],
        "tools": [WIRE_TOOL],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "thinking": {"type": "adaptive"},
        "max_tokens": 16000,
        "output_config": {"effort": "low"},
        "cache_control": EPHEMERAL,
    }
    # Key order and bytes: compact UTF-8 JSON with no sampling parameters.
    assert sent.content == _compact(expected).encode("utf-8")


def test_context_encoding_matches_the_responses_wire():
    """The same text, camera labels and PNG bytes as Responses, in Messages block types."""
    context = _request(3)
    messages = MessagesAgent(MODEL, max_output_tokens=64, transport=UNUSED_TRANSPORT)
    responses = ResponsesAgent("gpt-wire-test", transport=UNUSED_TRANSPORT)
    blocks = messages.prepare_request(context)["messages"][0]["content"]
    parts = responses.prepare_request(context)["input"][0]["content"]
    assert [block["type"] for block in blocks] == ["text", "text", "image"]
    assert [b["text"] for b in blocks if b["type"] == "text"] == [
        p["text"] for p in parts if p["type"] == "input_text"
    ]
    assert "data:image/png;base64," + blocks[2]["source"]["data"] == parts[2]["image_url"]


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
    return _compact([body["system"][0]["text"], body["tools"]]).encode("utf-8")


@pytest.mark.parametrize(
    "interface, ks", [("move_by", (1, 5, -1)), ("move_by_chunk", (1, 5))]
)
def test_static_system_and_tools_are_byte_identical_across_k_and_max_steps(interface, ks):
    agent = MessagesAgent(MODEL, max_output_tokens=16000, transport=UNUSED_TRANSPORT)
    bodies = [agent.prepare_request(_policy_context(interface, k, max_steps))
              for k in ks for max_steps in (6, 40, 1800)]
    assert len({_static(body) for body in bodies}) == 1
    body = bodies[0]
    # Derived from the Responses golden of the same interface: instructions and parameters.
    golden = json.loads((GOLDEN_DIR / f"{interface}_h5.json").read_text(encoding="utf-8"))
    assert body["system"] == [{"type": "text", "text": golden["body"]["instructions"],
                               "cache_control": EPHEMERAL}]
    assert body["tools"] == [
        {"name": tool["name"], "description": tool["description"],
         "input_schema": tool["parameters"]}
        for tool in golden["body"]["tools"]
    ]
    assert all("strict" not in tool for tool in body["tools"])
    assert hashlib.sha256(_static(body)).hexdigest() == PINNED_STATIC_SHA256[interface], (
        "static Messages bytes changed; review the diff deliberately"
    )


@pytest.mark.parametrize("model, effort", [
    ("claude-opus-5-5", "low"),
    ("claude-fable-5-1", "xhigh"),
    ("claude-sonnet-5-5", "max"),
    # Not an effort Claude accepts: sent unchanged, so the provider's error surfaces.
    ("claude-opus-5-5", "minimal"),
])
def test_configured_effort_is_sent_unchanged(model, effort):
    body = Wire(model=model).agent.prepare_request(_request(1, effort=effort))
    assert body["output_config"] == {"effort": effort}
    assert body["thinking"] == {"type": "adaptive"}


def test_a_provider_without_prompt_caching_sends_no_breakpoints():
    route = resolve(MODEL)
    route = dataclasses.replace(route, spec=dataclasses.replace(route.spec, prompt_caching=False))
    body = MessagesAgent(route, max_output_tokens=16000,
                         transport=UNUSED_TRANSPORT).prepare_request(_request(1))
    assert body["system"] == INSTRUCTIONS and "cache_control" not in body
    assert "cache_control" not in json.dumps(body)


def test_default_effort_sends_no_output_config():
    body = Wire().agent.prepare_request(_request(1, effort="default"))
    assert "output_config" not in body and body["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("options, effort", [
    ({}, "lowest"),
    ({"supported_efforts": ("low", "high")}, "minimal"),
], ids=["lowest_without_supported_efforts", "outside_supported_efforts"])
def test_unresolvable_effort_sends_nothing(api_key, options, effort):
    wire = Wire(_decision_reply("unused"), **options)
    with pytest.raises(BackendConfigurationError) as caught:
        wire.agent.generate(_request(1, effort=effort))
    assert failure_kind(caught.value) == "configuration"
    assert wire.sent == []


def test_provider_declaration():
    spec = resolve(MODEL).spec
    assert (spec.base_url, spec.base_url_check_env, spec.key_env, spec.auth_scheme) == (
        OFFICIAL, BASE_URL_ENV, KEY_ENV, "x-api-key",
    )
    assert spec.api_version == "2026-10-21" and spec.environment_variables == (
        KEY_ENV, BASE_URL_ENV,
    )
    assert spec.prompt_caching is True
    # Every Claude model ID is sent as written, with the same capabilities.
    for model in ("claude-opus-5-5", "claude-haiku-4-5", "claude-mythos-5-1"):
        route = resolve(model)
        assert route.wire_model == model
        assert route.capabilities == Capabilities(reasoning_mode=False, max_output_tokens=16000)


@pytest.mark.parametrize("model", ["Claude-opus-5-5"])
def test_malformed_claude_selectors_are_rejected_when_parsed(model):
    with pytest.raises(ValueError):
        resolve(model)
    with pytest.raises(ValueError):
        parse_configuration(["--mode", "agent", "--model", model])


def test_parser_records_the_anthropic_route_and_applies_its_capabilities(tmp_path):
    for model in SELECTORS:
        args = parse_configuration(["--mode", "agent", "--model", model])
        assert (args.model, dict(args.llm_route)) == (f"anthropic/{MODEL}", ANTHROPIC_ROUTE)
    with pytest.raises(ValueError, match="the anthropic provider does not support reasoning_mode"):
        parse_configuration(["--mode", "agent", "--model", MODEL, "--reasoning-mode", "pro"])
    accepted = parse_configuration(["--mode", "agent", "--model", MODEL,
                                    "--max-output-tokens", "128000"])
    agent = factory.create_policy(
        args, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "a",
        ActionSpec(),
    ).agent
    assert isinstance(agent, MessagesAgent)
    assert (agent.supported_efforts, agent.default_effort, agent.max_output_tokens) == (
        None, None, 16000,
    )
    recorder = agent.request_recorder
    assert (recorder.protocol, recorder.auth_env, recorder.auth_scheme) == (
        "messages", KEY_ENV, "x-api-key",
    )
    explicit = factory.create_policy(
        accepted, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "b",
        ActionSpec(),
    ).agent
    assert explicit.max_output_tokens == 128000


# ------------------------------------------------------- Part B: generate() over MockTransport


def test_accepted_call_is_returned_and_committed_verbatim(api_key, tmp_path):
    content = [_thinking("a"), _redacted("a"), _text("Raising first."), _tool_use("toolu_a")]
    wire = Wire(_reply(*content), _decision_reply("b"), log_dir=tmp_path / "model")
    reply = wire.agent.generate(_request(1))
    assert reply.tool_calls == (_native("toolu_a"),)
    assert reply.payload == {"tool_calls": [_native("toolu_a")]}
    assert [_compact(block) for block in reply.output_items] == [_compact(b) for b in content]
    assert reply.response_text == "Raising first."
    # Messages input_tokens excludes cache reads and writes; the canonical total includes them.
    assert reply.usage == {"uncached_input_tokens": 40, "cached_input_tokens": 100,
                           "cache_creation_input_tokens": 20, "output_tokens": 30,
                           "input_tokens": 160, "total_tokens": 190}
    metadata = reply.metadata
    assert (metadata["backend"], metadata["protocol"], metadata["provider"]) == (
        "messages", "messages", "anthropic",
    )
    assert (metadata["stop_reason"], metadata["response_model"], metadata["response_id"]) == (
        "tool_use", MODEL, "msg_test",
    )
    assert (metadata["thinking_type"], metadata["api_version"], metadata["reasoning_effort"]) == (
        "adaptive", "2026-10-21", "low",
    )
    assert metadata["prompt_caching"] is True
    assert metadata["thinking_replay"] == {"kept_blocks": 0, "stripped_blocks": 0,
                                           "first_stripped_path": None}
    telemetry = metadata["telemetry"]["usage"]
    assert (telemetry["input_tokens"], telemetry["availability"]) == (160, "reported")

    second = wire.agent.generate(_request(2, tool_results=[_result("toolu_a")]))
    assert second.metadata["previous_decision_ids"] == ["decision-1"]
    assert second.metadata["thinking_replay"]["kept_blocks"] == 2
    replayed = wire.body(1)["messages"]
    assert [message["role"] for message in replayed] == ["user", "assistant", "user"]
    assert replayed[0] == wire.body(0)["messages"][0]
    # Signed and redacted thinking come back byte for byte.
    assert _compact(replayed[1]["content"]) == _compact(content)
    # The tool result precedes the next observation in the same user message.
    assert replayed[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_a",
                                         "content": json.dumps({"status": "accepted"})}
    assert replayed[2]["content"][1] == _text(json.dumps({"observation": 2}))
    records = (tmp_path / "model" / "decisions.jsonl").read_text()
    assert KEY not in records and '"thinking_replay"' in records


@pytest.mark.parametrize(
    "status", [400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504, 529]
)
def test_every_non_200_status_is_a_resent_transport_failure(api_key, status):
    secret = "sk-ant-echoedsecret0123"
    echoed = {"type": "error", "error": {"type": "overloaded_error" if status == 529
                                         else "invalid_request_error",
                                         "message": f"invalid key {secret}"}}
    rejection = httpx.Response(status, headers={"Retry-After": "7"}, json=echoed)
    wire = Wire(rejection, _decision_reply("ok"))
    request = _request(1)
    with pytest.raises(BackendError) as caught:
        wire.agent.generate(request)
    error = caught.value
    assert type(error) is BackendError and failure_kind(error) == "transport"
    assert str(error) == f"Messages endpoint rejected request with HTTP {status}"
    assert error.metadata["http_status"] == status
    # Retry-After is honoured for 429 and 503 only; 529 overload is an ordinary backoff.
    assert error.metadata.get("retry_after_s", "absent") == (
        7.0 if status in (429, 503) else "absent"
    )
    assert error.metadata["telemetry"]["response"]["error"]["type"] == echoed["error"]["type"]
    # The rejected body is never kept: an echoed credential cannot reach the decision log.
    assert secret not in repr((str(error), error.metadata, error.usage))
    assert wire.agent.generate(request).tool_calls == (_native("toolu_ok"),)
    assert wire.sent[0].content == wire.sent[1].content


@pytest.mark.parametrize("reply, message", [
    (httpx.Response(200, content=b"{not json", headers={"content-type": "application/json"}),
     "malformed HTTP JSON envelope"),
    (httpx.Response(200, json=[_tool_use("toolu_x")]), "invalid HTTP response envelope"),
    (_reply(_text("I can't help with that."), stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber",
                          "explanation": "Declined by a classifier."}),
     "refused the request"),
    (_reply(_thinking("x"), _tool_use("toolu_x", arguments={"deltas": {}}),
            stop_reason="max_tokens"), "stop_reason 'max_tokens'"),
    (_reply(_text("Searching."), stop_reason="pause_turn"), "stop_reason 'pause_turn'"),
    (_reply(_tool_use("toolu_x"), stop_reason="stop_sequence"), "stop_reason 'stop_sequence'"),
    (_reply(_tool_use("toolu_x"), stop_reason="model_context_window_exceeded"),
     "stop_reason 'model_context_window_exceeded'"),
    (_reply(_tool_use("toolu_x"), stop_reason=None), "stop_reason None"),
    ({**_reply(), "content": None}, "invalid content array"),
    ({**_reply(_tool_use("toolu_x")), "type": "error"}, "invalid response envelope"),
    ({**_reply(_tool_use("toolu_x")), "role": "user"}, "invalid response envelope"),
    (_reply({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
             "input": {}}), "Unsupported native Messages response content"),
    (_reply({"type": "thinking", "thinking": ""}, _tool_use("toolu_x")),
     "thinking requires original text and signature"),
    (_reply({"type": "redacted_thinking"}, _tool_use("toolu_x")), "redacted thinking requires"),
    (_reply(_tool_use("toolu_x"), _tool_use("toolu_x")), "unique ID"),
    (_reply({**_tool_use("toolu_x"), "input": "{}"}), "unique ID, name and input object"),
    (_reply({**_tool_use("toolu_x"), "id": ""}), "unique ID, name and input object"),
    (_reply({"type": "text", "text": None}), "requires string text"),
    ({**_reply(_tool_use("toolu_x")), "usage": {"input_tokens": "many"}},
     "invalid token counts"),
    ({**_reply(_tool_use("toolu_x")), "usage": ["tokens"]}, "usage envelope must be an object"),
], ids=["malformed_json", "array_envelope", "refusal", "max_tokens", "pause_turn",
        "stop_sequence", "context_window", "no_stop_reason", "no_content", "error_type",
        "user_role", "server_tool_use", "unsigned_thinking", "redacted_without_data",
        "duplicate_ids", "string_input", "empty_id", "non_string_text", "invalid_usage",
        "non_object_usage"])
def test_unusable_response_is_a_transport_failure_that_commits_nothing(api_key, reply, message):
    wire = Wire(reply, _decision_reply("ok"), _decision_reply("next"))
    request = _request(1)
    with pytest.raises(BackendError, match=re.escape(message)) as caught:
        wire.agent.generate(request)
    assert type(caught.value) is BackendError and failure_kind(caught.value) == "transport"
    assert wire.agent._history == []
    # Nothing was committed: the resend is byte-identical and the next decision is clean.
    wire.agent.generate(request)
    assert wire.sent[0].content == wire.sent[1].content
    wire.agent.generate(_request(2, tool_results=[_result("toolu_ok")]))
    assert [message["role"] for message in wire.body()["messages"]] == [
        "user", "assistant", "user",
    ]


def test_refusal_records_its_category_without_its_text(api_key, tmp_path):
    refusal = _reply(_text("I can't help with that."), stop_reason="refusal",
                     stop_details={"type": "refusal", "category": "bio",
                                   "explanation": "Declined by a classifier."})
    wire = Wire(refusal, log_dir=tmp_path / "model")
    with pytest.raises(BackendError) as caught:
        wire.agent.generate(_request(1))
    metadata = caught.value.metadata
    assert metadata["provider_refusal"] is True
    assert metadata["stop_details"] == {"type": "refusal", "category": "bio"}
    assert metadata["telemetry"]["response"]["finish_reason"] == "refusal"
    assert metadata["telemetry"]["response"]["stop_details"] == {"type": "refusal",
                                                                 "category": "bio"}
    record = (tmp_path / "model" / "decisions.jsonl").read_text()
    assert "Declined by a classifier" not in record and "I can't help" not in record


@pytest.mark.parametrize("content, calls, text", [
    ([_thinking("x"), _text("Let me think about it.")], [], "Let me think about it."),
    ([_tool_use("toolu_x"), _tool_use("toolu_y")], [_native("toolu_x"), _native("toolu_y")], ""),
    ([_tool_use("toolu_x", name="shell")], [_native("toolu_x", name="shell")], ""),
], ids=["no_call", "two_calls", "unknown_tool"])
def test_wrong_call_count_is_a_format_error_after_commit(api_key, content, calls, text):
    stop = "tool_use" if calls else "end_turn"
    wire = Wire(_reply(*content, stop_reason=stop))
    first = _request(1, decision="d1")
    with pytest.raises(BackendFormatError) as caught:
        wire.agent.generate(first)
    error = caught.value
    assert type(error) is BackendFormatError and failure_kind(error) == "format"
    assert str(error) == "Call exactly one of the provided robot function tools"
    assert (error.metadata["tool_calls"], error.response_text) == (calls, text)
    # The completed output was committed: a repair replays it and must answer its calls.
    if calls:
        with pytest.raises(BackendConfigurationError, match="Pending tool calls"):
            wire.agent.prepare_request(_request(2, decision="d1"))
    rejections = [_result(call["call_id"], "rejected") for call in calls]
    repair = wire.agent.prepare_request(_request(2, decision="d1", tool_results=rejections))
    assert repair["messages"][0] == wire.body(0)["messages"][0]
    assert _compact(repair["messages"][1]) == _compact({"role": "assistant", "content": content})
    assert repair["messages"][2]["content"][:len(calls)] == [
        {"type": "tool_result", "tool_use_id": call["call_id"],
         "content": json.dumps({"status": "rejected"})} for call in calls
    ]
    assert repair["messages"][2]["content"][len(calls)] == _text(json.dumps({"observation": 2}))
    assert len(repair["messages"]) == 3


@pytest.mark.parametrize("excess", [0, 1])
def test_output_limit_counts_visible_text_and_arguments(api_key, excess):
    text = "Raising now."
    observed = len(text) + len(json.dumps(MOVE))
    wire = Wire(_reply(_thinking("a"), _text(text), _tool_use("toolu_a")),
                max_response_chars=observed - excess)
    if not excess:
        assert wire.agent.generate(_request(1)).tool_calls == (_native("toolu_a"),)
        return
    with pytest.raises(BackendOutputLimitError) as caught:
        wire.agent.generate(_request(1))
    error = caught.value
    assert (error.category, failure_kind(error)) == ("output_limit", "format")
    assert error.metadata["output_chars_observed"] == observed
    assert error.metadata["tool_calls"] == [_native("toolu_a")]
    with pytest.raises(BackendConfigurationError, match="Pending tool calls"):
        wire.agent.prepare_request(_request(2, decision="decision-1"))


def test_tool_use_id_reused_by_a_later_attempt_is_rejected_without_commit(api_key):
    wire = Wire(
        _reply(_tool_use("toolu_x"), _tool_use("toolu_y")),
        _reply(_thinking("r"), _tool_use("toolu_x")),
        _reply(_tool_use("toolu_z")),
    )
    with pytest.raises(BackendFormatError):
        wire.agent.generate(_request(1, decision="d1"))
    rejections = [_result("toolu_x", "rejected"), _result("toolu_y", "rejected")]
    repair = _request(2, decision="d1", tool_results=rejections)
    with pytest.raises(BackendError, match="reused a function call ID") as caught:
        wire.agent.generate(repair)
    assert type(caught.value) is BackendError and failure_kind(caught.value) == "transport"
    assert wire.agent.generate(repair).tool_calls == (_native("toolu_z"),)
    assert wire.sent[1].content == wire.sent[2].content


def _failing_recorder(request, http_request):
    raise OSError("audit disk full")


@pytest.mark.parametrize("options, fields, key, closed", [
    ({}, {}, None, False),
    ({}, {}, "two words", False),
    ({}, {}, "kéy-0123456789", False),
    ({}, {}, KEY, True),
    ({}, {"tools": ()}, KEY, False),
    ({}, {"tool_results": [_result("toolu_unknown")]}, KEY, False),
    ({"request_recorder": _failing_recorder}, {}, KEY, False),
], ids=["missing_key", "key_with_space", "non_ascii_key", "closed", "no_tools",
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
    if key is None:
        assert str(caught.value).startswith(f"{KEY_ENV} is not set; export the API key")


@pytest.mark.parametrize("configured, accepted", [
    (None, True),
    (OFFICIAL, True),
    (OFFICIAL + "/", True),
    ("https://messages.example.invalid/v1", False),
    ("http://api.anthropic.com/v1", False),
    # The SDK spelling: client libraries add /v1 themselves.
    ("https://api.anthropic.com", True),
    ("https://api.anthropic.com/", True),
    ("https://api.anthropic.com/v2", False),
])
def test_anthropic_base_url_must_name_the_official_endpoint(monkeypatch, configured, accepted):
    monkeypatch.setenv(KEY_ENV, KEY)
    if configured is not None:
        monkeypatch.setenv(BASE_URL_ENV, configured)
    wire = Wire(_decision_reply("a"))
    if accepted:
        wire.agent.generate(_request(1))
    else:
        with pytest.raises(BackendConfigurationError, match=BASE_URL_ENV) as caught:
            wire.agent.generate(_request(1))
        assert failure_kind(caught.value) == "configuration"
        assert "example.invalid" not in str(caught.value)
    assert len(wire.sent) == int(accepted)


@pytest.mark.parametrize("failures, sleeps, sources, accepted", [
    ([(529, {"Retry-After": "30"}), (500, {})], [1.125, 2.25], ["backoff", "backoff"], True),
    ([(429, {"Retry-After": "30"})], [30.0], ["retry_after"], True),
    ([(529, {})] * 3, [1.125, 2.25], ["backoff", "backoff"], False),
], ids=["overloaded_then_server_error", "rate_limited", "overloaded_exhausted"])
def test_policy_resends_http_failures_unchanged_after_backoff(
    api_key, monkeypatch, failures, sleeps, sources, accepted
):
    slept = []
    monkeypatch.setattr(policy_module, "_sleep", slept.append)
    monkeypatch.setattr(policy_module, "_jitter", lambda: 0.5)
    rejections = [httpx.Response(status, headers=headers, json={"type": "error", "error": {}})
                  for status, headers in failures]
    wire = Wire(_decision_reply("1"), *rejections, *([_decision_reply("2")] if accepted else []))
    policy = _policy(wire, max_attempts=3)
    _act(policy, 0)
    if accepted:
        _act(policy, 5)
    else:
        with pytest.raises(PolicyStop) as caught:
            policy.act(_observation(5))
        assert caught.value.name == "max_attempts"
    assert slept == sleeps
    attempts = policy.decisions[1:]
    assert [d["retry_wait_source"] for d in attempts] == [None, *sources]
    assert [d.get("error_kind") for d in attempts[:len(failures)]] == ["transport"] * len(failures)
    # Every attempt of the decision resent the same bytes, carrying the pending result once.
    assert len({request.content for request in wire.sent[1:]}) == 1
    results = [block for message in wire.body()["messages"] for block in message["content"]
               if block["type"] == "tool_result"]
    assert [block["tool_use_id"] for block in results] == ["toolu_1"]


# ------------------------------------------------------- Part C: prefix-bound thinking replay


def _run_decisions(wire, count, *, history_length, first=0):
    for index in range(first, first + count):
        results = [_result(f"toolu_{index - 1}")] if index else []
        wire.agent.generate(_request(index, history_length=history_length,
                                     tool_results=results))


def test_growing_window_keeps_every_thinking_block(api_key):
    wire = Wire(*(_decision_reply(str(n)) for n in range(4)))
    _run_decisions(wire, 4, history_length=10)
    body = wire.body()
    assert [block["signature"] for block in _thinking_blocks(body)] == [
        "signature-0", "signature-1", "signature-2",
    ]


def test_window_eviction_strips_thinking_and_records_it(api_key, tmp_path):
    wire = Wire(*(_decision_reply(str(n)) for n in range(4)), log_dir=tmp_path / "model")
    _run_decisions(wire, 3, history_length=2)
    # Decisions 0 and 1 still have their original prefix: both keep their thinking.
    assert len(_thinking_blocks(wire.body())) == 2
    wire.agent.generate(_request(3, history_length=2, tool_results=[_result("toolu_2")]))
    body = wire.body()
    # Decision 0 left the window, which edits the prefix of every retained turn.
    assert [message["role"] for message in body["messages"]] == [
        "user", "assistant", "user", "assistant", "user",
    ]
    assert _thinking_blocks(body) == []
    assert [message["content"] for message in body["messages"][1::2]] == [
        [_text(f"Plan {tag}."), _tool_use(f"toolu_{tag}")] for tag in ("1", "2")
    ]
    rows = [json.loads(line) for line in
            (tmp_path / "model" / "decisions.jsonl").read_text().splitlines()]
    assert [row["metadata"]["thinking_replay"] for row in rows] == [
        {"kept_blocks": 0, "stripped_blocks": 0, "first_stripped_path": None},
        {"kept_blocks": 1, "stripped_blocks": 0, "first_stripped_path": None},
        {"kept_blocks": 2, "stripped_blocks": 0, "first_stripped_path": None},
        {"kept_blocks": 0, "stripped_blocks": 2, "first_stripped_path": "messages.1.content.0"},
    ]
    # Only requests that really replay an earlier decision's thinking say so.
    assert [row["metadata"]["previous_decision_reasoning_replay"] for row in rows] == [
        False, True, True, False,
    ]


def test_resend_after_a_strip_is_byte_identical(api_key):
    overloaded = httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})
    wire = Wire(*(_decision_reply(str(n)) for n in range(3)), overloaded, _decision_reply("3"))
    _run_decisions(wire, 3, history_length=1)
    request = _request(3, history_length=1, tool_results=[_result("toolu_2")])
    preview = wire.agent.prepare_http_request(request)
    with pytest.raises(BackendError):
        wire.agent.generate(request)
    wire.agent.generate(request)
    assert wire.sent[-2].content == wire.sent[-1].content == preview.content
    assert _thinking_blocks(wire.body()) == []


def test_repair_keeps_its_decision_thinking_after_eviction(api_key):
    wire = Wire(
        _decision_reply("0"), _decision_reply("1"),
        _reply(_thinking("2a"), _tool_use("toolu_2a"), _tool_use("toolu_2b")),
        _decision_reply("2c"),
    )
    _run_decisions(wire, 2, history_length=1)
    with pytest.raises(BackendFormatError):
        wire.agent.generate(_request(2, decision="d2", history_length=1,
                                     tool_results=[_result("toolu_1")]))
    rejections = [_result("toolu_2a", "rejected"), _result("toolu_2b", "rejected")]
    reply = wire.agent.generate(_request(3, decision="d2", history_length=1,
                                         tool_results=rejections))
    # Decision 0 was evicted, so decision 1 is stripped; the repaired attempt keeps its own.
    assert reply.metadata["thinking_replay"] == {
        "kept_blocks": 1, "stripped_blocks": 1, "first_stripped_path": "messages.1.content.0",
    }
    # The kept block is the repair's own, not an earlier decision's.
    assert reply.metadata["previous_decision_reasoning_replay"] is False
    assert [block["signature"] for block in _thinking_blocks(wire.body())] == ["signature-2a"]


@pytest.mark.parametrize("history_length", [0, -1])
def test_history_zero_replays_nothing_and_repairs_keep_thinking(api_key, history_length):
    wire = Wire(
        _decision_reply("0"),
        _reply(_thinking("1a"), _text("No call yet."), stop_reason="end_turn"),
        _decision_reply("1b"),
    )
    first = wire.agent.generate(_request(0, history_length=history_length))
    assert first.metadata["thinking_replay"]["kept_blocks"] == 0
    with pytest.raises(BackendFormatError):
        wire.agent.generate(_request(1, decision="d1", history_length=history_length,
                                     tool_results=[_result("toolu_0")]))
    repair = wire.agent.generate(_request(2, decision="d1", history_length=history_length))
    assert repair.metadata["thinking_replay"] == {
        "kept_blocks": 1, "stripped_blocks": 0, "first_stripped_path": None,
    }
    body = wire.body()
    assert [message["role"] for message in body["messages"]] == ["user", "assistant", "user"]
    assert [block["signature"] for block in _thinking_blocks(body)] == ["signature-1a"]


def test_a_changed_system_prompt_strips_earlier_thinking(api_key):
    wire = Wire(_decision_reply("0"))
    wire.agent.generate(_request(0))
    following = dict(history_length=2, tool_results=[_result("toolu_0")])
    assert len(_thinking_blocks(wire.agent.prepare_request(_request(1, **following)))) == 1
    edited = wire.agent.prepare_request(_request(1, instructions="Edited.", **following))
    assert _thinking_blocks(edited) == []
    tools = (TOOL, {**TOOL, "name": "done"})
    assert _thinking_blocks(wire.agent.prepare_request(_request(1, tools=tools, **following))) == []


def test_a_turn_of_thinking_only_is_dropped_when_stripped():
    items = [
        {"role": "user", "content": [_text("a")]},
        {"role": "assistant", "content": [_thinking("x")], messages_module.BINDING: "stale"},
        {"role": "user", "content": [_text("b")]},
    ]
    bound, _, replay = messages_module._bind("system", [WIRE_TOOL], items)
    merged, positions = messages_module._coalesce(bound)
    assert merged == [{"role": "user", "content": [_text("a"), _text("b")]}]
    # The recorded path names where the block would be had it been kept.
    assert positions == [0, 1, 0]
    assert replay == {"kept_blocks": 0, "stripped_blocks": 1, "first": (1, 0),
                      "earlier_kept": 0}


def test_teacher_copy_move_strips_and_the_teacher_is_sent_once(api_key, tmp_path):
    teacher = _write_teacher(tmp_path / "teacher")
    wire = Wire(*(_decision_reply(str(n)) for n in range(3)))
    policy = _policy(wire, history_length=1, demo=True, demo_path=teacher)
    for step in (0, 5):
        _act(policy, step)
    # Within the window the earliest observation carries the teacher, and thinking is kept.
    assert len(_thinking_blocks(wire.body())) == 1
    preview = wire.agent.prepare_http_request(policy.prepare_request(_observation(10)))
    body = json.loads(preview.content)
    # Decision 0 is evicted: the teacher moves into the next retained observation.
    users = [message for message in body["messages"] if message["role"] == "user"]
    assert [sum('"demonstration"' in block.get("text", "") for block in user["content"])
            for user in users] == [1, 0]
    frames =[_pixels(block["source"]["data"]) for message in body["messages"]
              for block in message["content"] if block["type"] == "image"]
    for seed in (50, 51):
        assert frames.count(_image(seed).tobytes()) == 1
    assert _thinking_blocks(body) == []
    _act(policy, 10)
    assert wire.sent[-1].content == preview.content
    assert policy.decisions[-1]["backend"]["thinking_replay"] == {
        "kept_blocks": 0, "stripped_blocks": 1, "first_stripped_path": "messages.1.content.0",
    }


# ------------------------------------------- Part D: previews, archives and response telemetry


def test_request_preview_is_auth_free_and_reads_no_environment(monkeypatch, api_key):
    wire = Wire(_decision_reply("a"), _decision_reply("b"))
    wire.agent.generate(_request(1))
    following = _request(2, tool_results=[_result("toolu_a")])
    monkeypatch.delenv(KEY_ENV)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", ForbiddenEnviron())
        preview = wire.agent.prepare_http_request(following)
        again = wire.agent.prepare_http_request(following)
    assert not AUTH_HEADERS & set(preview.headers)
    assert preview.headers["anthropic-version"] == "2026-10-21"
    assert preview.content == again.content
    monkeypatch.setenv(KEY_ENV, KEY)
    wire.agent.generate(following)
    assert wire.sent[-1].content == preview.content


def test_preview_policy_archives_a_request_with_an_empty_environment(tmp_path, monkeypatch):
    for name in list(os.environ):
        monkeypatch.delenv(name)
    args = parse_configuration(["--mode", "preview", "--model", MODEL])
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
        "preview", "messages", f"{OFFICIAL}/messages",
    )
    assert (manifest["auth_env"], manifest["auth_scheme"]) == (KEY_ENV, "x-api-key")
    assert manifest["safe_headers"]["anthropic-version"] == "2026-10-21"
    assert [image["camera_label"] for image in manifest["images"]] == [
        "current/agentview", "current/robot0_eye_in_hand",
    ]
    readable = (directory / "request.md").read_text()
    assert "x-api-key: ${ANTHROPIC_API_KEY}" in readable and "Bearer" not in readable
    body = json.loads((directory / "request.body.json").read_bytes())
    assert body["model"] == MODEL and body["output_config"] == {"effort": "low"}


def test_scripted_preview_replies_commit_tool_use_without_thinking(tmp_path):
    from agentic_framework.harness.preview import OfflinePreviewAgent

    recorder = APIRequestRecorder(tmp_path / "archive", mode="preview", protocol="messages",
                                  auth_env=KEY_ENV, auth_scheme="x-api-key")
    agent = MessagesAgent(MODEL, max_output_tokens=64, request_recorder=recorder,
                          transport=UNUSED_TRANSPORT)
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
    assistant = bodies[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["tool_use"]
    assert assistant["content"][0]["id"] == "preview-1"
    assert bodies[1]["messages"][2]["content"][0]["tool_use_id"] == "preview-1"


def test_preview_commit_rejects_a_reused_call_id_without_changing_history():
    agent = MessagesAgent(MODEL, max_output_tokens=64, transport=UNUSED_TRANSPORT)

    def output(*blocks):
        return AgentOutput(payload={}, usage={}, metadata={}, elapsed_s=0.0, tool_calls=(),
                           output_items=tuple(blocks), response_text="")

    agent.accept_reply(_request(1, decision="d1"),
                       output(_tool_use("toolu_x"), _tool_use("toolu_y")))
    rejections = [_result("toolu_x", "rejected"), _result("toolu_y", "rejected")]
    repair = _request(2, decision="d1", tool_results=rejections)
    before = _compact(agent.prepare_request(repair))
    with pytest.raises(BackendError, match="reused a function call ID"):
        agent.accept_reply(repair, output(_thinking("2"), _tool_use("toolu_y")))
    assert _compact(agent.prepare_request(repair)) == before
    with pytest.raises(BackendError, match="Unsupported or mixed"):
        agent.accept_reply(repair, output({"type": "reasoning"}))


def _archive(tmp_path, body):
    recorder = APIRequestRecorder(tmp_path, protocol="messages", auth_env=KEY_ENV,
                                  auth_scheme="x-api-key")
    request = httpx.Request("POST", f"{OFFICIAL}/messages", json=body,
                            headers={"anthropic-version": "2026-10-21"})
    return recorder(_request(1), request)


def test_archive_validates_messages_bodies_by_protocol(tmp_path):
    body = MessagesAgent(MODEL, max_output_tokens=64,
                         transport=UNUSED_TRANSPORT).prepare_request(_request(1))
    assert _archive(tmp_path / "valid", body)["request_sequence"] == 1
    invalid = [
        {**body, "messages": [*body["messages"], {"role": "assistant", "content": [
            {"type": "document", "source": {"type": "text", "data": "x"}}]}]},
        {key: value for key, value in body.items() if key != "messages"},
    ]
    for index, candidate in enumerate(invalid):
        with pytest.raises(ValueError):
            _archive(tmp_path / f"invalid-{index}", candidate)


@pytest.mark.parametrize("raw, expected", [
    (USAGE, {"input_tokens": 160, "uncached_input_tokens": 40, "cached_input_tokens": 100,
             "cache_creation_input_tokens": 20, "output_tokens": 30, "total_tokens": 190,
             "availability": "reported"}),
    ({**USAGE, "cache_read_input_tokens": None, "cache_creation_input_tokens": None},
     {"input_tokens": None, "uncached_input_tokens": 40, "cached_input_tokens": None,
      "cache_creation_input_tokens": None, "output_tokens": 30, "total_tokens": None,
      "availability": "partial"}),
])
def test_canonical_usage_counts_cache_tokens_into_the_prompt(raw, expected):
    usage = canonical_usage(raw, "messages")
    assert {key: usage[key] for key in expected} == expected
    assert usage["issues"] == ([] if expected["availability"] == "reported" else [
        "input_tokens: total prompt unavailable because cache components are missing or invalid"
    ])


def test_binding_rejection_is_diagnosed_without_its_text():
    telemetry = new_telemetry("messages")
    message = ("messages.3.content.0: Invalid `signature` in `thinking` block. The block is "
               "bound to a different conversation. Remove the block.")
    observe_response_json(telemetry, {"type": "error", "error": {
        "type": "invalid_request_error", "message": message}})
    assert telemetry["response"]["error"] == {
        "type": "invalid_request_error", "path": "messages.3.content.0", "binding_mismatch": True,
    }
    other = new_telemetry("messages")
    observe_response_json(other, {"type": "error", "error": {
        "type": "overloaded_error", "message": "Overloaded"}})
    assert other["response"]["error"] == {"type": "overloaded_error", "binding_mismatch": False}
    responses = new_telemetry("responses")
    observe_response_json(responses, {"error": {"type": "x", "message": message}})
    assert responses["response"]["error"] == {"type": "x"}
