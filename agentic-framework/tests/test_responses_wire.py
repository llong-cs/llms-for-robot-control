"""Characterize the OpenAI Responses wire of ResponsesAgent, offline.

Part A compares request previews (``prepare_http_request``) with the golden files in
``golden/responses``. A golden stores the method, URL, headers and JSON body of one request.
Headers that depend on the local installation (content-length, user-agent, accept-encoding)
are left out, and every PNG data URL is replaced by
``png-pixels-sha256:<sha256 of the decoded RGB pixel bytes>`` so that goldens do not depend on
the Pillow or zlib version. Every other byte is compared exactly: the normalized request
content must equal the compact UTF-8 JSON serialization of the golden body, which is stored
decoded so that a regenerated golden produces a reviewable diff.

After an intended wire change, regenerate the goldens (the regenerating tests report as
skipped), review the diff, then rerun without the variable:

    AGENTIC_UPDATE_GOLDEN=1 python -B -m pytest -q agentic-framework/tests/test_responses_wire.py

The goldens hold httpx >= 0.28 request bytes: compact UTF-8 JSON bodies.

Every golden holds for both selector spellings, ``openai/<model>`` and the ``gpt-*`` shorthand.

Part B drives ``generate()`` through ``httpx.MockTransport``: acceptance and history commit,
failure classification, the history window, key resolution from the environment and model
naming.
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

from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness import policy as policy_module
from agentic_framework.harness.policy import LLMMotionPolicy
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
from agentic_framework.models.llm.errors import (
    BackendConfigurationError,
    BackendError,
    BackendFormatError,
    BackendOutputLimitError,
    failure_kind,
    retry_after_seconds,
)
from agentic_framework.models.llm.responses import ResponsesAgent

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "responses"
UPDATE_GOLDEN = "AGENTIC_UPDATE_GOLDEN"
# The official provider's key variable; the shared conftest removes it from every test.
KEY_ENV = "OPENAI_API_KEY"
KEY = "wire-test-key-0123456789"
MODEL = "gpt-wire-test"
# Both spellings route to the official provider and send the same bytes.
SELECTORS = (MODEL, f"openai/{MODEL}")
ENDPOINT = "https://responses.example.invalid/v1"
OFFICIAL = "https://api.openai.com/v1"
INSTRUCTION = "Lift the red cube."
AUTH_HEADERS = frozenset({"authorization", "proxy-authorization", "api-key", "x-api-key"})
# Derived from the installed httpx, its codecs and the PNG encoder, not from the agent.
INSTALLATION_HEADERS = frozenset({"accept-encoding", "content-length", "user-agent"})
PNG_DATA_URL = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]*)")
MOVE_ARGS = json.dumps({"deltas": {"dz": 0.001}, "note": "Raise the gripper slightly."})
TOOL = {"type": "function", "name": "move_by", "description": "Move.",
        "parameters": {"type": "object"}, "strict": False}
USAGE = {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 100},
         "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20},
         "total_tokens": 150}


# ---------------------------------------------------------------------------- helpers


def _image(seed):
    """Return a distinct deterministic 4x4 RGB frame."""
    return ((np.arange(48).reshape(4, 4, 3) * (seed + 1) + seed) % 256).astype(np.uint8)


def _pixel_token(pixels):
    return "png-pixels-sha256:" + hashlib.sha256(pixels.tobytes()).hexdigest()


def _png_token(match):
    data = base64.b64decode(match.group(1), validate=True)
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG" and image.mode == "RGB"
        return _pixel_token(np.asarray(image))


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _difference(expected, actual, path="body"):
    """Return the JSON path of the first difference between two decoded bodies, or None."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if list(expected) != list(actual):
            return f"{path} keys {list(expected)} != {list(actual)}"
        items = [(f"{path}.{key}", expected[key], actual[key]) for key in expected]
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length {len(expected)} != {len(actual)}"
        items = [(f"{path}[{i}]", a, b) for i, (a, b) in enumerate(zip(expected, actual))]
    else:
        return None if type(expected) is type(actual) and expected == actual else path
    return next((found for item in items if (found := _difference(item[1], item[2], item[0]))),
                None)


def _httpx_sends_compact_json():
    """Whether the installed httpx encodes ``json=`` bodies as compact UTF-8 (httpx >= 0.28)."""
    request = httpx.Request("POST", "https://wire.example.invalid/", json={"a": [1, "é"]})
    return request.content == '{"a":[1,"é"]}'.encode()


def _assert_golden(name, preview):
    """Compare one auth-free request preview with golden/responses/<name>.json."""
    if not _httpx_sends_compact_json():
        pytest.fail(f"the request goldens hold httpx>=0.28 bytes; httpx {httpx.__version__} "
                    "encodes JSON bodies differently", pytrace=False)
    assert not AUTH_HEADERS & set(preview.headers)
    content = preview.content
    assert preview.headers["content-length"] == str(len(content))
    text = PNG_DATA_URL.sub(_png_token, content.decode("utf-8"))
    record = {
        "method": preview.method,
        "url": str(preview.url),
        "headers": {k: v for k, v in sorted(preview.headers.items())
                    if k not in INSTALLATION_HEADERS},
        "body": json.loads(text),
    }
    path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get(UPDATE_GOLDEN) == "1":
        assert _compact(record["body"]) == text, "request content is not compact UTF-8 JSON"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pytest.skip(f"regenerated {path.name}; review the diff, then rerun without {UPDATE_GOLDEN}")
    if not path.is_file():
        pytest.fail(f"missing golden {path.name}; regenerate with {UPDATE_GOLDEN}=1")
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert [golden[k] for k in ("method", "url", "headers")] == [
        record[k] for k in ("method", "url", "headers")
    ]
    if text != _compact(golden["body"]):
        where = _difference(golden["body"], record["body"]) or "byte serialization"
        pytest.fail(f"{path.name}: request body differs at {where}; if intended, regenerate "
                    f"with {UPDATE_GOLDEN}=1 and review the diff", pytrace=False)


def _call(call_id, arguments=MOVE_ARGS, name="move_by"):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": arguments, "status": "completed"}


def _native(call_id, arguments=MOVE_ARGS, name="move_by"):
    """The normalized call a reply exposes for a raw _call item."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _reasoning(tag):
    return {"type": "reasoning", "id": f"rs_{tag}", "summary": [],
            "encrypted_content": f"opaque-reasoning-{tag}"}


def _message(text):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _reply(*output, **fields):
    return {"id": "resp_test", "object": "response", "model": MODEL, "status": "completed",
            "output": list(output), "usage": USAGE, **fields}


def _decision_reply(tag):
    return _reply(_reasoning(tag), _call(f"call_{tag}"))


def _result(call_id, status="accepted"):
    return {"type": "function_call_output", "call_id": call_id,
            "output": json.dumps({"status": status})}


def _request(index, *, decision=None, history_length=2, tool_results=(), effort="low",
             tools=(TOOL,)):
    text = json.dumps({"observation": index})
    return AgentContext(
        request_id=f"req-{index}",
        instructions="Choose one robot tool call.",
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
    """A ResponsesAgent on the official endpoint whose HTTP exchanges are scripted."""

    def __init__(self, *replies, preview=False, **options):
        self.replies = list(replies)
        self.sent: list[httpx.Request] = []
        self.previews: list[httpx.Request] = []
        self.agent = ResponsesAgent(
            options.pop("model", MODEL), transport=httpx.MockTransport(self._handle), **options
        )
        if preview:
            send = self.agent.generate

            def generate(request):
                self.previews.append(self.agent.prepare_http_request(request))
                return send(request)

            self.agent.generate = generate

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

    def assert_previews_sent(self):
        """Each preview equals the request generate() sent, apart from its authorization."""
        assert len(self.previews) == len(self.sent)
        for preview, sent in zip(self.previews, self.sent):
            assert (preview.method, preview.url, preview.content) == (
                sent.method, sent.url, sent.content
            )
            headers = dict(sent.headers)
            assert headers.pop("authorization") == f"Bearer {KEY}"
            assert dict(preview.headers) == headers


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
            "model_output": {"backend": {"backend": "responses"}},
        })
    path = directory / "trajectory.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "metadata": {"policy_family": "llm", "observation_profile": "control"},
        "outcome": {"complete": True, "status": "failed"},
        "turns": turns,
    }))
    return path


# ------------------------------------------------------------ Part A: golden request bodies


@pytest.mark.parametrize(
    "interface, ks", [("move_by", (1, 5, -1)), ("move_by_chunk", (1, 5))]
)
def test_first_request_matches_golden_for_every_k(interface, ks, selector):
    previews = []
    for k in ks:
        schedule = ScheduleConfig(h=5, k=k, max_steps=40, control_interface=interface)
        wire = Wire(model=selector)
        policy = _policy(wire, schedule=schedule)
        previews.append(wire.agent.prepare_http_request(policy.prepare_request(_observation(0))))
        assert wire.sent == []
    # K is never visible on the wire: instructions, tools and context are byte-identical.
    assert len({preview.content for preview in previews}) == 1
    _assert_golden(f"{interface}_h5", previews[0])


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra"])
def test_published_model_selectors_send_the_golden_bytes(model):
    """The configs' openai/gpt-6-astra and its shorthand differ from a golden only in the ID."""
    wire = Wire(model=model)
    policy = _policy(wire, schedule=ScheduleConfig(h=5, k=5, max_steps=40))
    preview = wire.agent.prepare_http_request(policy.prepare_request(_observation(0)))
    golden = json.loads((GOLDEN_DIR / "move_by_h5.json").read_text(encoding="utf-8"))
    assert (preview.method, str(preview.url)) == (golden["method"], golden["url"])
    assert PNG_DATA_URL.sub(_png_token, preview.content.decode("utf-8")) == _compact(
        {**golden["body"], "model": "gpt-6-astra"}
    )


@pytest.mark.parametrize("history_length", [0, 2])
def test_history_window_request_matches_golden(api_key, history_length, selector):
    wire = Wire(*(_decision_reply(str(n)) for n in (1, 2, 3, 4)), preview=True, model=selector)
    policy = _policy(wire, history_length=history_length)
    for step in (0, 5, 10):
        _act(policy, step)
    preview = wire.agent.prepare_http_request(policy.prepare_request(_observation(15)))
    _assert_golden(f"history_{history_length}", preview)
    _act(policy, 15)
    wire.assert_previews_sent()
    assert wire.sent[-1].content == preview.content


@pytest.mark.parametrize("history_length", [0, 2])
def test_format_repair_request_matches_golden(api_key, history_length, selector):
    missing_note = json.dumps({"deltas": {"dz": 0.001}})
    wire = Wire(
        _decision_reply("1"),
        _reply(_reasoning("2a"), _call("call_2a", arguments=missing_note)),
        _decision_reply("2b"),
        preview=True,
        model=selector,
    )
    policy = _policy(wire, history_length=history_length)
    _act(policy, 0)
    _act(policy, 5)
    assert [d["retry_reason"] for d in policy.decisions] == [None, None, "format"]
    wire.assert_previews_sent()
    _assert_golden(f"format_repair_history_{history_length}", wire.previews[2])


def test_demonstration_request_matches_golden(api_key, tmp_path, selector):
    wire = Wire(_decision_reply("1"), _decision_reply("2"), preview=True, model=selector)
    policy = _policy(wire, demo=True, demo_path=_write_teacher(tmp_path / "teacher"))
    _act(policy, 0)
    preview = wire.agent.prepare_http_request(policy.prepare_request(_observation(5)))
    _assert_golden("demonstration_history_2", preview)
    _act(policy, 5)
    wire.assert_previews_sent()
    # A stateless request carries the fixed teacher once, in its earliest observation.
    text = PNG_DATA_URL.sub(_png_token, preview.content.decode())
    for seed in (50, 51):
        assert text.count(_pixel_token(_image(seed))) == 1
    users = [item for item in json.loads(text)["input"] if item.get("role") == "user"]
    assert ['"demonstration"' in user["content"][0]["text"] for user in users] == [True, False]


# ------------------------------------------------------- Part B: generate() over MockTransport


def test_accepted_call_is_returned_and_committed_verbatim(api_key, tmp_path):
    output = [_reasoning("a"), _message("Raising first."), _call("call_a")]
    wire = Wire(_reply(*output), _decision_reply("b"), log_dir=tmp_path / "model")
    reply = wire.agent.generate(_request(1))
    assert reply.tool_calls == (_native("call_a"),)
    assert reply.payload == {"tool_calls": [_native("call_a")]}
    assert [_compact(item) for item in reply.output_items] == [_compact(i) for i in output]
    assert reply.response_text == "Raising first."
    assert reply.usage == {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                           "cached_input_tokens": 100, "reasoning_tokens": 20}
    assert (reply.metadata["http_status"], reply.metadata["response_id"]) == (200, "resp_test")
    sent = wire.sent[0]
    assert (sent.method, str(sent.url)) == ("POST", f"{OFFICIAL}/responses")
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    first = wire.body(0)
    assert {k: v for k, v in first.items() if k not in ("instructions", "input", "tools")} == {
        "model": MODEL, "store": False, "include": ["reasoning.encrypted_content"],
        "tool_choice": "required", "parallel_tool_calls": False, "reasoning": {"effort": "low"},
    }
    assert (first["tools"], len(first["input"])) == ([TOOL], 1)

    second = wire.agent.generate(_request(2, tool_results=[_result("call_a")]))
    assert second.metadata["previous_decision_ids"] == ["decision-1"]
    replayed = wire.body(1)["input"]
    expected = [first["input"][0], *output, _result("call_a")]
    assert [_compact(item) for item in replayed[:-1]] == [_compact(item) for item in expected]
    assert replayed[-1]["content"][0]["text"] == _request(2).text
    assert (tmp_path / "model" / "decisions.jsonl").is_file()
    assert all(KEY not in path.read_text() for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize(
    "status", [201, 204, 307, 400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504,
               529]
)
def test_every_non_200_status_is_a_resent_transport_failure(api_key, status):
    secret = "sk-echoedsecret0123"
    echoed = {"error": {"message": f"invalid key {secret}"}}
    rejection = httpx.Response(status, headers={"Retry-After": "7"}, json=echoed)
    wire = Wire(rejection, _decision_reply("ok"))
    request = _request(1)
    with pytest.raises(BackendError) as caught:
        wire.agent.generate(request)
    error = caught.value
    assert type(error) is BackendError and failure_kind(error) == "transport"
    assert str(error) == f"Responses endpoint rejected request with HTTP {status}"
    assert error.metadata["http_status"] == status
    assert error.metadata.get("retry_after_s", "absent") == (
        7.0 if status in (429, 503) else "absent"
    )
    # The rejected body is never kept: an echoed credential cannot reach the decision log.
    assert secret not in repr((str(error), error.metadata, error.usage))
    wire.agent.generate(request)
    assert wire.sent[0].content == wire.sent[1].content


@pytest.mark.parametrize("headers, expected", [
    ({"Retry-After": "3"}, 3.0),
    ({"retry-after-ms": "1500", "Retry-After": "9"}, 1.5),
    ({"retry-after-ms": "soon", "Retry-After": "9"}, 9.0),
    ({"Retry-After": "-5"}, 0.0),
    ({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, 0.0),
    ({"Retry-After": "nan"}, None),
    ({"Retry-After": "later"}, None),
    ({}, None),
])
def test_retry_after_parsing(headers, expected):
    assert retry_after_seconds(headers) == expected


@pytest.mark.parametrize("reply, kind", [
    (httpx.Response(200, content=b"{not json", headers={"content-type": "application/json"}),
     "transport"),
    (httpx.Response(200, json=[_call("call_x")]), "transport"),
    (_reply(_call("call_x"), status="incomplete",
            incomplete_details={"reason": "max_output_tokens"}), "transport"),
    (_reply(status="failed"), "transport"),
    (_reply({**_call("call_x"), "status": "in_progress"}), "transport"),
    (_reply({"type": "message", "role": "assistant",
             "content": [{"type": "refusal", "refusal": "I cannot help."}]}), "transport"),
    ({**_reply(), "output": None}, "transport"),
    (_reply(_call("call_x"), _call("call_x")), "transport"),
    (_reply({**_call("call_x"), "call_id": ""}), "transport"),
    (_reply({**_call("call_x"), "arguments": {"deltas": {"dz": 0.001}}}), "transport"),
    ({**_reply(_call("call_x")), "usage": {"input_tokens": "many"}}, "transport"),
    (httpx.ConnectError("connection refused"), "transport"),
    (httpx.ReadTimeout("read timed out"), "transport"),
    (httpx.RemoteProtocolError("peer closed connection"), "transport"),
    (httpx.LocalProtocolError("invalid request"), "internal"),
], ids=["malformed_json", "array_envelope", "incomplete", "failed", "incomplete_call",
        "refusal", "no_output_array", "duplicate_call_ids", "empty_call_id",
        "non_string_arguments", "invalid_usage", "connect_error", "read_timeout",
        "remote_protocol_error", "local_protocol_error"])
def test_failed_response_commits_nothing(api_key, reply, kind):
    wire = Wire(reply, _decision_reply("ok"), _decision_reply("next"))
    request = _request(1)
    with pytest.raises(Exception) as caught:
        wire.agent.generate(request)
    assert failure_kind(caught.value) == kind
    if not isinstance(reply, BaseException):
        assert type(caught.value) is BackendError
    wire.agent.generate(request)
    assert wire.sent[0].content == wire.sent[1].content
    wire.agent.generate(_request(2, tool_results=[_result("call_ok")]))
    assert [item.get("type", item.get("role")) for item in wire.body()["input"]] == [
        "user", "reasoning", "function_call", "function_call_output", "user"
    ]


@pytest.mark.parametrize("output, calls, text", [
    ([_reasoning("x"), _message("Let me think about it.")], [], "Let me think about it."),
    ([_call("call_x"), _call("call_y")], [_native("call_x"), _native("call_y")], ""),
    ([_call("call_x", name="shell")], [_native("call_x", name="shell")], ""),
], ids=["no_call", "two_calls", "unknown_tool"])
def test_wrong_call_count_is_a_format_error_after_commit(api_key, output, calls, text):
    wire = Wire(_reply(*output))
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
    expected = [wire.body(0)["input"][0], *output, *rejections]
    assert [_compact(item) for item in repair["input"][:-1]] == [_compact(i) for i in expected]
    assert repair["input"][-1]["content"][0]["text"] == _request(2).text


@pytest.mark.parametrize("excess", [0, 1])
def test_output_limit_counts_visible_text_and_arguments(api_key, excess):
    text = "Raising now."
    observed = len(text) + len(MOVE_ARGS)
    wire = Wire(_reply(_message(text), _call("call_a")), max_response_chars=observed - excess)
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


def _failing_recorder(request, http_request):
    raise OSError("audit disk full")


@pytest.mark.parametrize("options, fields, key, closed", [
    ({}, {}, None, False),
    ({}, {}, "two words", False),
    ({}, {}, "k\u00e9y-0123456789", False),
    ({}, {}, KEY, True),
    ({"supported_efforts": ("high",)}, {"effort": "low"}, KEY, False),
    ({}, {"effort": "lowest"}, KEY, False),
    ({}, {"tools": ()}, KEY, False),
    ({}, {"tool_results": [_result("call_unknown")]}, KEY, False),
    ({"request_recorder": _failing_recorder}, {}, KEY, False),
    ({"request_recorder": lambda request, http_request: ["artifact"]}, {}, KEY, False),
], ids=["missing_key", "key_with_space", "non_ascii_key", "closed", "unsupported_effort",
        "lowest_without_supported_efforts", "no_tools",
        "orphan_tool_result", "failing_recorder", "non_dict_recorder_artifact"])
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


@pytest.mark.parametrize("configured, accepted", [
    (None, True),
    (OFFICIAL, True),
    (OFFICIAL + "/", True),
    (ENDPOINT, False),
    ("http://api.openai.com/v1", False),
])
def test_openai_base_url_must_name_the_official_endpoint(monkeypatch, configured, accepted):
    monkeypatch.setenv(KEY_ENV, KEY)
    if configured is not None:
        monkeypatch.setenv("OPENAI_BASE_URL", configured)
    wire = Wire(_decision_reply("a"))
    if accepted:
        wire.agent.generate(_request(1))
    else:
        with pytest.raises(BackendConfigurationError, match="OPENAI_BASE_URL"):
            wire.agent.generate(_request(1))
    assert len(wire.sent) == int(accepted)


def test_call_id_reused_by_a_later_attempt_is_rejected_without_commit(api_key):
    wire = Wire(
        _reply(_call("call_x"), _call("call_y")),
        _reply(_call("call_x")),
        _reply(_call("call_z")),
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


def _output(*items):
    return AgentOutput(payload={}, usage={}, metadata={}, elapsed_s=0.0, tool_calls=(),
                       output_items=tuple(items), response_text="")


def test_preview_commit_rejects_a_reused_call_id_without_changing_history():
    """accept_reply applies generate()'s reused-call-ID rule, without any HTTP or key."""
    agent = ResponsesAgent(MODEL, transport=UNUSED_TRANSPORT)
    agent.accept_reply(_request(1, decision="d1"), _output(_call("call_x"), _call("call_y")))
    rejections = [_result("call_x", "rejected"), _result("call_y", "rejected")]
    repair = _request(2, decision="d1", tool_results=rejections)
    before = json.dumps(agent.prepare_request(repair))
    with pytest.raises(BackendError, match="reused a function call ID"):
        agent.accept_reply(repair, _output(_reasoning("2"), _call("call_y")))
    assert json.dumps(agent.prepare_request(repair)) == before
    agent.accept_reply(repair, _output(_call("call_z")))
    assert [item["call_id"] for item in agent.prepare_request(
        _request(3, tool_results=[_result("call_z")]))["input"]
        if item.get("type") == "function_call"] == ["call_x", "call_y", "call_z"]


@pytest.mark.parametrize("window, replayed", [
    (-1, []), (0, []), (1, ["3"]), (2, ["2a", "2b", "3"]), (5, ["1", "2a", "2b", "3"]),
])
def test_history_window_counts_decisions_and_replays_reasoning_verbatim(
    api_key, window, replayed
):
    wire = Wire(
        _decision_reply("1"),
        _reply(_reasoning("2a"), _message("No call yet.")),
        _decision_reply("2b"),
        _decision_reply("3"),
    )
    generate = wire.agent.generate
    generate(_request(1, history_length=window))
    with pytest.raises(BackendFormatError):
        generate(_request(2, decision="d2", history_length=window,
                          tool_results=[_result("call_1")]))
    generate(_request(3, decision="d2", history_length=window))
    generate(_request(4, history_length=window, tool_results=[_result("call_2b")]))
    body = wire.agent.prepare_request(
        _request(5, history_length=window, tool_results=[_result("call_3")])
    )
    reasoning = [_compact(item) for item in body["input"] if item.get("type") == "reasoning"]
    assert reasoning == [_compact(_reasoning(tag)) for tag in replayed]
    users = [item for item in body["input"] if item.get("role") == "user"]
    assert len(users) == len(replayed) + 1


@pytest.mark.parametrize("model", ["openai/gpt-wire-test", "gpt-wire-test", "OpenAI/gpt-wire-test"])
def test_openai_prefix_is_stripped_for_the_official_endpoint(model):
    agent = ResponsesAgent(model, transport=UNUSED_TRANSPORT)
    assert agent.model == "gpt-wire-test"
    assert agent.route == {
        "selector": "openai/gpt-wire-test", "provider": "openai", "protocol": "responses",
        "wire_model": "gpt-wire-test", "endpoint": OFFICIAL, "api_key_env": KEY_ENV,
        "base_url_env": "OPENAI_BASE_URL",
    }
    assert agent.prepare_request(_request(1))["model"] == "gpt-wire-test"
    preview = agent.prepare_http_request(_request(1))
    assert str(preview.url) == f"{OFFICIAL}/responses"


@pytest.mark.parametrize("model, message", [
    ("openai/", "names no model after openai/"),
    ("openai/vendor/model", "one model ID without '/' or whitespace"),
    ("openai/gpt wire", "one model ID without '/' or whitespace"),
    ("wire-test", "Unknown model 'wire-test'"),
])
def test_invalid_selector_is_rejected_before_any_request(model, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        ResponsesAgent(model, transport=UNUSED_TRANSPORT)


def test_request_preview_is_auth_free_and_reads_no_environment(monkeypatch, api_key):
    wire = Wire(_decision_reply("a"), _decision_reply("b"))
    wire.agent.generate(_request(1))
    following = _request(2, tool_results=[_result("call_a")])
    monkeypatch.delenv(KEY_ENV)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", ForbiddenEnviron())
        preview = wire.agent.prepare_http_request(following)
        again = wire.agent.prepare_http_request(following)
    assert not AUTH_HEADERS & set(preview.headers)
    assert preview.content == again.content
    # Previewing changed no history: the real request sends exactly the previewed bytes.
    monkeypatch.setenv(KEY_ENV, KEY)
    wire.agent.generate(following)
    assert wire.sent[-1].content == preview.content
    assert len(wire.sent) == 2


def test_api_key_is_resolved_from_the_environment_at_each_send(monkeypatch):
    wire = Wire(_decision_reply("a"), _decision_reply("b"))
    requests = [_request(1), _request(2, tool_results=[_result("call_a")])]
    for request, key in zip(requests, ("first-key-0001", "second-key-0002")):
        monkeypatch.setenv(KEY_ENV, key)
        wire.agent.generate(request)
        assert wire.sent[-1].headers["authorization"] == f"Bearer {key}"
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(BackendConfigurationError):
        wire.agent.generate(_request(3, tool_results=[_result("call_b")]))
    assert len(wire.sent) == 2


@pytest.mark.parametrize("options, effort, reasoning, extra", [
    ({}, "low", {"effort": "low"}, {}),
    ({"default_effort": "medium"}, "default", {"effort": "medium"}, {}),
    ({}, "default", None, {}),
    ({"supported_efforts": ("high", "minimal", "low")}, "lowest", {"effort": "minimal"}, {}),
    ({"reasoning_mode": "pro"}, "high", {"effort": "high", "mode": "pro"}, {}),
    ({"max_output_tokens": 512}, "low", {"effort": "low"}, {"max_output_tokens": 512}),
])
def test_reasoning_options_on_the_wire(options, effort, reasoning, extra):
    body = Wire(**options).agent.prepare_request(_request(1, effort=effort))
    assert body.get("reasoning") == reasoning
    assert {k: body[k] for k in ("max_output_tokens",) if k in body} == extra


@pytest.mark.parametrize("returned, verified", [("pro", True), ("standard", False)])
def test_returned_reasoning_mode_must_match_the_request(api_key, returned, verified):
    wire = Wire(_reply(_call("call_a"), reasoning={"mode": returned}), reasoning_mode="pro")
    if verified:
        assert wire.agent.generate(_request(1)).metadata["reasoning_mode_verified"] is True
        return
    with pytest.raises(BackendError, match="different reasoning mode") as caught:
        wire.agent.generate(_request(1))
    assert failure_kind(caught.value) == "transport"


@pytest.mark.parametrize("failures, sleeps, sources, accepted", [
    ([(429, {"Retry-After": "30"}), (500, {})], [30.0, 2.25], ["retry_after", "backoff"], True),
    ([(429, {"Retry-After": "900"})], [300.0], ["retry_after"], True),
    ([(503, {"Retry-After": "0"})] * 3, [1.125, 2.25], ["backoff", "backoff"], False),
])
def test_policy_resends_http_failures_unchanged_after_backoff(
    api_key, monkeypatch, failures, sleeps, sources, accepted
):
    slept = []
    monkeypatch.setattr(policy_module, "_sleep", slept.append)
    monkeypatch.setattr(policy_module, "_jitter", lambda: 0.5)
    rejections = [httpx.Response(status, headers=headers, json={"error": {}})
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
    assert [d["retry_wait_s"] for d in attempts] == [0.0, *sleeps]
    assert [d["retry_wait_source"] for d in attempts] == [None, *sources]
    assert [d.get("error_kind") for d in attempts[:len(failures)]] == ["transport"] * len(failures)
    # Every attempt of the decision resent the same bytes, carrying the pending receipt once.
    assert len({request.content for request in wire.sent[1:]}) == 1
    types = [item.get("type") for item in wire.body()["input"]]
    assert types.count("function_call_output") == 1


def test_policy_never_resends_configuration_errors():
    wire = Wire(_decision_reply("1"))
    policy = _policy(wire)
    with pytest.raises(BackendConfigurationError):
        policy.act(_observation(0))
    assert (wire.sent, policy.calls) == ([], 1)
    assert policy.decisions[0]["error_kind"] == "configuration"
