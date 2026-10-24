"""Per-decision attempt loop of LLMMotionPolicy and the runner's discarded-trial mapping."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from dataclasses import replace

import httpx
import numpy as np
import pytest
from inspect_robots.approver import AutoApprover
from inspect_robots.errors import PolicyStop
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import rollout
from inspect_robots.scene import Scene
from inspect_robots.types import Observation, StepResult

from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness import policy as policy_module
from agentic_framework.harness.controller import FrameworkController
from agentic_framework.harness.policy import LLMMotionPolicy, retry_wait
from agentic_framework.harness.run import result_row, unscore_discarded
from agentic_framework.harness.types import (
    ActionSpec,
    AgentOutput,
    ExecutionFeedback,
    MotionPlan,
    ScheduleConfig,
)
from agentic_framework.models.llm.errors import (
    BackendConfigurationError,
    BackendError,
    BackendFormatError,
    BackendOutputLimitError,
    http_status_error,
)
from agentic_framework.models.llm.scripted import ScriptedAgent

# A fixed jitter of 0.5 makes every backoff factor 1 + 0.25 * 0.5 = 1.125 (exact in binary).
JITTER = 0.5
ZERO_CALLS = "respond with exactly one native robot tool call"
CHUNK_SCHEDULE = ScheduleConfig(control_interface="move_by_chunk", h=1, k=1, max_steps=2)


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """Record every backoff wait instead of sleeping, with a fixed jitter."""
    recorded = []
    monkeypatch.setattr(policy_module, "_sleep", recorded.append)
    monkeypatch.setattr(policy_module, "_jitter", lambda: JITTER)
    return recorded


def _observation(step=0):
    return Observation(
        images={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state={
            "eef_pos": np.array([0.0, 0.0, 0.2]),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "gripper_width": np.array([0.04]),
        },
        instruction="Make a small movement.",
        extra={"env_step": step},
    )


class ToyRobot:
    """A deterministic embodiment with the LIBERO action contract and no simulator."""

    def __init__(self):
        self.info = LiberoEmbodiment(image_size=4).info
        self.last_reset_info = {"success": False}
        self.actions = []

    def reset(self, scene, *, seed=None):
        self.actions.clear()
        return _observation(0)

    def step(self, action):
        self.actions.append(action.data.copy())
        return StepResult(_observation(len(self.actions)))


def _motion(dx=0.001):
    """A valid scripted motion; ScriptedAgent adapts it to the configured motion tool."""
    arguments = {"deltas": {"dx": dx}, "note": "Nudge."}
    return {"commands": [{"name": "move_by", "arguments": arguments}]}


def _call(name, arguments, call_id):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _policy(script, *, max_attempts=3, schedule=None, **kwargs):
    agent = ScriptedAgent(script)
    policy = LLMMotionPolicy(
        agent, schedule or ScheduleConfig(), action_spec=ActionSpec(),
        max_attempts=max_attempts, **kwargs,
    )
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("trial", "Make a small movement."))
    return policy, agent


def _repair(error):
    return (
        "\nYour previous response failed validation: " + error
        + "\nMake one corrected native tool call for this unchanged observation."
    )


def _receipt(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _rejected(call_id, error):
    return _receipt(
        call_id, '{"status": "rejected", "executed": false, "error": ' + json.dumps(error) + "}"
    )


def _assert_resent(previous, resent):
    """A resend differs from the failed request only in its per-call request_id."""
    assert resent.request_id != previous.request_id
    assert replace(resent, request_id="", images={}) == replace(previous, request_id="", images={})
    assert resent.images.keys() == previous.images.keys()
    for name, image in previous.images.items():
        assert np.array_equal(resent.images[name], image)


def _attempt_fields(policy, *names):
    return [tuple(entry.get(name) for name in names) for entry in policy.decisions]


@pytest.mark.parametrize(
    ("failure", "error", "rejected_ids"),
    [
        pytest.param({"tool_calls": []}, ZERO_CALLS, (), id="zero-calls"),
        pytest.param(
            {"tool_calls": [
                _call("move_by", {"deltas": {"dx": 0.001}, "note": "a"}, "call-a"),
                _call("move_by", {"deltas": {"dy": 0.001}, "note": "b"}, "call-b"),
            ]},
            ZERO_CALLS, ("call-a", "call-b"), id="two-calls",
        ),
        pytest.param(
            {"tool_calls": [_call("teleport", {}, "call-a")]},
            "native robot tool is not enabled; use move_by, done or give_up", ("call-a",),
            id="unknown-tool",
        ),
        pytest.param(
            {"tool_calls": [
                _call("move_by", {"deltas": {"dx": 0.001}, "note": "n", "speed": 2}, "call-a"),
            ]},
            "unknown motion argument", ("call-a",), id="invalid-arguments",
        ),
        pytest.param(
            lambda: BackendFormatError(
                "response contained no tool call", response_text="Thinking aloud.",
                usage={"output_tokens": 3},
                metadata={"tool_calls": [_call("move_by", "{", "call-a")]},
            ),
            "response contained no tool call", ("call-a",), id="backend-format-error",
        ),
    ],
)
def test_format_failure_is_repaired_without_wait_on_the_same_snapshot(
    waits, failure, error, rejected_ids
):
    policy, agent = _policy([failure() if callable(failure) else failure, _motion()])

    result = policy._decide(_observation())

    assert isinstance(result, MotionPlan) and result.plan_id == "trial-00002"
    assert waits == []
    first, repair = agent.requests
    assert [first.request_id, repair.request_id] == ["trial-00001", "trial-00002"]
    assert first.decision_id == repair.decision_id == "trial-decision-00001"
    assert repair.text == first.text + _repair(error)
    assert repair.messages[:-1] == first.messages[:-1]
    assert repair.messages[-1]["content"] == [
        *first.messages[-1]["content"], {"type": "text", "text": _repair(error)},
    ]
    assert repair.images.keys() == first.images.keys()
    assert first.tool_results == ()
    assert repair.tool_results == tuple(_rejected(call_id, error) for call_id in rejected_ids)
    assert _attempt_fields(
        policy, "attempt", "error_kind", "validation_error", "outcome", "retry_reason",
        "retry_wait_s", "retry_wait_source",
    ) == [
        (1, "format", error, "format_error", None, 0.0, None),
        (2, None, None, "plan", "format", 0.0, None),
    ]
    if callable(failure):
        assert policy.decisions[0]["response_text"] == "Thinking aloud."


def test_motion_preflight_value_error_is_a_format_repair(waits):
    seen = []

    def validator(plan):
        seen.append(plan.plan_id)
        if len(seen) == 1:
            raise ValueError("target leaves the workspace")

    policy, agent = _policy([_motion(), _motion()], motion_plan_validator=validator)

    result = policy._decide(_observation())

    assert seen == ["trial-00001", "trial-00002"] and result.plan_id == "trial-00002"
    assert waits == []
    assert agent.requests[1].tool_results == (
        _rejected("trial-00001-call-0", "target leaves the workspace"),
    )
    assert agent.requests[1].text == agent.requests[0].text + _repair(
        "target leaves the workspace"
    )


def test_repeated_format_failures_carry_only_the_latest_feedback_and_receipts():
    unknown = "native robot tool is not enabled; use move_by, done or give_up"
    policy, agent = _policy([
        {"tool_calls": [_call("teleport", {}, "call-a")]},
        {"tool_calls": [_call("fly", {}, "call-b")]},
        _motion(),
    ])

    policy._decide(_observation())

    first, second, third = agent.requests
    assert second.text == first.text + _repair(unknown)
    assert third.text == first.text + _repair(unknown)
    assert second.tool_results == (_rejected("call-a", unknown),)
    assert third.tool_results == (_rejected("call-b", unknown),)


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        pytest.param(
            lambda: http_status_error("responses", httpx.Response(400)), "transport", id="http-400"
        ),
        pytest.param(lambda: TimeoutError("read timed out"), "transport", id="timeout"),
        pytest.param(lambda: ConnectionError("connection reset"), "transport", id="connection"),
        pytest.param(lambda: httpx.ReadTimeout("slow body"), "transport", id="httpx-timeout"),
        pytest.param(lambda: RuntimeError("encoder bug"), "internal", id="runtime-error"),
        pytest.param(lambda: KeyError("missing"), "internal", id="key-error"),
        pytest.param(
            lambda: httpx.LocalProtocolError("bad header"), "internal", id="httpx-local-protocol"
        ),
    ],
)
def test_transport_and_internal_failures_are_resent_unchanged_after_backoff(waits, failure, kind):
    policy, agent = _policy([failure(), _motion()])

    result = policy._decide(_observation())

    assert result.plan_id == "trial-00002"
    assert waits == [1.125]
    _assert_resent(*agent.requests)
    assert _attempt_fields(
        policy, "attempt", "error_kind", "retry_reason", "retry_wait_s", "retry_wait_source"
    ) == [(1, kind, None, 0.0, None), (2, None, kind, 1.125, "backoff")]


def test_resend_keeps_pending_results_and_repair_but_a_format_reply_consumes_them(waits):
    unknown = "native robot tool is not enabled; use move_by, done or give_up"
    policy, agent = _policy([
        _motion(),
        BackendError("HTTP 500"),
        {"tool_calls": [_call("teleport", {}, "call-x")]},
        TimeoutError("read timed out"),
        lambda request: _motion(),
    ], max_attempts=4)
    plan = policy._decide(_observation())
    policy.record_execution(ExecutionFeedback(plan.plan_id, "completed", steps_executed=5))

    policy._decide(_observation(5))

    accepted = _receipt("trial-00001-call-0", '{"plan_id": "trial-00001", "status": "accepted"}')
    _, transport, resent, repair, repair_resent = agent.requests
    assert transport.tool_results == (accepted,)
    _assert_resent(transport, resent)
    assert repair.tool_results == (_rejected("call-x", unknown),)
    assert repair.text == transport.text + _repair(unknown)
    _assert_resent(repair, repair_resent)
    assert {request.decision_id for request in agent.requests[1:]} == {"trial-decision-00002"}
    assert waits == [1.125, 2.25]


def test_transport_internal_and_format_failures_share_one_decision_budget(waits):
    policy, agent = _policy(
        [
            BackendError("HTTP 502"),
            RuntimeError("encoder bug"),
            {"tool_calls": []},
            TimeoutError("read timed out"),
            _motion(),
        ],
        max_attempts=5,
    )

    policy._decide(_observation())

    # Format repairs neither wait nor advance the backoff exponent.
    assert waits == [1.125, 2.25, 4.5]
    assert _attempt_fields(policy, "retry_reason", "retry_wait_s", "retry_wait_source") == [
        (None, 0.0, None),
        ("transport", 1.125, "backoff"),
        ("internal", 2.25, "backoff"),
        ("format", 0.0, None),
        ("transport", 4.5, "backoff"),
    ]


@pytest.mark.parametrize(("jitter", "factor"), [(0.0, 1.0), (JITTER, 1.125)])
def test_retry_wait_is_capped_exponential_backoff(monkeypatch, jitter, factor):
    monkeypatch.setattr(policy_module, "_jitter", lambda: jitter)

    values = [retry_wait(n) for n in range(1, 9)]

    assert values == [(min(60.0, 2 ** (n - 1) * factor), "backoff") for n in range(1, 9)]
    assert values[-1] == (60.0, "backoff")
    assert retry_wait(10**6) == (60.0, "backoff")


def test_retry_wait_prefers_a_longer_retry_after_up_to_300_seconds():
    assert retry_wait(1, 7) == (7.0, "retry_after")
    assert retry_wait(7, 100.0) == (100.0, "retry_after")
    assert retry_wait(1, 1000) == (300.0, "retry_after")
    assert retry_wait(1, 1.125) == (1.125, "backoff")
    assert retry_wait(3, 4.0) == (4.5, "backoff")
    for ignored in (None, float("nan"), True, "30"):
        assert retry_wait(1, ignored) == (1.125, "backoff")
    for invalid in (0, -1, True, 1.0, None):
        with pytest.raises(ValueError):
            retry_wait(invalid)


def test_retry_after_is_honoured_only_for_429_and_503(waits):
    policy, agent = _policy(
        [
            http_status_error("responses", httpx.Response(429, headers={"Retry-After": "7"})),
            http_status_error("responses", httpx.Response(503, headers={"retry-after-ms": "500"})),
            http_status_error("responses", httpx.Response(500, headers={"Retry-After": "120"})),
            http_status_error("responses", httpx.Response(429, headers={"Retry-After": "1000"})),
            _motion(),
        ],
        max_attempts=5,
    )

    policy._decide(_observation())

    assert waits == [7.0, 2.25, 4.5, 300.0]
    assert [entry["backend"] for entry in policy.decisions[:4]] == [
        {"retry_after_s": 7.0}, {"retry_after_s": 0.5}, {}, {"retry_after_s": 1000.0},
    ]
    assert [entry["retry_wait_source"] for entry in policy.decisions] == [
        None, "retry_after", "backoff", "backoff", "retry_after",
    ]
    assert policy.decisions[0]["error"] == (
        "BackendError: responses endpoint rejected request with HTTP 429"
    )


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        pytest.param(
            lambda: BackendConfigurationError("missing credentials"), "configuration",
            id="configuration",
        ),
        pytest.param(KeyboardInterrupt, "cancelled", id="keyboard-interrupt"),
        pytest.param(lambda: SystemExit(3), "cancelled", id="system-exit"),
        pytest.param(asyncio.CancelledError, "cancelled", id="asyncio-cancelled"),
        pytest.param(concurrent.futures.CancelledError, "cancelled", id="futures-cancelled"),
    ],
)
def test_configuration_errors_and_cancellation_are_never_retried(waits, failure, kind):
    error = failure()
    policy, agent = _policy([error, _motion()])

    with pytest.raises(type(error)) as raised:
        policy._decide(_observation())

    assert raised.value is error
    assert len(agent.requests) == 1
    assert waits == []
    assert _attempt_fields(policy, "attempt", "error_kind") == [(1, kind)]


def test_non_format_preflight_failure_ends_the_trial_without_resend(waits):
    def validator(plan):
        raise RuntimeError("simulator preflight failed")

    policy, agent = _policy([_motion(), _motion()], motion_plan_validator=validator)

    with pytest.raises(RuntimeError, match="simulator preflight failed"):
        policy._decide(_observation())

    assert len(agent.requests) == 1
    assert waits == []
    assert _attempt_fields(policy, "error_kind", "error") == [
        ("internal", "RuntimeError: simulator preflight failed"),
    ]


@pytest.mark.parametrize(
    ("max_attempts", "script", "expected_waits", "last"),
    [
        pytest.param(1, [BackendError("HTTP 500")], [], "BackendError: HTTP 500", id="single"),
        pytest.param(
            3, [{"tool_calls": []}, BackendError("HTTP 503"), RuntimeError("encoder bug")],
            [1.125], "RuntimeError: encoder bug", id="mixed",
        ),
        pytest.param(
            3, [BackendError("HTTP 500"), TimeoutError("timed out"), {"tool_calls": []}],
            [1.125, 2.25], f"format: {ZERO_CALLS}", id="format-last",
        ),
    ],
)
def test_exhausted_attempts_stop_the_episode_as_max_attempts(
    waits, max_attempts, script, expected_waits, last
):
    policy, agent = _policy(script, max_attempts=max_attempts)

    with pytest.raises(PolicyStop) as raised:
        policy.act(_observation())

    assert raised.value.name == "max_attempts"
    assert raised.value.detail == (
        f"decision trial-decision-00001 failed all {max_attempts} attempts (last: {last}); "
        "the episode is discarded"
    )
    assert len(agent.requests) == max_attempts
    # The last attempt is never followed by a wait.
    assert waits == expected_waits


@pytest.mark.parametrize("value", [0, -2, True, 2.0, None])
def test_max_attempts_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_attempts"):
        LLMMotionPolicy(ScriptedAgent([]), action_spec=ActionSpec(), max_attempts=value)


class _Recorder:
    """Trajectory recorder stand-in that can fail one kind of write."""

    managed = True
    path = "unused"

    def __init__(self, fail=None):
        self.fail = fail

    def inference_started(self, **kwargs):
        pass

    def append(self, *args):
        if self.fail == "append":
            raise OSError("disk full")

    def record_decision_timing(self, *args):
        if self.fail == "timing":
            raise OSError("timing sidecar unwritable")


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(lambda: BackendError("HTTP 502"), id="transport"),
        pytest.param(lambda: RuntimeError("encoder bug"), id="internal"),
    ],
)
def test_recording_failure_of_a_resendable_attempt_raises_the_recording_error(waits, failure):
    error = failure()
    policy, agent = _policy([error, _motion()])
    policy.trajectory_recorder = _Recorder(fail="append")

    with pytest.raises(OSError, match="disk full") as raised:
        policy._decide(_observation())

    assert raised.value.__cause__ is error
    assert not hasattr(error, "__notes__")
    assert policy.log_errors == []
    # An unrecorded failed attempt is never resent.
    assert len(agent.requests) == 1
    assert waits == []


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(KeyboardInterrupt, id="cancelled"),
        pytest.param(lambda: BackendConfigurationError("missing credentials"), id="configuration"),
    ],
)
def test_recording_failure_keeps_and_reraises_a_terminal_error(waits, failure):
    error = failure()
    policy, agent = _policy([error, _motion()])
    policy.trajectory_recorder = _Recorder(fail="append")

    with pytest.raises(type(error)) as raised:
        policy._decide(_observation())

    note = "Trajectory recording failed: OSError: disk full"
    assert raised.value is error
    assert raised.value.__notes__ == [note]
    assert policy.log_errors == [note]
    assert len(agent.requests) == 1
    assert waits == []


def test_recording_failure_after_a_failed_preflight_keeps_the_preflight_error(waits):
    error = RuntimeError("simulator preflight failed")

    def validator(plan):
        raise error

    policy, agent = _policy([_motion(), _motion()], motion_plan_validator=validator)
    policy.trajectory_recorder = _Recorder(fail="append")

    with pytest.raises(RuntimeError) as raised:
        policy._decide(_observation())

    note = "Trajectory recording failed: OSError: disk full"
    assert raised.value is error and raised.value.__notes__ == [note]
    assert len(agent.requests) == 1


def test_decision_timing_failure_is_noted_on_the_primary_error():
    error = BackendConfigurationError("missing credentials")
    policy, _ = _policy([error])
    policy.trajectory_recorder = _Recorder(fail="timing")

    with pytest.raises(BackendConfigurationError) as raised:
        policy._decide(_observation())

    note = "Trajectory recording failed: OSError: timing sidecar unwritable"
    assert raised.value is error and raised.value.__notes__ == [note]
    assert policy.log_errors == [note]


def test_policy_metrics_count_every_kind_of_failed_attempt(waits):
    policy, _ = _policy(
        [
            BackendOutputLimitError("response exceeded 100 chars", usage={"output_tokens": 100}),
            {"tool_calls": []},
            BackendError("HTTP 502", usage={"input_tokens": 7}),
            RuntimeError("encoder bug"),
            AgentOutput(_motion(), usage={"input_tokens": 11, "output_tokens": 5}),
            TimeoutError("read timed out"),
            BackendConfigurationError("missing credentials"),
        ],
        max_attempts=5,
    )
    policy._decide(_observation())
    with pytest.raises(BackendConfigurationError):
        policy._decide(_observation(5))

    metrics = policy.metrics()

    assert waits == [1.125, 2.25, 1.125]
    expected = {
        "model_calls": 7,
        "max_attempts": 5,
        "transport_retry_calls": 2,
        "transport_failures": 2,
        "internal_retry_calls": 1,
        "internal_failures": 1,
        "retry_wait_s": 4.5,
        "configuration_failures": 1,
        "planning_calls": 2,
        "format_repair_calls": 2,
        "output_limit_errors": 1,
        "format_errors": 2,
        "outcomes": {"output_limit": 1, "format_error": 1, "error": 4, "plan": 1},
        "usage": {"output_tokens": 105, "input_tokens": 18},
        "inference_control_steps": [0, 5],
        "log_errors": [],
    }
    assert {key: metrics[key] for key in expected} == expected


def _rollout(script, *, max_attempts=3):
    agent = ScriptedAgent(script)
    robot = ToyRobot()
    policy = LLMMotionPolicy(
        agent, CHUNK_SCHEDULE, action_spec=ActionSpec(), max_attempts=max_attempts
    )
    policy.bind(robot.info)
    scene = Scene("scene-0", "Make a small movement.", metadata={"init_state_index": 0})
    started = time.monotonic()
    try:
        record = rollout(
            policy, robot, scene, max_steps=CHUNK_SCHEDULE.max_steps, seed=0, epoch=0,
            controller=FrameworkController(CHUNK_SCHEDULE, spec=policy.spec),
            approver=AutoApprover(), sink=NullSink(),
        )
        return record, robot, policy, result_row(record, scene, policy, started, 20.0)
    finally:
        policy.close()


def test_attempt_budget_resets_after_each_accepted_decision(waits):
    record, robot, policy, row = _rollout([
        BackendError("HTTP 500"),
        {"tool_calls": []},
        _motion(0.001),
        TimeoutError("read timed out"),
        TimeoutError("read timed out"),
        _motion(-0.0005),
    ])

    assert record.termination_reason == "max_steps" and len(robot.actions) == 2
    # Both the attempt count and the backoff exponent restart for the second decision.
    assert waits == [1.125, 1.125, 2.25]
    assert _attempt_fields(policy, "logical_decision_id", "attempt") == [
        ("scene-0-decision-00001", 1), ("scene-0-decision-00001", 2),
        ("scene-0-decision-00001", 3), ("scene-0-decision-00002", 1),
        ("scene-0-decision-00002", 2), ("scene-0-decision-00002", 3),
    ]
    assert row["status"] == "success" and row["oracle_success"] is False
    # Only a discarded row is marked as never evaluated.
    assert row.get("task_success_evaluated", True) is True


def test_max_attempts_termination_becomes_a_discarded_unscored_row(waits):
    record, robot, policy, row = _rollout([
        _motion(),
        BackendError("HTTP 500"),
        BackendError("HTTP 500"),
        BackendError("HTTP 500"),
    ])

    assert len(robot.actions) == 1
    assert (record.status, record.termination_reason) == ("success", "max_attempts")
    assert record.metadata["policy_stop"]["name"] == "max_attempts"
    assert row["status"] == "discarded"
    assert row["termination_reason"] == "max_attempts"
    assert row["oracle_success"] is None
    assert row["task_success_evaluated"] is False
    assert row["success_phase"] is None
    assert (row["policy"]["model_calls"], row["policy"]["transport_failures"]) == (4, 3)


def _eval_log(samples):
    return EvalLog(
        version=EvalLog.SCHEMA_VERSION,
        status="success",
        eval=EvalSpec("task", "llm-motion", "toy", "2026-01-01T00:00:00Z", "0.0.0"),
        results=EvalResults(
            total_scenes=len(samples),
            total_trials=sum(len(sample.epochs) for sample in samples),
            metrics={"success_at_end": 0.25, "episode_length": 70.0},
        ),
        stats=EvalStats("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", 60.0, total_steps=0),
        samples=tuple(samples),
    )


def test_unscore_discarded_drops_discarded_epochs_from_every_aggregate():
    mixed = SceneResult(
        "a", "success", reduced={"success_at_end": 0.5, "episode_length": 60.0},
        epochs=({"success_at_end": 1.0, "episode_length": 40.0},
                {"success_at_end": 0.0, "episode_length": 80.0}),
        termination_reasons=("success", "max_attempts"),
    )
    only_discarded = SceneResult(
        "b", "success", reduced={"success_at_end": 0.0, "episode_length": 100.0},
        epochs=({"success_at_end": 0.0, "episode_length": 100.0},),
        termination_reasons=("max_attempts",),
    )
    kept = SceneResult(
        "c", "success", reduced={"success_at_end": 0.0, "episode_length": 50.0},
        epochs=({"success_at_end": 0.0, "episode_length": 50.0},),
        termination_reasons=("max_steps",),
    )

    log = unscore_discarded(_eval_log([mixed, only_discarded, kept]))

    a, b, c = log.samples
    assert a.epochs == ({"success_at_end": 1.0, "episode_length": 40.0}, {})
    assert a.reduced == {"success_at_end": 1.0, "episode_length": 40.0}
    assert (b.epochs, b.reduced) == (({},), {})
    assert c is kept
    assert log.results.metrics == {"success_at_end": 0.5, "episode_length": 45.0}
    untouched = _eval_log([kept])
    assert unscore_discarded(untouched) is untouched
