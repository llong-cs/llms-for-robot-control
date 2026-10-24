"""Golden ``decisions.jsonl`` logs of two offline rollouts.

A golden is the decision log a trial directory receives, one row per line in the policy's
own serialization, with clock readings and installation-dependent byte counts replaced by
placeholders after their types are checked. ``offline_example`` runs the shipped
``examples/offline.py``. ``responses_repair_retry`` drives ResponsesAgent over
``httpx.MockTransport`` through a format repair, a backoff resend and a Retry-After resend.

After an intended change to the decision record, regenerate the goldens (the regenerating
tests report as skipped), review the diff, then rerun without the variable:

    AGENTIC_UPDATE_GOLDEN=1 python -B -m pytest -q agentic-framework/tests/test_decision_golden.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
from inspect_robots.approver import AutoApprover
from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import rollout
from inspect_robots.scene import Scene
from inspect_robots.types import Observation, StepResult

from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness import policy as policy_module
from agentic_framework.harness.controller import FrameworkController
from agentic_framework.harness.policy import LLMMotionPolicy
from agentic_framework.harness.types import ActionSpec, ScheduleConfig
from agentic_framework.models.llm.responses import ResponsesAgent

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "decisions"
UPDATE_GOLDEN = "AGENTIC_UPDATE_GOLDEN"
# The official provider's key variable; the shared conftest removes it from every test.
KEY_ENV = "OPENAI_API_KEY"
KEY = "decision-golden-key-0123456789"
INSTRUCTION = "Make two small movements."
SCHEDULE = ScheduleConfig(control_interface="move_by_chunk", h=1, k=1, max_steps=2)
# Measured clocks, wall-clock timestamps and sizes of PNG- or httpx-encoded bodies.
SECONDS = frozenset({
    "request_wall_s", "context_build_s", "request_prepare_s", "elapsed_s",
    "backend_reported_elapsed_s", "policy_postprocessing_s", "policy_elapsed_s",
    "client_elapsed_s", "backend_elapsed_s", "request_start_wall_s", "round_trip_s",
    "http_latency_s",
})
TIMESTAMPS = frozenset({"request_started_at", "request_finished_at"})
UNIX_TIMES = frozenset({"started_at_unix_s", "ended_at_unix_s"})
BYTE_COUNTS = frozenset({
    "request_body_bytes", "response_body_bytes", "request_bytes", "response_bytes",
    "content-length",
})


def _placeholder(key, value):
    """Check one volatile value and return its placeholder, or None to keep the value."""
    if value is None:
        return None
    if key in SECONDS:
        assert type(value) is float and math.isfinite(value) and value >= 0, (key, value)
        return "<seconds>"
    if key in TIMESTAMPS:
        assert datetime.fromisoformat(value).utcoffset() == timedelta(0), (key, value)
        return "<utc>"
    if key in UNIX_TIMES:
        assert type(value) is float and value > 0, (key, value)
        return "<unix-time>"
    if key in BYTE_COUNTS:
        assert type(value) in (int, str) and str(value).isdigit() and int(value) > 0, (key, value)
        return "<bytes>"
    return None


def _normalized(value):
    if isinstance(value, dict):
        return {key: _placeholder(key, item) or _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def _serialized(row):
    """The policy's decisions.jsonl serialization of one row."""
    return json.dumps(row, ensure_ascii=False, allow_nan=False)


def _difference(expected, actual, path):
    """Return the JSON path of the first difference between two decoded values, or None."""
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


def _assert_golden(name, log):
    """Compare a trial's decisions.jsonl, normalized, with golden/decisions/<name>.jsonl."""
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines and all(_serialized(json.loads(line)) == line for line in lines)
    rows = [_normalized(json.loads(line)) for line in lines]
    text = "".join(_serialized(row) + "\n" for row in rows)
    path = GOLDEN_DIR / f"{name}.jsonl"
    if os.environ.get(UPDATE_GOLDEN) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        pytest.skip(f"regenerated {path.name}; review the diff, then rerun without {UPDATE_GOLDEN}")
    if not path.is_file():
        pytest.fail(f"missing golden {path.name}; regenerate with {UPDATE_GOLDEN}=1")
    golden = path.read_text(encoding="utf-8")
    if text != golden:
        expected = [json.loads(line) for line in golden.splitlines()]
        where = _difference(expected, rows, "rows") or "byte serialization"
        pytest.fail(f"{path.name}: decision log differs at {where}; if intended, regenerate "
                    f"with {UPDATE_GOLDEN}=1 and review the diff", pytrace=False)


def _load_example(repo_root):
    """Import examples/offline.py without bytecode or the sys.path entries it inserts."""
    spec = importlib.util.spec_from_file_location(
        "offline_example", repo_root / "examples" / "offline.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:], sys.dont_write_bytecode = saved_path, saved_flag
    return module


class ToyRobot:
    """The offline example's motionless robot: one blank camera and a fixed state."""

    def __init__(self):
        self.info = LiberoEmbodiment(image_size=4).info
        self.last_reset_info = {"success": False}
        self.actions = []

    def observation(self):
        return Observation(
            images={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            state={"eef_pos": np.array([0.0, 0.0, 0.2]),
                   "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                   "gripper_width": np.array([0.04])},
            instruction=INSTRUCTION, extra={"env_step": len(self.actions)},
        )

    def reset(self, scene, *, seed=None):
        self.actions.clear()
        return self.observation()

    def step(self, action):
        self.actions.append(action.data.copy())
        return StepResult(self.observation())


def _reply(tag, arguments, *, text=None):
    """A completed Responses envelope with reasoning, optional text and one chunk call."""
    output = [{"type": "reasoning", "id": f"rs_{tag}", "summary": [],
               "encrypted_content": f"opaque-reasoning-{tag}"}]
    if text is not None:
        output.append({"type": "message", "id": f"msg_{tag}", "role": "assistant",
                       "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]})
    output.append({"type": "function_call", "id": f"fc_{tag}", "call_id": f"call_{tag}",
                   "name": "move_by_chunk", "arguments": json.dumps(arguments),
                   "status": "completed"})
    return httpx.Response(200, json={
        "id": f"resp_{tag}", "object": "response", "model": "gpt-golden-test",
        "status": "completed", "output": output,
        "usage": {"input_tokens": 900, "input_tokens_details": {"cached_tokens": 600},
                  "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 25},
                  "total_tokens": 940},
    })


def test_offline_example_decision_log_matches_golden(repo_root, tmp_path, monkeypatch, capsys):
    example = _load_example(repo_root)
    trial = tmp_path / "trial"
    policy_class = example.LLMMotionPolicy
    monkeypatch.setattr(example, "LLMMotionPolicy",
                        lambda *args, **kwargs: policy_class(*args, log_dir=trial, **kwargs))
    example.main()
    assert json.loads(capsys.readouterr().out)["termination"] == "max_steps"
    _assert_golden("offline_example", trial / "decisions.jsonl")


def test_responses_repair_and_resend_decision_log_matches_golden(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(policy_module, "_sleep", waits.append)
    monkeypatch.setattr(policy_module, "_jitter", lambda: 0.5)
    monkeypatch.setenv(KEY_ENV, KEY)
    replies = [
        _reply("1", {"deltas": {"dx": [0.001]}, "note": "Nudge toward +x."}, text="Moving."),
        _reply("2a", {"deltas": {"dx": [-0.0005]}}),
        httpx.Response(500, json={"error": {"message": "overloaded"}}),
        httpx.Response(429, headers={"Retry-After": "3"}, json={"error": {}}),
        _reply("2b", {"deltas": {"dx": [-0.0005]}, "note": "Back off slightly."}),
    ]
    agent = ResponsesAgent(
        "gpt-golden-test", transport=httpx.MockTransport(lambda request: replies.pop(0)),
    )
    trial = tmp_path / "trial"
    robot = ToyRobot()
    policy = LLMMotionPolicy(agent, SCHEDULE, action_spec=ActionSpec(), max_attempts=4,
                             log_dir=trial)
    policy.bind(robot.info)
    try:
        record = rollout(
            policy, robot, Scene("golden", INSTRUCTION), max_steps=SCHEDULE.max_steps, seed=0,
            epoch=0, controller=FrameworkController(SCHEDULE, spec=policy.spec),
            approver=AutoApprover(), sink=NullSink(),
        )
    finally:
        policy.close()
    assert (record.termination_reason, len(robot.actions), replies) == ("max_steps", 2, [])
    assert waits == [1.125, 3.0]
    assert [(row["retry_reason"], row.get("error_kind"), row.get("outcome"))
            for row in policy.decisions] == [
        (None, None, "plan"), (None, "format", "format_error"), ("format", "transport", None),
        ("transport", "transport", None), ("transport", None, "plan"),
    ]
    assert all(KEY not in path.read_text() for path in trial.rglob("*") if path.is_file())
    _assert_golden("responses_repair_retry", trial / "decisions.jsonl")
