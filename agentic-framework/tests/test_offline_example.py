"""The shipped offline example: a scripted policy/controller loop with no network request."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime

import numpy as np

INSTRUCTION = "Make two small movements."
NOTE = "Make one small movement, then observe again."
SCRIPTED_DX = (0.001, -0.0005)
# Normalized actions: each physical dx divided by ActionSpec().pose_scale (0.05 m).
EXPECTED_ACTIONS = [[0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
TIMESTAMP_KEYS = ("request_started_at", "request_finished_at")
DURATION_KEYS = (
    "request_wall_s", "elapsed_s", "context_build_s", "request_prepare_s",
    "policy_postprocessing_s", "policy_elapsed_s",
)
REQUIRED_ROW_KEYS = frozenset({
    "request_id", "logical_decision_id", "observation_step", "kind", "attempt",
    "retry_reason", "retry_wait_s", "step_budget", "context_text", "context_messages",
    "tool_results", "tool_calls", "outcome", "backend", "usage", "image_sha256",
    "telemetry", *TIMESTAMP_KEYS, *DURATION_KEYS,
})


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


def _assert_report(report):
    assert set(report) == {
        "model_calls", "network_requests", "scripted_decisions", "physical_steps",
        "termination", "actions",
    }
    # The policy's model calls, as policy-metrics.json counts them, are all scripted.
    assert report["model_calls"] == 2 and report["network_requests"] == 0
    assert report["scripted_decisions"] == 2
    assert report["physical_steps"] == 2
    assert report["termination"] == "max_steps"
    assert np.allclose(report["actions"], EXPECTED_ACTIONS, rtol=0, atol=1e-12)


def _receipt(index):
    """The only model-visible result of an accepted motion: its plan id and status."""
    return {
        "type": "function_call_output",
        "call_id": f"offline-example-{index:05d}-call-0",
        "output": json.dumps({"plan_id": f"offline-example-{index:05d}", "status": "accepted"}),
    }


def _stable(row):
    """Check volatile timing fields, then drop them so the row can be compared exactly."""
    started, finished = (datetime.fromisoformat(row[key]) for key in TIMESTAMP_KEYS)
    assert started.tzinfo is not None and started <= finished
    for key in DURATION_KEYS:
        assert type(row[key]) is float and math.isfinite(row[key]) and row[key] >= 0, key
    return {
        key: value for key, value in row.items()
        if key not in (*TIMESTAMP_KEYS, *DURATION_KEYS, "telemetry")
    }


def test_example_script_reports_two_scripted_steps(repo_root):
    completed = subprocess.run(
        [sys.executable, "-B", "examples/offline.py"], cwd=repo_root,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    _assert_report(json.loads(completed.stdout))


def test_scripted_loop_decision_log(repo_root, tmp_path, monkeypatch, capsys):
    """The example loop with a trial log directory writes one stable record per decision."""
    example = _load_example(repo_root)
    trial = tmp_path / "trial"
    policies = []
    policy_class = example.LLMMotionPolicy

    def with_log_dir(*args, **kwargs):
        policies.append(policy_class(*args, log_dir=trial, **kwargs))
        return policies[-1]

    monkeypatch.setattr(example, "LLMMotionPolicy", with_log_dir)
    example.main()
    _assert_report(json.loads(capsys.readouterr().out))
    (policy,) = policies
    assert isinstance(policy.agent, example.ScriptedAgent)
    requests = policy.agent.requests

    rows = [json.loads(line) for line in (trial / "decisions.jsonl").read_text().splitlines()]
    assert rows == json.loads((trial / "decisions.json").read_text())
    assert len(rows) == len(requests) == 2
    blank = hashlib.sha256(np.zeros((4, 4, 3), dtype=np.uint8).tobytes()).hexdigest()
    for index, (row, request, dx) in enumerate(zip(rows, requests, SCRIPTED_DX), start=1):
        assert REQUIRED_ROW_KEYS <= set(row)
        assert not {"error", "error_kind", "validation_error"} & set(row)
        stable = _stable(row)
        request_id = f"offline-example-{index:05d}"
        budget = {"steps_remaining": 3 - index, "max_steps": 2}
        expected = {
            "request_id": request_id,
            "logical_decision_id": f"offline-example-decision-{index:05d}",
            "observation_step": index - 1,
            "kind": "plan",
            "attempt": 1,
            "retry_reason": None,
            "retry_wait_s": 0.0,
            "retry_wait_source": None,
            "step_budget": budget,
            "outcome": "plan",
            "usage": {},
            "image_sha256": {"current/agentview": blank},
            "tool_results": [_receipt(index - 1)] if index > 1 else [],
        }
        assert {key: stable[key] for key in expected} == expected
        (call,) = stable["tool_calls"]
        assert {key: call[key] for key in ("type", "call_id", "name")} == {
            "type": "function_call", "call_id": f"{request_id}-call-0", "name": "move_by_chunk",
        }
        assert json.loads(call["arguments"]) == {
            "deltas": {"dx": [dx], "gripper": [0.0]}, "note": NOTE,
        }
        # The logged context is byte-identical to the request text the backend received.
        assert row["context_text"] == request.text
        assert json.loads(request.text) == {
            "instruction": INSTRUCTION,
            "current_observation": {
                "state": {
                    "eef_pos": [0.0, 0.0, 0.2], "eef_quat": [0.0, 0.0, 0.0, 1.0],
                    "gripper_width": [0.04],
                },
                "step_budget": budget,
                "images": ["current/agentview"],
            },
        }
        assert [dict(result) for result in request.tool_results] == expected["tool_results"]

    first, second = requests
    assert first.instructions == second.instructions
    assert first.tools == second.tools
    assert [tool["name"] for tool in first.tools] == ["move_by_chunk", "done", "give_up"]

    transcript = [
        {key: value for key, value in row.items() if key != "elapsed_s"}
        for row in policy.transcript()
    ]
    assert transcript == [
        {"request_id": f"offline-example-{index:05d}", "step": index - 1, "kind": "plan",
         "outcome": "plan", "error": None}
        for index in (1, 2)
    ]

    execution = json.loads((trial / "execution.json").read_text())
    assert [action["t"] for action in execution["actions"]] == [0, 1]
    assert np.allclose(
        [action["action"] for action in execution["actions"]], EXPECTED_ACTIONS, rtol=0, atol=1e-12
    )
    assert [
        (item["plan_id"], item["status"], item["steps_executed"]) for item in execution["feedback"]
    ] == [("offline-example-00001", "exhausted", 1), ("offline-example-00002", "episode_ended", 1)]

    metrics = json.loads((trial / "policy-metrics.json").read_text())
    assert {
        key: metrics[key] for key in (
            "model_calls", "planning_calls", "format_repair_calls", "transport_retry_calls",
            "internal_retry_calls", "retry_wait_s", "outcomes", "inference_control_steps",
            "executed_actions", "log_errors",
        )
    } == {
        "model_calls": 2, "planning_calls": 2, "format_repair_calls": 0,
        "transport_retry_calls": 0, "internal_retry_calls": 0, "retry_wait_s": 0.0,
        "outcomes": {"plan": 2},
        "inference_control_steps": [0, 1], "executed_actions": 2, "log_errors": [],
    }
    assert metrics["context"]["observation_profile"] == "control"
    assert metrics["context"]["demo"] is False

