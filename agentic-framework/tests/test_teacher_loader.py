"""Teacher demonstrations loaded from trajectories written by the real recorder.

Each teacher is recorded offline: a scripted backend replays fixed native tool calls
with the reply metadata a model backend records, and a toy robot moves by the
commanded increment. Rejection cases edit a copy of a recorded document.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

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
from agentic_framework.harness.demonstration import Demonstration
from agentic_framework.harness.policy import LLMMotionPolicy
from agentic_framework.harness.types import (
    ActionSpec,
    AgentOutput,
    AugmentationConfig,
    MemoryConfig,
    ScheduleConfig,
)
from agentic_framework.models.llm.errors import BackendError
from agentic_framework.models.llm.scripted import ScriptedAgent

INSTRUCTION = "Nudge the gripper along x."
TEACHER_MODEL = "teacher-sentinel-model"
USAGE = {"input_tokens": 918273, "output_tokens": 37}
TEACHER_TEXT = "Moving a little toward +x first."
FIRST_NOTE, SECOND_NOTE = "Step toward +x.", "Back off slightly."
REJECTION = "native robot tool is not enabled; use move_by_chunk, done or give_up"
REPAIR = (
    f"Your previous response failed validation: {REJECTION}\n"
    "Make one corrected native tool call for this unchanged observation."
)
QUATERNION = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Record backoff waits instead of sleeping; a test that expects one patches its own."""
    waits = []
    monkeypatch.setattr(policy_module, "_sleep", waits.append)
    yield
    assert waits == [], f"unexpected backoff waits: {waits}"


def _state(x):
    return {"eef_pos": [x, 0.0, 0.2], "eef_quat": QUATERNION, "gripper_width": [0.04]}


class _Robot:
    """Single-arm toy robot: the end effector moves by each commanded increment."""

    def __init__(self):
        self.info = LiberoEmbodiment(image_size=4).info
        self.last_reset_info = {"success": False}
        self.steps = 0
        self.position = np.array([0.0, 0.0, 0.2])

    def observation(self):
        return Observation(
            images={"agentview": np.full((4, 4, 3), 40 * self.steps, dtype=np.uint8)},
            state={"eef_pos": self.position.copy(), "eef_quat": np.array(QUATERNION),
                   "gripper_width": np.array([0.04])},
            instruction=INSTRUCTION, extra={"env_step": self.steps},
        )

    def reset(self, scene, *, seed=None):
        self.steps, self.position = 0, np.array([0.0, 0.0, 0.2])
        return self.observation()

    def step(self, action):
        self.steps += 1
        self.position = self.position + np.asarray(ActionSpec().pose_scale[:3]) * action.data[:3]
        return StepResult(self.observation())


def _metadata():
    """Backend metadata as the Responses wire records it."""
    return {"backend": "responses", "model": TEACHER_MODEL,
            "history_length_semantics": "window_size"}


def _reply(name, arguments, metadata, text=""):
    """Respond with one native call, its usage and its backend metadata, like a model."""

    def respond(request):
        call = {"type": "function_call", "call_id": f"{request.request_id}-call-0",
                "name": name, "arguments": json.dumps(arguments)}
        return AgentOutput({"tool_calls": [call]}, usage=dict(USAGE), metadata=metadata,
                          tool_calls=(call,), output_items=(call,), response_text=text)

    return respond


def _motion(dx, note):
    return {"deltas": {"dx": [dx]}, "note": note}


def _script():
    """Accepted motion, a rejected call, then its repaired motion at the same observation."""
    metadata = [_metadata() for _ in range(3)]
    return [
        _reply("move_by_chunk", _motion(0.001, FIRST_NOTE), metadata[0], TEACHER_TEXT),
        _reply("teleport", {"x": 1}, metadata[1]),
        _reply("move_by_chunk", _motion(-0.0005, SECOND_NOTE), metadata[2]),
    ]


def _schedule(interface="move_by_chunk"):
    return ScheduleConfig(control_interface=interface, h=1, k=1, max_steps=2)


def _record(directory, script, *, history_length=2, max_attempts=3, demo_path=None,
            finalize=True):
    """Record one two-step move_by_chunk episode as --record-trajectory does."""
    schedule = _schedule()
    robot = _Robot()
    policy = LLMMotionPolicy(
        ScriptedAgent(script), schedule, action_spec=ActionSpec(),
        augmentation=AugmentationConfig(memory=MemoryConfig(history_length=history_length)),
        max_attempts=max_attempts, record_trajectory=True, log_dir=directory,
        demo=demo_path is not None, demo_path=demo_path, demo_content="full",
    )
    policy.bind(robot.info)
    try:
        record = rollout(
            policy, robot, Scene("teacher", INSTRUCTION), max_steps=2, seed=0, epoch=0,
            controller=FrameworkController(schedule, spec=policy.spec),
            approver=AutoApprover(), sink=NullSink(),
        )
        if finalize:
            policy.on_trial_end(record, str(directory), "offline")
    finally:
        policy.close()
    return directory


def _document(directory):
    return json.loads((directory / "trajectory.json").read_text())


def _edited(source, destination, edit):
    """Copy a recorded teacher with its images and apply ``edit`` to its document."""
    shutil.copytree(source, destination)
    document = _document(destination)
    edit(document)
    (destination / "trajectory.json").write_text(json.dumps(document))
    return destination


def _edit_requests(document, edit):
    """Edit each recorded request payload, keeping any trailing repair feedback."""
    for turn in document["turns"]:
        text = turn["request"]["text"]
        payload, end = json.JSONDecoder().raw_decode(text)
        edit(payload)
        turn["request"]["text"] = json.dumps(payload, separators=(",", ":")) + text[end:]


def _load(path, content):
    return Demonstration(path, observation_profile="control", demo_content=content)


def _student(demo_path, content, *, interface="move_by_chunk", spec=None):
    policy = LLMMotionPolicy(
        ScriptedAgent([]), _schedule(interface), action_spec=spec or ActionSpec(),
        demo=True, demo_path=demo_path, demo_content=content,
    )
    policy.bind(_Robot().info)
    return policy


def _results(request):
    return [(r["call_id"], r["output"]["status"]) for r in request.get("tool_results", [])]


def test_full_mode_projects_the_recorded_exchange(tmp_path):
    teacher = _record(tmp_path / "teacher", _script())
    demo = _load(teacher, "full")
    assert demo.contract["control_interface"] == "move_by_chunk"
    assert demo.contract["history_length"] == 2
    assert demo.contract["chunk_target_reference"] == "cumulative_from_call_start"
    payload, images = demo.snapshot()
    assert set(payload) == {"guidance", "requests"}
    assert payload["guidance"].startswith("TEACHER DEMONSTRATION: The following is a recorded")

    frames = {"agentview": "demo/00000/agentview"}, {"agentview": "demo/00001/agentview"}

    def observation(x, remaining, cameras):
        return {"instruction": INSTRUCTION, "state": _state(x),
                "step_budget": {"steps_remaining": remaining, "max_steps": 2},
                "images": list(cameras.values()), "cameras": cameras}

    def output(request_id, name, arguments):
        return {"call_id": f"{request_id}-call-0", "name": name, "arguments": arguments}

    assert payload["requests"] == [
        {"observation": observation(0.0, 2, frames[0]),
         "model_output": {"text": TEACHER_TEXT, "tool_calls": [
             output("teacher-00001", "move_by_chunk", _motion(0.001, FIRST_NOTE))]}},
        {"observation": observation(0.001, 1, frames[1]),
         "tool_results": [{"call_id": "teacher-00001-call-0",
                           "output": {"plan_id": "teacher-00001", "status": "accepted"}}],
         "model_output": {"tool_calls": [output("teacher-00002", "teleport", {"x": 1})]}},
        {"observation": observation(0.001, 1, frames[1]),
         "tool_results": [{"call_id": "teacher-00002-call-0",
                           "output": {"status": "rejected", "executed": False,
                                      "error": REJECTION}}],
         "validation_feedback": REPAIR,
         "model_output": {"tool_calls": [
             output("teacher-00003", "move_by_chunk", _motion(-0.0005, SECOND_NOTE))]}},
    ]
    # The retry at step 1 shares one frame; resizing never applies to 4x4 images.
    assert sorted(images) == ["demo/00000/agentview", "demo/00001/agentview"]
    assert [int(images[name].max()) for name in sorted(images)] == [0, 40]
    assert demo.resized_image_count == 0
    assert _load(teacher / "trajectory.json", "full").sha256 == demo.sha256


def test_observation_mode_keeps_one_state_per_inference_point(tmp_path):
    teacher = _record(tmp_path / "teacher", _script())
    demo = _load(teacher, "observations")
    assert demo.contract == {"observation_profile": "control", "control_interface": "move_by_chunk"}
    payload, _ = demo.snapshot()
    assert payload["requests"] == [
        {"observation": {"instruction": INSTRUCTION, "state": _state(x),
                         "images": [f"demo/{index:05d}/agentview"],
                         "cameras": {"agentview": f"demo/{index:05d}/agentview"}}}
        for index, x in enumerate((0.0, 0.001))
    ]
    assert payload["guidance"].startswith("TEACHER DEMONSTRATION: This is a recorded sequence")


@pytest.mark.parametrize(
    ("history_length", "expected"),
    [
        pytest.param(2, [[], ["accepted"], ["rejected"]], id="history"),
        pytest.param(0, [[], [], ["rejected"]], id="no-history"),
    ],
)
def test_delivered_tool_results_follow_recorded_history(tmp_path, history_length, expected):
    teacher = _record(tmp_path / "teacher", _script(), history_length=history_length)
    requests = _load(teacher, "full").snapshot()[0]["requests"]
    calls = [r["model_output"]["tool_calls"][0]["call_id"] for r in requests]
    sources = {"accepted": calls[0], "rejected": calls[1]}
    assert [_results(r) for r in requests] == [
        [(sources[status], status) for status in statuses] for statuses in expected
    ]
    # Repair feedback stays in the request whether or not its rejection was delivered.
    assert [r.get("validation_feedback") for r in requests] == [None, None, REPAIR]


@pytest.mark.parametrize(("history_length", "expected"), [(0, []), (2, ["accepted"])])
def test_transport_resend_is_skipped_and_not_a_repair(
    tmp_path, monkeypatch, history_length, expected
):
    monkeypatch.setattr(policy_module, "_sleep", lambda seconds: None)
    script = _script()
    script[1] = BackendError("simulated outage", metadata=_metadata())
    teacher = _record(tmp_path / "teacher", script, history_length=history_length)
    assert [t["model_output"].get("error_kind") for t in _document(teacher)["turns"]] == [
        None, "transport", None,
    ]
    requests = _load(teacher, "full").snapshot()[0]["requests"]
    assert len(requests) == 2
    assert "validation_feedback" not in requests[1]
    assert [status for _, status in _results(requests[1])] == expected


def _set_backend(key, value):
    """Edit only the final turn's backend metadata: the loader checks every turn."""
    def edit(document):
        document["turns"][-1]["model_output"]["backend"][key] = value
    return edit


def _set(path, value):
    def edit(document):
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return edit


def _drop_step_budget(document):
    _edit_requests(document, lambda payload: payload["current_observation"].pop("step_budget"))


def _drop_chunk_reference(document):
    document["metadata"]["schedule"].pop("chunk_target_reference")


def _first_turn_only(document):
    document["turns"] = document["turns"][:1]


SCRIPTED = "produced by a scripted backend"


@pytest.mark.parametrize(
    ("edit", "errors"),
    [
        pytest.param(_set(("schema_version",), 1),
                     {"full": "schema_version=2", "observations": "schema_version=2"},
                     id="schema-v1"),
        pytest.param(_set_backend("backend", "scripted"),
                     {"full": SCRIPTED, "observations": SCRIPTED}, id="scripted-label"),
        pytest.param(_set_backend("backend", "offline_request_preview"),
                     {"full": SCRIPTED, "observations": SCRIPTED}, id="preview-label"),
        pytest.param(_set_backend("scripted_preview_response", True),
                     {"full": SCRIPTED, "observations": SCRIPTED}, id="preview-flag"),
        pytest.param(_first_turn_only,
                     {"full": "accepted motion call followed by another observation",
                      "observations": "at least two distinct inference points"},
                     id="no-observed-motion"),
        pytest.param(_set(("metadata", "policy_family"), "vla"),
                     {"full": "Only LLM trajectories"}, id="not-llm"),
        pytest.param(_set(("metadata", "observation_profile"), "privileged"),
                     {"full": "differs from the live observation profile 'control'"},
                     id="other-profile"),
        pytest.param(_drop_step_budget, {"full": "has no step_budget"}, id="no-step-budget"),
        pytest.param(_set(("metadata", "schedule"), {"h": -1, "k": -1}),
                     {"full": "legacy H=-1 teachers"}, id="legacy-horizon"),
        pytest.param(_drop_chunk_reference, {"full": "chunk_target_reference is missing"},
                     id="legacy-chunk-targets"),
    ],
)
def test_unusable_teachers_are_rejected(tmp_path, edit, errors):
    teacher = _record(tmp_path / "teacher", _script())
    edited = _edited(teacher, tmp_path / "edited", edit)
    for content, message in errors.items():
        with pytest.raises(ValueError, match=message):
            _load(edited, content)


def test_discarded_recording_is_never_a_teacher(tmp_path):
    """A decision that fails max_attempts discards the episode, and with it the teacher."""
    teacher = _record(tmp_path / "teacher", _script()[:2], max_attempts=1)
    outcome = _document(teacher)["outcome"]
    assert (outcome["complete"], outcome["status"], outcome["termination_reason"]) == (
        True, "discarded", "max_attempts",
    )
    assert outcome["oracle_success"] is None
    for content in ("full", "observations"):
        with pytest.raises(ValueError, match="discarded"):
            _load(teacher, content)


def test_unfinalized_recording_is_rejected(tmp_path):
    teacher = _record(tmp_path / "teacher", _script(), finalize=False)
    assert _document(teacher)["outcome"]["complete"] is False
    for content in ("full", "observations"):
        with pytest.raises(ValueError, match="outcome.complete=true"):
            _load(teacher, content)


def test_scripted_backend_recording_is_rejected(tmp_path):
    plans = [{"decision": "replace", "commands": [{"name": "move_by", "arguments": json.dumps(
        {"deltas": {"dx": dx}, "note": FIRST_NOTE})}]} for dx in (0.001, -0.0005)]
    teacher = _record(tmp_path / "teacher", plans)
    for content in ("full", "observations"):
        with pytest.raises(ValueError, match=SCRIPTED):
            _load(teacher, content)


def test_teacher_that_used_a_demonstration_is_rejected(tmp_path):
    first = _record(tmp_path / "first", _script())
    nested = _record(tmp_path / "nested", _script(), demo_path=first)
    assert all('"demonstration"' in t["request"]["text"] for t in _document(nested)["turns"])
    with pytest.raises(ValueError, match="nested demos are rejected"):
        _load(nested, "full")


def test_full_teacher_must_match_the_student_control_contract(tmp_path):
    teacher = _record(tmp_path / "teacher", _script())
    _student(teacher, "full").close()
    with pytest.raises(ValueError, match="control_interface is incompatible"):
        _student(teacher, "full", interface="move_by")
    coarse = ActionSpec(pose_scale=(0.1, 0.1, 0.1, 0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="action_spec is incompatible.*: pose_scale$"):
        _student(teacher, "full", spec=coarse)
    # Observation sequences carry no actions, so the teacher interface does not matter.
    _student(teacher, "observations", interface="move_by").close()


def test_observation_mode_strips_a_privileged_teacher(tmp_path):
    teacher = _record(tmp_path / "teacher", _script())

    def privileged(document):
        document["metadata"]["observation_profile"] = "privileged"
        _edit_requests(document, lambda payload: payload["current_observation"].update(
            privileged={"schema_version": 2, "objects": {"PRIVILEGED-SENTINEL": [1.0]}}))

    edited = _edited(teacher, tmp_path / "privileged", privileged)
    with pytest.raises(ValueError, match="differs from the live observation profile"):
        _load(edited, "full")
    payload, _ = _load(edited, "observations").snapshot()
    assert len(payload["requests"]) == 2
    assert "PRIVILEGED-SENTINEL" not in json.dumps(payload)
    assert "privileged" not in json.dumps(payload["requests"])


def test_observation_mode_keeps_only_public_robot_state(tmp_path):
    """Any other state key a teacher request carried is dropped from an observation sequence."""
    teacher = _record(tmp_path / "teacher", _script())

    def extra_state(document):
        _edit_requests(document, lambda payload: payload["current_observation"]["state"].update(
            cube_pose_sentinel=[9.0, 9.0, 9.0]))

    edited = _edited(teacher, tmp_path / "extra-state", extra_state)
    payload, _ = _load(edited, "observations").snapshot()
    assert [entry["observation"]["state"] for entry in payload["requests"]] == [
        _state(0.0), _state(0.001),
    ]
    assert "cube_pose_sentinel" not in json.dumps(payload)


@pytest.mark.parametrize("content", ["full", "observations"])
def test_student_context_contains_only_what_the_teacher_saw(tmp_path, content):
    teacher = _record(tmp_path / "teacher", _script())
    policy = _student(teacher, content)
    robot = _Robot()
    policy.reset(Scene("student", "Push the block."))
    observation = robot.reset(None)
    request = policy.prepare_request(replace(observation, instruction="Push the block."))
    policy.close()

    payload = json.loads(request.text)
    assert list(payload) == ["demonstration", "instruction", "current_observation"]
    assert payload["instruction"] == "Push the block."
    assert payload["demonstration"] == _load(teacher, content).snapshot()[0]
    assert list(request.images) == [
        "demo/00000/agentview", "demo/00001/agentview", "current/agentview",
    ]
    requests = payload["demonstration"]["requests"]
    allowed = {"observation", "model_output", "tool_results", "validation_feedback"}
    for entry in requests:
        assert set(entry) <= (allowed if content == "full" else {"observation"})
        assert set(entry["observation"]) == (
            {"instruction", "state", "images", "cameras"}
            | ({"step_budget"} if content == "full" else set())
        )
        if content == "full":
            assert set(entry["model_output"]) <= {"text", "tool_calls"}
            for call in entry["model_output"]["tool_calls"]:
                assert set(call) == {"call_id", "name", "arguments"}
            for result in entry.get("tool_results", []):
                assert set(result) == {"call_id", "output"}
    visible = json.dumps(requests)
    # Recorded metadata, usage, backend identity, outcome and execution audit stay out.
    hidden = (
        TEACHER_MODEL, str(USAGE["input_tokens"]), "responses", "history_length_semantics",
        "ScriptedAgent", "oracle_success", "termination_reason", "exhausted",
        "episode_ended", "executed_actions", "steps_executed", "telemetry", "elapsed",
        "trajectory-images", "usage",
    )
    assert [word for word in hidden if word in visible] == []
    if content == "observations":
        actions = ("teleport", "move_by_chunk", FIRST_NOTE, TEACHER_TEXT, "accepted", "call_id")
        assert [word for word in actions if word in visible] == []


@pytest.mark.parametrize("content", ["full", "observations"])
def test_scripted_demonstration_label_is_rejected(tmp_path, content):
    teacher = _record(tmp_path / "teacher", _script())

    def relabel(document):
        for turn in document["turns"]:
            turn["model_output"]["backend"]["backend"] = "scripted_demonstration"

    edited = _edited(teacher, tmp_path / "relabeled", relabel)
    with pytest.raises(ValueError, match="scripted"):
        _load(edited, content)
