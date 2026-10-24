"""Static instructions and tool schemas are blind to K and the step budget."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from inspect_robots.approver import AutoApprover
from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import rollout
from inspect_robots.scene import Scene
from inspect_robots.types import Observation, StepResult

from agentic_framework.configuration.profiles import config_directory
from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness import policy as policy_module
from agentic_framework.harness.controller import FrameworkController
from agentic_framework.harness.policy import LLMMotionPolicy
from agentic_framework.harness.types import AgentOutput, ScheduleConfig

H = 3
K_VALUES = {"move_by": (1, 3, 5, -1), "move_by_chunk": (1, 3, 5)}
MAX_STEPS = (6, 220, 1800)
SECRET = "hidden-object-pose-7f3a"
INSTRUCTION = "Hold still, then stop."
# sha256 of the static bytes for the LIBERO embodiment with H=5 and default options.
# The prompt is part of the experimental condition: change these only on purpose.
PINNED_SHA256 = {
    "move_by": "3e3c49eb07720c169c4e6bce99b23c1566f4716b23b5b4a796385ad61c738402",
    "move_by_chunk": "72827a77429807c98997c2d2a6a47ed23cfc4ded5295bf83180eadb89f87c185",
}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Record backoff waits instead of sleeping; a test that expects one patches its own."""
    waits = []
    monkeypatch.setattr(policy_module, "_sleep", waits.append)
    yield
    assert waits == [], f"unexpected backoff waits: {waits}"


class ToyRobot:
    """A motionless LIBERO-shaped robot that also exposes simulator truth it must not leak."""

    def __init__(self):
        embodiment = LiberoEmbodiment(image_size=4)
        self.info, self.spec = embodiment.info, embodiment.action_spec
        self.last_reset_info = {"success": False}
        self.actions = []

    def observation(self):
        return Observation(
            images={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            state={
                "eef_pos": np.array([0.0, 0.0, 0.2]),
                "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "gripper_width": np.array([0.04]),
                "object_pos": np.array([0.31, 0.27, 0.05]),
            },
            instruction=INSTRUCTION,
            extra={
                "env_step": len(self.actions),
                "privileged": {"objects": {SECRET: [0.31, 0.27, 0.05]}},
                "privileged_libero": {"success": SECRET},
            },
        )

    def reset(self, scene, *, seed=None):
        self.actions.clear()
        return self.observation()

    def step(self, action):
        self.actions.append(action.data.copy())
        return StepResult(self.observation())


class RecordingAgent:
    """Replay native tool calls and record each request with the steps executed before it."""

    model = "offline-scripted"

    def __init__(self, robot, calls):
        self.robot, self.calls, self.requests = robot, list(calls), []

    def generate(self, request):
        self.requests.append((request, len(self.robot.actions)))
        name, arguments = self.calls.pop(0)
        call = {"type": "function_call", "call_id": f"call-{len(self.requests)}", "name": name,
                "arguments": json.dumps(arguments)}
        return AgentOutput({"tool_calls": [call]}, tool_calls=(call,))

    def close(self):
        pass


def _static(request) -> bytes:
    """The bytes every request repeats: instructions and tool schemas, in wire key order."""
    tools = json.dumps(list(request.tools), ensure_ascii=False, separators=(",", ":"))
    return (request.instructions + "\n" + tools).encode()


def _current(request) -> dict:
    """The model-visible observation, excluding any appended format-repair feedback."""
    return json.loads(request.messages[0]["content"][0]["text"])["current_observation"]


def _run(interface, k, max_steps):
    """Run one offline trial: an invalid call, its repair, a hold motion, then done."""
    robot = ToyRobot()
    schedule = ScheduleConfig(h=H, k=k, max_steps=max_steps, control_interface=interface)
    hold = {"dx": 0.0} if interface == "move_by" else {"dx": [0.0] * H}
    motion = (interface, {"deltas": hold, "note": "Hold the pose."})
    agent = RecordingAgent(robot, [
        ("teleport", {}), motion, motion, ("done", {"summary": "Held.", "hindsight": "none"}),
    ])
    policy = LLMMotionPolicy(agent, schedule, action_spec=robot.spec)
    policy.bind(robot.info)
    try:
        record = rollout(
            policy, robot, Scene("invariance", INSTRUCTION), max_steps=max_steps, seed=0, epoch=0,
            controller=FrameworkController(schedule, spec=policy.spec),
            approver=AutoApprover(), sink=NullSink(),
        )
    finally:
        policy.close()
    return record, agent.requests


def _prepared(interface, *, h=H, k=1, max_steps=220, env_step=0, **options):
    robot = ToyRobot()
    schedule = ScheduleConfig(h=h, k=k, max_steps=max_steps, control_interface=interface)
    policy = LLMMotionPolicy(RecordingAgent(robot, []), schedule, action_spec=robot.spec,
                             **options)
    policy.bind(robot.info)
    policy.reset(Scene("invariance", INSTRUCTION))
    robot.actions.extend([None] * env_step)
    return policy, policy.prepare_request(robot.observation())


@pytest.mark.parametrize("interface", ["move_by", "move_by_chunk"])
def test_static_prompt_is_byte_identical_across_k_and_max_steps(interface):
    static = set()
    for k in K_VALUES[interface]:
        for max_steps in MAX_STEPS:
            record, requests = _run(interface, k, max_steps)
            # A rejected call, its repair and a later decision: never a vacuous comparison.
            assert record.termination_reason in ("done", "max_steps")
            assert len(requests) >= 3
            static.update(_static(request) for request, _ in requests)
    assert len(static) == 1
    _, preview = _prepared(interface)
    assert static == {_static(preview)}


@pytest.mark.parametrize("interface", ["move_by", "move_by_chunk"])
def test_k_and_step_limits_never_appear_in_the_static_prompt(interface):
    _, sentinel = _prepared(interface, k=97, max_steps=4321)
    _, reference = _prepared(interface, k=1, max_steps=220)
    assert _static(sentinel) == _static(reference)
    text = _static(sentinel).decode()
    numbers = set(re.findall(r"\d+", text))
    assert not {"97", "4321"} & numbers
    assert not re.search(r"\bK\b|replan", text)


@pytest.mark.parametrize("interface", sorted(PINNED_SHA256))
def test_static_prompt_bytes_are_pinned(interface):
    _, request = _prepared(interface, h=5)
    digest = hashlib.sha256(_static(request)).hexdigest()
    assert digest == PINNED_SHA256[interface], "static prompt changed; review the diff deliberately"


@pytest.mark.parametrize(("interface", "k"), [("move_by", 1), ("move_by", -1),
                                              ("move_by_chunk", 3)])
def test_every_request_reports_the_remaining_step_budget(interface, k):
    for max_steps in (6, 220):
        _, requests = _run(interface, k, max_steps)
        for request, executed in requests:
            assert _current(request)["step_budget"] == {
                "steps_remaining": max_steps - executed, "max_steps": max_steps,
            }
            assert "env_step" not in request.text
        # The format repair resends the same observation; later decisions see less budget.
        executed = [steps for _, steps in requests]
        assert executed[:2] == [0, 0] and executed[1:] == sorted(set(executed[1:]))
    policy, request = _prepared(interface, k=k, max_steps=40, env_step=39)
    assert _current(request)["step_budget"] == {"steps_remaining": 1, "max_steps": 40}
    assert "step_budget" in request.instructions
    robot = ToyRobot()
    robot.actions.extend([None] * 40)
    with pytest.raises(ValueError, match="at least one remaining policy control step"):
        policy.prepare_request(robot.observation())


def test_privileged_observations_are_absent_by_default():
    defaults = json.loads((config_directory() / "runtime-defaults.json").read_text())
    assert defaults["observation"]["profile"] == "control"
    for interface in ("move_by", "move_by_chunk"):
        policy, _ = _prepared(interface)
        assert policy.context.observation_profile == "control"
        _, requests = _run(interface, 1, 220)
        for request, _ in requests:
            current = _current(request)
            assert set(current) == {"state", "step_budget", "images"}
            assert set(current["state"]) == {"eef_pos", "eef_quat", "gripper_width"}
            wire = json.dumps([request.text, request.messages, request.instructions,
                               list(request.tools)])
            assert "privileged" not in wire
            assert SECRET not in wire
