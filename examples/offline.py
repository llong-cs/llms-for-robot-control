#!/usr/bin/env python3
"""Exercise a complete policy/controller loop without a simulator or model call."""
# Imports after the path setup below deliberately use this checkout.
# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "agentic-framework/src", ROOT / "agentic-framework/vendor/inspect-robots/src"):
    sys.path.insert(0, str(folder))

from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness.controller import FrameworkController
from agentic_framework.harness.policy import LLMMotionPolicy
from agentic_framework.harness.types import ActionSpec, ScheduleConfig
from agentic_framework.models.llm.backend import ScriptedBackend
from inspect_robots.approver import AutoApprover
from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import rollout
from inspect_robots.scene import Scene
from inspect_robots.types import Observation, StepResult


class ToyRobot:
    """A deterministic robot used solely to illustrate the framework interface."""

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
            instruction="Make two small movements.", extra={"env_step": len(self.actions)},
        )

    def reset(self, scene, *, seed=None):
        self.actions.clear()
        return self.observation()

    def step(self, action):
        self.actions.append(action.data.copy())
        return StepResult(self.observation())


def main():
    schedule = ScheduleConfig(control_interface="move_by_chunk", h=1, k=1, max_steps=2)
    plans = [{"decision": "replace", "commands": [{"name": "move_by", "arguments": json.dumps({
        "deltas": {"dx": dx, "gripper": 0.0}, "note": "Make one small movement, then observe again."})}],
        "plan_id": "", "summary": "", "reason": "", "hindsight": "none"} for dx in (0.001, -0.0005)]
    backend = ScriptedBackend(plans)
    robot = ToyRobot()
    policy = LLMMotionPolicy(backend, schedule, action_tools="move_by", action_spec=ActionSpec())
    policy.bind(robot.info)
    try:
        record = rollout(policy, robot, Scene("offline-example", "Make two small movements."),
                         max_steps=2, seed=42, epoch=0,
                         controller=FrameworkController(schedule, spec=policy.spec),
                         approver=AutoApprover(), sink=NullSink())
        if record.termination_reason != "max_steps" or len(robot.actions) != 2:
            raise RuntimeError("The policy/controller loop did not execute both actions")
        for request in backend.requests:
            payload = json.loads(request.text)
            if "privileged" in payload["current_observation"]:
                raise RuntimeError("Privileged observations must be disabled")
        print(json.dumps({"model_calls": 0, "scripted_decisions": len(backend.requests),
                          "physical_steps": len(robot.actions),
                          "termination": record.termination_reason,
                          "actions": [value.tolist() for value in robot.actions]}, indent=2))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
