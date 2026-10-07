"""Native robot tool contracts; motion values are physical WORLD increments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from agentic_framework.harness.motion_limits import (
    motion_pose_bounds,
    validate_gripper_bounds,
    validate_motion_pose,
)
from agentic_framework.harness.types import (
    ActionSpec,
    MotionCommand,
    MotionPlan,
    ScheduleConfig,
    Stop,
)
from agentic_framework.harness.upstream_adapter import upstream_tool_fields

FIELDS = {"move_by": ("dx", "dy", "dz", "rx", "ry", "rz")}


def _selected(action_tools: str) -> tuple[str, ...]:
    if action_tools != "move_by":
        raise ValueError("action_tools must be move_by; absolute motion tools were removed")
    return ("move_by",)


def motion_tool_name(config: ScheduleConfig) -> str:
    """Expose the explicitly configured robot control interface."""
    return config.control_interface


def output_schema(
    *,
    schedule: ScheduleConfig | None = None,
    h: int | None = None,
    control_interface: str = "move_by",
    max_commands: int | None = None,
    action_tools: str = "move_by",
    notes: bool = True,
    reflection: bool = True,
    spec: ActionSpec | None = None,
) -> dict:
    """Describe the former compact output format for historical records only."""
    del notes, reflection, spec
    _selected(action_tools)
    if sum(v is not None for v in (schedule, h, max_commands)) > 1:
        raise ValueError("set only one of schedule, h, or max_commands")
    horizon = schedule.h if schedule is not None else h if h is not None else max_commands
    horizon = 5 if horizon is None else horizon
    interface = schedule.control_interface if schedule is not None else control_interface
    ScheduleConfig(h=horizon, control_interface=interface)
    call = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "enum": ["move_by", "done", "give_up"],
            },
            "arguments_json": {"type": "string", "description": "JSON-encoded tool arguments."},
        },
        "required": ["tool_name", "arguments_json"],
        "additionalProperties": False,
    }
    if interface == "move_by":
        return call
    return {
        "type": "object",
        "properties": {
            "calls": {"type": "array", "items": call, "minItems": 1, "maxItems": horizon}
        },
        "required": ["calls"],
        "additionalProperties": False,
    }


def robot_tool_definitions(
    config: ScheduleConfig,
    spec: ActionSpec,
    *,
    notes: bool = True,
    reflection: bool = True,
    action_tools: str = "move_by",
) -> list[dict]:
    """Describe one motion step using Inspect's function contracts.

    The interface-specific pose envelope and unscaled gripper bounds are shared
    with decoder/controller validation. ``native_tool_schemas`` converts scalar
    parameters into aligned arrays only for the chunk interface.
    """
    _selected(action_tools)
    inherited = upstream_tool_fields()
    endpoints = motion_pose_bounds(spec, config)
    low, high = [min(pair) for pair in endpoints], [max(pair) for pair in endpoints]
    descriptions = []
    for start, label, unit in ((0, "translation", "m"), (3, "rotation", "rad")):
        names = FIELDS["move_by"][start:start + 3]
        group = endpoints[start:start + 3]
        if all(pair == group[0] for pair in group):
            a, b = group[0]
            descriptions.append(f"{label} ({'/'.join(names)}): [{a:g}, {b:g}] {unit}")
        else:
            # Custom native input bounds may further constrain individual axes.
            descriptions.extend(f"{name}: [{a:g}, {b:g}]"
                                for name, (a, b) in zip(names, group, strict=True))
    bounds = ", ".join(descriptions) + f", gripper: [{spec.gripper_low:g}, {spec.gripper_high:g}]."
    if config.control_interface == "move_by":
        movement = (
            "Move BY the given displacement per dimension. The executor freezes the "
            "target from the measured pose at this call's start. After every physical step "
            "it observes and corrects the remaining position and orientation error toward "
            "that SAME target; it does not add the original displacement again. "
            "The framework passes the full remaining pose error without framework clipping; "
            "the native controller retains its own input clipping and adjustments. "
            "Joint and actuator limits still apply, and target reachability is not guaranteed. "
            "Unnamed pose dimensions are held at their start targets. "
            "Bounds below describe the total target offset, not one-step travel, a fixed "
            "execution duration, or a guaranteed path. "
            'To hold the current pose, use deltas {"dx": 0} and omit gripper; this advances '
            "at least one control step and checks the held pose. "
        )
    else:
        movement = (
            "At call acceptance, integrate all pose increments from the initial measured "
            "pose into a fixed absolute waypoint sequence. Each increment extends the "
            "previous planned waypoint using the embodiment's coordinate and rotation "
            "conventions. "
        )
        if config.motion_time_scale == 1:
            movement += (
                "On each tick, correct from the current measured pose toward the "
                "next fixed waypoint through the native delta controller using the full "
                "remaining pose error without framework clipping. "
                "Each waypoint receives ONE control step "
                f"({1 / spec.control_hz:g} seconds) before advancing, even if it was not reached. "
            )
        else:
            movement += (
                f"The executor subdivides each planned segment into {config.motion_time_scale} "
                "equal translation and shortest-arc orientation steps, preserving every "
                "original waypoint endpoint. On each tick, correct from the current "
                "measured pose toward the next interpolated target through the native delta "
                "controller using the full remaining pose error without framework "
                "clipping. Each interpolated target receives "
                f"one control step ({1 / spec.control_hz:g} seconds) before advancing, even "
                "if it was not reached. "
                f"Each original array index therefore spans {config.motion_time_scale} "
                "control periods. The explicit gripper command at that index is repeated "
                "unchanged for every subdivision; it is not divided. Omission retains the "
                "usual gripper hold behavior. This stretches the planned timing, without "
                "guaranteeing the same measured path or a proportional change in speed. "
            )
        movement += (
            "Tracking error does not rebase the remaining waypoints. Zero pose increments "
            "repeat the previous planned waypoint (the initial pose at index zero), so "
            "they can continue correcting toward it. Pose and gripper values at the same "
            "array index act together. No extra tracking or gripper settling steps are "
            "added; use repeated zero pose increments or gripper commands across indices "
            "if more time is needed. Each pose increment must obey the per-dimension "
            "operational single-step limits below; these bounds apply to requested "
            "increments, not feedback corrections. The native controller retains its own "
            "input clipping and adjustments. Joint and actuator limits still apply; "
            "these bounds do not guarantee reachability or actual travel under inertia "
            "or contact. The entire chunk is validated before execution. "
        )
    if config.control_interface == "move_by":
        movement += (
            "Pose offsets and an explicit gripper value may share one move_by; both are applied "
            "together from its first control step. Use separate calls only when you intend "
            "separate motions. "
        )
    if spec.gripper_mode == "absolute":
        movement += (
            "The gripper dimension is an absolute joint target, not a displacement. "
            "Omitting gripper holds its observed controller target; explicit zero is a target. "
        )
    parameters = {
        "type": "object",
        "properties": {
            "deltas": {
                "type": "object",
                "description": "Map of dimension name to value. Valid names: dx, dy, dz, rx, ry, rz, gripper. "
                "Use finite numeric values and omit unneeded dimensions.",
                "properties": {
                    name: {"type": "number"} for name in (*FIELDS["move_by"], "gripper")
                },
                "additionalProperties": False,
                "minProperties": 1,
            }
        },
        "required": ["deltas"],
        "additionalProperties": False,
    }
    parameters["properties"]["deltas"].update(
        properties={
            **{
                name: {"type": "number", "minimum": a, "maximum": b}
                for name, a, b in zip(FIELDS["move_by"], low, high, strict=True)
            },
            "gripper": {
                "type": "number",
                "minimum": spec.gripper_low,
                "maximum": spec.gripper_high,
            },
        },
        additionalProperties=False,
        minProperties=1,
    )
    if spec.arm_names:
        parameters["properties"]["arm"] = {
            "type": "string", "enum": list(spec.arm_names),
            "description": "Select the arm for this motion; all other arms hold their pose and gripper target.",
        }
        parameters["required"].append("arm")
        movement += "Specify arm for every call. Only that arm receives these deltas. "
    if notes:
        parameters["properties"]["note"] = inherited["note"]
        parameters["required"].append("note")
    result = [
        {
            "type": "function",
            "name": "move_by",
            "description": movement + "Bounds for each component: " + bounds,
            "parameters": parameters,
            "strict": False,
        }
    ]
    for name in ("done", "give_up"):
        function = inherited[name]
        if not reflection:
            function["parameters"]["properties"]["hindsight"]["description"] = (
                "Set hindsight to 'none'."
            )
        result.append({"type": "function", **function, "strict": False})
    return result


def native_tool_schemas(
    schedule: ScheduleConfig,
    spec: ActionSpec,
    *,
    notes: bool = True,
    reflection: bool = True,
    action_tools: str = "move_by",
) -> list[dict]:
    """Expose one native motion call or a stopping call for each model turn."""
    definitions = robot_tool_definitions(
        schedule, spec, notes=notes, reflection=reflection, action_tools=action_tools
    )
    if schedule.control_interface == "move_by_chunk":
        motion = definitions[0]
        step_parameters = motion["parameters"]
        dimensions = step_parameters["properties"]["deltas"]["properties"]
        arrays = {}
        for name, item in dimensions.items():
            item = deepcopy(item)
            arrays[name] = {
                "type": "array",
                "items": {"anyOf": [item, {"type": "null"}]} if name == "gripper" else item,
                "minItems": schedule.h,
                "maxItems": schedule.h,
            }
        arrays["gripper"]["description"] = (
            "One gripper value per step; null omits the command for that step and "
            "holds the current driver target. Zero remains an explicit value."
        )
        if schedule.motion_time_scale > 1:
            arrays["gripper"]["description"] = (
                "One gripper value per original array index, repeated unchanged for all "
                f"{schedule.motion_time_scale} subdivisions. Null omits the command and "
                "holds the current driver target. Zero remains an explicit value."
            )
        motion["name"] = motion_tool_name(schedule)
        motion["parameters"] = {
            "type": "object",
            "properties": {
                "deltas": {
                    "type": "object",
                    "description": (
                        "Dimension-to-array mapping. Matching indices form one pose increment "
                        "and optional gripper command. Omitted pose dimensions contribute zero "
                        "increment. Every index needs "
                        "at least one numeric dimension."
                    ),
                    "properties": arrays,
                    "additionalProperties": False,
                    "minProperties": 1,
                    # With no pose array, every gripper entry must be a real command.
                    "anyOf": [
                        *({"required": [name]} for name in FIELDS["move_by"]),
                        {
                            "required": ["gripper"],
                            "properties": {"gripper": {"items": deepcopy(dimensions["gripper"])}}
                        },
                    ],
                }
            },
            "required": ["deltas"],
            "additionalProperties": False,
        }
        if spec.arm_names:
            motion["parameters"]["properties"]["arm"] = deepcopy(step_parameters["properties"]["arm"])
            motion["parameters"]["required"].append("arm")
        if notes:
            motion["parameters"]["properties"]["note"] = {
                **deepcopy(step_parameters["properties"]["note"]),
                "description": (
                    "In one or two sentences, explain what you observe, what the entire "
                    f"sequence of {schedule.h} targets is intended to achieve, and why you chose it."
                ),
                "minLength": 1,
                "pattern": r"\S",
            }
            motion["parameters"]["required"].append("note")
        motion["description"] = (
            "Submit an ordered sequence in one move_by_chunk call. "
            f"Supply an array of exactly {schedule.h} values for each requested dimension in deltas. "
            + motion["description"]
        )
    return definitions


def instructions(
    config: ScheduleConfig,
    spec: ActionSpec,
    *,
    notes: bool = True,
    reflection: bool = True,
    action_tools: str = "move_by",
    approver: str = "none",
    observation_profile: str = "control",
    embodiment_name: str = "libero",
    is_simulated: bool = True,
) -> str:
    """Guide native robot tool use and measured execution without output wrappers."""
    _selected(action_tools)
    if observation_profile not in ("control", "openpi_matched", "privileged"):
        raise ValueError("unknown observation profile")
    if approver not in ("none", "clamp"):
        raise ValueError("unknown approver")
    motion_name = motion_tool_name(config)
    intro = (
        "Choose the next robot action from the supplied observations, proprioception, "
        f"instructions and tool results. Invoke only the provided {motion_name}, done or give_up "
        "robot tools. Do not use shell commands, files, browsing, image-editing tools, "
        "or delegation. Judge progress from the actual tool execution result and the "
        "next observation. Acceptance of a target does not establish arrival, a grasp "
        "or task success."
    )
    if config.control_interface == "move_by":
        selection = (
            "Invoke exactly one robot tool per turn: move_by, done or give_up. "
            "Follow the move_by tool definition for argument format and execution semantics."
        )
    else:
        selection = (
            "Invoke exactly one robot tool per turn: move_by_chunk, done or give_up. "
            "Follow the move_by_chunk tool definition for array format and execution semantics. "
            "Do not mix motion and stopping calls."
        )
    compare = (
        "openpi_state and images"
        if observation_profile == "openpi_matched"
        else "eef_pos, eef_quat and images"
    )
    if spec.arm_names:
        compare = "each arm's prefixed eef_pos, eef_quat and images"
    identity = "simulated robot" if is_simulated else "robot"
    control = [
        f"You are controlling a {identity} embodiment named '{embodiment_name}' through tool calls. "
        "Each observation gives you the current proprioceptive state and camera images. "
        "Work toward the user's goal in small, deliberate motions; re-check each new observation "
        "before choosing the next motion.",
    ]
    if notes:
        control.append(
            (
                "Every move_by call must include a note: "
                if config.control_interface == "move_by"
                else "Every move_by_chunk call must include a single note for the entire call: "
            )
            + "in one or two sentences, say what you observe in the current observation"
            + (
                " and why you chose this motion. "
                if config.control_interface == "move_by"
                else f", what you intend to achieve through the entire sequence of {config.h} targets, "
                "and why you chose these motions. "
            )
            + "The user reads these notes to follow what "
            "you see and decide, so write them for a human reader."
        )
    if approver == "clamp":
        control.append(
            "The configured approver clamps actions to the declared action-space bounds."
        )
    control.extend(
        [
            selection,
            "Generated calls may be only partly executed and their targets may remain unmet. "
            f"At every inference, compare the current {compare} with earlier observations "
            "and tool calls to infer how far the motion progressed before deciding what "
            "to do next. Never assume previous commands achieved their targets. "
            "Any unexecuted actions from the previous response are discarded at the next "
            "turn. Choose new actions from the current observation and measured state.",
            "For grasping, choose a reachable local contact region and approach angle, "
            "considering its size relative to the gripper's opening range using according "
            "to your observations. Check lateral alignment, finger orientation and contact "
            "height in the available views.",
            "Choose when to close from the current observation and intended finger contacts. "
            "If alignment needs visual confirmation, inspect a fresh observation before closing. "
            "Use small motions near contact so the next observation can guide corrections.",
            "A closing command or narrow finger gap alone does not establish a grasp; "
            "pay attention to whether the object actually follows as the gripper moves. "
            "The gripper may take multiple steps to fully close/open.",
            "You can always swtich angle/orientation to obtain a clear view.",
            "When the goal is achieved call done; if it cannot be achieved call give_up. "
            "Calling done stops the trial but cannot declare success; the benchmark scorer "
            "judges success independently.",
            "Each observation's step_budget reports steps_remaining out of max_steps physical "
            "control steps; the trial fails if the task is not accomplished before the steps "
            "run out.",
        ]
    )
    if reflection:
        control.append(
            "Note what you are learning about this rig and task as you go: done and give_up "
            "ask what you wish you had known from the start."
        )
    else:
        control.append("Set hindsight to 'none'.")
    libero = embodiment_name == "libero"
    rig = "LIBERO Franka Panda" if libero else f"Embodiment '{embodiment_name}'"
    embodiment = [
        f"{rig}, {spec.control_hz:g} Hz. Actions [dx,dy,dz,rx,ry,rz,gripper] use "
        "end-effector displacements in WORLD coordinates: translation in meters and "
        "rotation as an axis-angle vector in radians. Mode-specific ranges are given "
        f"in the {motion_name} definition. +z moves upward; use observed state and camera motion "
        "to establish x/y directions. The adapter converts these physical units to "
        "normalized OSC inputs.",
        "The gripper is a numeric command: negative opens, positive closes, and "
        f"{spec.gripper_neutral:g}/omission holds the gripper driver's current target; it "
        "does not repeat the previous opening/closing command.",
    ]
    if observation_profile in ("control", "privileged"):
        embodiment.append(
            "eef_pos is the controlled end-effector position in WORLD meters. eef_quat "
            "is [x,y,z,w]; gripper_width is finger separation in meters; joint_pos contains "
            "arm joint angles in radians."
        )
        compare = "eef_pos, eef_quat and images"
    else:
        embodiment.append(
            "openpi_state contains WORLD end-effector position in meters, absolute hand-body "
            "axis-angle orientation in radians, then two signed finger joint positions in "
            "meters. These orientation values describe state, not action increments. "
            "Use the finger joint positions and images to judge opening and closing."
        )
        compare = "openpi_state and images"
    if libero:
        embodiment.append(
            "Initialization applies an open gripper command during the 10 settling steps; "
            "use the observed finger state for its actual opening."
        )
        embodiment.append(
            "agentview is the external camera; robot0_eye_in_hand is the wrist camera. "
            "Both RGB images use the reference LIBERO evaluator's 180-degree orientation correction."
        )
    if observation_profile != "privileged":
        embodiment.append("Infer objects and contacts visually from the supplied observations.")
    if spec.gripper_mode == "absolute":
        rotation = (
            "Euler XYZ increments" if spec.rotation == "euler_xyz" else "axis-angle increments"
        )
        frame = spec.frame.upper()
        open_target = spec.gripper_low if spec.gripper_open is None else spec.gripper_open
        close_target = spec.gripper_high if spec.gripper_close is None else spec.gripper_close
        embodiment = [
            f"{rig}, {spec.control_hz:g} Hz. {motion_name} requests end-effector translation in "
            f"meters along {frame} axes, and {rotation} in radians. Mode-specific ranges "
            f"are in the {motion_name} definition. The adapter converts these physical goal "
            "offsets to native controller inputs using the declared scale and signs.",
            "The gripper is an ABSOLUTE gripper joint target in radians, not a "
            f"displacement or a signed open/close command. {open_target:g} is the open target; "
            f"{close_target:g} is the close target. Omit gripper to keep the observed driver "
            "target; an explicit zero requests opening and does not mean hold.",
            "eef_pos and eef_quat give the controlled end-effector pose in WORLD coordinates; "
            "eef_quat is [x,y,z,w]. joint_pos contains arm joint angles in radians. "
            "gripper_pos is the measured gripper joint angle in radians; gripper_target, "
            "when supplied, is the driver's current absolute joint target."
            + (" Infer objects and contacts visually." if observation_profile != "privileged" else ""),
        ]
        compare = "eef_pos, eef_quat and images"
        if spec.control_notes.strip():
            embodiment.append(
                spec.control_notes.strip().replace(
                    "both are native RGB views without the LIBERO image rotation.",
                    "both are native RGB views.",
                )
            )
    if spec.arm_names:
        rotation = "Euler XYZ increments" if spec.rotation == "euler_xyz" else "axis-angle increments"
        frame = spec.frame.upper()
        open_target = spec.gripper_low if spec.gripper_open is None else spec.gripper_open
        close_target = spec.gripper_high if spec.gripper_close is None else spec.gripper_close
        arms = ", ".join(spec.arm_names)
        embodiment = [
            f"{rig}, {spec.control_hz:g} Hz. Select arm ({arms}) explicitly on every "
            f"{motion_name} call; one call controls one arm. "
            + ("A chunk uses the same arm for all its indices. "
               if config.control_interface == "move_by_chunk" else "")
            + "The other arm receives zero pose offsets and holds its observed "
            "gripper target. Both arms still evolve physically, so inspect both in the next observation.",
            f"Pose deltas are translation in meters along {frame} axes and {rotation} in radians. "
            "Rotation increments pre-multiply the current end-effector orientation in the declared "
            f"frame. Mode-specific ranges are in the {motion_name} definition.",
            f"The gripper is an ABSOLUTE target in {spec.gripper_units}, with allowed range "
            f"[{spec.gripper_low:g}, {spec.gripper_high:g}]: {open_target:g} opens and "
            f"{close_target:g} closes. Omission holds the selected arm's observed controller "
            "target. Explicit zero is a target, never a hold instruction.",
            "Each robot state key has its arm prefix: for example left_eef_pos and right_eef_pos "
            "are controlled end-effector positions in WORLD meters; left_eef_quat and "
            "right_eef_quat are WORLD quaternions [x,y,z,w]. The corresponding joint_pos "
            "values are arm joint angles in radians. gripper_pos is measured opening in "
            f"{spec.gripper_units}; gripper_target is the last controller target. "
            "A target is not a measurement or evidence of a grasp.",
            "Infer task objects, contacts and task progress visually from the supplied RGB images.",
        ]
        if spec.control_notes.strip():
            embodiment.append(spec.control_notes.strip())
        compare = "each arm's prefixed eef_pos, eef_quat and images"
    if observation_profile == "privileged":
        embodiment.append(
            "This episode additionally provides privileged schema v2: named simulator objects "
            "and articulations, with source provenance, WORLD-meter positions and xyzw "
            "quaternions. Read the supplied field semantics before interpreting poses or joints. "
            "The regions, goals, and evaluation sections each declare available, data, source, "
            "and reason. Use their data only when available=true; unavailable means unknown, "
            "not an empty set, a false predicate, or task failure. Do not assume any task region, "
            "goal predicate, or success check exists when it is unavailable. "
            "Use measured simulator values together with images to plan and verify motions. "
            "Object body origins are not necessarily grasp points; check geometry, contact, "
            "and gripper placement in the images."
        )
    return (
        intro
        + "\n\nROBOT CONTROL INSTRUCTIONS:\n"
        + "\n".join(control)
        + "\n\nEmbodiment notes:\n"
        + "\n".join(embodiment)
    )


def _args(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except RecursionError:
            # Deep nesting within max_response_chars is invalid model output.
            raise ValueError("tool arguments are nested too deeply") from None
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


def _chunk_steps(args: dict, h: int, *, notes: bool, spec: ActionSpec | None = None) -> list[dict]:
    """Validate aligned public arrays before expanding the internal step primitives."""
    if set(args) - ({"deltas", "note", "arm"} if spec and spec.arm_names else {"deltas", "note"}):
        raise ValueError("move_by_chunk accepts only deltas arrays, a single note, and the configured arm selector")
    deltas = args.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError("move_by_chunk.deltas must be a nonempty dimension-to-array object")
    if set(deltas) - {*FIELDS["move_by"], "gripper"}:
        raise ValueError("unknown motion dimension")
    for dimension, values in deltas.items():
        if not isinstance(values, list) or len(values) != h:
            raise ValueError(f"move_by_chunk.deltas.{dimension} must be an array of exactly {h} values")
        if dimension != "gripper" and any(value is None for value in values):
            raise ValueError("only gripper entries may be null; pose entries must be finite numeric")
    explanation = args.get("note")
    if "note" in args:
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("move_by_chunk.note must be a nonempty string for the entire call")
    elif notes:
        raise ValueError("motion note is required by this configuration")
    return [
        {
            "deltas": {dimension: values[index] for dimension, values in deltas.items()},
            **({"arm": args["arm"]} if "arm" in args else {}),
            **({"note": explanation} if explanation is not None else {}),
        }
        for index in range(h)
    ]


def decode_tool_calls(
    calls,
    config: ScheduleConfig,
    *,
    plan_id: str,
    observation_step: int,
    notes: bool = True,
    reflection: bool = True,
    action_tools: str = "move_by",
    spec: ActionSpec | None = None,
):
    """Validate one native call and its entire plan before any action can execute."""
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or len(calls) != 1:
        raise ValueError("respond with exactly one native robot tool call")
    call = calls[0]
    if not isinstance(call, Mapping) or call.get("type") != "function_call":
        raise ValueError("robot tool call must be a native function_call")
    if not isinstance(call.get("call_id"), str) or not call["call_id"].strip():
        raise ValueError("native tool call requires a nonempty call_id")
    name = call.get("name")
    _selected(action_tools)
    expected_motion = motion_tool_name(config)
    if name not in (expected_motion, "done", "give_up"):
        raise ValueError(f"native robot tool is not enabled; use {expected_motion}, done or give_up")
    if not isinstance(call.get("arguments"), str):
        raise ValueError("native tool arguments must be a JSON string")
    args = _args(call["arguments"])
    if name == expected_motion:
        steps = [args] if config.control_interface == "move_by" else _chunk_steps(args, config.h, notes=notes, spec=spec)
        payload = {
            "decision": "replace",
            "commands": [{"name": "move_by", "arguments": step} for step in steps],
        }
    else:
        key = "summary" if name == "done" else "reason"
        if set(args) != {key, "hindsight"}:
            raise ValueError(f"{name} requires only {key} and hindsight")
        payload = {"decision": name, "commands": [], **args}
    return decode(
        payload,
        config,
        plan_id=plan_id,
        observation_step=observation_step,
        notes=notes,
        reflection=reflection,
        action_tools=action_tools,
        spec=spec,
    )


def canonical_payload(payload: Any) -> dict:
    """Normalize internal plans and keep old compact replay records readable.

    Parsing and validating every call occurs before returning any plan. This
    function never executes actions, and rejected tool names remain rejected by
    decode even for legacy recordings.
    """
    if not isinstance(payload, dict):
        raise ValueError("tool response must be a JSON object")
    if "decision" in payload:
        return payload
    if "calls" in payload:
        if set(payload) != {"calls"} or not isinstance(payload["calls"], list):
            raise ValueError("calls must be the only field and contain a list")
        calls = payload["calls"]
    else:
        calls = [payload]
    if not calls:
        raise ValueError("at least one robot call is required")
    decoded = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"tool_name", "arguments_json"}:
            raise ValueError("each call requires tool_name and arguments_json only")
        if not isinstance(call["tool_name"], str) or not isinstance(call["arguments_json"], str):
            raise ValueError("tool_name and arguments_json must be strings")
        decoded.append((call["tool_name"], _args(call["arguments_json"])))
    terminal = [n for n, _ in decoded if n in ("done", "give_up")]
    if terminal:
        if len(decoded) != 1:
            raise ValueError("stopping cannot be mixed with other calls")
        name, args = decoded[0]
        key = "summary" if name == "done" else "reason"
        if set(args) != {key, "hindsight"}:
            raise ValueError(f"{name} requires only {key} and hindsight")
        return {"decision": name, "commands": [], **args}
    return {
        "decision": "replace",
        "commands": [{"name": name, "arguments": args} for name, args in decoded],
    }


def decode(
    payload,
    config: ScheduleConfig,
    *,
    plan_id: str,
    observation_step: int,
    notes: bool = True,
    reflection: bool = True,
    action_tools: str = "move_by",
    spec: ActionSpec | None = None,
):
    """Validate the whole decision before returning any executable motion plan."""
    del reflection
    payload = canonical_payload(payload)
    selected = _selected(action_tools)
    spec = spec or ActionSpec()
    if not isinstance(payload, dict):
        raise ValueError("tool response must be a JSON object")
    decision = payload.get("decision")
    permitted = {"decision", "commands", "plan_id", "summary", "reason", "hindsight"}
    if set(payload) - permitted:
        raise ValueError(
            "tool response contains unknown fields; return only the required tool call fields"
        )
    commands = payload.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError("motion tool calls must be a list")
    if decision in ("done", "give_up"):
        if commands:
            raise ValueError("done or give_up must be returned alone without motion tool calls")
        key = "summary" if decision == "done" else "reason"
        detail, hindsight = payload.get(key), payload.get("hindsight")
        if not isinstance(detail, str) or not isinstance(hindsight, str):
            raise ValueError(f"{decision} requires {key} and hindsight strings")
        return Stop(decision, detail, hindsight)
    if decision != "replace":
        raise ValueError("tool_name must be move_by, done, or give_up")
    expected = 1 if config.control_interface == "move_by" else config.h
    if len(commands) != expected:
        raise ValueError(f"response must contain exactly {expected} move_by motion tool calls")
    parsed = []
    for command in commands:
        if not isinstance(command, dict) or set(command) != {"name", "arguments"}:
            raise ValueError("each motion tool call must contain only tool_name and arguments_json")
        name = command["name"]
        if not isinstance(name, str) or name not in selected:
            raise ValueError("motion tool is not enabled")
        args = _args(command["arguments"])
        key = "deltas"
        if set(args) - ({"deltas", "note", "arm"} if spec.arm_names else {"deltas", "note"}):
            raise ValueError("unknown motion argument")
        arm = args.get("arm")
        if spec.arm_names and (not isinstance(arm, str) or arm not in spec.arm_names):
            raise ValueError("motion arm must select one of: " + ", ".join(spec.arm_names))
        values = args.get(key)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{key} must be a nonempty object")
        if set(values) - {*FIELDS[name], "gripper"}:
            raise ValueError("unknown motion dimension")
        values = {label: value for label, value in values.items() if value is not None}
        if not values:
            raise ValueError(f"{key} must contain a non-null motion dimension or gripper")
        converted = {}
        for label, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be finite numeric")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError("numeric conversion overflow") from exc
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite numeric")
            converted[label] = value
        validate_motion_pose(converted, spec, config)
        if "gripper" in converted:
            validate_gripper_bounds(converted["gripper"], spec)
        note = args.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError("note must be a string")
        if notes and (note is None or not note.strip()):
            raise ValueError("motion note is required by this configuration")
        parsed.append(MotionCommand(name, converted, note, arm))
    return MotionPlan(plan_id, tuple(parsed), config.h, observation_step)
