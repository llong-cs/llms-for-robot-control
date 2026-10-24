"""Native tool-call decoding, motion envelopes, schedules and native model profiles."""

from __future__ import annotations

import json
import math

import pytest

from agentic_framework.configuration.arguments import parse_configuration
from agentic_framework.configuration.profiles import (
    list_model_profiles,
    load_model_profile,
    profile_path,
    validate_native_horizon,
)
from agentic_framework.harness.tools import (
    decode,
    decode_tool_calls,
    native_tool_schemas,
    parse_tool_call,
    to_plan,
)
from agentic_framework.harness.types import (
    ActionSpec,
    MotionCommand,
    MotionPlan,
    ScheduleConfig,
    Stop,
    ToolCall,
)

H = 4
POSE = ("dx", "dy", "dz", "rx", "ry", "rz")
# Distinct per-axis limits expose any axis mix-up; the native input range (+-1 x pose_scale)
# is wider than every limit, so the limits alone define the single-step envelope.
SPEC = ActionSpec(single_step_pose_limit=(0.001, 0.002, 0.003, 0.01, 0.02, 0.03))
STEP_BOUND = dict(zip(POSE, (0.001, 0.002, 0.003, 0.01, 0.02, 0.03), strict=True))
MOVE_BY_BOUND = dict(zip(POSE, (0.004, 0.008, 0.012, 0.04, 0.08, 0.12), strict=True))  # H x limit
MOVE_BY = ScheduleConfig(h=H, k=H, control_interface="move_by")
CHUNK = ScheduleConfig(h=H, k=H, control_interface="move_by_chunk")
NOTE = "Hold the pose and look again."


def _call(name, arguments, *, call_id="call-1"):
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": encoded}


def _decode(calls, schedule, *, spec=SPEC, **options):
    return decode_tool_calls(
        calls, schedule, plan_id="plan-1", observation_step=7, spec=spec, **options
    )


def _move(deltas, *, schedule=MOVE_BY, spec=SPEC, **extra):
    return _decode([_call("move_by", {"deltas": deltas, "note": NOTE, **extra})], schedule,
                   spec=spec)


def _chunk(deltas, *, spec=SPEC, h=H, **extra):
    schedule = ScheduleConfig(h=h, k=h, control_interface="move_by_chunk")
    return _decode([_call("move_by_chunk", {"deltas": deltas, "note": NOTE, **extra})],
                   schedule, spec=spec)


def _motion_schema(schedule, spec=SPEC):
    return native_tool_schemas(schedule, spec)[0]["parameters"]["properties"]["deltas"]


def _edges(bound):
    """Return (accepted, rejected) values at, just inside and just outside +-bound."""
    accepted = (bound, -bound, math.nextafter(bound, 0.0), math.nextafter(-bound, 0.0))
    rejected = (math.nextafter(bound, math.inf), math.nextafter(-bound, -math.inf))
    return accepted, rejected


@pytest.mark.parametrize(
    ("calls", "message"),
    [
        ([], "exactly one native robot tool call"),
        ([_call("move_by", {"deltas": {"dx": 0}, "note": NOTE}, call_id=f"c{i}") for i in (1, 2)],
         "exactly one native robot tool call"),
        ([_call("done", {"summary": "s", "hindsight": "h"}),
          _call("move_by", {"deltas": {"dx": 0}, "note": NOTE}, call_id="call-2")],
         "exactly one native robot tool call"),
        ('{"name": "move_by"}', "exactly one native robot tool call"),
        ([{"type": "message", "call_id": "c", "name": "move_by", "arguments": "{}"}],
         "native function_call"),
        ([{**_call("move_by", {"deltas": {"dx": 0}, "note": NOTE}), "call_id": " "}],
         "nonempty call_id"),
        ([{**_call("move_by", {}), "arguments": {"deltas": {"dx": 0}, "note": NOTE}}],
         "JSON string"),
        ([_call("move_by", "[1, 2]")], "JSON object"),
    ],
    ids=["none", "two-motions", "stop-and-motion", "string", "not-function-call",
         "blank-call-id", "decoded-arguments", "array-arguments"],
)
def test_exactly_one_native_function_call_is_required(calls, message):
    with pytest.raises(ValueError, match=message):
        _decode(calls, MOVE_BY)


@pytest.mark.parametrize(("schedule", "motion", "foreign"),
                         [(MOVE_BY, "move_by", "move_by_chunk"),
                          (CHUNK, "move_by_chunk", "move_by")])
def test_tool_names_are_limited_to_the_interface_and_stop_tools(schedule, motion, foreign):
    for name in (foreign, "move_to", "shell", "Done", ""):
        with pytest.raises(ValueError, match=f"not enabled; use {motion}, done or give_up"):
            _decode([_call(name, {"deltas": {"dx": 0}, "note": NOTE})], schedule)
    tools = [tool["name"] for tool in native_tool_schemas(schedule, SPEC)]
    assert tools == [motion, "done", "give_up"]
    stop = {"summary": "finished", "hindsight": "none"}
    assert _decode([_call("done", stop)], schedule) == Stop("done", "finished", "none")


@pytest.mark.parametrize("axis", POSE)
def test_move_by_envelope_is_h_times_the_single_step_limit(axis):
    bound = MOVE_BY_BOUND[axis]
    accepted, rejected = _edges(bound)
    for value in accepted:
        plan = _move({axis: value})
        assert plan == MotionPlan(
            "plan-1", (MotionCommand("move_by", {axis: value}, NOTE, None),), H, 7
        )
    for value in rejected:
        with pytest.raises(ValueError, match=rf"{axis} exceeds the H-scaled target limit \(H=4\)"):
            _move({axis: value})
    # The advertised schema and the decoder share one envelope.
    assert _motion_schema(MOVE_BY)["properties"][axis] == {
        "type": "number", "minimum": -bound, "maximum": bound,
    }


def test_move_by_envelope_scales_with_h_and_keeps_decimal_endpoints():
    spec = ActionSpec(single_step_pose_limit=(0.022, 0.001, 0.001, 0.01, 0.01, 0.01))
    for h, bound in ((1, 0.022), (5, 0.11), (7, 0.154)):
        schedule = ScheduleConfig(h=h, k=1)
        # 0.022 * 5 in binary floating point is 0.10999999999999999; 0.11 must be accepted.
        assert isinstance(_move({"dx": bound}, schedule=schedule, spec=spec), MotionPlan)
        with pytest.raises(ValueError, match="dx exceeds"):
            _move({"dx": math.nextafter(bound, 1.0)}, schedule=schedule, spec=spec)
        assert _motion_schema(schedule, spec)["properties"]["dx"]["maximum"] == bound


def test_native_input_range_tightens_the_single_step_envelope():
    # dx: +-1 x 0.0005 is tighter than the 0.001 limit; the tighter value is scaled by H.
    spec = ActionSpec(pose_scale=(0.0005, 0.05, 0.05, 0.5, 0.5, 0.5),
                      single_step_pose_limit=SPEC.single_step_pose_limit)
    assert isinstance(_move({"dx": 0.002}, spec=spec), MotionPlan)
    with pytest.raises(ValueError, match=r"dx exceeds .* \[-0.002, 0.002\]"):
        _move({"dx": 0.0021}, spec=spec)


def test_move_by_chunk_requires_exactly_h_values_per_dimension():
    values = [0.001, -0.001, 0.0, 0.0005]
    plan = _chunk({"dx": values, "gripper": [1.0, None, -1.0, None]})
    expected = (
        {"dx": 0.001, "gripper": 1.0}, {"dx": -0.001}, {"dx": 0.0, "gripper": -1.0},
        {"dx": 0.0005},
    )
    assert plan == MotionPlan(
        "plan-1", tuple(MotionCommand("move_by", step, NOTE, None) for step in expected), H, 7
    )
    for wrong in (values[:-1], [*values, 0.0], [], 0.001):
        with pytest.raises(ValueError, match="deltas.dx must be an array of exactly 4 values"):
            _chunk({"dx": wrong})
    with pytest.raises(ValueError, match="deltas.gripper must be an array of exactly 4 values"):
        _chunk({"dx": values, "gripper": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="only gripper entries may be null"):
        _chunk({"dx": [0.0, None, 0.0, 0.0]})
    # An index with nothing but an omitted gripper command is no step at all.
    with pytest.raises(ValueError, match="non-null motion dimension or gripper"):
        _chunk({"gripper": [1.0, None, 1.0, 1.0]})
    with pytest.raises(ValueError, match="accepts only deltas arrays"):
        _chunk({"dx": values}, arm="left")
    schema = _motion_schema(CHUNK)["properties"]["dx"]
    assert (schema["minItems"], schema["maxItems"]) == (H, H)


@pytest.mark.parametrize("axis", POSE)
def test_move_by_chunk_values_obey_the_unscaled_single_step_limit(axis):
    bound = STEP_BOUND[axis]
    accepted, rejected = _edges(bound)
    plan = _chunk({axis: list(accepted)})
    assert [command.values[axis] for command in plan.commands] == list(accepted)
    for value in (*rejected, MOVE_BY_BOUND[axis]):
        # Every index is validated before execution, including the last one.
        with pytest.raises(ValueError, match=f"{axis} exceeds the single-step limit"):
            _chunk({axis: [0.0, 0.0, 0.0, value]})
    assert _motion_schema(CHUNK)["properties"][axis]["items"] == {
        "type": "number", "minimum": -bound, "maximum": bound,
    }


@pytest.mark.parametrize("interface", ["move_by", "move_by_chunk"])
def test_gripper_bounds_are_unscaled(interface):
    spec = ActionSpec(gripper_low=-0.5, gripper_neutral=0.0, gripper_high=0.75,
                      input_low=(-1.0,) * 6 + (-0.5,), input_high=(1.0,) * 6 + (0.75,))

    def move(value):
        if interface == "move_by":
            return _move({"gripper": value}, spec=spec)
        return _chunk({"gripper": [value] * H}, spec=spec)

    for value in (-0.5, 0.75, 0.0):
        assert all(c.values["gripper"] == value for c in move(value).commands)
    for value in (math.nextafter(0.75, 1.0), math.nextafter(-0.5, -1.0), 0.75 * H):
        with pytest.raises(ValueError, match=r"gripper is out of gripper bounds \[-0.5, 0.75\]"):
            move(value)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ('{"deltas": {"dx": NaN}, "note": "n"}', "dx must be finite numeric"),
        ('{"deltas": {"dy": Infinity}, "note": "n"}', "dy must be finite numeric"),
        ('{"deltas": {"dz": 1e400}, "note": "n"}', "dz must be finite numeric"),
        ('{"deltas": {"rx": true}, "note": "n"}', "rx must be finite numeric"),
        ('{"deltas": {"ry": "0.001"}, "note": "n"}', "ry must be finite numeric"),
        ('{"deltas": {"rz": 1' + "0" * 400 + '}, "note": "n"}', "numeric conversion overflow"),
        ('{"deltas": {"dx": null}, "note": "n"}', "non-null motion dimension"),
        ('{"deltas": {}, "note": "n"}', "deltas must be a nonempty object"),
        ('{"deltas": {"dw": 0.0}, "note": "n"}', "unknown motion dimension"),
        ('{"deltas": {"dx": 0.0}, "note": "n", "speed": 1}', "unknown motion argument"),
        ('{"deltas": {"dx": 0.0}}', "motion note is required"),
        ('{"deltas": {"dx": 0.0}, "note": "  "}', "motion note is required"),
        ('{"deltas": {"dx": 0.0}, "note": 3}', "note must be a string"),
        ('{"deltas": {"dx": 0.0}', "Expecting"),
    ],
)
def test_move_by_rejects_malformed_values(arguments, message):
    with pytest.raises(ValueError, match=message):
        _decode([_call("move_by", arguments)], MOVE_BY)


def test_notes_are_optional_only_when_disabled():
    plan = _decode([_call("move_by", {"deltas": {"dx": 0.0}})], MOVE_BY, notes=False)
    assert plan.commands == (MotionCommand("move_by", {"dx": 0.0}, None, None),)
    chunk = _decode([_call("move_by_chunk", {"deltas": {"dx": [0.0] * H}})], CHUNK, notes=False)
    assert [command.note for command in chunk.commands] == [None] * H
    with pytest.raises(ValueError, match="motion note is required"):
        _decode([_call("move_by_chunk", {"deltas": {"dx": [0.0] * H}})], CHUNK)


@pytest.mark.parametrize("schedule", [MOVE_BY, CHUNK], ids=["move_by", "move_by_chunk"])
def test_done_and_give_up_return_stops_with_detail_and_hindsight(schedule):
    done = _decode([_call("done", {"summary": "Block placed.", "hindsight": "Go slower."})],
                   schedule)
    assert done == Stop("done", "Block placed.", "Go slower.")
    give_up = _decode([_call("give_up", {"reason": "Out of reach.", "hindsight": "none"})],
                      schedule)
    assert give_up == Stop("give_up", "Out of reach.", "none")
    for name, arguments, message in (
        ("done", {"reason": "r", "hindsight": "h"}, "done requires only summary and hindsight"),
        ("done", {"summary": "s"}, "done requires only summary and hindsight"),
        ("done", {"summary": "s", "hindsight": "h", "success": True}, "requires only"),
        ("give_up", {"summary": "s", "hindsight": "h"}, "give_up requires only reason"),
        ("done", {"summary": "s", "hindsight": None}, "requires summary and hindsight strings"),
        ("give_up", {"reason": 1, "hindsight": "h"}, "requires reason and hindsight strings"),
    ):
        with pytest.raises(ValueError, match=message):
            _decode([_call(name, arguments)], schedule)


def _outcome(function):
    """Return a call's result, or its ValueError message for a byte-exact comparison."""
    try:
        return function()
    except ValueError as exc:
        return ("ValueError", str(exc))


@pytest.mark.parametrize(
    ("schedule", "calls", "failing_stage"),
    [
        (MOVE_BY, [_call("move_by", {"deltas": {"dx": 0.004}, "note": NOTE})], None),
        (CHUNK, [_call("move_by_chunk", {"deltas": {"dx": [0.001] * H}, "note": NOTE})], None),
        (CHUNK, [_call("done", {"summary": "s", "hindsight": "h"})], None),
        (MOVE_BY, [], "parse"),
        (MOVE_BY, [{"type": "message", "call_id": "c", "name": "move_by", "arguments": "{}"}],
         "parse"),
        (MOVE_BY, [_call("move_by_chunk", {"deltas": {"dx": [0.0] * H}, "note": NOTE})], "parse"),
        (MOVE_BY, [_call("move_by", "[1, 2]")], "parse"),
        (MOVE_BY, [_call("move_by", '{"deltas": {"dx": 0.0}')], "parse"),
        (CHUNK, [_call("move_by_chunk", {"deltas": {"dx": [0.0] * (H - 1)}, "note": NOTE})],
         "parse"),
        (CHUNK, [_call("move_by_chunk", {"deltas": {"dx": [0.0] * H}})], "parse"),
        (CHUNK, [_call("move_by_chunk", {"deltas": {"dx": [0.0] * H}, "note": NOTE},
                       call_id=" ")], "parse"),
        (MOVE_BY, [_call("move_by", {"deltas": {"dx": 0.0041}, "note": NOTE})], "to_plan"),
        (MOVE_BY, [_call("move_by", {"deltas": {"dx": 0.0}})], "to_plan"),
        (MOVE_BY, [_call("move_by", {"deltas": {"dx": 0.0}, "note": NOTE, "speed": 1})],
         "to_plan"),
        (CHUNK, [_call("move_by_chunk", {"deltas": {"dx": [0.0, 0.0, 0.0, 0.0011]},
                                         "note": NOTE})], "to_plan"),
        (CHUNK, [_call("move_by_chunk", {"deltas": {"gripper": [None] * H}, "note": NOTE})],
         "to_plan"),
        (CHUNK, [_call("done", {"summary": "s"})], "to_plan"),
        (MOVE_BY, [_call("give_up", {"reason": 1, "hindsight": "h"})], "to_plan"),
    ],
)
def test_parse_tool_call_then_to_plan_equals_decode_tool_calls(schedule, calls, failing_stage):
    expected = _outcome(lambda: _decode(calls, schedule))
    parsed = _outcome(lambda: parse_tool_call(calls, schedule, spec=SPEC))
    if failing_stage == "parse":
        assert isinstance(expected, tuple) and parsed == expected
        return
    assert isinstance(parsed, ToolCall)
    planned = _outcome(lambda: to_plan(parsed, schedule, plan_id="plan-1", observation_step=7,
                                       spec=SPEC))
    assert planned == expected
    assert isinstance(expected, tuple) == (failing_stage == "to_plan")


@pytest.mark.parametrize(
    ("schedule", "call"),
    [
        (MOVE_BY, _call("move_by", '{"note": "Hold.",  "deltas": {"dx": 0.0}}')),
        (CHUNK, _call("move_by_chunk", {"deltas": {"dx": [0.0] * H}, "note": NOTE})),
        (MOVE_BY, _call("done", {"summary": "s", "hindsight": "h"})),
        (CHUNK, _call("give_up", {"reason": "r", "hindsight": "h"}, call_id="call-9")),
    ],
)
def test_tool_call_record_round_trips_the_native_call(schedule, call):
    parsed = parse_tool_call([call], schedule, spec=SPEC)
    assert parsed == ToolCall(call["call_id"], call["name"], json.loads(call["arguments"]),
                              call["arguments"])
    assert parsed.record() == call
    assert parse_tool_call([parsed.record()], schedule, spec=SPEC) == parsed


def test_compact_records_decode_through_the_same_validation():
    arguments = json.dumps({"deltas": {"dz": 0.012}, "note": NOTE})
    plan = decode({"tool_name": "move_by", "arguments_json": arguments}, MOVE_BY,
                  plan_id="old", observation_step=3, spec=SPEC)
    assert plan == MotionPlan("old", (MotionCommand("move_by", {"dz": 0.012}, NOTE, None),), H, 3)
    with pytest.raises(ValueError, match="dz exceeds"):
        decode({"tool_name": "move_by", "arguments_json": json.dumps(
            {"deltas": {"dz": 0.0121}, "note": NOTE})}, MOVE_BY, plan_id="old",
            observation_step=3, spec=SPEC)
    stop = {"tool_name": "done", "arguments_json": '{"summary": "s", "hindsight": "h"}'}
    assert decode(stop, MOVE_BY, plan_id="old", observation_step=3) == Stop("done", "s", "h")
    with pytest.raises(ValueError, match="stopping cannot be mixed"):
        decode({"calls": [stop, {"tool_name": "move_by", "arguments_json": arguments}]},
               MOVE_BY, plan_id="old", observation_step=3, spec=SPEC)
    with pytest.raises(ValueError, match="motion tool is not enabled"):
        decode({"tool_name": "move_to", "arguments_json": arguments}, MOVE_BY,
               plan_id="old", observation_step=3, spec=SPEC)


def test_schedule_config_horizon_contract():
    assert ScheduleConfig() == ScheduleConfig(h=5, k=5, control_interface="move_by")
    assert ScheduleConfig(h=3, k=-1, control_interface="move_by").k == -1
    assert ScheduleConfig(h=3, k=8, control_interface="move_by_chunk").k == 8
    with pytest.raises(ValueError, match="K=-1 requires control_interface=move_by"):
        ScheduleConfig(h=3, k=-1, control_interface="move_by_chunk")
    for h in (0, -1, True, 1.0, "5", None):
        with pytest.raises(ValueError, match="h must be a positive integer"):
            ScheduleConfig(h=h)
    for k in (0, -2, True, 1.0, None):
        with pytest.raises(ValueError, match="k must be -1 or a positive integer"):
            ScheduleConfig(k=k)
    with pytest.raises(ValueError, match="control_interface must be move_by or move_by_chunk"):
        ScheduleConfig(control_interface="move_to")
    for name in ("max_steps", "max_motion_steps", "motion_time_scale"):
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            ScheduleConfig(**{name: 0})


def test_native_profiles_load_without_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_PROJECT_ROOT", str(tmp_path))
    profiles = list_model_profiles()
    assert {"molmoact2-droid", "molmoact2-libero", "pi05-droid", "pi05-libero"} <= set(profiles)
    for profile_id, profile in profiles.items():
        defaults = profile["defaults"]
        assert profile["model"]["expected_metadata"]["action_horizon"] == defaults["h"]
        checkpoint = profile_path(profile["model"]["checkpoint"])
        assert checkpoint.is_relative_to(tmp_path) and not checkpoint.exists(), profile_id
        # A native profile runs through move_by_chunk with its own H and positive K.
        ScheduleConfig(h=defaults["h"], k=defaults["k"], max_steps=defaults["max_steps"],
                       control_interface="move_by_chunk")
        validate_native_horizon(profile, defaults["h"])
        with pytest.raises(ValueError, match=f"requires H={defaults['h']}"):
            validate_native_horizon(profile, defaults["h"] + 1)
    droid = load_model_profile("molmoact2-droid")
    assert droid["family"] == "molmoact2"
    assert droid["environment"]["benchmark"] == "maniskill"
    assert {key: droid["defaults"][key] for key in ("h", "k", "control_hz")} == {
        "h": 15, "k": 15, "control_hz": 15,
    }
    assert load_model_profile("molmoact2") == load_model_profile("MOLMO") == droid


def test_molmoact2_droid_resolves_its_native_cadence():
    args = parse_configuration(["--model-profile", "molmoact2-droid", "--mode", "vla"])
    assert (args.h, args.k, args.control_hz) == (15, 15, 15)
    assert (args.control_interface, args.policy_family, args.benchmark) == (
        "move_by_chunk", "molmoact2", "maniskill",
    )
    with pytest.raises(ValueError, match=r"requires H/K/Hz=\(15, 15, 15\)"):
        parse_configuration(["--model-profile", "molmoact2-droid", "--mode", "vla", "--k", "5"])
    custom = ["--model-profile", "molmoact2-droid", "--mode", "vla", "--policy-profile", "custom"]
    assert parse_configuration([*custom, "--k", "5"]).k == 5
    with pytest.raises(ValueError, match="requires H=15 from its model profile"):
        parse_configuration([*custom, "--h", "10"])
    with pytest.raises(ValueError, match="control_interface=move_by_chunk"):
        parse_configuration(["--model-profile", "molmoact2-droid", "--mode", "vla",
                             "--control-interface", "move_by"])
