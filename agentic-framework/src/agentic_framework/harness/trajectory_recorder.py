"""Versioned measured trajectories and model decisions, using one recorder.

The embodiment proxy records physical samples; policy hooks describe decisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_framework.common.io import plain, write_json
from agentic_framework.environments.recording_state import (
    recording_state_catalog,
    validate_recording_state,
)
from agentic_framework.harness.done_verification import VERIFICATION_PHASE, episode_phases
from agentic_framework.harness.task_outcome import (
    TaskFeedbackRecordingError,
    sanitize_task_feedback,
    sanitize_task_section,
    task_evaluation,
    task_progress,
)
from agentic_framework.harness.trajectory_telemetry import (
    summarize_requests,
    usage_receipt,
    utc_now,
)
from agentic_framework.harness.types import DISCARDED_STATUS, MAX_ATTEMPTS_TERMINATION
from agentic_framework.models.llm.backend import _redacted


class TrajectoryRecorder:
    def __init__(self, directory, metadata, *, managed=False, record_environment_state=None):
        if record_environment_state is None:
            record_environment_state = managed
        if type(record_environment_state) is not bool:
            raise TypeError("record_environment_state must be Boolean")
        if managed and not record_environment_state:
            raise ValueError("managed trajectory recording always includes environment state")
        self.record_environment_state = record_environment_state
        self._latest_task_evaluation = None
        self._initial_task_evaluation = None
        self._environment_feedback = {}
        self._environment_feedback_integrity = None
        self._environment_catalog = None
        self.directory = Path(directory)
        self.path = self.directory / "trajectory.json"
        self.steps_path = self.directory / "trajectory-steps.jsonl"
        self.events_path = self.directory / "trajectory-events.jsonl"
        self.managed = managed
        self.document = {
            "schema_version": 2,
            "steps_file": self.steps_path.name,
            "events_file": self.events_path.name,
            "step_count": 0,
            "event_count": 0,
            "decision_timings": [],
            "usage_adjustments": [],
            "metadata": _redacted(plain(metadata)),
            "turns": [],
            "outcome": {"oracle_success": False, "complete": False},
        }
        self.document["metadata"]["environment_state_recording"] = {
            "enabled": record_environment_state, "schema_version": 2,
            "capability": "recording_state", "policy_visible": False,
            "frame": "world", "position_unit": "m", "quaternion_order": "xyzw",
            "status": "pending" if record_environment_state else "disabled",
        }
        self._calls = {}
        self._plans = {}
        self._stop_turn = None
        self._native_turns = {}
        # Recorded done-verification hold rows; they always form the trace's suffix.
        self._verification_rows = 0
        self._started = time.monotonic()
        self._cpu_started = time.process_time()
        self._reset_elapsed_s = None
        self._step_elapsed_s = 0.0
        self._snapshot_elapsed_s = 0.0
        self._video_elapsed_s = None
        self.document["metadata"]["telemetry"] = {
            "schema_version": 1, "started_at": utc_now(),
            "relative_wall_time_origin": "recorder construction",
            "policy_request_wall_time_origin": "policy reset",
            "clock": "monotonic elapsed seconds; UTC for cross-process audit only",
            "token_accounting": "provider-reported counts; absent is null, not zero",
            "cost_accounting": "reported provider amount only; estimates belong to offline reports",
            "simulator_timing": "adapter round trip including rendering/IPC if inside adapter; not pure physics time",
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.steps_path.exists() or self.events_path.exists():
            raise FileExistsError(f"trajectory artifacts already exist in {self.directory}")
        self.steps_path.touch(exist_ok=False)
        self.events_path.touch(exist_ok=False)
        self.record_event("recorder_started")
        self.flush()

    def record_event(self, event_type, **fields):
        event = {"event": event_type, "sequence": self.document["event_count"],
                 "at": utc_now(), "wall_time_s": time.monotonic() - self._started,
                 **fields}
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(_redacted(plain(event)), ensure_ascii=False, allow_nan=False) + "\n")
        self.document["event_count"] += 1

    def inference_started(self, *, request_id, logical_decision_id, observation_step,
                          kind, attempt=1, request_summary=None):
        self.record_event("inference_started", request_id=request_id,
                          logical_decision_id=logical_decision_id, observation_step=observation_step,
                          kind=kind, attempt=attempt, request=request_summary or {})
        self.flush()

    def record_decision_timing(self, logical_decision_id, elapsed_s):
        self.document["decision_timings"].append({
            "logical_decision_id": logical_decision_id, "elapsed_s": elapsed_s,
            "scope": "policy decision including context construction, all repair and retry requests, validation and audit writes",
        })
        self.flush()

    def record_usage_adjustment(self, usage, *, source):
        self.document["usage_adjustments"].append({
            "at": utc_now(), "source": source, "request_id": None,
            "usage": usage_receipt(plain(usage), source=source, scope="late_session_usage"),
        })
        self._summarize()
        self.flush()

    def _summarize(self):
        self.document["telemetry_summary"] = summarize_requests(
            self.document["turns"], self.document["usage_adjustments"],
        )
        finalized_at = self.document.get("runtime", {}).get("finalized_at")
        if finalized_at is None and self.document["outcome"].get("complete"):
            finalized_at = utc_now()
        self.document["runtime"] = {
            "recording_wall_s": time.monotonic() - self._started,
            "framework_process_cpu_s": time.process_time() - self._cpu_started,
            "framework_process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "resources_scope": "framework process only; excludes remote model/simulator/GPU processes; peak RSS is process lifetime on Linux",
            "reset_wall_s": self._reset_elapsed_s,
            "simulator_step_wall_s": self._step_elapsed_s,
            "environment_snapshot_wall_s": self._snapshot_elapsed_s,
            "video_export_wall_s": self._video_elapsed_s,
            "sampled_at": utc_now(), "finalized_at": finalized_at,
        }

    def video_timing(self, elapsed_s):
        self._video_elapsed_s = elapsed_s
        self._summarize()
        self.flush()

    def update_metadata(self, **metadata):
        self.document["metadata"].update(_redacted(plain(metadata)))
        self.flush()

    def _image(self, frame):
        frame = np.asarray(frame)
        digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
        relative = Path("trajectory-images") / f"{digest}.png"
        path = self.directory / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(path)
        return str(relative)

    def _request_value(self, value):
        """Persist exact native inputs without embedding huge image lists in JSON."""
        if isinstance(value, Image.Image):
            value = np.asarray(value)
        if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3:
            return {"image_path": self._image(value), "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, Mapping):
            return {str(k): self._request_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._request_value(v) for v in value]
        return plain(value)

    def record_reset(self, observation, *, scene, seed, reset_info, control_hz,
                     environment_state=None, environment_state_error=None, timing=None):
        if self.document["step_count"]:
            raise ValueError("one trajectory recorder must be used for exactly one reset")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("trajectory recording requires a finite positive control_hz")
        reset_info, reset_integrity = sanitize_task_feedback(reset_info, path="simulator_reset")
        if not isinstance(reset_info, Mapping):
            reset_integrity["issues"].append({"path": "simulator_reset", "error": "expected_mapping", "type": type(reset_info).__name__})
            reset_integrity.update(complete=False, status="partial")
            reset_info = {"invalid_native_reset_info": reset_info}
        if not reset_integrity["complete"]:
            self.document["metadata"]["simulator_reset_integrity"] = reset_integrity
        environment_state = sanitize_task_section(environment_state)
        state = self._measured_state(observation)
        fingerprint = hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.update_metadata(
            scene_id=scene.id, scene=plain(scene),
            suite=scene.metadata.get("suite"), task_id=scene.metadata.get("task_id"),
            init_state_index=scene.metadata.get("init_state_index"),
            environment_seed=seed,
            initial_state_sha256=reset_info.get("initial_state_sha256"),
            reset_observation_sha256=fingerprint, simulator_reset=reset_info,
            control_hz=float(control_hz), pose_frame="world", quaternion_order="xyzw",
            pose_source="observation.state.eef_pos/eef_quat (environment control point)",
            units={"eef_pos": "m", "eef_quat_xyzw": "unit_quaternion", "gripper_width": "m", "joint_pos": "rad", "sim_time_s": "s"},
        )
        if "left_eef_pos" in state or "right_eef_pos" in state:
            self.update_metadata(
                pose_source="observation.state.<arm>_eef_pos/<arm>_eef_quat (environment control points)",
                arm_names=[arm for arm in ("left", "right") if f"{arm}_eef_pos" in state],
            )
        if environment_state is not None:
            self.update_metadata(initial_environment_state_sha256=hashlib.sha256(
                json.dumps(environment_state, sort_keys=True, allow_nan=False).encode()
            ).hexdigest())
        self._environment_feedback = {"phase": "reset", "info": reset_info}
        self._environment_feedback_integrity = reset_integrity
        self._sample(observation, environment_state=environment_state,
                     environment_state_error=environment_state_error, timing=timing)
        self.record_event("reset_completed", observation_step=0, timing=timing or {})
        self.flush()

    @staticmethod
    def _measured_state(observation):
        state = observation.state or {}
        width = state.get("gripper_width")
        if width is not None:
            width = np.asarray(width, dtype=float)
            if width.size != 1:
                raise ValueError("measured gripper_width must contain one scalar")
            width = float(width.reshape(-1)[0])
        result = {
            "eef_pos": plain(state.get("eef_pos")),
            "eef_quat_xyzw": plain(state.get("eef_quat")),
            "gripper_width": width,
            "joint_pos": plain(state.get("joint_pos")),
        }
        # Keep each dual-arm sensor separate; never collapse one arm into a
        # single end-effector trajectory or retain task-object state here.
        for arm in ("left", "right"):
            for field in ("eef_pos", "eef_quat", "joint_pos", "gripper_pos", "gripper_target", "gripper_width"):
                key = f"{arm}_{field}"
                if key in state:
                    output_key = key + "_xyzw" if field == "eef_quat" else key
                    result[output_key] = plain(state[key])
        return result

    def environment_snapshot(self, embodiment):
        """Capture a side-channel sample, preserving physical samples on read failure."""
        if not self.record_environment_state:
            return None, None
        report = self.document["metadata"]["environment_state_recording"]
        try:
            capabilities = getattr(embodiment.info, "capabilities", ())
            reader = getattr(embodiment, "recording_state", None)
            if "recording_state" not in capabilities or not callable(reader):
                raise ValueError("environment does not support recording_state")
            report["capability"] = "recording_state"
            state = sanitize_task_section(reader())
            state = plain(state)
            state = self._validate_environment_state(state, control_hz=embodiment.info.control_hz)
            report.update(status="available", schema_version=state["schema_version"],
                          object_names=sorted(state["objects"]),
                          body_names=sorted(state.get("bodies", {})),
                          coverage=state.get("coverage"))
            return state, None
        except BaseException as error:
            report.update(status="incomplete" if self.document["step_count"] else "unavailable",
                          error=f"{type(error).__name__}: {error}")
            report.setdefault("missing_steps", []).append(self.document["step_count"])
            return None, error

    def _validate_environment_state(self, state, *, control_hz=None):
        state = validate_recording_state(
            state, expected_step=self.document["step_count"],
            control_hz=(self.document["metadata"].get("control_hz")
                        if control_hz is None else control_hz),
        )
        catalog = recording_state_catalog(state)
        if self._environment_catalog is not None and catalog != self._environment_catalog:
            raise ValueError("recorded object/body/joint catalog changed during the episode")
        self._environment_catalog = catalog
        return state

    def _update_task_outcome(self):
        """Preserve the newest native task feedback without inventing a score."""
        latest = copy.deepcopy(self._latest_task_evaluation)
        if latest is None:
            latest = task_evaluation(None, {}, step=None, sim_time_s=None)
        latest["is_final_control_step"] = (
            latest.get("control_step") is not None
            and latest["control_step"] == self.document["step_count"] - 1
        )
        progress = task_progress(latest)
        self.document["outcome"].update(
            task_evaluation=latest,
            initial_task_evaluation=copy.deepcopy(self._initial_task_evaluation),
            task_progress=progress,
            progress_score=progress["value"],
        )

    def _sample(self, observation, action=None, result=None, *, environment_state=None,
                environment_state_error=None, timing=None):
        step = self.document["step_count"]
        environment_state = sanitize_task_section(environment_state)
        meta = plain(dict(action.meta)) if action is not None else {}
        plan_id = meta.get("plan_id")
        # A done-verification hold is not a model decision: no call or turn produced it,
        # including its final step, which carries the timeout request_stop.
        verification = meta.get("evaluation_phase") == VERIFICATION_PHASE
        if self._verification_rows and not verification:
            raise ValueError("a policy-phase step cannot follow a done-verification hold")
        turn = (None if verification else
                self._stop_turn if meta.get("request_stop") else self._plans.get(plan_id))
        request_id = None if verification else (
            meta.get("request_id") or (turn["request_id"] if turn else None)
        )
        row = {
            "step": step, "sim_time_s": step / self.document["metadata"]["control_hz"],
            "observation_time_s": plain(observation.state_time),
            "wall_time_s": time.monotonic() - self._started,
            **self._measured_state(observation),
            "action": plain(action.data) if action is not None else None,
            "action_meta": meta, "request_id": request_id,
            "decision_id": turn.get("logical_decision_id") if turn else request_id,
            "recorded_at": utc_now(), "timing": timing or {},
        }
        if timing:
            if step == 0:
                self._reset_elapsed_s = timing.get("reset_wall_s")
            else:
                self._step_elapsed_s += timing.get("simulator_step_wall_s", 0.0)
            self._snapshot_elapsed_s += timing.get("environment_snapshot_wall_s", 0.0)
        if self.record_environment_state:
            row["environment_state"] = environment_state
            if environment_state is None:
                row["environment_state_error"] = (
                    f"{type(environment_state_error).__name__}: {environment_state_error}"
                    if environment_state_error is not None else "recording state was not supplied"
                )
        if result is not None:
            row["result"], self._environment_feedback_integrity = sanitize_task_feedback({
                "terminated": bool(result.terminated), "truncated": bool(result.truncated),
                "termination_reason": result.termination_reason, "reward": result.reward,
                "info": result.info,
            }, path="result")
        if result is not None:
            self._environment_feedback = {"phase": "step", **row["result"]}
        evaluation = task_evaluation(
            environment_state, self._environment_feedback, step=step,
            sim_time_s=row["sim_time_s"], feedback_integrity=self._environment_feedback_integrity,
        )
        row["task_evaluation"] = evaluation
        self._latest_task_evaluation = evaluation
        if step == 0:
            self._initial_task_evaluation = copy.deepcopy(evaluation)
        payload = json.dumps(_redacted(row), ensure_ascii=False, allow_nan=False)
        with self.steps_path.open("a") as stream:
            stream.write(payload + "\n")
        self.document["step_count"] += 1
        self._verification_rows += int(verification)
        self._update_task_outcome()
        integrity = evaluation.get("integrity", {})
        if integrity.get("complete") is False:
            report = self.document["metadata"].setdefault("task_feedback_recording", {
                "status": "incomplete", "invalid_steps": [],
            })
            report["invalid_steps"].append(step)
            report["latest_issues"] = integrity["issues"]
        self.flush()
        if integrity.get("complete") is False:
            raise TaskFeedbackRecordingError(
                "Native task feedback contained invalid fields; confirmed physical sample was saved "
                "with null values and task_evaluation.integrity issues"
            )

    def record_step(self, action, result, *, environment_state=None, environment_state_error=None,
                    timing=None):
        if not self.document["step_count"]:
            raise ValueError("trajectory reset must be recorded before any step")
        self._sample(result.observation, action, result, environment_state=environment_state,
                     environment_state_error=environment_state_error, timing=timing)
        turn = self._native_turns.get(action.meta.get("request_id"))
        if turn is not None:
            turn["next_observation"] = self._measured_state(result.observation)
            turn["next_observation_step"] = self.document["step_count"] - 1
            turn["execution"] = {
                "steps_executed": sum(1 for line in turn["executed_actions"]) + 1,
                "termination_reason": result.termination_reason,
            }
            self.flush()

    def record_native_inference(self, event, *, request, observation):
        value = _redacted(plain(event))
        request_id = value["request_id"]
        turn = {
            "kind": "native", "request_id": request_id, "logical_decision_id": request_id,
            "observation_step": value["observation_step"], "attempt": 1,
            "request_wall_s": value.get("request_wall_s"), "elapsed_s": value.get("elapsed_s"),
            "request": _redacted(self._request_value(request)),
            "observation": {"instruction": observation.instruction, **self._measured_state(observation)},
            "model_output": {key: item for key, item in value.items() if key != "telemetry"},
            "telemetry": value.get("telemetry", {}),
            "executed_actions": [], "tool_results": [],
        }
        self.record_event("inference_finished", request_id=request_id,
                          observation_step=value["observation_step"],
                          status=turn["telemetry"].get("status", "error" if "error" in value else "completed"))
        self.document["turns"].append(turn)
        self._native_turns[request_id] = turn
        self.flush()

    def record_native_action(self, t, action, observation):
        turn = self._native_turns.get(action.meta.get("request_id"))
        if turn is not None:
            turn["executed_actions"].append({"t": t, "action": plain(action.data), "meta": plain(dict(action.meta))})
            self.flush()

    def record_native_execution(self, feedback, observation, t):
        value = plain(feedback)
        turn = self._native_turns.get(value.get("request_id", value.get("plan_id")))
        if turn is not None:
            turn["execution"] = value
            turn["next_observation"] = self._measured_state(observation)
            turn["next_observation_step"] = t
            self.flush()

    def snapshot(self, context, observation, *, step_budget=None):
        """Record an observation; a request's snapshot adds the step budget it sent."""
        snapshot, images = context._snapshot(observation)
        paths = {name: self._image(frame) for name, frame in images.items()}
        return {
            "instruction": observation.instruction or "",
            **plain(snapshot),
            **({"step_budget": plain(step_budget)} if step_budget is not None else {}),
            "images": paths,
        }

    def append(self, entry, observation, context, results):
        output_keys = (
            "payload", "tool_calls", "response_text", "backend", "validation_error",
            "error", "error_kind", "outcome", "usage", "elapsed_s",
        )
        output = {key: entry[key] for key in output_keys if key in entry}
        output.setdefault("tool_calls", [])
        turn = {
            "kind": "llm",
            "request_id": entry["request_id"],
            "logical_decision_id": entry["logical_decision_id"],
            "observation_step": entry["observation_step"],
            "attempt": entry["attempt"],
            "retry_reason": entry.get("retry_reason"),
            # Wall time waited before a transport or internal resend; not inference time.
            "retry_wait_s": entry.get("retry_wait_s", 0.0),
            "request_wall_s": entry.get("request_wall_s"),
            "elapsed_s": entry.get("elapsed_s"),
            "telemetry": _redacted(plain(entry.get("telemetry", {}))),
            "observation": self.snapshot(context, observation, step_budget=entry.get("step_budget")),
            "request": {
                "text": entry["context_text"],
                "messages": entry["context_messages"],
                "tool_results": entry["tool_results"],
                "reasoning_effort": entry["reasoning_requested"],
            },
            "model_output": _redacted(plain(output)),
            "tool_results": [],
            "executed_actions": [],
        }
        self.record_event("inference_finished", request_id=entry["request_id"],
                          observation_step=entry["observation_step"],
                          status=turn["telemetry"].get("status", entry.get("outcome", "error")))
        self.document["turns"].append(turn)
        for call in output["tool_calls"]:
            if isinstance(call.get("call_id"), str):
                self._calls[call["call_id"]] = turn
        if entry.get("outcome") == "plan":
            self._plans[entry["request_id"]] = turn
        if entry.get("outcome") == "stop":
            self._stop_turn = turn
        self.results(results)
        self.flush()

    def results(self, results):
        for result in results:
            # A transport failure may leave the prior model-facing acknowledgement
            # pending. It must never replace measured physical execution feedback.
            if json.loads(result["output"]).get("status") == "accepted":
                continue
            turn = self._calls.get(result.get("call_id"))
            if turn is None:
                continue
            existing = turn["tool_results"]
            existing[:] = [r for r in existing if r["call_id"] != result["call_id"]]
            existing.append(_redacted(plain(result)))

    def action(self, t, action):
        # Controller meta carries the plan ID; keep applied low-level actions.
        if action.meta.get("evaluation_phase") == VERIFICATION_PHASE:
            return  # A done-verification hold belongs to no model turn.
        plan_id = action.meta.get("plan_id")
        turn = (self._stop_turn if action.meta.get("request_stop")
                else self._plans.get(plan_id))
        if turn is not None:
            turn["executed_actions"].append({"t": t, "action": plain(action.data), "meta": plain(dict(action.meta))})

    def execution(self, feedback, observation, context, t):
        """Record the audit-only execution result of one plan; never model input."""
        value = plain(feedback)
        turn = self._plans.get(value["plan_id"])
        if turn is None:
            raise ValueError("trajectory execution has no recorded plan")
        calls = turn["model_output"]["tool_calls"]
        if len(calls) != 1:
            raise ValueError("executed trajectory turn requires exactly one tool call")
        status = value.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("trajectory execution feedback requires a nonempty status")
        output = {
            "status": status,
            "executed": value["steps_executed"] > 0,
            **{key: item for key, item in value.items() if key != "status"},
        }
        self.results([{
            "type": "function_call_output",
            "call_id": calls[0]["call_id"],
            "output": json.dumps(output, ensure_ascii=False, allow_nan=False),
        }])
        turn["next_observation"] = self.snapshot(context, observation)
        turn["next_observation_step"] = t
        self.flush()

    def stop_execution(self, observation, context, t, reason, steps_executed, *, control_action):
        """Record what followed a native stop receipt; audit-only, never model input.

        The native model receives a stopped acknowledgement before the rollout
        acts on it. ``control_action`` is ``neutral_stop_step`` when the stop ends
        the episode with one neutral step (``steps_executed`` says whether it
        completed, including faults before any simulator step), or
        ``done_verification`` when a done call starts done verification instead;
        its hold steps are not model actions and are never attributed to this turn.
        """
        if control_action not in ("neutral_stop_step", "done_verification"):
            raise ValueError(f"unknown stop control action: {control_action!r}")
        turn = self._stop_turn
        if turn is None:
            raise ValueError("trajectory stop execution has no recorded stop call")
        calls = turn["model_output"]["tool_calls"]
        if len(calls) != 1:
            raise ValueError("trajectory stop requires exactly one tool call")
        turn["native_tool_results"] = list(turn["tool_results"])
        self.results([{
            "type": "function_call_output",
            "call_id": calls[0]["call_id"],
            "output": json.dumps({
                "status": "stopped",
                "termination_reason": reason,
                "executed": steps_executed > 0,
                "steps_executed": steps_executed,
                "control_action": control_action,
            }, ensure_ascii=False, allow_nan=False),
        }])
        turn["next_observation"] = self.snapshot(context, observation)
        turn["next_observation_step"] = t
        self.flush()

    def finish(self, record):
        self.document["metadata"]["environment_seed"] = record.seed
        if "simulator_reset" in record.metadata:
            reset_info, integrity = sanitize_task_feedback(record.metadata["simulator_reset"], path="simulator_reset")
            self.document["metadata"]["simulator_reset"] = reset_info
            if not integrity["complete"]:
                self.document["metadata"]["simulator_reset_integrity"] = integrity
        inference_steps = [turn["observation_step"] for turn in self.document["turns"] if turn.get("attempt", 1) == 1]
        # An episode that exhausted max_attempts is discarded: not evaluated, never a teacher.
        discarded = (
            record.status == "success" and record.termination_reason == MAX_ATTEMPTS_TERMINATION
        )
        status = DISCARDED_STATUS if discarded else record.status
        oracle_success = None if discarded else bool(
            record.steps and record.steps[-1].result.terminated
            and record.steps[-1].result.termination_reason == "success"
        )
        executed = max(0, self.document["step_count"] - 1)
        phases = episode_phases(record, oracle_success=oracle_success)
        self.document["outcome"] = {
            "complete": True,
            "oracle_success": oracle_success,
            "initially_solved": bool(record.metadata.get("initially_solved", False)),
            "status": status,
            "termination_reason": record.termination_reason,
            "error": record.error,
            "executed_control_steps": executed,
            # The dense trace is policy rows followed by done-verification hold rows.
            "policy_control_steps": executed - self._verification_rows,
            "verification_control_steps": self._verification_rows,
            "verification_triggered": phases["verification_triggered"],
            "success_phase": phases["success_phase"],
            "done_verification": plain(
                record.metadata.get("controller_totals", {}).get("done_verification")
            ),
            "inference_control_steps": inference_steps,
            "inference_wall_s": sum(turn.get("elapsed_s") or 0 for turn in self.document["turns"]),
        }
        if discarded:
            self.document["outcome"]["task_success_evaluated"] = False
        self._update_task_outcome()
        self._summarize()
        self.record_event("trial_finalized", status=status,
                          termination_reason=record.termination_reason)
        self.flush()

    def fail(self, exc):
        """Preserve setup/reset failures even when Inspect has no trial record."""
        executed = max(0, self.document["step_count"] - 1)
        self.document["outcome"].update(
            complete=True, status="cancelled" if isinstance(exc, KeyboardInterrupt) else "error",
            termination_reason="cancelled" if isinstance(exc, KeyboardInterrupt) else "error",
            error=f"{type(exc).__name__}: {exc}",
            executed_control_steps=executed,
            policy_control_steps=executed - self._verification_rows,
            verification_control_steps=self._verification_rows,
        )
        self._update_task_outcome()
        self._summarize()
        self.record_event("trial_failed", error=f"{type(exc).__name__}: {exc}")
        self.flush()

    def flush(self):
        write_json(self.path, self.document)


class RecordingEmbodiment:
    """Observe successful reset/step returns without changing their semantics."""

    def __init__(self, embodiment, recorder):
        self.embodiment = embodiment
        self.recorder = recorder

    def __getattr__(self, name):
        return getattr(self.embodiment, name)

    def reset(self, scene, *, seed=None):
        self.recorder.record_event("reset_started", seed=seed)
        started = time.monotonic()
        try:
            observation = self.embodiment.reset(scene, seed=seed)
        except BaseException as exc:
            self.recorder._reset_elapsed_s = time.monotonic() - started
            try:
                self.recorder.record_event("reset_failed", elapsed_s=time.monotonic() - started,
                                           error=f"{type(exc).__name__}: {exc}")
            except BaseException as recording_error:
                exc.add_note(
                    f"Trajectory recording failed: {type(recording_error).__name__}: {recording_error}"
                )
            raise
        reset_elapsed = time.monotonic() - started
        snapshot_started = time.monotonic()
        state, error = self.recorder.environment_snapshot(self.embodiment)
        snapshot_elapsed = time.monotonic() - snapshot_started
        try:
            self.recorder.record_reset(
                observation, scene=scene, seed=seed,
                reset_info=getattr(self.embodiment, "last_reset_info", {}),
                control_hz=self.embodiment.info.control_hz,
                environment_state=state, environment_state_error=error,
                timing={"reset_wall_s": reset_elapsed, "environment_snapshot_wall_s": snapshot_elapsed},
            )
        except BaseException as recording_error:
            if error is None:
                raise
            error.add_note(
                f"Trajectory recording failed: {type(recording_error).__name__}: {recording_error}"
            )
        if error is not None:
            raise error
        return observation

    def step(self, action):
        started = time.monotonic()
        try:
            result = self.embodiment.step(action)
        except BaseException as exc:
            try:
                self.recorder.record_event("step_failed", elapsed_s=time.monotonic() - started,
                                           next_control_step=self.recorder.document["step_count"],
                                           error=f"{type(exc).__name__}: {exc}")
            except BaseException as recording_error:
                exc.add_note(
                    f"Trajectory recording failed: {type(recording_error).__name__}: {recording_error}"
                )
            raise
        step_elapsed = time.monotonic() - started
        snapshot_started = time.monotonic()
        state, error = self.recorder.environment_snapshot(self.embodiment)
        try:
            self.recorder.record_step(
                action, result, environment_state=state, environment_state_error=error,
                timing={"simulator_step_wall_s": step_elapsed,
                        "environment_snapshot_wall_s": time.monotonic() - snapshot_started},
            )
        except BaseException as recording_error:
            if error is None:
                raise
            error.add_note(
                f"Trajectory recording failed: {type(recording_error).__name__}: {recording_error}"
            )
        if error is not None:
            raise error
        return result
