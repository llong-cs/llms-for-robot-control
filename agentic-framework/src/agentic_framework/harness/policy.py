"""A small tool policy retaining Inspect's lifecycle and audit interfaces."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from inspect_robots.errors import PolicyError, PolicyStop
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo

from agentic_framework.common.io import plain, write_json
from agentic_framework.harness.context import ContextBuilder, _feedback, _step, context_messages
from agentic_framework.harness.demonstration import (
    DEFAULT_DEMO_CONTENT,
    DEFAULT_DEMO_IMAGE_MAX_SIDE,
    DEFAULT_DEMO_MODE,
    repair_feedback,
)
from agentic_framework.harness.tools import decode_tool_calls, instructions, native_tool_schemas
from agentic_framework.harness.trajectory_telemetry import llm_receipt, utc_now
from agentic_framework.harness.types import (
    CHUNK_TARGET_REFERENCE,
    MAX_ATTEMPTS_TERMINATION,
    ActionSpec,
    AugmentationConfig,
    ModelRequest,
    MotionPlan,
    ScheduleConfig,
    Stop,
    pack_plan,
)
from agentic_framework.models.llm.backend import (
    RESENT_FAILURE_KINDS,
    BackendFormatError,
    _redacted,
    failure_kind,
)
from agentic_framework.observability.progress import request_progress

# Wait before resending a decision after a transport/provider or internal
# failure. The simulator stays paused during inference, so only wall time passes.
# The n-th resent failure of one decision (transport and internal together)
# waits 2**(n-1) s times a jitter factor in [1, 1.25), capped at 60 s; a
# provider's longer Retry-After (HTTP 429/503) is honoured up to 300 s. The last
# attempt of a decision is never followed by a wait, and format repairs never wait.
RETRY_BACKOFF_BASE_S = 1.0
RETRY_BACKOFF_MAX_S = 60.0
RETRY_AFTER_MAX_S = 300.0
# Module-level so offline tests can replace the clock; production uses the real one.
_sleep = time.sleep
_jitter = random.random


def retry_wait(failures, retry_after_s=None):
    """Return (seconds, source) to wait after a decision's n-th resent failure.

    ``source`` is "retry_after" when the provider's Retry-After exceeds the
    backoff, else "backoff".
    """
    if type(failures) is not int or failures < 1:
        raise ValueError("failures must be a positive integer")
    nominal = RETRY_BACKOFF_BASE_S * 2 ** min(failures - 1, 16)
    backoff = min(RETRY_BACKOFF_MAX_S, nominal * (1 + 0.25 * _jitter()))
    if (
        type(retry_after_s) in (int, float)
        and retry_after_s == retry_after_s
        and retry_after_s > backoff
    ):
        return min(RETRY_AFTER_MAX_S, float(retry_after_s)), "retry_after"
    return backoff, "backoff"


def _exception_mapping(exc, name):
    """Return an exception's attached usage/metadata mapping, or {} for anything else."""
    value = getattr(exc, name, None)
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class AgentPolicyConfig(PolicyConfig):
    model: str = ""
    h: int = 5
    k: int = 5
    control_interface: str = "move_by"
    reasoning_effort: str = "low"
    augmentation: dict = field(default_factory=dict)
    motion_time_scale: int = 1


class LLMMotionPolicy(PolicyBase):
    requires_motion_controller = True
    supports_motion_plan_validation = True

    def __init__(
        self,
        backend,
        schedule=None,
        *,
        action_spec,
        augmentation=None,
        observation_profile="control",
        demo=False,
        demo_path=None,
        demo_mode=DEFAULT_DEMO_MODE,
        demo_content=DEFAULT_DEMO_CONTENT,
        demo_image_max_side=DEFAULT_DEMO_IMAGE_MAX_SIDE,
        record_trajectory=None,
        cameras=(),
        max_images=16,
        max_context_chars=64000,
        action_tools="move_by",
        approver="none",
        max_attempts=3,
        log_dir=None,
        motion_plan_validator=None,
    ):
        if motion_plan_validator is not None and not callable(motion_plan_validator):
            raise TypeError("motion_plan_validator must be callable or None")
        self.motion_plan_validator = motion_plan_validator
        self.backend = backend
        self.schedule = schedule or ScheduleConfig()
        self.augmentation = augmentation or AugmentationConfig()
        if not isinstance(self.augmentation, AugmentationConfig):
            raise TypeError("augmentation must be AugmentationConfig")
        if not isinstance(action_spec, ActionSpec):
            # No default: a fallback would silently replace the embodiment's
            # measured single-step limits with unrelated global values.
            raise TypeError("LLMMotionPolicy requires the embodiment's explicit ActionSpec")
        self.spec = action_spec
        # The context shares the policy's spec from construction; bind() never swaps it.
        self.context = ContextBuilder(
            self.augmentation,
            self.spec,
            observation_profile=observation_profile,
            demo=demo,
            demo_path=demo_path,
            demo_mode=demo_mode,
            demo_content=demo_content,
            demo_image_max_side=demo_image_max_side,
            cameras=cameras,
            max_images=max_images,
            max_context_chars=max_context_chars,
        )
        if action_tools != "move_by":
            raise ValueError("action_tools must be move_by; absolute motion tools were removed")
        reasoning = self.augmentation.reasoning
        self.reasoning_effort = reasoning.effective_effort
        self.notes = reasoning.enabled and reasoning.notes
        self.reflection = reasoning.enabled and reasoning.hindsight
        self.action_tools, self.approver = action_tools, approver
        # Attempts of ONE logical decision; the count restarts after each accepted
        # decision. Format repairs and transport/provider or internal resends all consume one.
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = max_attempts
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if record_trajectory is not None and type(record_trajectory) is not bool:
            raise TypeError("record_trajectory must be a boolean")
        self.record_trajectory = bool(record_trajectory)
        self.trajectory_recorder = None
        self._recording_observation = None
        self.config = AgentPolicyConfig(
            action_horizon=1 if self.schedule.control_interface == "move_by" else self.schedule.h,
            replan_interval=(
                None if self.schedule.k == -1
                else self.schedule.k * self.schedule.motion_time_scale
            ),
            model=getattr(backend, "model", type(backend).__name__),
            h=self.schedule.h,
            k=self.schedule.k,
            motion_time_scale=self.schedule.motion_time_scale,
            control_interface=self.schedule.control_interface,
            reasoning_effort=self.reasoning_effort,
            augmentation=asdict(self.augmentation),
        )
        self.calls = 0
        self.decisions = []
        self.feedback = []
        self.actions = []
        self.usage = Counter()
        self._transcript_cursor = 0
        self._closed = False
        self._last_inference_latency_s = 0.0
        self._scene_id = "trial"
        self.log_errors = []
        self._trial_started = time.monotonic()
        self._decision_count = 0
        self._pending_call = None
        self._tool_results = []

    def bind(self, embodiment_info):
        self._embodiment_name = embodiment_info.name
        self.info = PolicyInfo(
            "llm-motion",
            embodiment_info.action_space,
            embodiment_info.observation_space,
            embodiment_info.control_hz,
        )
        bounds = embodiment_info.action_space
        semantics = getattr(bounds, "semantics", None)
        declared = self.spec
        if tuple(getattr(bounds, "shape", ())) != (len(declared.input_low),) or semantics is None:
            raise ValueError("LLMMotionPolicy requires seven action dimensions per declared arm")
        if (
            semantics.control_mode != "eef_delta_pose"
            or semantics.frame != declared.frame
            or semantics.rotation_repr != declared.rotation
            or semantics.gripper != "continuous"
        ):
            raise ValueError(
                "LLMMotionPolicy requires eef_delta_pose in the declared frame/rotation "
                "and a continuous gripper"
            )
        dimensions = ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")
        expected_labels = (
            tuple(f"{arm}_{name}" for arm in declared.arm_names for name in dimensions)
            if declared.arm_names else dimensions
        )
        if tuple(semantics.dim_labels or ()) != expected_labels:
            raise ValueError("OSC action dimensions must match each declared arm's pose/gripper block")
        if (
            embodiment_info.control_hz is None
            or not np.isfinite(embodiment_info.control_hz)
            or embodiment_info.control_hz <= 0
        ):
            raise ValueError("LLMMotionPolicy requires a finite positive control_hz")
        low = tuple(float(value) for value in bounds.low)
        high = tuple(float(value) for value in bounds.high)
        if (
            declared.input_low != low
            or declared.input_high != high
            or declared.control_hz != embodiment_info.control_hz
            or declared.gripper_low != low[-1]
            or declared.gripper_high != high[-1]
        ):
            raise ValueError("declared action_spec does not match the live embodiment contract")
        if self.context.observation_profile == "openpi_matched" and embodiment_info.name not in (
            "libero",
        ):
            raise ValueError("openpi_matched is a LIBERO-specific observation profile")
        if self.context.observation_profile == "privileged" and embodiment_info.name not in (
            "libero", "maniskill-franka-droid", "maniskill-ditto-franka-droid",
        ):
            raise ValueError("privileged is supported only for LIBERO and ManiSkill DROID families")
        if self.context.demonstration is not None and self.context.demo_content == "full":
            # The teacher contract is checked here and never enters the context.
            contract = self.context.demonstration.contract
            if contract["control_interface"] != self.schedule.control_interface:
                raise ValueError(
                    "demonstration control_interface is incompatible with the live policy"
                )
            demo_spec = contract["action_spec"]
            live_spec = plain(asdict(self.spec))
            semantic_fields = (
                "pose_scale", "pose_sign", "frame", "rotation", "input_low", "input_high",
                "gripper_mode", "gripper_low", "gripper_high", "gripper_neutral",
                "gripper_open", "gripper_close", "control_hz",
            )
            if self.spec.arm_names:
                semantic_fields += ("arm_names", "gripper_units")
            if self.spec.gripper_measured_range is not None:
                semantic_fields += ("gripper_measured_range",)
            mismatched = [name for name in semantic_fields
                          if demo_spec.get(name) != live_spec[name]]
            if mismatched:
                raise ValueError(
                    "demonstration action_spec is incompatible with the live embodiment: "
                    + ", ".join(mismatched)
                )
            # H can differ: current tool definitions govern every live action.
        prompt_options = dict(
            notes=self.notes,
            reflection=self.reflection,
            action_tools=self.action_tools,
            approver=self.approver,
            observation_profile=self.context.observation_profile,
            embodiment_name=embodiment_info.name,
            is_simulated=bool(getattr(embodiment_info, "is_simulated", True)),
        )
        self._instructions = instructions(self.schedule, self.spec, **prompt_options)

    def on_trial_start(self, scene_id, epoch, log_dir, run_id):
        if self.log_dir is None:
            self.log_dir = Path(log_dir)
        # Sidecars are best-effort; create their directory lazily when writing.

    def reset(self, scene):
        self._end_rollout()
        reset = getattr(self.backend, "reset", None)
        if reset is not None:
            reset()
        self._pending_call = None
        self._tool_results = []
        self._scene_id = scene.id
        if self.trajectory_recorder is not None and not self.trajectory_recorder.managed:
            self.trajectory_recorder = None
        self._recording_observation = None
        self._recording_scene = plain(scene)
        self._decision_count = 0
        self._last_inference_latency_s = 0.0
        self._trial_started = time.monotonic()
        self.log_errors.clear()
        self.context.reset()
        self.calls = 0
        self.decisions.clear()
        self.feedback.clear()
        self.actions.clear()
        self.usage.clear()
        self._transcript_cursor = 0
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.update_metadata(**self._recording_metadata())

    def _schedule_metadata(self):
        schedule = asdict(self.schedule)
        if self.schedule.control_interface == "move_by_chunk":
            schedule["chunk_target_reference"] = CHUNK_TARGET_REFERENCE
        return schedule

    def _recording_metadata(self):
        return {
            "scene_id": self._scene_id, "scene": self._recording_scene,
            "embodiment": self._embodiment_name, "policy_family": "llm",
            "model": self.config.model, "augmentation": asdict(self.augmentation),
            "observation_profile": self.context.observation_profile,
            "demo_mode": self.context.demo_mode,
            "demo_content": self.context.demo_content,
            "action_spec": asdict(self.spec), "schedule": self._schedule_metadata(),
            "instructions": self._instructions,
            "max_attempts": self.max_attempts,
            "tool_schemas": native_tool_schemas(
                schedule=self.schedule, action_tools=self.action_tools,
                notes=self.notes, reflection=self.reflection, spec=self.spec,
            ),
        }

    def _append(self, entry):
        call_started = entry.pop("_call_started_mono", None)
        if call_started is not None:
            call_span = time.monotonic() - call_started
            entry["policy_postprocessing_s"] = max(0.0, call_span - entry["elapsed_s"])
            entry["policy_elapsed_s"] = (call_span + entry.get("context_build_s", 0)
                                          + entry.get("request_prepare_s", 0))
        entry["telemetry"] = llm_receipt(entry, model=self.config.model)
        # Telemetry lives outside model_output; recorded backend metadata is audit only.
        if isinstance(entry.get("backend"), dict):
            entry["backend"].pop("telemetry", None)
        entry = _redacted(plain(entry))
        self.decisions.append(entry)
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.append(entry, self._recording_observation, self.context, self._tool_results)
        if self.log_dir is not None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / "decisions.jsonl").open("a") as f:
                    f.write(json.dumps(plain(entry), ensure_ascii=False, allow_nan=False) + "\n")
            except (OSError, TypeError, ValueError) as exc:
                self._log_failure(exc)

    def _step_budget(self, observation):
        """Model-visible physical step budget at the time of a request.

        The rollout passes the executed policy-phase control steps as env_step;
        done-verification holds follow the last model call and never count here.
        """
        max_steps = self.schedule.max_steps
        remaining = max_steps - _step(observation)
        if remaining < 1:
            raise ValueError("a model request requires at least one remaining policy control step")
        return {"steps_remaining": remaining, "max_steps": max_steps}

    def prepare_request(self, observation, status=None, *, request_id=None) -> ModelRequest:
        """Preview the next request without accepting a decision or spending a call.

        Bind and reset the policy as for execution first. Context selection is
        identical to execution; even its audit scratch state is restored here.
        """
        audit = self.context.last_audit
        try:
            text, images = self.context.build(
                observation, status=status, step_budget=self._step_budget(observation)
            )
        finally:
            self.context.last_audit = audit
        return self._request_from_snapshot(
            text,
            images,
            request_id=(
                f"{self._scene_id}-{self.calls + 1:05d}" if request_id is None else request_id
            ),
        )

    def _request_from_snapshot(
        self, text, images, *, request_id, validation_error=None, decision_id=None
    ) -> ModelRequest:
        messages = context_messages(text, images)
        if validation_error:
            repair = repair_feedback(validation_error)
            text += repair
            messages[-1]["content"].append({"type": "text", "text": repair})
        return ModelRequest(
            request_id,
            self._instructions,
            text,
            images,
            native_tool_schemas(
                schedule=self.schedule,
                action_tools=self.action_tools,
                notes=self.notes,
                reflection=self.reflection,
                spec=self.spec,
            ),
            self.reasoning_effort,
            messages=messages,
            history_length=self.augmentation.memory.history_length,
            tool_results=tuple(copy.deepcopy(self._tool_results)),
            decision_id=decision_id or f"{self._scene_id}-decision-{self._decision_count + 1:05d}",
        )

    def _note_recording_error(self, primary, recording_error):
        message = f"Trajectory recording failed: {type(recording_error).__name__}: {recording_error}"
        primary.add_note(message)
        if message not in self.log_errors:
            self.log_errors.append(message)

    def _decide(self, observation, status=None):
        started = time.monotonic()
        self._decision_count += 1
        primary_error = None
        try:
            return self._decide_impl(observation, status)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            # Includes all repair and retry requests, the backoff waits before
            # resends, and terminal/failed calls too.
            self._last_inference_latency_s = time.monotonic() - started
            if self.trajectory_recorder is not None:
                try:
                    self.trajectory_recorder.record_decision_timing(
                        f"{self._scene_id}-decision-{self._decision_count:05d}",
                        self._last_inference_latency_s,
                    )
                except BaseException as recording_error:
                    if primary_error is None:
                        raise
                    self._note_recording_error(primary_error, recording_error)

    def _decide_impl(self, observation, status=None):
        if self.record_trajectory or self.trajectory_recorder is not None:
            if self.trajectory_recorder is None:
                if self.log_dir is None:
                    raise ValueError("record_trajectory requires a trial log directory")
                from agentic_framework.harness.trajectory_recorder import TrajectoryRecorder
                self.trajectory_recorder = TrajectoryRecorder(self.log_dir, {
                    **self._recording_metadata(), "recording_mode": "decisions_only",
                })
            self._recording_observation = observation
        context_started = time.monotonic()
        step_budget = self._step_budget(observation)
        text, images = self.context.build(observation, status=status, step_budget=step_budget)
        context_elapsed = time.monotonic() - context_started
        validation_error = None
        # Why this attempt exists: None (first), "format" (repair), or "transport"
        # or "internal" (the previous attempt's failure; the same request resent).
        retry_reason = None
        last_failure = None
        resent_failures = 0
        # Wall time waited before this attempt was sent (resends only).
        retry_wait_s, retry_wait_source = 0.0, None
        logical_id = f"{self._scene_id}-decision-{self._decision_count:05d}"
        # The attempt budget is per decision: an accepted decision starts a new count.
        for attempt in range(1, self.max_attempts + 1):
            self.calls += 1
            request_id = f"{self._scene_id}-{self.calls:05d}"
            step = int(observation.extra.get("env_step", 0))
            prepare_started = time.monotonic()
            # A resend reuses the previous attempt's request context: the same
            # observation, tool results and any pending repair feedback.
            request = self._request_from_snapshot(
                text,
                images,
                request_id=request_id,
                validation_error=validation_error,
                decision_id=logical_id,
            )
            entry = {
                "request_id": request_id,
                "logical_decision_id": logical_id,
                "observation_step": step,
                "kind": "plan",
                "attempt": attempt,
                "retry_reason": retry_reason,
                "retry_wait_s": retry_wait_s,
                "retry_wait_source": retry_wait_source,
                "step_budget": step_budget,
                "image_sha256": {
                    k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in images.items()
                },
                "context_text": request.text,
                "context_messages": request.messages,
                "tool_results": request.tool_results,
                "reasoning_requested": self.reasoning_effort,
                "augmentation": asdict(self.augmentation),
                "context_audit": dict(self.context.last_audit),
            }
            call_start = time.monotonic()
            entry["request_wall_s"] = call_start - self._trial_started
            entry.update(
                _call_started_mono=call_start, request_started_at=utc_now(),
                context_build_s=context_elapsed if attempt == 1 else 0.0,
                request_prepare_s=call_start - prepare_started,
                request_summary={
                    "reasoning_effort": self.reasoning_effort,
                    "history_length": request.history_length,
                    "context_chars": len(request.text), "instruction_chars": len(request.instructions),
                    "image_count": len(images),
                    "image_uncompressed_bytes": sum(image.nbytes for image in images.values()),
                    "image_shapes": {name: list(image.shape) for name, image in images.items()},
                    "tool_count": len(request.tools), "context_reused": attempt > 1,
                    "retry_reason": retry_reason, "retry_wait_s": retry_wait_s,
                    "scope": "current policy request; backend wire history counted separately",
                },
            )
            if self.trajectory_recorder is not None:
                self.trajectory_recorder.inference_started(
                    request_id=request_id, logical_decision_id=logical_id,
                    observation_step=step, kind="llm", attempt=attempt,
                    request_summary=entry["request_summary"],
                )
            call_start = time.monotonic()
            entry.update(_call_started_mono=call_start, request_started_at=utc_now(),
                         request_wall_s=call_start - self._trial_started)
            try:
                with request_progress(getattr(self, "runtime_progress", None)):
                    reply = self.backend.generate(request)
                backend_call_elapsed = time.monotonic() - call_start
                entry["request_finished_at"] = utc_now()
                self._tool_results.clear()
            except BackendFormatError as exc:
                backend_call_elapsed = time.monotonic() - call_start
                entry["request_finished_at"] = utc_now()
                self._tool_results.clear()
                validation_error = str(exc)
                self._reject_calls(
                    getattr(exc, "metadata", {}).get("tool_calls", ()), validation_error
                )
                usage = dict(getattr(exc, "usage", {}))
                self.usage.update({k: v for k, v in usage.items() if type(v) is int})
                entry.update(
                    validation_error=validation_error,
                    error_kind="format",
                    outcome="output_limit" if exc.category == "output_limit" else "format_error",
                    response_text=getattr(exc, "response_text", ""),
                    tool_calls=plain(getattr(exc, "metadata", {}).get("tool_calls", ())),
                    usage=usage,
                    backend=dict(getattr(exc, "metadata", {})),
                    elapsed_s=backend_call_elapsed,
                )
                self._append(entry)
                retry_reason, last_failure = "format", f"format: {validation_error}"
                retry_wait_s, retry_wait_source = 0.0, None
                continue
            except BaseException as exc:
                backend_call_elapsed = time.monotonic() - call_start
                entry["request_finished_at"] = utc_now()
                # Configuration errors (local setup found before sending, or a
                # Codex isolation breach) and cancellation end the trial at once.
                # Every transport/provider failure (any HTTP status) and internal
                # exception is resent as the next attempt; the provider may
                # already have billed it.
                kind = failure_kind(exc)
                usage = _exception_mapping(exc, "usage")
                self.usage.update({k: v for k, v in usage.items() if type(v) is int})
                entry.update(
                    error=f"{type(exc).__name__}: {exc}",
                    error_kind=kind,
                    usage=usage,
                    backend=_exception_mapping(exc, "metadata"),
                    elapsed_s=backend_call_elapsed,
                )
                recorded = True
                try:
                    self._append(entry)
                except BaseException as recording_error:
                    # A failed audit write must not turn cancellation or the
                    # original model failure into a different rollout outcome,
                    # and an unrecorded failed attempt is never retried.
                    self._note_recording_error(exc, recording_error)
                    recorded = False
                if kind not in RESENT_FAILURE_KINDS or not recorded:
                    raise
                # No model output exists, so the resent request carries no
                # rejection receipt; pending tool results stay undelivered.
                retry_reason, last_failure = kind, entry["error"]
                resent_failures += 1
                retry_wait_s, retry_wait_source = 0.0, None
                if attempt < self.max_attempts:
                    # Back off before resending; the simulator stays paused.
                    retry_wait_s, retry_wait_source = retry_wait(
                        resent_failures, entry["backend"].get("retry_after_s")
                    )
                    _sleep(retry_wait_s)
                continue
            entry.update(
                payload=plain(reply.payload),
                tool_calls=plain(reply.tool_calls),
                usage=dict(reply.usage),
                backend=dict(reply.metadata),
                elapsed_s=backend_call_elapsed,
                backend_reported_elapsed_s=reply.elapsed_s,
            )
            if reply.response_text:
                entry["response_text"] = reply.response_text
            self.usage.update({k: v for k, v in reply.usage.items() if type(v) is int})
            try:
                result = decode_tool_calls(
                    reply.tool_calls,
                    self.schedule,
                    plan_id=request_id,
                    observation_step=step,
                    notes=self.notes,
                    reflection=self.reflection,
                    action_tools=self.action_tools,
                    spec=self.spec,
                )
                if isinstance(result, MotionPlan) and self.motion_plan_validator is not None:
                    self.motion_plan_validator(result)
            except (ValueError, TypeError, OverflowError) as exc:
                validation_error = str(exc)
                self._reject_calls(reply.tool_calls, validation_error)
                entry.update(
                    validation_error=validation_error, error_kind="format", outcome="format_error"
                )
                self._append(entry)
                retry_reason, last_failure = "format", f"format: {validation_error}"
                retry_wait_s, retry_wait_source = 0.0, None
                continue
            except BaseException as exc:
                # Not a format failure (e.g. a failed simulator preflight). The backend
                # has already accepted this reply, so it is not resent; record the
                # billed reply before the trial ends.
                entry.update(error=f"{type(exc).__name__}: {exc}", error_kind=failure_kind(exc))
                try:
                    self._append(entry)
                except BaseException as recording_error:
                    self._note_recording_error(exc, recording_error)
                raise
            entry["outcome"] = "stop" if isinstance(result, Stop) else "plan"
            self._append(entry)
            call = reply.tool_calls[0]
            if isinstance(result, MotionPlan):
                self._pending_call = {"call_id": call["call_id"], "plan_id": result.plan_id}
            else:
                self._tool_results.append(
                    self._tool_result(
                        call["call_id"],
                        {
                            "status": "stopped",
                            "tool": result.name,
                            "detail": result.detail,
                        },
                    )
                )
                self._end_rollout()
            return result
        raise PolicyStop(
            MAX_ATTEMPTS_TERMINATION,
            f"decision {logical_id} failed all {self.max_attempts} attempts "
            f"(last: {last_failure}); the episode is discarded",
        )

    def act(self, observation):
        result = self._decide(observation)
        if isinstance(result, Stop):
            raise PolicyStop(result.name, result.detail, result.hindsight)
        if not isinstance(result, MotionPlan):
            raise PolicyError("planning requires a new plan or stop")
        chunk = pack_plan(result, action_size=len(self.spec.input_low))
        from dataclasses import replace

        return replace(chunk, inference_latency_s=self._last_inference_latency_s)

    def record_execution(self, feedback):
        self.feedback.append(plain(feedback))
        if self._pending_call is not None:
            receipt = _feedback(feedback)
            if receipt.get("plan_id") != self._pending_call["plan_id"]:
                raise PolicyError("Execution receipt does not match the pending native tool call")
            self._tool_results.append(self._tool_result(self._pending_call["call_id"], receipt))
            self._pending_call = None

    def record_execution_result(self, feedback, observation, t):
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.execution(feedback, observation, self.context, t)

    def record_stop_execution_result(self, observation, t, reason, steps_executed, *,
                                     control_action):
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.stop_execution(
                observation, self.context, t, reason, steps_executed,
                control_action=control_action,
            )

    def record_action(self, t, action, observation):
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.action(t, action)
        self.actions.append(
            {"t": t, "action": plain(action.data), "meta": plain(dict(action.meta))}
        )

    @staticmethod
    def _tool_result(call_id, output):
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(plain(output), ensure_ascii=False, allow_nan=False),
        }

    def _reject_calls(self, calls, error):
        for call in calls:
            call_id = call.get("call_id")
            if isinstance(call_id, str) and call_id:
                self._tool_results.append(
                    self._tool_result(
                        call_id,
                        {
                            "status": "rejected",
                            "executed": False,
                            "error": error,
                        },
                    )
                )

    def _end_rollout(self):
        if self._pending_call is not None:
            self._tool_results.append(
                self._tool_result(
                    self._pending_call["call_id"],
                    {
                        "status": "interrupted",
                        "reason": "rollout ended before an execution receipt",
                    },
                )
            )
            self._pending_call = None
        if self._tool_results:
            if self.trajectory_recorder is not None:
                # Accepted receipts are replaced by measured execution results.
                terminal = [r for r in self._tool_results if json.loads(r["output"]).get("status") != "accepted"]
                self.trajectory_recorder.results(terminal)
                self.trajectory_recorder.flush()
            record = getattr(self.backend, "record_tool_results", None)
            if record is not None:
                record(tuple(self._tool_results))
            self._tool_results.clear()
        end = getattr(self.backend, "end_rollout", None)
        if end is not None:
            final_usage = end()
            if isinstance(final_usage, dict):
                self.usage.update({k: v for k, v in final_usage.items() if type(v) is int})
                if final_usage and self.trajectory_recorder is not None:
                    self.trajectory_recorder.record_usage_adjustment(
                        final_usage, source=type(self.backend).__name__ + ".end_rollout",
                    )

    @staticmethod
    def _transcript_rows(decisions):
        return [
            {
                "request_id": d["request_id"],
                "step": d["observation_step"],
                "kind": d["kind"],
                "outcome": d.get("outcome"),
                "elapsed_s": d["elapsed_s"],
                "error": d.get("error", d.get("validation_error")),
            }
            for d in decisions
        ]

    def transcript(self):
        # Full wire records live in sidecars; summaries are detached from live state.
        return self._transcript_rows(self.decisions)

    def transcript_delta(self):
        result = self._transcript_rows(self.decisions[self._transcript_cursor :])
        self._transcript_cursor = len(self.decisions)
        return result

    def on_trial_end(self, record, log_dir, run_id):
        self._end_rollout()
        if self.trajectory_recorder is not None and not self.trajectory_recorder.managed:
            self.trajectory_recorder.finish(record)
        self._flush()
        record.metadata["agentic"] = self.metrics()

    def metrics(self):
        offline_preview = bool(getattr(self.backend, "is_offline_preview", False))
        decisions = [d for d in self.decisions if d["attempt"] == 1]
        intervals = [
            {
                "from_request": a["request_id"],
                "to_request": b["request_id"],
                "control_steps": b["observation_step"] - a["observation_step"],
                "sim_time_s": (b["observation_step"] - a["observation_step"])
                / self.spec.control_hz,
                "wall_time_s": b["request_wall_s"] - a["request_wall_s"],
            }
            for a, b in zip(decisions, decisions[1:])
        ]
        return {
            "model": self.config.model,
            "model_calls": 0 if offline_preview else self.calls,
            **(
                {"scripted_requests": self.calls, "model_call_source": "scripted_preview"}
                if offline_preview
                else {}
            ),
            # Per decision; format_repair_calls + transport_retry_calls +
            # internal_retry_calls count the requests beyond each decision's first attempt.
            "max_attempts": self.max_attempts,
            "transport_retry_calls": sum(
                d.get("retry_reason") == "transport" for d in self.decisions
            ),
            "transport_failures": sum(d.get("error_kind") == "transport" for d in self.decisions),
            # Unexpected exceptions while building or sending a request or parsing
            # its response (for example a framework bug), resent like transport failures.
            "internal_retry_calls": sum(
                d.get("retry_reason") == "internal" for d in self.decisions
            ),
            "internal_failures": sum(d.get("error_kind") == "internal" for d in self.decisions),
            # Wall time waited before resends; never part of inference_wall_s.
            "retry_wait_s": sum(d.get("retry_wait_s", 0.0) for d in self.decisions),
            # Requests that began a new native thread after a failure lost the
            # rollout's conversation (Codex with memory): from there the episode
            # continues without its earlier turns. An audit flag for memory ablations.
            "conversation_restarts": sum(
                d.get("backend", {}).get("conversation_restarted") is True for d in self.decisions
            ),
            "configuration_failures": sum(
                d.get("error_kind") == "configuration" for d in self.decisions
            ),
            "native_retry_notifications": sum(
                d.get("backend", {}).get("native_retry_notifications", 0) for d in self.decisions
            ),
            "planning_calls": sum(d["kind"] == "plan" for d in decisions),
            "polling_calls": 0,
            "format_repair_calls": sum(d.get("retry_reason") == "format" for d in self.decisions),
            "decision_intervals": intervals,
            "inference_control_steps": [d["observation_step"] for d in decisions],
            "output_limit_errors": sum(d.get("outcome") == "output_limit" for d in self.decisions),
            "simulation_paused_during_inference": True,
            "log_errors": list(self.log_errors),
            "outcomes": dict(Counter(d.get("outcome", "error") for d in self.decisions)),
            "format_errors": sum("validation_error" in d for d in self.decisions),
            "usage": dict(self.usage),
            "inference_wall_s": sum(d["elapsed_s"] for d in self.decisions),
            "executed_actions": len(self.actions),
            "feedback": plain(self.feedback),
            "schedule": self._schedule_metadata(),
            "action_spec": asdict(self.spec),
            "augmentation": asdict(self.augmentation),
            "trajectory_recording": str(self.trajectory_recorder.path) if self.trajectory_recorder is not None else None,
            "context": {
                "demo": self.context.demo,
                "demo_mode": self.context.demo_mode,
                "demo_content": self.context.demo_content,
                "demo_path": str(self.context.demo_path) if self.context.demo_path is not None else None,
                "demo_image_max_side": self.context.demo_image_max_side,
                "observation_profile": self.context.observation_profile,
                "cameras": self.context.cameras,
                "max_images": self.context.max_images,
                "max_context_chars": self.context.max_context_chars,
            },
        }

    def _log_failure(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        if message not in self.log_errors:
            self.log_errors.append(message)
            warnings.warn(
                f"Policy sidecar logging degraded: {message}", RuntimeWarning, stacklevel=2
            )

    def _flush(self):
        if self.log_dir is not None:
            try:
                write_json(
                    self.log_dir / "execution.json",
                    {"actions": self.actions, "feedback": self.feedback},
                )
                write_json(self.log_dir / "decisions.json", self.decisions)
                write_json(self.log_dir / "policy-metrics.json", self.metrics())
            except (OSError, TypeError, ValueError) as exc:
                self._log_failure(exc)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._end_rollout()
            finally:
                self.backend.close()
        finally:
            self._flush()
