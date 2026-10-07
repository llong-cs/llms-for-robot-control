"""Small, best-effort runtime snapshots for external evaluation monitors."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from inspect_robots.logging.sink import NullSink


def _utc_now():
    return datetime.now(UTC).isoformat()


class EvaluationProgress(NullSink):
    """Publish scalar progress without depending on trajectory recording.

    Step updates are throttled; lifecycle and request transitions are immediate.
    The file always contains one complete JSON object. A failed snapshot write
    must not change policy execution, exception handling, or the task outcome.
    """

    def __init__(self, output, *, max_steps, control_hz, verification_max_steps=0,
                 interval_s=1.0):
        self.path = Path(output) / "progress.json"
        self.interval_s = interval_s
        self._last_write = float("-inf")
        self.state = {}
        self.start_trial(None, max_steps=max_steps, control_hz=control_hz,
                         verification_max_steps=verification_max_steps)

    def start_trial(self, scene_id, *, max_steps, control_hz, verification_max_steps=0):
        now = _utc_now()
        self.state = {
            "schema_version": 1,
            "scene_id": scene_id,
            "phase": "initializing",
            # steps counts every executed step; policy_steps are capped by max_steps and
            # verification_steps are done-verification holds after a model done call.
            "steps": 0,
            "policy_steps": 0,
            "verification_steps": 0,
            "max_steps": max_steps,
            "verification_max_steps": verification_max_steps,
            "control_hz": control_hz,
            "model_calls": 0,
            "completed_model_calls": 0,
            "last_inference_s": None,
            "pending_request_started_at": None,
            "started_at": now,
            "phase_started_at": now,
            "updated_at": now,
        }
        self._publish(force=True)

    def _publish(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_write < self.interval_s:
            return
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            self.state["updated_at"] = _utc_now()
            temporary.write_text(json.dumps(self.state, allow_nan=False) + "\n")
            temporary.replace(self.path)
            self._last_write = now
        except Exception:
            # Telemetry is optional, including on full or unavailable storage.
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass

    def phase(self, value, **fields):
        if self.state["phase"] != value:
            self.state["phase_started_at"] = _utc_now()
        self.state.update(fields, phase=value)
        self._publish(force=True)

    def request_started(self):
        self.state["model_calls"] += 1
        self.phase("inference", pending_request_started_at=_utc_now())

    def request_finished(self, elapsed_s):
        self.state["completed_model_calls"] += 1
        self.phase("executing", pending_request_started_at=None,
                   last_inference_s=max(0.0, elapsed_s))

    def on_trial_start(self, scene_id, epoch):
        self.phase("resetting")

    def log_step(self, t, observation, action, result):
        verification = action.meta.get("evaluation_phase") == "verification"
        self.state["steps"] = t + 1
        self.state["verification_steps"] += int(verification)
        self.state["policy_steps"] = self.state["steps"] - self.state["verification_steps"]
        phase = "verification" if verification else "executing"
        if self.state["phase"] != phase:
            self.phase(phase)
        else:
            self._publish()

    def _step_fields(self, steps, verification):
        return {"steps": steps, "policy_steps": steps - verification,
                "verification_steps": verification}

    def on_trial_end(self, record):
        totals = record.metadata.get("controller_totals", {})
        steps = record.metadata.get("executed_control_steps", totals.get("actual_steps", len(record.steps)))
        verification = totals.get("verification_control_steps", self.state["verification_steps"])
        self.phase("finalizing", **self._step_fields(steps, verification),
                   pending_request_started_at=None)

    def finish(self, row=None, *, phase=None):
        fields = {"pending_request_started_at": None}
        if row is not None:
            fields.update(self._step_fields(
                row.get("steps", self.state["steps"]),
                row.get("verification_control_steps", self.state["verification_steps"]),
            ))
            for key in ("oracle_success", "termination_reason"):
                if key in row:
                    fields[key] = row[key]
            if phase is None:
                phase = {
                    "success": "completed", "cancelled": "cancelled", "discarded": "discarded",
                }.get(row.get("status"), "failed")
        self.phase(phase or "failed", **fields)


@contextmanager
def request_progress(progress):
    """Count actual sends and finished attempts, including repairs and errors."""
    if progress is not None:
        try:
            progress.request_started()
        except Exception:
            pass
    started = time.monotonic()
    try:
        yield
    finally:
        if progress is not None:
            try:
                progress.request_finished(time.monotonic() - started)
            except Exception:
                pass
