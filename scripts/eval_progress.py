"""Bounded, thread-safe terminal feedback for independent evaluator workers.

Only the small progress snapshot is read. Prompts, model responses, credentials,
exception bodies, and simulator logs are never forwarded to the terminal.
"""
from __future__ import annotations

import json
import math
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

MAX_SNAPSHOT_BYTES = 64 * 1024
_PHASES = frozenset({
    "initializing", "resetting", "inference", "executing", "verification",
    "finalizing", "completed", "failed", "cancelled", "preview", "discarded",
})


def _safe_label(value, limit=100):
    """Keep trusted launcher labels on one terminal line, including in pipes."""
    return re.sub(r"[^A-Za-z0-9 .,:/_+()=\[\]-]", "?", str(value))[:limit]


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value) and 0 <= value <= 1e12:
                return value
        except OverflowError:
            pass
    return None


def _counter(value):
    return value if type(value) is int and 0 <= value <= 10**12 else None


def _utc_seconds(value):
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _duration(value):
    seconds = max(0, int(value))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def _snapshot(path):
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
        if len(raw) > MAX_SNAPSHOT_BYTES:
            return None, "unreadable"
        value = json.loads(raw)
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                or value["schema_version"] != 1 or value.get("phase") not in _PHASES):
            return None, "unreadable"
        return value, None
    except FileNotFoundError:
        return None, "unavailable"
    except (OSError, ValueError, TypeError, RecursionError):
        return None, "unreadable"


def _worker_key(worker):
    return str(worker.get("id", "worker")) if isinstance(worker, dict) else str(worker)


def _worker_label(worker):
    if not isinstance(worker, dict):
        return _safe_label(worker)
    if "devices" not in worker:
        return _safe_label(worker.get("id", "evaluation"), 40)
    devices = worker.get("devices", [])
    if not isinstance(devices, (list, tuple)):
        devices = []
    gpu = ",".join(_safe_label(device, 45) for device in devices)
    return f"{_safe_label(worker.get('id', 'worker'), 40)} GPU {gpu or '?'}"


def _job_key(job):
    return str(job.get("id", "task")) if isinstance(job, dict) else str(job)


def _job_label(job):
    if not isinstance(job, dict):
        return _safe_label(job)
    task = _counter(job.get("task_id"))
    if task is None:
        return _safe_label(job.get("id", "task"))
    pieces = [_safe_label(job.get("suite", ""), 32), f"task {task}"]
    if job.get("difficulty"):
        pieces.append(_safe_label(job["difficulty"], 16))
    initial = _counter(job.get("init_start"))
    if initial is not None:
        pieces.append(f"init {initial}")
    repeat_id = _counter(job.get("repeat_id"))
    if repeat_id is not None:
        pieces.append(f"repeat {repeat_id}")
    return " ".join(piece for piece in pieces if piece)


class ProgressReporter:
    """Serialize concise progress lines without coupling feedback to evaluation.

    ``monotonic`` controls rate limiting and ``wall_time`` supplies Unix seconds
    for UTC timestamps and snapshot ages. Both clocks are injectable for tests.
    A broken output stream disables feedback without interrupting the workers.
    """

    def __init__(self, stream=None, *, monotonic=time.monotonic, wall_time=time.time,
                 interval_s=10.0, heartbeat_s=15.0):
        if (_number(interval_s) is None or _number(heartbeat_s) is None
                or interval_s <= 0 or heartbeat_s <= 0):
            raise ValueError("Progress reporting intervals must be positive finite seconds")
        self._stream = sys.stdout if stream is None else stream
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._interval = float(interval_s)
        self._heartbeat_interval = float(heartbeat_s)
        self._lock = threading.Lock()
        self._last = {}
        self._enabled = True

    def _emit(self, worker, text):
        if not self._enabled:
            return False
        stamp = datetime.fromtimestamp(self._wall_time(), timezone.utc).strftime("%H:%M:%S UTC")
        try:
            self._stream.write(f"[{stamp}] {_worker_label(worker)} | {text}\n")
            self._stream.flush()
        except Exception:
            # Feedback must never turn a completed or progressing trial into a failure.
            self._enabled = False
            return False
        return True

    def event(self, worker, message):
        """Immediately print a trusted launcher lifecycle message."""
        with self._lock:
            return self._emit(worker, _safe_label(message, 400))

    def heartbeat(self, worker, phase, *, elapsed_s, force=False):
        """Print a rate-limited wait for server loading or endpoint validation."""
        with self._lock:
            if not self._enabled:
                return False
            key = ("heartbeat", _worker_key(worker), str(phase))
            now = self._monotonic()
            if not force and now - self._last.get(key, -math.inf) < self._heartbeat_interval:
                return False
            self._last[key] = now
            elapsed = _number(elapsed_s)
            duration = "?" if elapsed is None else _duration(elapsed)
            return self._emit(worker, f"{_safe_label(phase, 100)} | waiting {duration}")

    def job(self, worker, job, snapshot_path, *, elapsed_s, budget, force=False):
        """Read one bounded snapshot and report its allowlisted metrics only.

        Snapshot age is reported independently from process lifetime. In
        particular, a live process or an unchanged snapshot does not establish
        that inference is healthy or that robot control steps are progressing.
        """
        with self._lock:
            if not self._enabled:
                return False
            key = ("job", _worker_key(worker), _job_key(job))
            now = self._monotonic()
            if not force and now - self._last.get(key, -math.inf) < self._interval:
                return False
            self._last[key] = now
            value, unavailable = _snapshot(snapshot_path)
            elapsed = _number(elapsed_s)
            wall = "?" if elapsed is None else _duration(elapsed)
            pieces = [_job_label(job)]
            if value is None:
                pieces += ["initializing", f"progress snapshot {unavailable}", f"wall {wall}"]
                return self._emit(worker, " | ".join(pieces))
            wall_now = self._wall_time()
            updated = _utc_seconds(value.get("updated_at"))
            phase = value["phase"]
            if phase == "completed":
                success = value.get("oracle_success")
                if success is True:
                    phase += ": task succeeded"
                elif success is False:
                    phase += ": task failed"
            elif phase == "failed":
                phase = "failed: runtime error (see eval.log)"
            elif phase == "discarded":
                phase = "discarded: reached max_attempts (excluded from results)"
            elif phase == "verification":
                phase = "verification: holding after done, no model calls"
            pieces.append(phase)
            # steps counts every executed step; policy steps are capped by max_steps and
            # done-verification holds are shown separately against their own limit.
            steps = _counter(value.get("steps"))
            policy_steps = _counter(value.get("policy_steps"))
            verification_steps = _counter(value.get("verification_steps"))
            verification_max = _counter(value.get("verification_max_steps"))
            maximum = _counter(value.get("max_steps"))
            if maximum is None:
                maximum = _counter(budget)
            shown = policy_steps if policy_steps is not None else steps
            steps_text = f"steps {'?' if shown is None else shown}/{'?' if maximum is None else maximum}"
            if verification_steps or value["phase"] == "verification":
                steps_text += (f" (+{'?' if verification_steps is None else verification_steps}"
                               f"/{'?' if verification_max is None else verification_max} verification)")
            pieces.append(steps_text)
            hz = _number(value.get("control_hz"))
            if steps is not None and hz:
                pieces.append(f"sim {steps / hz:.1f}s")
            calls = _counter(value.get("model_calls"))
            completed_calls = _counter(value.get("completed_model_calls"))
            if calls is not None:
                calls_text = f"requests {calls}"
                if completed_calls is not None:
                    calls_text += f" ({completed_calls} complete)"
                pieces.append(calls_text)
            pending = _utc_seconds(value.get("pending_request_started_at"))
            if pending is not None:
                pieces.append(f"inference waiting {_duration(wall_now - pending)}")
            inference_s = _number(value.get("last_inference_s"))
            if inference_s is not None:
                pieces.append(f"last inference {inference_s:.1f}s")
            pieces.append(f"wall {wall}")
            if updated is not None:
                pieces.append(f"snapshot age {_duration(wall_now - updated)}")
            else:
                pieces.append("snapshot age unknown")
            return self._emit(worker, " | ".join(pieces))
