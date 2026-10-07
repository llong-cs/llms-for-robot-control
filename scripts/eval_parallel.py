"""Run independent evaluation trials on a fixed pool of visible GPU groups.

The framework remains responsible for trial execution and recording. Each worker
owns one optional model service and its evaluator process groups; cancellation
finishes the evaluator before closing the model service. Every evaluation writes
into a new, empty output directory; there is no resume.

The first SIGINT (Ctrl+C), SIGQUIT (Ctrl+\\), SIGTERM or SIGHUP starts a
graceful interrupted shutdown. From then on, during fail-fast cancellation, once
every trial has finished, and after normal completion or failure, those signals
only print a notice: evaluator and server cleanup, where every wait has a
timeout, and the final receipts always run. SIGKILL, or another signal the
launcher does not handle, still aborts that cleanup and can orphan process groups.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from scripts import workspace as w
from scripts.eval_progress import ProgressReporter
from scripts.eval_server import managed_server


class EvaluationCancelled(RuntimeError):
    """The owning evaluation requested cancellation before this job completed."""


SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGHUP)
SHUTDOWN_NOTICE = "Shutdown in progress; waiting for evaluators and model servers to stop"
# Evaluator cleanup: SIGINT to finalize recordings, then group SIGKILL and a bounded reap.
EVALUATOR_FINALIZE_S = 30
EVALUATOR_REAP_S = 15


def _signal_name(number):
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


class ShutdownSignals:
    """Own SIGINT, SIGQUIT, SIGTERM and SIGHUP while an evaluation holds child processes.

    Before shutdown, the first of these signals starts it: the handler marks
    shutdown and raises KeyboardInterrupt in the main thread, so the launcher
    takes its interrupted path. Inside ``deferred()`` that KeyboardInterrupt is
    held until the block ends. Every later signal only writes a one-line notice
    to stderr and returns. The same holds while any ``settled`` event is set
    (fail-fast cancellation, or every trial finished and only teardown remains),
    because the outcome is already decided, and after ``shutting_down`` is set
    on normal completion or failure. Every evaluator and server cleanup wait has
    a timeout, so cleanup cannot be interrupted and still ends. SIGKILL, or a
    signal not handled here, can still abort it. Signals that are already
    ignored (for example SIGHUP under nohup) stay ignored. Previous handlers are
    restored on exit. Outside the main thread nothing is installed, because
    Python runs handlers only there.
    """

    def __init__(self, *settled):
        self.shutting_down = False
        self.received = None
        self.ignored = 0
        self._settled = settled
        self._deferring = False
        self._pending = False
        self._previous = {}

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        try:
            for number in SHUTDOWN_SIGNALS:
                if signal.getsignal(number) != signal.SIG_IGN:
                    self._previous[number] = signal.signal(number, self._handle)
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, *_):
        self.shutting_down = True
        self._restore()
        return False

    def _restore(self):
        previous, self._previous = self._previous, {}
        for number, handler in previous.items():
            # None means a handler installed outside Python; the default is the closest match.
            signal.signal(number, signal.SIG_DFL if handler is None else handler)

    @contextmanager
    def deferred(self):
        """Hold a first signal until the block ends, then raise KeyboardInterrupt.

        A KeyboardInterrupt raised inside ``executor.submit`` can leave a started
        worker thread that neither its future nor the executor tracks, so
        nothing would wait for it to stop its model server.
        """
        self._deferring = True
        try:
            yield
        finally:
            self._deferring = False
        if self._pending:
            self._pending = False
            raise KeyboardInterrupt

    def _handle(self, number, _frame):
        if not self.shutting_down and not any(event.is_set() for event in self._settled):
            self.shutting_down = True
            self.received = _signal_name(number)
            if self._deferring:
                self._pending = True
                return
            raise KeyboardInterrupt
        try:
            self.ignored += 1
            # os.write avoids re-entering a buffered stream the main thread may hold.
            os.write(2, f"{SHUTDOWN_NOTICE} ({_signal_name(number)} ignored).\n".encode())
        except BaseException:
            # A notice must never interrupt cleanup, even when the terminal is gone.
            pass


class _Countdown:
    """Set ``done`` after ``count`` calls to ``count_down``; safe across worker threads."""

    def __init__(self, count):
        self._lock = threading.Lock()
        self._remaining = count
        self.done = threading.Event()

    def count_down(self):
        with self._lock:
            self._remaining -= 1
            if self._remaining <= 0:
                self.done.set()


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _without_options(command, valued):
    """Remove a bounded set of known valued options without rewriting other args."""
    result = []
    skip = False
    for argument in command:
        if skip:
            skip = False
            continue
        name, separator, _ = str(argument).partition("=")
        if name in valued:
            skip = not separator
        else:
            result.append(str(argument))
    if skip:
        raise ValueError("An evaluation option is missing its value")
    return result


def _transport(plan):
    server = plan.get("server") or {}
    profile = server.get("profile") or {}
    kind = profile.get("transport", {}).get("kind")
    if kind:
        return kind
    return "http" if plan["resolved"].get("policy", {}).get("url") else "websocket"


def _ports(host, count):
    """Choose distinct local ports; readiness also checks ownership before spawn."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Managed servers require a loopback host")
    sockets = []
    try:
        for _ in range(count):
            probe = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(probe)
            probe.bind((host, 0))
        return [probe.getsockname()[1] for probe in sockets]
    finally:
        for probe in sockets:
            probe.close()


def _pool(plan):
    """Validate the job assignment and choose one port per worker; never writes."""
    groups = plan.get("gpu_groups")
    jobs = plan.get("jobs")
    if not isinstance(groups, list) or not groups or any(not isinstance(group, list) or not group for group in groups):
        raise ValueError("Parallel evaluation requires nonempty GPU groups")
    flat_devices = [device for group in groups for device in group]
    if any(not isinstance(device, str) or not device for device in flat_devices):
        raise ValueError("GPU group device selectors must be nonempty strings")
    if len(flat_devices) != len(set(flat_devices)):
        raise ValueError("GPU groups must not overlap")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Parallel evaluation requires at least one planned trial")
    seen = set()
    for job in jobs:
        identifier = job.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identifier) or identifier in seen:
            raise ValueError("Job identifiers must be unique safe directory names")
        seen.add(identifier)
        for name in ("task_id", "init_start"):
            if type(job.get(name)) is not int or job[name] < 0:
                raise ValueError(f"Job {name} must be a nonnegative integer")
        if "repeat_id" in job and (type(job["repeat_id"]) is not int or job["repeat_id"] < 0):
            raise ValueError("Job repeat_id must be a nonnegative integer")
        if not isinstance(job.get("suite"), str) or not job["suite"]:
            raise ValueError("Each job requires a suite")
        if not isinstance(job.get("difficulty"), str) or not job["difficulty"]:
            raise ValueError("Each job requires a difficulty")
    if any(str(argument).split("=", 1)[0] == "--gpu" for argument in plan["command"]):
        raise ValueError("Select GPUs with CUDA_VISIBLE_DEVICES, not --gpu")
    server = plan.get("server") or {}
    managed = bool(server and not server.get("external"))
    host = server.get("host", server.get("basehost", "127.0.0.1"))
    ports = _ports(host, len(groups)) if managed else [server.get("port")] * len(groups)
    return {
        "schema_version": 1,
        "created_at": _timestamp(),
        "workers": [
            {"id": f"worker-{index:03d}", "index": index, "devices": group,
             "host": host, "port": ports[index],
             "job_ids": [job["id"] for job in jobs[index::len(groups)]]}
            for index, group in enumerate(groups)
        ],
    }


def _job_launch(plan, job, worker):
    """Produce a self-contained framework config and one-trial public command."""
    root = Path(plan["output_dir"])
    directory = root / "jobs" / job["id"]
    target = root / "evaluation" / job["id"]
    config = copy.deepcopy(plan["resolved"])
    config.setdefault("environment", {}).update(
        suite=job["suite"], task_ids=str(job["task_id"]), trials=1,
        init_start=job["init_start"], difficulty=job["difficulty"],
    )
    config["environment"].pop("gpu", None)
    if "repeat_id" in job:
        config.setdefault("run", {})["repeat_id"] = job["repeat_id"]
    config.setdefault("output", {})["directory"] = str(target)
    server = plan.get("server")
    policy = config.setdefault("policy", {})
    if server and not server.get("external"):
        policy["host"], policy["port"] = worker["host"], worker["port"]
        if _transport(plan) == "http":
            template = policy.get("url") or server["profile"]["transport"].get("url") or "http://127.0.0.1/act"
            parsed = urlsplit(template)
            authority = f"[{worker['host']}]" if ":" in worker["host"] else worker["host"]
            policy["url"] = urlunsplit((parsed.scheme or "http", f"{authority}:{worker['port']}",
                                        parsed.path or "/act", parsed.query, ""))
    if server:
        if _transport(plan) == "http":
            policy.pop("host", None)
            policy.pop("port", None)
        else:
            policy.pop("url", None)
    replaced_options = {
        "--config", "--suite", "--task-ids", "--trials", "--init-start",
        "--difficulty", "--output-dir", "--policy-port", "--policy-host", "--policy-url",
    }
    if "repeat_id" in job:
        replaced_options.add("--repeat-id")
    command = _without_options(plan["command"], replaced_options)
    command += ["--config", str(directory / "config.json"), "--suite", job["suite"],
                "--task-ids", str(job["task_id"]), "--trials", "1",
                "--init-start", str(job["init_start"]), "--difficulty", job["difficulty"],
                "--output-dir", str(target)]
    if "repeat_id" in job:
        command += ["--repeat-id", str(job["repeat_id"])]
    return config, command


def _stop_evaluator(process):
    """Finalize recordings while their model service remains alive; every wait is bounded."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=EVALUATOR_FINALIZE_S)
        except subprocess.TimeoutExpired:
            pass
    # A child simulator may outlive its evaluator parent. This group is owned
    # by this launch, so clear surviving children even after the parent exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=EVALUATOR_REAP_S)
    except subprocess.TimeoutExpired:
        # Only a process in uninterruptible sleep outlives SIGKILL. It must not
        # keep the model server running or hold back the final receipts.
        raise RuntimeError(f"Evaluator pid {process.pid} was not reaped "
                           f"{EVALUATOR_REAP_S}s after SIGKILL") from None


def _run_job(plan, job, worker, stop_event):
    from scripts.eval_resources import device_environment

    root = Path(plan["output_dir"])
    directory = root / "jobs" / job["id"]
    # run_parallel already wrote this job's config.json from the same plan.
    _, command = _job_launch(plan, job, worker)
    status = {
        "job_id": job["id"], "worker_id": worker["id"], "status": "starting",
        "started_at": _timestamp(), "devices": worker["devices"], "command": command,
        "evaluation_output": str(root / "evaluation" / job["id"]),
    }
    _write(directory / "status.json", status)
    process = None
    progress = plan.get("_progress")
    started = time.monotonic()
    budget = plan["resolved"].get("execution", {}).get("max_steps")
    if budget is None:
        budget = plan.get("control_step_budgets", {}).get(job["suite"])
    snapshot = root / "evaluation" / job["id"] / "progress.json"
    try:
        if stop_event.is_set():
            raise EvaluationCancelled("Evaluation stopped before this trial started")
        env = device_environment(w.child_environment(), worker["devices"])
        env["PYTHONUNBUFFERED"] = "1"
        if progress is not None:
            progress.event(worker, f"Starting task {job['task_id']} ({job['difficulty']}, init {job['init_start']}); "
                           f"log: {directory / 'eval.log'}")
        with (directory / "eval.log").open("w") as log:
            process = subprocess.Popen(command, cwd=plan["cwd"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status="running", pid=process.pid)
            _write(directory / "status.json", status)
            while True:
                if progress is not None:
                    progress.job(worker, job, snapshot, elapsed_s=time.monotonic() - started, budget=budget)
                if process.poll() is not None:
                    code = process.returncode
                    break
                if stop_event.is_set():
                    raise EvaluationCancelled("Another trial failed or evaluation was interrupted")
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            status.update(status="completed" if code == 0 else "failed", returncode=code)
            if code:
                stop_event.set()
                status["error"] = f"Evaluator exited with code {code}; see {directory / 'eval.log'}"
    except EvaluationCancelled as exc:
        status.update(status="interrupted", error=str(exc), returncode=130)
    except BaseException as exc:
        stop_event.set()
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", returncode=2)
    finally:
        if process is not None:
            try:
                status["process_returncode"] = _stop_evaluator(process)
            except BaseException as exc:
                stop_event.set()
                status.update(status="failed", error=f"Evaluator cleanup failed: {type(exc).__name__}: {exc}",
                              returncode=2, process_returncode=None)
        status["finished_at"] = _timestamp()
        _write(directory / "status.json", status)
        if progress is not None:
            progress.job(worker, job, snapshot, elapsed_s=time.monotonic() - started, budget=budget, force=True)
            progress.event(worker, f"Trial {job['id']}: {status['status']} "
                           f"(exit {status['returncode']}); log: {directory / 'eval.log'}")
    return status


def _worker(plan, worker, jobs, stop_event):
    from scripts.eval_resources import device_environment

    directory = Path(plan["output_dir"]) / "workers" / worker["id"]
    directory.mkdir(parents=True, exist_ok=True)
    status = {"worker_id": worker["id"], "devices": worker["devices"],
              "started_at": _timestamp(), "status": "starting", "job_ids": worker["job_ids"]}
    context = nullcontext()
    progress = plan.get("_progress")
    countdown = plan.get("_trial_countdown")
    server = plan.get("server")
    server_started = None
    try:
        if stop_event.is_set():
            raise EvaluationCancelled("Evaluation stopped before this worker started")
        if not worker["job_ids"]:
            status["status"] = "idle"
            return status
        if server and not server.get("external"):
            server_output = directory / "server"
            command = _without_options(server["command"], {"--port", "--host", "--output-dir"})
            command += ["--port", str(worker["port"]), "--host", worker["host"],
                        "--output-dir", str(server_output)]
            base_env = dict(server.get("environment", server.get("env", {})))
            base_env["CUDA_VISIBLE_DEVICES"] = ",".join(device for group in plan["gpu_groups"] for device in group)
            env = device_environment(base_env, worker["devices"])
            env["PYTHONUNBUFFERED"] = "1"
            server_started = time.monotonic()
            if progress is not None:
                progress.event(worker, f"Loading {server['profile']['id']} "
                               f"(timeout {plan['_parallel_server_timeout']:g}s).")
            def report_wait(elapsed_s):
                if progress is not None:
                    progress.heartbeat(worker, "model loading", elapsed_s=elapsed_s)
            context = managed_server(command, env, output_dir=server_output, profile=server["profile"],
                                     host=worker["host"], port=worker["port"], timeout_s=plan["_parallel_server_timeout"],
                                     cancel_event=stop_event, on_wait=report_wait)
            status["server_output"] = str(server_output)
        _write(directory / "status.json", status)
        with context:
            if progress is not None and server_started is not None:
                progress.event(worker, f"Model ready after {time.monotonic() - server_started:.1f}s.")
            status["status"] = "running"
            _write(directory / "status.json", status)
            for identifier in worker["job_ids"]:
                if stop_event.is_set():
                    break
                result = _run_job(plan, jobs[identifier], worker, stop_event)
                if countdown is not None:
                    countdown.count_down()
                if result["status"] != "completed":
                    status.update(status=result["status"], error=result.get("error"))
                    break
            else:
                status["status"] = "completed"
            if status["status"] == "running":
                status["status"] = "interrupted"
    except EvaluationCancelled as exc:
        status.update(status="interrupted", error=str(exc))
    except BaseException as exc:
        was_cancelled = stop_event.is_set()
        stop_event.set()
        status.update(status="interrupted" if was_cancelled else "failed",
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        status["finished_at"] = _timestamp()
        _write(directory / "status.json", status)
        if progress is not None and status["status"] in ("failed", "interrupted"):
            progress.event(worker, f"Worker {status['status']}; details: {directory / 'status.json'}")
    return status


def _aggregate(plan, states):
    root = Path(plan["output_dir"])
    rows, missing, invalid, unstarted = [], [], [], []
    preview = plan["resolved"].get("run", {}).get("mode") == "preview"
    for job in plan["jobs"]:
        identifier = job["id"]
        job_status = states[identifier]
        if job_status["status"] == "not_started":
            unstarted.append(identifier)
        path = root / "evaluation" / identifier / "results.json"
        if not path.is_file():
            missing.append(identifier)
            continue
        try:
            values = _read(path)
            if not isinstance(values, list) or len(values) != 1 or any(not isinstance(row, dict) for row in values):
                raise ValueError("Each trial must write exactly one result object")
            for row in values:
                json.dumps(row, allow_nan=False)
                if not isinstance(row.get("status"), str):
                    raise ValueError("Result status must be a string")
                for name in ("steps", "wall_time_s"):
                    value = row.get(name, 0)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Result {name} must be finite and nonnegative")
                policy = row.get("policy", {})
                if not isinstance(policy, dict):
                    raise ValueError("Result policy metrics must be an object")
                calls = policy.get("model_calls", 0)
                if type(calls) is not int or calls < 0:
                    raise ValueError("Result model_calls must be a nonnegative integer")
                if row["status"] == "success" and not preview and type(row.get("oracle_success")) is not bool:
                    raise ValueError("Completed trial must report an explicit boolean oracle_success")
                if row["status"] == "discarded" and row.get("oracle_success") is not None:
                    raise ValueError("Discarded trial must not report oracle_success")
                rows.append({**row, "evaluation_job_id": identifier,
                             "evaluation_job_status": job_status["status"],
                             **({"evaluation_repeat_id": job["repeat_id"]} if "repeat_id" in job else {}),
                             "source_output_dir": str(path.parent), "source_results_file": str(path)})
        except (ValueError, OSError, TypeError) as exc:
            invalid.append({"job_id": identifier, "error": f"{type(exc).__name__}: {exc}"})
    normal = [row for row in rows if row.get("status") == "success" and row["evaluation_job_status"] == "completed"]
    # Trials discarded after max_attempts are listed and counted only; no metric uses them.
    discarded = [row for row in rows
                 if row.get("status") == "discarded" and row["evaluation_job_status"] == "completed"]
    successes = sum(row.get("oracle_success") is True for row in normal)
    summary = {
        "schema_version": 1, "planned_trials": len(plan["jobs"]),
        "base_trials": plan.get("base_trials", len(plan["jobs"])), "repeats": plan.get("repeats", 1),
        "trials": sum(row.get("status") != "discarded" for row in rows),
        "completed_trials": len(normal),
        "discarded_trials": len(discarded),
        "discarded_jobs": [row["evaluation_job_id"] for row in discarded],
        "completed_jobs": sum(s["status"] == "completed" for s in states.values()),
        "infrastructure_failed_jobs": [key for key, value in states.items() if value["status"] == "failed"],
        "interrupted_jobs": [key for key, value in states.items() if value["status"] == "interrupted"],
        "not_started_jobs": unstarted, "missing_result_jobs": missing, "invalid_results": invalid,
        "steps": sum(row.get("steps", 0) for row in normal),
        # Done-verification holds (after an LLM done call) are counted apart from policy steps.
        "policy_control_steps": sum(row.get("policy_control_steps", row.get("steps", 0))
                                    for row in normal),
        "verification_control_steps": sum(row.get("verification_control_steps", 0)
                                          for row in normal),
        "verification_trials": sum(row.get("verification_triggered") is True for row in normal),
        "wall_time_s": sum(row.get("wall_time_s", 0) for row in normal),
        "model_calls": sum(row.get("policy", {}).get("model_calls", 0) for row in normal),
        "task_success_evaluated": not preview,
        "successes": None if preview else successes,
        "verification_successes": None if preview else sum(
            row.get("oracle_success") is True and row.get("success_phase") == "verification"
            for row in normal),
        "success_rate": successes / len(normal) if normal and not preview else None,
        "normal_task_failures": None if preview else len(normal) - successes,
        "initially_solved": sum(row.get("initially_solved") is True for row in normal),
    }
    if preview:
        summary.update(preview=True, api_calls=0)
    _write(root / "evaluation" / "results.json", rows)
    _write(root / "evaluation" / "summary.json", summary)
    return summary


def run_parallel(plan, *, server_timeout=900):
    """Run one worker per selected GPU group and return an aggregate exit code.

    The device groups and job assignment are validated before any output mutation
    or service launch. The output directory must be new or empty; an existing
    evaluation is never resumed or overwritten.
    """
    from scripts.eval import public_plan

    if not math.isfinite(server_timeout) or server_timeout <= 0:
        raise ValueError("Model server timeout must be positive and finite")
    root = Path(plan["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output directory is not empty: {root}; choose a new output directory")
    # Validate the job assignment before creating output.
    pool = _pool(plan)
    for python in [plan["command"][0], *(
        [plan["server"]["command"][0]] if plan.get("server") and not plan["server"].get("external") else []
    )]:
        if not Path(python).is_file():
            raise ValueError(f"Interpreter does not exist: {python}")
    progress = ProgressReporter()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pool.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this evaluation output") from exc
        # Close the check-to-lock race: only this launch's lock file may exist.
        if any(path.name != ".pool.lock" for path in root.iterdir()):
            raise ValueError(f"Output directory is not empty: {root}; choose a new output directory")
        _write(root / "pool-plan.json", pool)
        _write(root / "eval-config.json", plan["config"])
        _write(root / "eval-plan.json", public_plan(plan))
        _write(root / "resolved-config.json", plan["resolved"])
        # Once every trial has finished, only teardown remains and the outcome is fixed.
        countdown = _Countdown(len(plan["jobs"]))
        current = dict(plan, _parallel_server_timeout=server_timeout, _progress=progress,
                       _trial_countdown=countdown)
        active_workers = sum(bool(worker["job_ids"]) for worker in pool["workers"])
        progress.event({"id": "evaluation"},
                       f"{len(plan['jobs'])} total trials; {active_workers} active workers; "
                       f"{len(pool['workers']) - active_workers} idle GPU groups. "
                       "Progress updates every 10s; model-loading heartbeat every 15s.")
        stop_event = threading.Event()
        jobs = {job["id"]: job for job in plan["jobs"]}
        status = {"status": "starting", "started_at": _timestamp(), "launcher_pid": os.getpid(),
                  "worker_count": len(pool["workers"]), "planned_trials": len(jobs),
                  "pool_plan": str(root / "pool-plan.json")}
        assigned = {identifier: worker for worker in pool["workers"] for identifier in worker["job_ids"]}
        configs = {job["id"]: _job_launch(plan, job, assigned[job["id"]])[0] for job in plan["jobs"]}
        states = {job["id"]: {"job_id": job["id"], "worker_id": assigned[job["id"]]["id"],
                              "status": "not_started", "reason": "Queued for this evaluation"}
                  for job in plan["jobs"]}
        with ShutdownSignals(stop_event, countdown.done) as signals:
            executor = None
            futures = []
            worker_states = []
            outcome = None
            try:
                _write(root / "eval-status.json", status)
                for identifier, config in configs.items():
                    directory = root / "jobs" / identifier
                    _write(directory / "config.json", config)
                    (directory / "eval.log").touch()
                    _write(directory / "status.json", states[identifier])
                server = plan.get("server")
                if server and server.get("external") and plan["resolved"].get("run", {}).get("mode") != "preview":
                    from scripts.eval import external_metadata
                    progress.event({"id": "evaluation"}, "Verifying the external model service.")
                    _write(root / "external-server-metadata.json", external_metadata(plan))
                    progress.event({"id": "evaluation"}, "External model identity verified.")
                executor = ThreadPoolExecutor(max_workers=len(pool["workers"]), thread_name_prefix="evaluation")
                # Every started worker must have a future that shutdown waits on.
                with signals.deferred():
                    for worker in pool["workers"]:
                        futures.append(executor.submit(_worker, current, worker, jobs, stop_event))
                status["status"] = "running"
                _write(root / "eval-status.json", status)
                for future in as_completed(futures):
                    worker_states.append(future.result())
                    if worker_states[-1]["status"] != "idle":
                        finished_workers = sum(state["status"] == "completed" for state in worker_states)
                        progress.event({"id": "evaluation"}, f"{finished_workers}/{active_workers} active workers finished.")
            except BaseException as exc:
                # First statement: from here a signal only prints a notice.
                signals.shutting_down = True
                outcome = exc
            signals.shutting_down = True
            return _shutdown(plan, status, states, outcome=outcome, signals=signals, stop_event=stop_event,
                             executor=executor, futures=futures, worker_states=worker_states,
                             progress=progress)


def _cleanup_failed(errors, step, exc):
    """Record one failed shutdown step; later steps still run."""
    errors.append(f"{step}: {type(exc).__name__}: {exc}")
    try:
        traceback.print_exception(exc)
    except BaseException:
        pass


def _shutdown(plan, status, states, *, outcome, signals, stop_event, executor, futures,
              worker_states, progress):
    """Finish one evaluation; every step runs even when an earlier step fails.

    Runs with ``signals.shutting_down`` set, so SIGINT, SIGQUIT, SIGTERM and
    SIGHUP cannot interrupt it. Cleanup errors are recorded in eval-status.json and
    printed. The run then fails with exit code 1, or keeps 130 when it was
    interrupted. Only a failure to write eval-status.json itself is raised.
    """
    root = Path(plan["output_dir"])
    jobs = plan["jobs"]
    errors = []
    interrupted = isinstance(outcome, KeyboardInterrupt)
    if outcome is not None:
        stop_event.set()
        if interrupted:
            source = signals.received or "user"
            status.update(status="interrupted", error=f"Evaluation interrupted by {source}")
            progress.event({"id": "evaluation"},
                           f"Interrupted by {source}. Finalizing recordings and stopping owned services. "
                           "Further Ctrl+C, SIGQUIT, SIGTERM and SIGHUP are ignored until cleanup finishes.")
        else:
            status.update(status="failed", error=f"{type(outcome).__name__}: {outcome}")
    if executor is not None:
        # Worker-owned evaluator cleanup runs before managed_server exits. Keep
        # ownership until every worker has reaped its processes and written its
        # receipts: waiting on the futures first means Thread.join never runs
        # while a worker is still busy (an interrupted join marks it stopped).
        try:
            wait(futures)
            executor.shutdown(wait=True)
        except BaseException as exc:
            _cleanup_failed(errors, "Waiting for workers", exc)
    for identifier in states:
        path = root / "jobs" / identifier / "status.json"
        try:
            # A missing receipt means shutdown began before it was written.
            value = _read(path) if path.is_file() else dict(states[identifier])
            if not isinstance(value, dict) or not isinstance(value.get("status"), str):
                raise ValueError("Job status must be an object with a status string")
        except BaseException as exc:
            _cleanup_failed(errors, f"Reading job status {identifier}", exc)
            states[identifier] = dict(states[identifier], status="failed",
                                      error=f"Unreadable job status: {type(exc).__name__}: {exc}")
            continue
        states[identifier] = value
        if value["status"] == "not_started":
            value["reason"] = "Evaluation cancelled before this job started"
            try:
                _write(path, value)
            except BaseException as exc:
                _cleanup_failed(errors, f"Writing job status {identifier}", exc)
    summary = None
    try:
        summary = _aggregate(plan, states)
    except BaseException as exc:
        _cleanup_failed(errors, "Aggregating results", exc)
    # A run whose every trial was discarded produced no valid trial; most likely a
    # persistent provider failure, not model behaviour, so it fails like one.
    all_discarded = bool(jobs) and summary is not None and summary["discarded_trials"] == len(jobs)
    if interrupted:
        code = 130
    elif errors or summary is None or status["status"] == "failed" or any(
        worker.get("status") == "failed" for worker in worker_states
    ) or any(value["status"] != "completed" for value in states.values()) or (
        summary["missing_result_jobs"] or summary["invalid_results"]
        or summary["completed_trials"] + summary["discarded_trials"] != len(jobs)
    ) or all_discarded:
        code = 1
        status["status"] = "failed"
        if all_discarded and "error" not in status:
            status["error"] = (
                f"Every trial ({len(jobs)}) was discarded: a decision failed all max_attempts "
                "attempts in each. Every HTTP error, including a provider rejection (wrong key, "
                "model id or parameters), and every internal exception inside the backend "
                "call is resent, so a misconfigured provider or a recurring framework error "
                "ends here; check the recorded model errors in the job trajectories"
            )
    else:
        code = 0
        status["status"] = "completed"
    status.update(returncode=code, finished_at=_timestamp())
    if summary is not None:
        status.update(completed_jobs=summary["completed_jobs"], completed_trials=summary["completed_trials"],
                      discarded_trials=summary["discarded_trials"], successes=summary["successes"],
                      summary=str(root / "evaluation" / "summary.json"))
    if errors:
        status["cleanup_errors"] = errors
    status_error = None
    try:
        _write(root / "eval-status.json", status)
    except BaseException as exc:
        status_error = exc
    if summary is None:
        progress.event({"id": "evaluation"}, f"{status['status']}: no summary was written. "
                       f"Details: {root / 'eval-status.json'}")
    else:
        success_text = (f"{summary['successes']} task successes, {summary['normal_task_failures']} task failures"
                        if summary["task_success_evaluated"] else "preview only; no task success scoring")
        progress.event({"id": "evaluation"},
                       f"{status['status']}: {summary['completed_trials']}/{len(jobs)} valid trials; "
                       f"{summary['discarded_trials']} discarded (max_attempts); "
                       f"{success_text}. Summary: {root / 'evaluation' / 'summary.json'}")
    if errors:
        progress.event({"id": "evaluation"}, f"{len(errors)} cleanup step(s) failed. "
                       f"Details: {root / 'eval-status.json'} and stderr")
    if status_error is not None:
        raise status_error
    return code
