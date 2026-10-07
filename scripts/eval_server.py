"""Bounded ownership of a model server started for one evaluation run.

No inference request is sent here. Readiness requires both the server's fresh
receipt and its actual HTTP/WebSocket handshake to identify the selected model.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _check_port(host, port):
    """Fail without touching the owner of an already bound local endpoint."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Managed servers require a loopback host")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Model server port is unavailable: {host}:{port}") from exc


def _validate_metadata(metadata, profile, source):
    if not isinstance(metadata, dict) or metadata.get("status") != "loaded":
        raise ValueError(f"{source} does not report status=loaded")
    for key, expected in profile["model"]["expected_metadata"].items():
        if metadata.get(key) != expected:
            raise ValueError(f"{source} disagrees with model profile field {key}")
    for key, expected in profile["model"].get("optional_metadata", {}).items():
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"{source} disagrees with model profile field {key}")
    return metadata


def _health(profile, host, port, timeout_s):
    if profile["transport"]["kind"] == "http":
        authority = f"[{host}]" if ":" in host else host
        # The server is local: inherited proxy variables must not reroute probes.
        opener = build_opener(ProxyHandler({}))
        request = Request(f"http://{authority}:{port}/healthz", method="GET")
        with opener.open(request, timeout=timeout_s) as response:
            return json.load(response)
    if profile["transport"]["kind"] == "websocket":
        from agentic_framework.models.vla.openpi import OpenPiClient

        client = OpenPiClient(host, port, timeout_s=timeout_s)
        try:
            return client.get_server_metadata()
        finally:
            client.close()
    raise ValueError("Unsupported model server transport")


def ready_metadata(profile, host, port, timeout_s=5):
    """Return live model identity after one bounded, read-only health handshake."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Health timeout must be positive and finite")
    return _validate_metadata(_health(profile, host, port, timeout_s), profile, "Live server metadata")


def _stop(process):
    """Signal only the process group created with start_new_session=True."""
    summary = {"process_group": process.pid, "signals": [], "returncode": None}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
            summary["signals"].append(sig.name)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        else:
            # Also clear surviving distributed workers if their parent exited
            # promptly after TERM. KILL is harmless if its group has disappeared.
            process.wait(timeout=15)
    summary["returncode"] = process.returncode
    return summary


@contextmanager
def managed_server(command, env, *, output_dir, profile, host, port, timeout_s=900, cancel_event=None, on_wait=None):
    """Start an owned native server, yield verified metadata, and always stop it.

    Existing metadata/logs are rejected so an old run cannot satisfy readiness.
    Failures leave ``server.log`` and ``server-lifecycle.json`` for diagnosis.
    The caller owns GPU selection and the matching evaluation configuration.
    An optional on_wait callback receives elapsed loading seconds; reporting
    failures never prevent server readiness checks or owned-process cleanup.
    """
    if not command or not Path(command[0]).expanduser().is_file():
        raise FileNotFoundError("Model server interpreter does not exist")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Model server timeout must be positive and finite")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Model server port must be within 1..65535")
    output_dir = Path(output_dir).expanduser().resolve()
    metadata_path = output_dir / "server_metadata.json"
    lifecycle_path = output_dir / "server-lifecycle.json"
    log_path = output_dir / "server.log"
    if any(path.exists() for path in (
        metadata_path, lifecycle_path, log_path
    )):
        raise FileExistsError(f"Model server output already contains a run: {output_dir}")
    _check_port(host, port)
    output_dir.mkdir(parents=True, exist_ok=True)
    lifecycle = {
        "schema_version": 1,
        "profile": profile["id"],
        "host": host,
        "port": port,
        "started_at": _timestamp(),
        "status": "starting",
        "pid": None,
        "ready_at": None,
        "stopped_at": None,
    }
    _write(lifecycle_path, lifecycle)
    process = None
    failed = False
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            lifecycle["pid"] = process.pid
            _write(lifecycle_path, lifecycle)
            loading_started = time.monotonic()
            deadline = loading_started + timeout_s
            last_probe_error = None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Model server startup cancelled")
                if on_wait is not None:
                    try:
                        on_wait(max(0.0, time.monotonic() - loading_started))
                    except Exception:
                        # Console feedback must never change model execution.
                        pass
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"Model server exited before readiness (code {returncode}); see {log_path}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f"; last probe: {last_probe_error}" if last_probe_error else ""
                    raise TimeoutError(f"Model server readiness timed out; see {log_path}{detail}")
                if metadata_path.is_file():
                    # Some older launchers write their receipt in place. A
                    # partially written JSON must not abort an otherwise healthy
                    # load; identity mismatches are still rejected immediately.
                    try:
                        receipt = json.loads(metadata_path.read_text())
                    except (OSError, json.JSONDecodeError) as exc:
                        receipt = {}
                        last_probe_error = f"{type(exc).__name__}: {exc}"
                    if not isinstance(receipt, dict):
                        raise ValueError("Server receipt must be a JSON object")
                    if receipt.get("status") == "loaded":
                        _validate_metadata(receipt, profile, "Server receipt")
                        if "port" in receipt and receipt["port"] != port:
                            raise ValueError("Server receipt has an unexpected port")
                        try:
                            metadata = ready_metadata(profile, host, port, min(2.0, remaining))
                        except (OSError, URLError, TimeoutError, ConnectionError) as exc:
                            last_probe_error = f"{type(exc).__name__}: {exc}"
                        else:
                            _validate_metadata(metadata, profile, "Live server metadata")
                            for key in (
                                "pid", "port", "checkpoint_tree_sha256",
                                "checkpoint_source_manifest_sha256", "model_code_sha256",
                            ):
                                if key in receipt and metadata.get(key) != receipt[key]:
                                    raise ValueError(f"Live server disagrees with fresh receipt field {key}")
                            if process.poll() is not None:
                                raise RuntimeError("Model server exited during readiness verification")
                            lifecycle.update(status="ready", ready_at=_timestamp())
                            _write(lifecycle_path, lifecycle)
                            yield metadata
                            break
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        failed = True
        lifecycle.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        cleanup_error = None
        if process is not None:
            try:
                lifecycle["cleanup"] = _stop(process)
            except BaseException as exc:
                cleanup_error = exc
                lifecycle["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                lifecycle["status"] = "cleanup_failed"
        lifecycle["stopped_at"] = _timestamp()
        if not failed and cleanup_error is None:
            lifecycle["status"] = "stopped"
        _write(lifecycle_path, lifecycle)
        if cleanup_error is not None and not failed:
            raise cleanup_error
