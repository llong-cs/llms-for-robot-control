"""Export each recorded trial as a native-camera MP4 at simulation speed.

Uses the pinned Inspect Robots native composite encoder. Videos contain the
reset image followed by every recorded post-action image. Camera order follows
the benchmark: external view first, then the wrist camera. Model latency is excluded.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from inspect_robots._video import (
    _encode_composite_mp4,
    default_fps,
    discover_streams,
    resolve_frames_dir,
)
from inspect_robots.frames import _safe

from agentic_framework.common.project import project_root
from agentic_framework.harness.types import MAX_ATTEMPTS_TERMINATION

CAMERAS = ("agentview", "robot0_eye_in_hand")
FINAL_STATUSES = {"success", "error", "cancelled"}


def _find_ffmpeg() -> str:
    """Choose an installed encoder with libx264 from PATH or the imageio runtime."""
    override = os.environ.get("VLM4ROBOTICS_FFMPEG")
    import imageio_ffmpeg

    candidates = (
        [override]
        if override
        else [
            shutil.which("ffmpeg"),
            imageio_ffmpeg.get_ffmpeg_exe(),
        ]
    )
    checked = set()
    for executable in candidates:
        if not executable or executable in checked:
            continue
        checked.add(executable)
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            continue
        try:
            result = subprocess.run(
                [executable, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and any(
            len(fields := line.split()) > 1 and fields[1] == "libx264"
            for line in result.stdout.splitlines()
        ):
            return executable
    raise FileNotFoundError(
        "No installed ffmpeg with libx264 was found; set VLM4ROBOTICS_FFMPEG "
        "to an H.264-capable binary"
    )


def _evaluation_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {path}")
    if (path / "evaluation/evals").is_dir():
        return path / "evaluation"
    if not (path / "evals").is_dir():
        raise ValueError(f"Expected a runner evaluation directory containing evals/: {path}")
    return path


def _final_logs_in(scope: Path) -> list[tuple[Path, dict[str, Any]]]:
    logs = []
    for path in sorted(scope.rglob("*.json")):
        if {"frames", "wire", "actions", "transcripts", "codex"}.intersection(
            path.relative_to(scope).parts
        ):
            continue
        if path.name.endswith(".live.json"):
            continue
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict) or document.get("status") not in FINAL_STATUSES:
            continue
        if not all(key in document for key in ("eval", "stats", "samples")):
            continue
        if not isinstance(document["samples"], list):
            raise ValueError(f"Invalid samples in final log: {path}")
        logs.append((path, document))
    if not logs:
        raise ValueError(f"No final evaluation logs found below {scope}")
    return logs


def _expected_steps(scores: dict, log: dict, sample_count: int, epoch_count: int) -> int | None:
    value = scores.get("episode_length")
    if value is None and sample_count == 1 and epoch_count == 1:
        value = log["stats"].get("total_steps")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise ValueError(f"Invalid recorded episode length: {value!r}")
    return int(value)


def _final_logs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return _final_logs_in(root / "evals")


def _destination(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination == project_root():
        raise ValueError("Select a video subdirectory, for example outputs/videos")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_manifest(destination: Path, records: list[dict]) -> None:
    payload = (json.dumps(records, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(destination / "manifest.json", payload)


def _validate_ownership(logs: list[tuple[Path, dict]], previous: list[dict]) -> None:
    """Reject conflicting identities and sanitized filenames before writing any video."""
    owners: dict[tuple[str, int], Path] = {}
    filenames: dict[str, tuple[str, int]] = {}

    def claim(scene_id: Any, epoch: Any, log_path: Any, *, existing: bool) -> None:
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"Invalid scene ID in {log_path}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"Invalid epoch in video manifest: {epoch!r}")
        if not isinstance(log_path, (str, Path)) or not str(log_path):
            raise ValueError("Missing log path in video manifest")
        identity = (scene_id, epoch)
        owner = Path(log_path).expanduser().resolve()
        if identity in owners and (existing or owners[identity] != owner):
            raise ValueError(f"Multiple final logs for {scene_id} epoch {epoch}")
        filename = _safe(f"{scene_id}-e{epoch}")
        if filename in filenames and filenames[filename] != identity:
            raise ValueError(f"Video filename collision for {scene_id} epoch {epoch}")
        owners[identity] = owner
        filenames[filename] = identity

    for record in previous:
        if not isinstance(record, dict):
            raise ValueError("Invalid record in video manifest")
        claim(record.get("scene_id"), record.get("epoch"), record.get("log_path"), existing=True)
    requested: set[tuple[str, int]] = set()
    for log_path, log in logs:
        for sample in log["samples"]:
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample in {log_path}")
            epochs = sample.get("epochs")
            if not isinstance(epochs, list) or not epochs:
                raise ValueError(f"No recorded trial epochs for {sample.get('scene_id')}")
            for epoch, _scores in enumerate(epochs):
                scene_id = sample.get("scene_id")
                claim(scene_id, epoch, log_path, existing=False)
                identity = (scene_id, epoch)
                if identity in requested:
                    raise ValueError(f"Multiple final logs for {scene_id} epoch {epoch}")
                requested.add(identity)


def _export_logs(logs: list[tuple[Path, dict]], destination: Path) -> list[dict]:
    ffmpeg = _find_ffmpeg()
    records: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for log_path, log in logs:
        stored_dir = log["stats"].get("frames_dir")
        if not isinstance(stored_dir, str) or not stored_dir:
            raise ValueError(f"No recorded frames in {log_path}; run with --record-trajectory")
        frames_dir = resolve_frames_dir(stored_dir, log_path)
        if frames_dir is None:
            raise FileNotFoundError(f"Recorded frames directory is missing for {log_path}")
        streams, strays = discover_streams(frames_dir)
        if strays:
            raise ValueError(f"Unrecognized frame files in {frames_dir}: {strays[0].name}")
        info = log["eval"].get("embodiment_info", {})
        fps, fps_source = default_fps(info)
        if fps_source != "control_hz from log" or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Expected a positive recorded control_hz in {log_path}")
        declared_cameras = info.get("observation_space", {}).get("cameras", [])
        available = [camera["name"] for camera in declared_cameras]
        is_maniskill = (
            log["eval"].get("embodiment", "").startswith("maniskill-")
            or "external_cam" in available
        )
        cameras = ("external_cam", "wrist_cam") if is_maniskill else CAMERAS
        if not is_maniskill and fps != 20.0:
            raise ValueError(f"Expected recorded LIBERO control_hz=20 in {log_path}")
        samples = log["samples"]
        for sample in samples:
            scene_id = sample.get("scene_id")
            if not isinstance(scene_id, str) or not scene_id:
                raise ValueError(f"Invalid scene ID in {log_path}")
            epochs = sample.get("epochs")
            if not isinstance(epochs, list) or not epochs:
                raise ValueError(f"No recorded trial epochs for {scene_id}")
            metadata = sample.get("scene_metadata") or {}
            for epoch, scores in enumerate(epochs):
                identity = (scene_id, epoch)
                if identity in seen:
                    raise ValueError(f"Multiple final logs for {scene_id} epoch {epoch}")
                seen.add(identity)
                prefix = _safe(f"{scene_id}-e{epoch}")
                keys = tuple(f"{prefix}_{_safe(camera)}" for camera in cameras)
                if any(key not in streams for key in keys):
                    raise ValueError(
                        f"All registered policy cameras are required for {scene_id} epoch {epoch}"
                    )
                ordered = [(key, streams[key]) for key in keys]
                steps = tuple(step for step, _path in ordered[0][1])
                if steps != tuple(range(len(steps))):
                    raise ValueError(
                        f"Missing or duplicate frame steps for {scene_id} epoch {epoch}"
                    )
                if any(tuple(step for step, _path in stream) != steps for _key, stream in ordered[1:]):
                    raise ValueError(f"Camera timelines differ for {scene_id} epoch {epoch}")
                if not isinstance(scores, dict):
                    raise ValueError(f"Invalid epoch scores for {scene_id} epoch {epoch}")
                expected = _expected_steps(scores, log, len(samples), len(epochs))
                if expected is not None and len(steps) != expected + 1:
                    raise ValueError(
                        f"Incomplete frame coverage for {scene_id}: "
                        f"{len(steps)} frames, expected reset + {expected} simulator steps"
                    )
                encoded = _encode_composite_mp4(ordered, fps, ffmpeg)
                if encoded is None:
                    raise RuntimeError(f"Native video encoding failed for {scene_id} epoch {epoch}")
                payload, surviving_keys, emitted_steps = encoded
                if surviving_keys != keys or emitted_steps != steps:
                    raise ValueError(f"Native encoder dropped cameras or frames for {scene_id}")
                filename = f"{prefix}.mp4"
                video_path = destination / filename
                _atomic_write(video_path, payload)
                success = scores.get("success_at_end")
                reasons = sample.get("termination_reasons") or []
                reason = reasons[epoch] if epoch < len(reasons) else None
                if reason == MAX_ATTEMPTS_TERMINATION:
                    # A discarded episode is never evaluated.
                    success = None
                policy_config = log["eval"].get("policy_config") or {}
                record = {
                    "path": str(video_path),
                    "scene_id": scene_id,
                    "epoch": epoch,
                    "suite": metadata.get("suite"),
                    "task_id": metadata.get("task_id"),
                    "init_state_index": metadata.get("init_state_index"),
                    "instruction": sample.get("instruction"),
                    "category": metadata.get("category"),
                    "model": policy_config.get("model"),
                    "eval_status": log["status"],
                    "sample_status": sample.get("status"),
                    "robot_success": None if success is None else bool(success == 1),
                    "termination_reason": reason,
                    "cameras_left_to_right": list(cameras),
                    "fps": fps,
                    "frame_count": len(steps),
                    "simulator_steps": len(steps) - 1,
                    "includes_reset_frame": True,
                    "duration_s": len(steps) / fps,
                    "simulation_elapsed_s": (len(steps) - 1) / fps,
                    "timing": "one reset frame then one frame per simulator step; no model latency",
                    "coverage": "verified_episode_length"
                    if expected is not None
                    else "stored_steps",
                    "log_path": str(log_path),
                    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                    "frames_dir": str(frames_dir.resolve()),
                    "video_sha256": hashlib.sha256(payload).hexdigest(),
                    "encoder": "inspect_robots._video._encode_composite_mp4",
                    "ffmpeg": ffmpeg,
                }
                records.append(record)
    return records


def export_videos(evaluation_dir: Path, output_dir: Path | None = None) -> list[dict]:
    """Write one MP4 per final trial and return its reproducibility metadata.

    ``evaluation_dir`` may contain ``evals/`` or ``evaluation/evals/``; nested
    batch evals are discovered recursively. The default destination is
    ``<evaluation-dir>/videos``. Returned paths are absolute and the destination
    ``manifest.json`` stores the same list. Complete registered camera streams are required:
    missing steps, missing cameras, and dropped frames fail explicitly.
    Failed/cancelled trials remain exportable with their original status.
    """
    root = _evaluation_root(Path(evaluation_dir))
    destination = _destination(Path(output_dir) if output_dir is not None else root / "videos")
    logs = _final_logs(root)
    with (destination / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _validate_ownership(logs, [])
        records = _export_logs(logs, destination)
        _write_manifest(destination, records)
    return records


def export_trial_videos(trial_dir: Path, output_dir: Path) -> list[dict]:
    """Export only one finalized trial and merge its records into the manifest.

    ``trial_dir`` contains its final JsonLogSink log and recorded frames (normally
    ``<run>/evals/<scene>``). Success, error, and cancelled logs are accepted,
    including a reset-only trial. Existing records from other trials are retained;
    repeating the same trial replaces its entries. An identity owned by a
    different final log or a sanitized filename collision is rejected before
    encoding. Concurrent finalizations serialize manifest updates and all writes
    use atomic replacement. Errors propagate to the recording lifecycle caller.
    """
    scope = Path(trial_dir).expanduser().resolve()
    if not scope.is_dir():
        raise ValueError(f"Trial directory does not exist: {scope}")
    logs = _final_logs_in(scope)
    destination = _destination(Path(output_dir))
    with (destination / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = destination / "manifest.json"
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        if not isinstance(previous, list):
            raise ValueError(f"Invalid video manifest: {manifest}")
        _validate_ownership(logs, previous)
        records = _export_logs(logs, destination)
        replaced = {(record["scene_id"], record["epoch"]) for record in records}
        combined = [
            record for record in previous if (record["scene_id"], record["epoch"]) not in replaced
        ]
        combined.extend(records)
        combined.sort(key=lambda record: (record["scene_id"], record["epoch"]))
        _write_manifest(destination, combined)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        records = export_videos(args.evaluation_dir, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Video export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(records, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
