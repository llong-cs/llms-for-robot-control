"""Finalize the rendered-video part of a managed trajectory recording."""

from __future__ import annotations

import os
import time
from pathlib import Path

from agentic_framework.observability.videos import export_trial_videos


def finalize_video(recorder, output: Path, *, preview: bool = False) -> dict:
    """Attach a portable receipt without hiding the original rollout failure.

    Encoding failures belong to the recording receipt, not the robot's task
    outcome. The runner marks the attempt incomplete while retaining both.
    """
    started = time.monotonic()
    if preview:
        receipt = {"status": "not_applicable", "reason": "preview"}
    elif recorder.document["step_count"] == 0:
        receipt = {"status": "unavailable", "reason": "no_reset_frame"}
    else:
        try:
            records = export_trial_videos(recorder.directory, output / "videos")
            if len(records) != 1:
                raise ValueError("one trajectory must have exactly one rendered video")
            record = records[0]
            if record["frame_count"] != recorder.document["step_count"]:
                raise ValueError("video frames do not match the dense trajectory samples")
            receipt = {
                "status": "saved",
                "path": Path(os.path.relpath(record["path"], recorder.directory)).as_posix(),
                **{key: record[key] for key in (
                    "fps", "frame_count", "simulator_steps", "includes_reset_frame",
                    "duration_s", "simulation_elapsed_s", "timing", "cameras_left_to_right",
                    "video_sha256",
                )},
            }
        except Exception as exc:
            receipt = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    recorder.document["video"] = receipt
    recorder.video_timing(time.monotonic() - started)
    return receipt
