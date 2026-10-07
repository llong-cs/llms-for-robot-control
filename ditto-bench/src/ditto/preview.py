"""Render real simulator initial states; no model calls or generated concept art."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .catalog import TASKS
from .difficulties import expand_difficulties
from .paths import preview_root

TITLES = ["Extraction", "Odd-object grasping", "Insertion", "Standing stability", "Tool-assisted drawer", "Ring release"]
SUBTITLES = ["Pull the board completely out", "Put the object in the basket", "Match, insert and release", "Stand on the gray base and release", "Use the tool to open the drawer", "Remove the open ring from the bridge"]


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[0]
    return value.astype(np.uint8)


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if hasattr(value, "detach"):
        return _json(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def contact_sheet(output):
    width, height, gutter = 768, 620, 24
    sheet = Image.new("RGB", (width * 3 + gutter * 4, height * 2 + 150), "#edf0ed")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 18), "Ditto Bench / Six physical-interaction tasks", font=_font(34), fill="#173d40")
    draw.text((gutter, 69), "ManiSkill · Franka FR3 + Robotiq · Actual simulator reset states", font=_font(21), fill="#526b6b")
    for task in TASKS:
        path = output / f"{task.task_id + 1:02d}_{task.slug}.png"
        if not path.exists():
            continue
        col, row = task.task_id % 3, task.task_id // 3
        x, y = gutter + col * (width + gutter), 125 + row * height
        draw.rounded_rectangle((x, y, x + width, y + height - 18), radius=14, fill="white")
        frame = Image.open(path).convert("RGB")
        frame.thumbnail((width, 484))
        sheet.paste(frame, (x + (width - frame.width) // 2, y))
        draw.text((x + 20, y + 491), f"Task {task.task_id}  {TITLES[task.task_id]}",
                  font=_font(29), fill="#173d40")
        draw.text((x + 20, y + 542), SUBTITLES[task.task_id], font=_font(23), fill="#526b6b")
    sheet.save(output / "suite_overview.png")
    sheet.save(output / "suite_overview.jpg", quality=88)
    return output / "suite_overview.png"


def comparison_sheet(output, levels, selected):
    """Six task rows and four difficulty columns, all actual simulator renders."""
    width, row_height, label_width, top = 520, 330, 210, 112
    sheet = Image.new("RGB", (label_width + width * len(levels), top + row_height * len(selected)), "#edf0ed")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 15), "Ditto Bench / Difficulty comparison", font=_font(30), fill="#173d40")
    draw.text((20, 55), "Actual simulator reset states · Same paired seed", font=_font(18), fill="#526b6b")
    for column, level in enumerate(levels):
        draw.text((label_width + column * width + 18, 78), level.upper(), font=_font(23), fill="#173d40")
    for row, index in enumerate(selected):
        task = TASKS[index]
        y = top + row * row_height
        draw.text((18, y + 100), f"Task {index}", font=_font(27), fill="#173d40")
        draw.text((18, y + 143), TITLES[index], font=_font(16), fill="#173d40")
        for column, level in enumerate(levels):
            x = label_width + column * width
            path = output / level / f"{index + 1:02d}_{task.slug}.png"
            if not path.is_file():
                continue
            frame = Image.open(path).convert("RGB")
            frame.thumbnail((width - 12, row_height - 25))
            sheet.paste(frame, (x + (width - frame.width) // 2, y + (row_height - frame.height) // 2))
    sheet.save(output / "difficulty_comparison.png")
    sheet.save(output / "difficulty_comparison.jpg", quality=92)
    return output / "difficulty_comparison.png"


def finalize_preview(output):
    levels = tuple(level for level in expand_difficulties("all") if (output / level).is_dir())
    reports = []
    for level in levels:
        for task in TASKS:
            report = output / level / f"{task.task_id + 1:02d}_{task.slug}.json"
            if report.is_file():
                reports.append(json.loads(report.read_text()))
    (output / "parameters_manifest.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    selected = sorted({report["task"]["task_id"] for report in reports})
    return comparison_sheet(output, levels, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", default="all", help="Zero-based task IDs separated by commas, or all")
    parser.add_argument("--task-randomize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--difficulty", default="easy", help="easy/medium/hard/xhard or all")
    parser.add_argument("--gpu", required=True, help="GPU index assigned to the preview renderer")
    parser.add_argument("--output-dir", type=Path, default=preview_root(), help="Directory for rendered images, the contact sheet and render manifest")
    parser.add_argument("--seed", type=int, default=42, help="Reset seed used for each selected task (default: 42)")
    parser.add_argument("--shader", default="rt-fast", help="Shader pack for policy cameras (default: rt-fast)")
    parser.add_argument("--primary-camera", choices=("render_camera", "external_cam", "wrist_cam"),
                        default="external_cam", help="Camera shown in contact sheets; default: actual external policy camera")
    args = parser.parse_args()
    levels = expand_difficulties(args.difficulty)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from . import make_env
    selected = range(6) if args.task_ids == "all" else [int(x) for x in args.task_ids.split(",")]
    for level in levels:
        level_output = args.output_dir / level
        level_output.mkdir(parents=True, exist_ok=True)
        reports = []
        for index in selected:
            task = TASKS[index]
            print(f"Rendering {task.env_id}", flush=True)
            env = make_env(index, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
                           reward_mode="none", sim_backend="cpu", num_envs=1,
                           difficulty=level, randomize=args.task_randomize, sensor_configs={"shader_pack": args.shader})
            try:
                observation, info = env.reset(seed=args.seed)
                prefix = f"{index + 1:02d}_{task.slug}"
                frame = (_array(env.unwrapped.render_rgb_array(camera_name="render_camera"))
                         if args.primary_camera == "render_camera"
                         else _array(observation["sensor_data"][args.primary_camera]["rgb"]))
                Image.fromarray(frame).save(level_output / f"{prefix}.png")
                Image.fromarray(frame).save(level_output / f"{prefix}.jpg", quality=85)
                # This is an actual policy view; human rendering uses the official camera.
                overview = _array(observation["sensor_data"]["external_cam"]["rgb"])
                Image.fromarray(overview).save(level_output / f"{prefix}_robot.png")
                for camera in ("external_cam", "wrist_cam"):
                    image = _array(observation["sensor_data"][camera]["rgb"])
                    Image.fromarray(image).save(level_output / f"{prefix}_{camera}.png")
                report = {"task": asdict(task), "difficulty": level, "seed": args.seed,
                          "control_hz": env.unwrapped.control_freq, "sim_hz": env.unwrapped.sim_freq,
                          "initial_metrics": _json(info), "task_spec": _json(env.unwrapped.task_spec),
                          "asset_records": _json(env.unwrapped.asset_records),
                          "policy_called": False, "sim_backend": "cpu",
                          "policy_shader_pack": args.shader,
                          "human_camera": "official render_camera (1280x720, rt)",
                          "robot_view_camera": "external_cam", "display_camera": args.primary_camera,
                          "description": "Actual simulator reset state; no policy rollout." }
                (level_output / f"{prefix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
                reports.append(report)
                print(f"Saved {prefix}; success={_json(info['success'])}", flush=True)
            finally:
                env.close()
        (level_output / "render_manifest.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
        print(contact_sheet(level_output), flush=True)
    print(finalize_preview(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
