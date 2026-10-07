"""Standing objects, tools and drawer handles."""
from __future__ import annotations

import numpy as np
from transforms3d.quaternions import quat2mat

from ..assets import box, capsule_between, cylinder, part_mesh, ring_parts

ORANGE = (0.88, 0.31, 0.10, 1.0)
BLUE = (0.04, 0.28, 0.90, 1.0)
GRAY = (0.48, 0.50, 0.54, 1.0)
GOLD = (0.80, 0.60, 0.22, 1.0)


def surface_vertices(parts):
    return np.concatenate([np.asarray(part_mesh(p).vertices, dtype=np.float32)
                           for p in parts if p.collision])


def resting_origin(parts, quaternion, xy, clearance=0.002):
    points = surface_vertices(parts) @ quat2mat(quaternion).T
    return (*xy, float(clearance - points[:, 2].min()))


# Local base-to-top axis of every stand_parts object: the gray base is a local-z
# cylinder whose bottom face is the local z = 0 plane, and the stem and upper
# body lie above it along +z (local z 0.011 to 0.151 m for all difficulties).
STAND_BASE_TO_TOP_AXIS = (0.0, 0.0, 1.0)


def stand_parts(difficulty):
    """Vary base radius, upper-body shape, then visible eccentricity."""
    radius = 0.032 if difficulty == "easy" else 0.016
    shift = -0.008 if difficulty == "xhard" else 0.0
    # The same centered grasping stem joins both upper body designs.
    parts = [cylinder(0.012, 0.0365, center=(0, 0, 0.0475), color=ORANGE)]
    if difficulty in ("easy", "medium"):
        parts.append(box((0.030, 0.030, 0.033), center=(0, 0, 0.115), color=ORANGE))
        parts.append(box((0.023, 0.023, 0.0015), center=(0, 0, 0.1495), color=GOLD))
        family = "broad_disk_square_prism" if difficulty == "easy" else "narrow_disk_square_prism"
    else:
        parts.append(box((0.050, 0.024, 0.033), center=(0, shift, 0.115), color=ORANGE))
        parts.append(box((0.043, 0.017, 0.0015), center=(0, shift, 0.1495), color=GOLD))
        family = "narrow_disk_offset_box" if shift else "narrow_disk_centered_box"
    # The gray base is always the last part; the task treats all others as the upper part.
    parts.append(cylinder(radius, 0.006, center=(0, 0, 0.006), color=GRAY))
    return parts, family


def tool_parts(difficulty):
    """Paired tools have identical solids: no added hook, slot or visual cue."""
    if difficulty in ("easy", "medium", "hard"):
        return [capsule_between((-0.089, 0, 0), (0.089, 0, 0), 0.006,
                                color=BLUE)], "thin_straight_rod"
    return [box((0.095, 0.022, 0.003), color=BLUE)], "wide_thin_board"


def tool_holder_parts(difficulty):
    """Tall passive cup exposes the upper grip while retaining a real open bore."""
    inner_radius = 0.008 if difficulty in ("easy", "medium", "hard") else 0.024
    height, floor, wall = 0.110, 0.002, 0.0025
    color = (0.52, 0.46, 0.37, 1.0)
    outer_radius = inner_radius + wall
    parts = [cylinder(outer_radius, floor / 2, center=(0, 0, floor / 2), color=color)]
    parts.extend(ring_parts(inner_radius, outer_radius, (height - floor) / 2,
                            center=(0, 0, (height + floor) / 2), color=color))
    return parts, {
        "shape": "open_cylindrical_cup", "height_m": height,
        "inner_radius_m": inner_radius, "outer_radius_m": outer_radius,
        "floor_thickness_m": floor, "wall_thickness_m": wall,
        "friction": 0.3, "color_rgba": list(color),
        "initial_lean_deg": 0.0,
        "support": "passive contact only; no attachment to tool",
        "fixture_motion": "kinematic; pose changes only at reset or snapshot restore",
    }


def drawer_handle_parts(difficulty):
    """All levels share the same shallow, slender bridge handle."""
    gap, span, radius, z = 0.033, 0.080, 0.004, 0.090
    side_radius = 0.004
    y = span / 2 + side_radius
    x = -(0.150 + gap + radius)
    parts = [capsule_between((x, -y, z), (x, y, z), radius, color=GOLD)]
    parts.extend(capsule_between((-0.147, sy, z), (x, sy, z), side_radius,
                                color=GOLD) for sy in (-y, y))
    return parts, {"handle_family": "bridge", "handle_clear_gap_m": gap,
                   "handle_clear_width_m": span, "handle_height_m": z,
                   "handle_bar_radius_m": radius, "handle_support_radius_m": side_radius}
