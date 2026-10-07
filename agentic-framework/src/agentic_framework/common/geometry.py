"""WORLD-frame axis-angle and measured end-effector pose math."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from inspect_robots.types import Observation


def vector(value: Any, size: int, label: str = "vector") -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return result


def unit_quaternion(value: Any) -> np.ndarray:
    """Validate and normalize an xyzw quaternion."""
    q = vector(value, 4, "quaternion")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be finite and nonzero")
    return q / norm


def quaternion_multiply(left: Any, right: Any) -> np.ndarray:
    """Hamilton product, xyzw order; the left rotation is applied last."""
    a, b = unit_quaternion(left), unit_quaternion(right)
    return unit_quaternion(
        np.r_[
            a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
            a[3] * b[3] - np.dot(a[:3], b[:3]),
        ]
    )


def axis_angle_quaternion(value: Any) -> np.ndarray:
    rotvec = vector(value, 3, "axis-angle")
    angle = float(np.linalg.norm(rotvec))
    if not np.isfinite(angle):
        raise ValueError("axis-angle magnitude must be finite")
    if angle < 1e-12:
        return unit_quaternion(np.r_[0.5 * rotvec, 1.0])
    return unit_quaternion(np.r_[rotvec * (math.sin(angle / 2) / angle), math.cos(angle / 2)])


def rotation_error(target: Any, current: Any) -> np.ndarray:
    """Shortest WORLD axis-angle taking the current orientation to the target."""
    q = unit_quaternion(current)
    error = quaternion_multiply(target, np.r_[-q[:3], q[3]])
    if error[3] < 0:
        error = -error
    sine = float(np.linalg.norm(error[:3]))
    if sine < 1e-12:
        return 2.0 * error[:3]
    return error[:3] * (2.0 * np.arctan2(sine, error[3]) / sine)


def pose(observation: Observation) -> tuple[np.ndarray, np.ndarray]:
    return vector(observation.state["eef_pos"], 3, "eef_pos"), unit_quaternion(
        observation.state["eef_quat"]
    )


def euler_xyz_quaternion(value: Any) -> np.ndarray:
    """Match native euler_angles_to_matrix(angles, 'XYZ'): Rx @ Ry @ Rz."""
    angles = vector(value, 3, "Euler XYZ")
    rotations = [axis_angle_quaternion(np.eye(3)[i] * angle) for i, angle in enumerate(angles)]
    return quaternion_multiply(quaternion_multiply(rotations[0], rotations[1]), rotations[2])


def quaternion_rotate(quaternion: Any, value: Any) -> np.ndarray:
    q, point = unit_quaternion(quaternion), vector(value, 3, "rotated vector")
    return point + 2 * np.cross(q[:3], np.cross(q[:3], point) + q[3] * point)


def quaternion_to_euler_xyz(value: Any) -> np.ndarray:
    """Inverse of Rx @ Ry @ Rz, choosing z=0 at either gimbal singularity."""
    q = unit_quaternion(value)
    matrix = np.column_stack([quaternion_rotate(q, axis) for axis in np.eye(3)])
    sine = float(np.clip(matrix[0, 2], -1.0, 1.0))
    middle = math.asin(sine)
    if abs(math.cos(middle)) > 1e-7:
        first = math.atan2(-matrix[1, 2], matrix[2, 2])
        last = math.atan2(-matrix[0, 1], matrix[0, 0])
    else:
        first = math.atan2(matrix[2, 1], matrix[1, 1])
        last = 0.0
    return np.array([first, middle, last])
