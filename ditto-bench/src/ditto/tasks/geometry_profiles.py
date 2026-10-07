"""Small exact polygon operations for shape-matched insertion evaluation.

The same polygon vertices define collision profiles and evaluator boundaries.
No intended yaw is encoded: compatible rotations succeed through actual fit.
"""
from __future__ import annotations

import math

import numpy as np


def convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(points) <= 2:
        return np.asarray(points, dtype=float).reshape(-1, 2)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def clip_polygon(polygon, normal, limit):
    """Clip a convex polygon to dot(point, normal) <= limit."""
    polygon = np.asarray(polygon, dtype=float)
    if not len(polygon):
        return polygon.reshape(-1, 2)
    normal = np.asarray(normal, dtype=float)
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = np.dot(a, normal) - limit, np.dot(b, normal) - limit
        if da <= 1e-12:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            out.append(a + (b - a) * da / (da - db))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def offset_convex(polygon, distance):
    polygon = np.asarray(polygon, dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack((edges[:, 1], -edges[:, 0]), axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    limits = np.sum(normals * polygon, axis=1) + distance
    return np.asarray([np.linalg.solve(np.stack((normals[i-1], normals[i])),
                                      (limits[i-1], limits[i])) for i in range(len(polygon))])


def insertion_profile(difficulty):
    if difficulty in ("easy", "medium"):
        angles = np.linspace(0, 2*math.pi, 48, endpoint=False)
        profile = .024 * np.stack((np.cos(angles), np.sin(angles)), axis=1)
        clearance = .005 if difficulty == "easy" else .002
        kind = "circle"
    elif difficulty == "hard":
        profile = np.asarray([[-.028, -.018], [.028, -.018], [.028, .018], [-.028, .018]])
        clearance, kind = .002, "rectangle"
    else:
        profile = np.asarray([[-.024, -.024], [.024, -.024], [.024, -.006],
                              [0, -.006], [0, .024], [-.024, .024]])
        clearance, kind = .002, "l"
    if kind == "l":
        c = clearance
        socket = np.asarray([[-.024-c, -.024-c], [.024+c, -.024-c], [.024+c, -.006+c],
                             [c, -.006+c], [c, .024+c], [-.024-c, .024+c]])
    else:
        socket = offset_convex(profile, clearance)
    return kind, profile.astype(np.float32), socket.astype(np.float32), clearance


def profile_contains_polygon(points, socket, kind, tolerance=0.0005):
    """A convex projected body part must fit the full aperture, including L notch."""
    points = np.asarray(points, dtype=float)
    if not len(points):
        return True
    if kind != "l":
        edges = np.roll(socket, -1, axis=0) - socket
        normals = np.stack((edges[:, 1], -edges[:, 0]), axis=1)
        norms = np.linalg.norm(normals, axis=1)
        return bool(np.all(points @ normals.T <= np.sum(socket * normals, axis=1) + tolerance * norms + 1e-9))
    low, high = socket.min(axis=0), socket.max(axis=0)
    if not np.all((points >= low-tolerance-1e-9) & (points <= high+tolerance+1e-9)):
        return False
    # Checking vertices alone misses an edge crossing the missing quadrant.
    notch_x, notch_y = socket[3]
    hull = convex_hull(points)
    forbidden = clip_polygon(hull, (-1, 0), -(notch_x+tolerance))
    forbidden = clip_polygon(forbidden, (0, -1), -(notch_y+tolerance))
    if not len(forbidden):
        return True
    # Also reject a degenerate section/edge crossing the notch interior.
    interior = forbidden.mean(axis=0)
    return not (interior[0] > notch_x+tolerance+1e-9 and interior[1] > notch_y+tolerance+1e-9)


def submerged_vertices(vertices, edges, top):
    """Exact vertices of a convex body clipped by z <= top."""
    vertices = np.asarray(vertices, dtype=float)
    lower = vertices[vertices[:, 2] <= top + 1e-9]
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    crossings = (a[:, 2] < top) & (b[:, 2] > top) | (b[:, 2] < top) & (a[:, 2] > top)
    a, b = a[crossings], b[crossings]
    if len(a):
        hit = a + (b-a) * ((top-a[:, 2])/(b[:, 2]-a[:, 2]))[:, None]
        lower = np.concatenate((lower, hit))
    return lower
