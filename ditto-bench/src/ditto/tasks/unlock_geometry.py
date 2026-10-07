"""Real open rings and a closed, fixed bridge for the disengagement task.

Progress measures the three-dimensional distance from the mouth center to the
nearest transverse exit of the beam. Aperture fit remains a separate diagnostic.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..assets import box, part_mesh, prism, quat_euler

BLUE = (0.04, 0.28, 0.90, 1.0)
GRAY = (0.47, 0.49, 0.51, 1.0)
OUTER_RADIUS = .084
INNER_RADIUS = .070
HALF_THICKNESS = .014
GAP_HALF = .016
# Initial pitch rotates the local Z major axis into the tabletop plane. This
# gives the larger ellipse a stable resting orientation with its mouth upward.
ELLIPSE_OUTER_RADII = (.100, .120)
ELLIPSE_INNER_RADII = (.086, .106)
BEAM_HALF = np.asarray((.040, .160, .008), dtype=np.float32)
BRIDGE_ORIGIN = np.asarray((-.40, .08, .124), dtype=np.float32)


def box_vertices(half, center=(0, 0, 0)):
    import itertools
    return np.asarray([[center[i]+s[i]*half[i] for i in range(3)]
                       for s in itertools.product((-1, 1), repeat=3)], dtype=np.float32)


def ring_parts_for_level(level):
    """Square rings, or a larger elliptical ring with a real open mouth."""
    offset = .040 if level == "easy" else 0.
    if level == "xhard":
        outer_x, outer_z = ELLIPSE_OUTER_RADII
        inner_x, inner_z = ELLIPSE_INNER_RADII
        # Separate inner/outer cut angles put both ends of each cut face at
        # exactly z=+/-16 mm. The real 32 mm aperture remains open in collision.
        outer_cut = math.asin(GAP_HALF / outer_z)
        inner_cut = math.asin(GAP_HALF / inner_z)
        outer_angles = np.linspace(math.pi+outer_cut, 3*math.pi-outer_cut, 65)
        inner_angles = np.linspace(math.pi+inner_cut, 3*math.pi-inner_cut, 65)
        outer = np.column_stack((outer_x*np.cos(outer_angles), outer_z*np.sin(outer_angles)))
        inner = np.column_stack((inner_x*np.cos(inner_angles), inner_z*np.sin(inner_angles)))
        parts = [prism([inner[i], outer[i], outer[i+1], inner[i+1]], HALF_THICKNESS,
                       quat=quat_euler(math.pi/2, 0, 0), color=BLUE)
                 for i in range(len(outer)-1)]
        family = "open_elliptical_ring"
        pitch = math.pi/2
        # Include every actual outer vertex in the virtual fill so its convex
        # outline contains the entire physical ring, as well as the open gap.
        angles = np.unique(np.r_[np.linspace(0, 2*math.pi, 96, endpoint=False),
                                 np.mod(outer_angles, 2*math.pi)])
        polygon = np.column_stack((outer_x*np.cos(angles), outer_z*np.sin(angles)))
        # Midpoint between the two physical cut-face centroids.
        mouth_center = [float((outer[0, 0]+inner[0, 0])/2), 0, 0]
    else:
        outer_x = outer_z = OUTER_RADIUS
        inner_x = inner_z = INNER_RADIUS
        wall = OUTER_RADIUS-INNER_RADIUS
        parts = [box((OUTER_RADIUS, HALF_THICKNESS, wall/2), center=(0, 0, z), color=BLUE)
                 for z in (-OUTER_RADIUS+wall/2, OUTER_RADIUS-wall/2)]
        parts.append(box((wall/2, HALF_THICKNESS, INNER_RADIUS),
                         center=(OUTER_RADIUS-wall/2, 0, 0), color=BLUE))
        for lo, hi in ((-INNER_RADIUS, offset-GAP_HALF), (offset+GAP_HALF, INNER_RADIUS)):
            parts.append(box((wall/2, HALF_THICKNESS, (hi-lo)/2),
                             center=(-OUTER_RADIUS+wall/2, 0, (lo+hi)/2), color=BLUE))
        family = "open_square_ring"
        # Put hard's mouth above the beam, visible before the required rotation.
        pitch = math.pi/2 if level == "hard" else 0.
        polygon = np.asarray([[-OUTER_RADIUS,-OUTER_RADIUS], [OUTER_RADIUS,-OUTER_RADIUS],
                              [OUTER_RADIUS,OUTER_RADIUS], [-OUTER_RADIUS,OUTER_RADIUS]])
        mouth_center = [-.077, 0, offset]
    return parts, {"family": family, "initial_pitch_rad": pitch,
                   "mouth_center_local_m": mouth_center,
                   "outer_envelope_m": [2*outer_x, 2*HALF_THICKNESS, 2*outer_z],
                   "inner_envelope_m": [2*inner_x, 2*HALF_THICKNESS, 2*inner_z],
                   "mouth_clear_width_m": 2*GAP_HALF, "filled_outline_xz": polygon}


def bridge_parts():
    """Top and two full supports close the topology against the table."""
    # World origin is at the beam centre. Legs run to the table, not floating
    # pegs from whose ends a ring could simply slide without mouth alignment.
    top = box(tuple(float(v) for v in BEAM_HALF), color=GRAY)
    leg_half = (.040, .010, (.124-.008)/2)
    parts = [top]
    for y in (-.150, .150):
        parts.append(box(leg_half, center=(0, y, -.008-leg_half[2]), color=GRAY))
    return parts


def vertices(parts):
    return np.concatenate([np.asarray(part_mesh(p).vertices) for p in parts if p.collision]).astype(np.float32)


def filled_prism(polygon):
    """Convex virtual fill used only to ask if *any* fixture still threads it."""
    p = np.asarray(polygon)
    points = np.concatenate([np.column_stack((p[:,0], np.full(len(p), y), p[:,1]))
                             for y in (-HALF_THICKNESS, HALF_THICKNESS)])
    edge2 = np.roll(p, -1, axis=0)-p
    edges = np.concatenate([np.column_stack((edge2[:,0], np.zeros(len(p)), edge2[:,1])), [[0,1,0]]])
    normals = np.concatenate([np.column_stack((-edge2[:,1], np.zeros(len(p)), edge2[:,0])), [[0,1,0]]])
    return points.astype(np.float32), edges.astype(np.float32), normals.astype(np.float32)


def convex_prism_box_separation(points, edges, normals, world_rotation, world_position,
                                 box_centers, box_halves):
    """Exact separating-axis test, including edge cross-products, in ring frame.

    Returns each fixture box's largest interval separation. Positive means a
    separating plane exists; no contact/hole identity is assumed.
    """
    dtype, device = world_rotation.dtype, world_rotation.device
    def t(x): return torch.as_tensor(x, dtype=dtype, device=device)
    p = t(points)
    e = t(edges)
    n = t(normals)
    # Rows below are world box axes represented in ring coordinates.
    box_axes = world_rotation
    cross = torch.linalg.cross(e[None,:,None,:], box_axes[:,None,:,:], dim=-1).flatten(1,2)
    axes = torch.cat([n[None].expand(len(world_rotation),-1,-1), box_axes, cross], dim=1)
    lengths = torch.linalg.vector_norm(axes, dim=-1)
    valid = lengths > 1e-7
    axes = axes / lengths.clamp(min=1e-7)[...,None]
    projected = torch.einsum('vc,bac->bav', p, axes)
    lo, hi = projected.amin(-1), projected.amax(-1)
    centers = (t(box_centers)[None]-world_position[:,None]) @ world_rotation
    center_projections = torch.einsum('bjc,bac->bja', centers, axes)
    # dot local axis against each fixed world axis gives projection radius.
    axis_world = axes @ world_rotation.transpose(1,2)
    radius = torch.einsum('bac,jc->bja', axis_world.abs(), t(box_halves))
    separation = torch.maximum(center_projections-radius-hi[:,None], lo[:,None]-center_projections-radius)
    return torch.where(valid[:,None], separation, -torch.inf).amax(-1)


def mouth_exit_distance(ring_rotation, ring_position, mouth_z=0., *, mouth_center_local_m=None):
    """Euclidean mouth-center distance to either finite beam exit centerline.

    Inputs use the canonical bridge frame. The two exits are the beam's local
    X faces, not its Y ends, which are closed by the legs. Their centerlines run
    along Y at beam-center height. Ring half-depth is reserved beside each leg,
    so an exit target never places part of the mouth inside a support.
    """
    dtype, device = ring_position.dtype, ring_position.device
    origin = torch.as_tensor(BRIDGE_ORIGIN, dtype=dtype, device=device)
    mouth_local = torch.as_tensor((-.077, 0, mouth_z) if mouth_center_local_m is None
                                 else mouth_center_local_m, dtype=dtype, device=device)
    mouth = ring_position + torch.einsum("bij,j->bi", ring_rotation, mouth_local)
    # The supports occupy the final 20 mm at either longitudinal beam end.
    free_half_length = float(BEAM_HALF[1]) - .020 - HALF_THICKNESS
    exit_y = mouth[:, 1].clamp(origin[1]-free_half_length, origin[1]+free_half_length)
    candidates = torch.stack([
        torch.stack((torch.full_like(exit_y, origin[0]+side*float(BEAM_HALF[0])),
                     exit_y, torch.full_like(exit_y, origin[2])), dim=-1)
        for side in (-1, 1)
    ], dim=1)
    distances = torch.linalg.vector_norm(mouth[:, None]-candidates, dim=-1)
    nearest_distance, nearest_side = distances.min(-1)
    return nearest_distance, distances, nearest_side, mouth


def mouth_alignment_error(beam_world_vertices, ring_rotation, ring_position, mouth_z,
                          *, outer_half_width_m=OUTER_RADIUS, inner_half_height_m=INNER_RADIUS):
    """Maximum aperture violation of the true beam/central-ring-plane section.

    Mouth points toward local -X. Its tangential opening is 32 mm. The inward
    alignment corridor is bounded by the opposite outer edge; moving the ring
    arbitrarily far away, or tilting its plane off the beam, cannot cancel the
    error. Longitudinal motion through a correctly aligned mouth has no score
    bonus. This is a fit diagnostic only, not the progress score.
    """
    local = (beam_world_vertices[None]-ring_position[:,None]) @ ring_rotation
    # Cube vertex numbering is itertools.product((-1,1), repeat=3).
    indices = [(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
    a = local[:,[i for i,j in indices]]
    b = local[:,[j for i,j in indices]]
    delta = b-a
    crossing = (a[...,1]*b[...,1] <= 0) & (delta[...,1].abs() > 1e-8)
    fraction = -a[...,1]/torch.where(delta[...,1].abs()>1e-8,delta[...,1],torch.ones_like(delta[...,1]))
    section = a+fraction[...,None]*delta
    # Infinite beam projection would allow an unrelated distant ring to score;
    # a real finite cross-section must intersect the ring's central plane.
    any_section = crossing.any(-1)
    tangential = (section[...,2]-mouth_z).abs()-GAP_HALF
    # Wide longitudinal interval covers passage through the mouth, but not a
    # remote noninteracting pose. This remains a single distance to a corridor.
    longitudinal = torch.maximum(section[...,0]-outer_half_width_m,
                                 -outer_half_width_m-.016-section[...,0])
    violation = torch.maximum(tangential, longitudinal).clamp(min=0)
    error = torch.where(crossing, violation, -torch.inf).amax(-1)
    plane_distance = torch.where(local[...,1].amin(-1)>0, local[...,1].amin(-1), -local[...,1].amax(-1)).clamp(min=0)
    return torch.where(any_section, error, inner_half_height_m+plane_distance), any_section
