"""Load measured motion references and derive operational per-step limits.

Schema 5 separates unscaled calibration ceilings in ``validation_limits`` from
agent operational translation/rotation bounds. Calibration ceilings are enforced
as configuration limits, not claimed as globally established physical maxima.
Legacy schemas 3 and 4 are normalized without changing their operational bounds.
Schema 3 derives operational bounds by rounding reduced references downward;
newer versions store the two grouped bounds explicitly. No form guarantees
reachability or replaces native command scaling.
Task families with identical physics share a profile; unsupported combinations
fail explicitly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from decimal import ROUND_DOWN, Decimal, localcontext
from pathlib import Path

from agentic_framework.configuration.profiles import config_directory

MOTION_PROFILES_PATH = config_directory() / "motion-profiles.json"
_MATCH_FIELDS = ("environment", "embodiment", "control_mode", "control_hz", "frame", "rotation")
_POSE_FIELDS = ("dx", "dy", "dz", "rx", "ry", "rz")


def _validation_limits(profile, schema_version):
    """Normalize calibration ceilings without inventing physical maximum evidence."""
    if schema_version < 5:
        if "validation_limits" in profile:
            raise ValueError("Stored validation_limits require schema_version=5")
        limits = {
            "basis": "measured_local_envelope",
            "single_step_pose_limit": copy.deepcopy(profile.get("measured_single_step_pose_limit")),
            "global_physical_maximum_established": False,
        }
    else:
        if "measured_single_step_pose_limit" in profile:
            raise ValueError("measured_single_step_pose_limit is derived; store only validation_limits in schema 5")
        limits = profile.get("validation_limits")
        if not isinstance(limits, dict) or set(limits) != {
            "basis", "single_step_pose_limit", "global_physical_maximum_established",
        }:
            raise ValueError("validation_limits requires basis, single_step_pose_limit and global_physical_maximum_established")
        if limits["basis"] != "measured_local_envelope":
            raise ValueError("validation_limits.basis must be measured_local_envelope")
        if limits["global_physical_maximum_established"] is not False:
            raise ValueError("validation_limits cannot claim an established global physical maximum from local calibration")
    profile["validation_limits"] = limits
    # Keep old recorded metadata consumers compatible without a second stored source.
    profile["measured_single_step_pose_limit"] = copy.deepcopy(limits["single_step_pose_limit"])
    return limits["single_step_pose_limit"]


def load_motion_profile(*, environment, embodiment, control_mode, control_hz, frame, rotation,
                        path=None):
    """Select an exact embodiment/controller/frequency profile with source provenance.

    There is no generic fallback or implicit frequency rescaling. All tasks use
    their embodiment's profile unless a distinct measured controller profile is
    registered; task names do not invent different robot dynamics.
    """
    source = Path(path) if path is not None else MOTION_PROFILES_PATH
    raw = source.read_bytes()
    document = json.loads(raw)
    schema_version = document.get("schema_version")
    if schema_version not in (3, 4, 5) or not isinstance(document.get("profiles"), list):
        raise ValueError("Unsupported motion profile configuration; expected schema_version=3, 4 or 5")
    if (isinstance(control_hz, bool) or not isinstance(control_hz, (int, float))
            or not math.isfinite(control_hz) or control_hz <= 0):
        raise ValueError("Motion profile control_hz must be positive and finite")
    query = dict(environment=environment, embodiment=embodiment, control_mode=control_mode,
                 control_hz=control_hz, frame=frame, rotation=rotation)
    matches = []
    for profile in document["profiles"]:
        match = profile.get("match", {})
        if set(match) != set(_MATCH_FIELDS):
            raise ValueError("Motion profiles must match environment, embodiment, mode, Hz, frame and rotation")
        if all(query[key] in value if isinstance(value, list) else query[key] == value
               for key, value in match.items()):
            matches.append(profile)
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in query.items())
        raise ValueError(f"Expected exactly one measured motion profile for {detail}; found {len(matches)}")
    result = copy.deepcopy(matches[0])
    availability = result.get("availability", "available")
    if availability not in ("available", "unavailable"):
        raise ValueError("Motion profile availability must be available or unavailable")
    grouped_fields = ("single_step_translation_limit", "single_step_rotation_limit")
    derived_fields = ("single_step_pose_limit", *grouped_fields)
    if schema_version == 3 and any(field in result for field in derived_fields):
        raise ValueError("Operational motion limits are derived; store only the measured reference and fraction")
    if schema_version >= 4 and "single_step_pose_limit" in result:
        raise ValueError("single_step_pose_limit is derived; store only the two grouped operational limits")
    values = _validation_limits(result, schema_version)
    fraction = result.get("motion_limit_fraction")
    if availability == "unavailable":
        if (values is not None or fraction is not None
                or not isinstance(result.get("reason"), str) or not result["reason"].strip()):
            raise ValueError("Unavailable motion profiles require null reference/fraction and an explicit reason")
        if schema_version >= 4 and any(field not in result or result[field] is not None
                                       for field in grouped_fields):
            raise ValueError("Unavailable motion profiles require null grouped operational limits")
        result.update({field: None for field in derived_fields})
    else:
        if (not isinstance(values, list) or len(values) != 6
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in values)):
            raise ValueError("Motion profile validation_limits.single_step_pose_limit requires six positive finite values")
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("Motion profile motion_limit_fraction must be finite and in (0, 1]")
        scale = Decimal(str(fraction))
        scaled_references = []
        for value in values:
            reference = Decimal(str(value))
            with localcontext() as context:
                context.prec = max(28, len(reference.as_tuple().digits) + len(scale.as_tuple().digits))
                scaled_references.append(reference * scale)
        shared_limits = []
        for start, field in zip((0, 3), grouped_fields, strict=True):
            minimum = min(scaled_references[start:start + 3])
            if schema_version >= 4:
                limit = result.get(field)
                if (isinstance(limit, bool) or not isinstance(limit, (int, float))
                        or not math.isfinite(limit) or limit <= 0):
                    raise ValueError(f"Motion profile {field} must be positive finite numeric data")
                derived = float(limit)
            else:
                with localcontext() as context:
                    context.prec = max(28, len(minimum.as_tuple().digits))
                    quantum = Decimal(1).scaleb(minimum.adjusted())
                    derived = float(minimum.quantize(quantum, rounding=ROUND_DOWN))
            if not math.isfinite(derived) or derived <= 0:
                raise ValueError("Derived motion limits must be positive finite representable floats")
            # Check the unscaled calibration ceiling independently of the margin.
            # H-scaled operational targets consequently stay within H times these ceilings.
            for axis in range(start, start + 3):
                ceiling = values[axis]
                if Decimal(str(derived)) > Decimal(str(ceiling)):
                    raise ValueError(
                        f"Motion profile {result['id']} {field}={derived:g} exceeds "
                        f"the calibrated single-step ceiling for {_POSE_FIELDS[axis]}={ceiling:g}"
                    )
            if Decimal(str(derived)) > minimum:
                raise ValueError(f"Motion profile {field} exceeds measured reference ceiling")
            shared_limits.append(derived)
        translation, rotation_limit = shared_limits
        result["single_step_translation_limit"] = translation
        result["single_step_rotation_limit"] = rotation_limit
        result["single_step_pose_limit"] = [translation] * 3 + [rotation_limit] * 3
    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("method"):
        raise ValueError("Measured motion profiles require their measurement provenance")
    result["configuration"] = {
        "path": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "selected_for": query,
    }
    return result


def motion_pose_limits(profile):
    """Return derived operational limits, refusing to substitute native target scales."""
    if profile.get("availability", "available") != "available":
        raise ValueError(f"Motion profile {profile['id']} is unavailable: {profile['reason']}")
    return tuple(profile["single_step_pose_limit"])
