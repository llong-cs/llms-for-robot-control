"""Stateless sampling protocol shared by clients and isolated model services.

Version 1 hashes the UTF-8 string ``seed_protocol_v1:<episode>:<request>``;
the first four SHA-256 bytes interpreted big-endian form the sampling seed.
Request indices begin at zero after every episode reset.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

SEED_PROTOCOL = "seed_protocol_v1"
SEED_KEY = "sampling"


def sampling_request(policy_seed: int, request_index: int) -> dict:
    if type(policy_seed) is not int or policy_seed < 0:
        raise ValueError("policy_seed must be a nonnegative integer")
    if type(request_index) is not int or request_index < 0:
        raise ValueError("request_index must be a nonnegative integer")
    digest = hashlib.sha256(f"{SEED_PROTOCOL}:{policy_seed}:{request_index}".encode()).digest()
    return {
        "protocol": SEED_PROTOCOL,
        "policy_seed": policy_seed,
        "request_index": request_index,
        "seed": int.from_bytes(digest[:4], "big"),
    }


def validate_sampling(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("sampling must be an object")
    expected = sampling_request(value.get("policy_seed"), value.get("request_index"))
    if dict(value) != expected:
        raise ValueError("sampling seed protocol, derived seed, or fields mismatch")
    return expected
