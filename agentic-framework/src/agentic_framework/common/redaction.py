"""Mask credentials and inline images in records before they are written."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

_SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "secret",
        "credential",
        "credentials",
    }
)
_BEARER = re.compile(r"(?i)Bearer\s+[^\s\"']+")
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def scrub(text: str) -> str:
    """Replace bearer tokens and sk- keys in one string."""
    text = _BEARER.sub("Bearer [redacted]", text)
    return _SK_KEY.sub("[redacted]", text)


def redacted(value: Any) -> Any:
    """Return a copy with secret-named keys masked and every string scrubbed.

    Image data URLs are replaced by their sha256 digest.
    """
    if isinstance(value, Mapping):
        return {
            str(k): "[redacted]" if str(k).lower() in _SECRET_KEYS else redacted(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redacted(v) for v in value]
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return "[image sha256=" + hashlib.sha256(value.encode()).hexdigest() + "]"
        return scrub(value)
    return value
