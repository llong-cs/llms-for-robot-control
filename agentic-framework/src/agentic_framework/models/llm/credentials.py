"""Read an explicitly selected private dotenv file without changing the environment."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def read_credentials_file(path: str | Path) -> dict[str, str | None]:
    """Require a user-owned mode-600 dotenv file; absent files allow environment auth.

    Call only when sending a request, never while constructing or previewing one.
    Interpolation is disabled so credential values cannot expand other variables.
    """
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Credential source must not be a symlink")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("Credential source must be a regular file")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError("Credential file must be owned by this user and mode 600")

    from dotenv import dotenv_values

    return dict(dotenv_values(path, interpolate=False))
