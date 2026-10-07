"""Resolve mutable resources inside the project without a server-specific layout."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Use an explicit project root, the source checkout, or the caller's directory."""
    override = os.environ.get("AGENTIC_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().absolute()
    framework = Path(__file__).resolve().parents[3]
    if (framework / "pyproject.toml").is_file() and (framework / "src/agentic_framework").is_dir():
        return framework.parent
    return Path.cwd()


def project_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Anchor relative paths while preserving interpreter and credential symlinks."""
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else (base or project_root()) / path))


def environment_python(name: str) -> Path:
    """Return an isolated interpreter in the project's environment directory."""
    return project_root() / "envs" / name / "bin/python"
