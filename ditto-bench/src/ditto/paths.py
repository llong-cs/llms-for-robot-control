"""Locate project-local assets and outputs without importing simulator packages."""
import os
from datetime import UTC, datetime
from pathlib import Path


def project_root():
    configured = os.environ.get("AGENTIC_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().absolute()
    for parent in Path(__file__).resolve().parents:
        if (parent / "agentic-framework/pyproject.toml").is_file() and (
            parent / "ditto-bench/pyproject.toml"
        ).is_file():
            return parent
    return Path.cwd()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def asset_root():
    return project_path(os.environ.get("DITTO_ASSET_DIR", "data/generated/ditto-bench/v1"))


def molmo_root():
    return project_path(os.environ.get("MOLMOACT2_SOURCE_ROOT", "third_party/molmoact2"))


def robot_asset_root():
    return project_path(os.environ.get("DROID_ASSET_ROOT", "data/molmoact2-sim-eval-assets"))


def preview_root():
    return project_root() / "outputs/ditto-bench" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S-%fZ-preview")
