"""Standard-library profile loading for isolated model-server environments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Model environments need only these standard-library configuration modules.
# Resolve the canonical checkout rather than an unrelated editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agentic_framework.configuration.profiles import load_model_profile  # noqa: E402


def load_server_profile(value, default):
    """Accept a profile id/path or an already resolved profile for this family."""
    profile = value if isinstance(value, dict) else load_model_profile(value or default)
    expected_family = load_model_profile(default)["family"]
    if profile["family"] != expected_family:
        raise ValueError(f"This server requires model family {expected_family!r}")
    return profile


def profile_parser(description, default, argv=None):
    """Choose a profile before applying its defaults; CLI flags still override it."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--model-profile",
        default=default,
        help=f"Model profile id or JSON path (default: {default})",
    )
    selected, _ = selector.parse_known_args(argv)
    profile = load_server_profile(selected.model_profile, default)
    parser = argparse.ArgumentParser(
        description=description,
        parents=[selector],
        allow_abbrev=False,
    )
    parser.set_defaults(profile=profile)
    return parser, profile


def project_root():
    """Find the project containing this script without importing a model runtime."""
    override = os.environ.get("AGENTIC_PROJECT_ROOT")
    return Path(override).expanduser().absolute() if override else Path(__file__).resolve().parents[3]


def profile_path(value):
    """Resolve profile and command-line paths against the containing project."""
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else project_root() / path))


def configure_project_environment():
    """Use local caches while preserving explicit environment settings."""
    for name, relative in {
        "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch",
        "PIP_CACHE_DIR": "cache/pip",
        "UV_CACHE_DIR": "cache/uv",
        "UV_PYTHON_INSTALL_DIR": "cache/python",
    }.items():
        os.environ.setdefault(name, str(project_root() / relative))


def server_metadata(profile):
    """Publish the profile's model contract alongside independently verified hashes."""
    metadata = {
        **profile["model"].get("optional_metadata", {}),
        **profile["model"]["expected_metadata"],
        **profile["server"].get("metadata", {}),
        "model_profile": profile["id"],
    }
    metadata.setdefault("replan_steps", profile["defaults"]["k"])
    metadata.setdefault("native_control_hz", profile["defaults"]["control_hz"])
    return metadata
