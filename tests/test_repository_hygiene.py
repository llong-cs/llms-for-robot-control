"""Publication hygiene for everything a commit of this checkout can publish.

The checks cover tracked files plus untracked files that are not ignored, because both reach
the public repository with the next ``git add``. A tracked file already deleted from the working
tree is left out: that ``git add`` records its removal.
"""

from __future__ import annotations

import json
import subprocess
from fnmatch import fnmatch
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Private provider modules follow this naming rule and are never published.
PRIVATE_PROVIDERS = "agentic-framework/src/agentic_framework/models/llm/providers/private_*"


def _paths(*options: str) -> set[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", *options], cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode()
    return {path for path in listed.split("\0") if path}


@cache
def _published() -> tuple[str, ...] | None:
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True,
            text=True, check=False,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        return tuple(sorted(_paths("--cached", "--others", "--exclude-standard")
                            - _paths("--deleted")))
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.fixture(scope="module")
def published() -> tuple[str, ...]:
    files = _published()
    if files is None:
        pytest.skip("Repository hygiene needs a git checkout")
    return files


def test_no_tracked_file_matches_an_ignore_rule(published):
    """Force-added files bypass .gitignore, which keeps local and private material out."""
    ignored = _paths("--cached", "--ignored", "--exclude-per-directory=.gitignore")
    assert sorted(ignored - _paths("--deleted")) == []


def test_no_private_provider_is_published(published):
    assert [name for name in published if fnmatch(name, PRIVATE_PROVIDERS)] == []


def _privileged(value, parent: str | None = None) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if item == "privileged" and (
                key == "observation_profile" or (key == "profile" and parent == "observation")
            ):
                return True
            if _privileged(item, key):
                return True
    elif isinstance(value, list):
        return any(_privileged(item, parent) for item in value)
    return False


def test_no_config_enables_privileged_observations(published):
    enabled = []
    for name in published:
        path = ROOT / name
        if name.endswith(".json") and path.is_file():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as error:
                pytest.fail(f"{name} is not valid JSON: {error}")
            if _privileged(document):
                enabled.append(name)
    assert enabled == []
