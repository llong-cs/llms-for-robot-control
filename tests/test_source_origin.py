"""The suite must exercise this checkout's sources, not another installed or editable copy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "agentic_framework": "agentic-framework/src",
    "inspect_robots": "agentic-framework/vendor/inspect-robots/src",
    "ditto": "ditto-bench/src",
    "scripts": ".",
}


@pytest.mark.parametrize(("package", "source"), sorted(SOURCES.items()))
def test_packages_are_imported_from_this_checkout(package, source):
    spec = importlib.util.find_spec(package)
    if spec is None and package == "ditto":
        pytest.skip("The suite package is imported only by eval.py subprocesses")
    assert spec is not None and spec.origin is not None, package
    origin = Path(spec.origin).resolve()
    expected = (ROOT / source / package).resolve()
    assert origin.is_relative_to(expected), (
        f"{package} is imported from {origin}, not from {expected}; put this checkout's "
        "source directories first on PYTHONPATH"
    )
