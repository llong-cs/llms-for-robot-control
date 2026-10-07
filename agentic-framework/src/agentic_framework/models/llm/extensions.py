"""Discover optional LLM backends without importing unselected providers.

An installed extension registers its name in the ``agentic_framework.llm_backends``
entry-point group. Its entry point must expose ``configure(args)`` and
``create(args, directory)``. Configuration validates and resolves options without
reading credentials or making network requests; creation returns a policy backend.
"""

from __future__ import annotations

from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "agentic_framework.llm_backends"
BUILTIN_BACKENDS = ("codex", "responses")


def _extensions():
    found = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        if not entry.name or entry.name in BUILTIN_BACKENDS:
            raise ValueError(f"Invalid optional backend name: {entry.name!r}")
        if entry.name in found:
            raise ValueError(f"Multiple extensions register backend {entry.name!r}")
        found[entry.name] = entry
    return found


def backend_names():
    """List available names from package metadata; no provider code is loaded."""
    return tuple(sorted((*BUILTIN_BACKENDS, *_extensions())))


def load_backend_extension(name):
    """Load and check the selected installed backend extension."""
    entry = _extensions().get(name)
    if entry is None:
        raise ValueError(
            f"Backend {name!r} is not installed; available backends: "
            + ", ".join(backend_names())
        )
    extension = entry.load()
    for method in ("configure", "create"):
        if not callable(getattr(extension, method, None)):
            raise ValueError(f"Backend {name!r} must expose callable {method}()")
    return extension
