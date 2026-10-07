"""Compatibility CLI for ``python -m agentic_framework.run``.

Implementation and internal imports live in :mod:`agentic_framework.harness.run`.
"""
from agentic_framework.harness.run import main

if __name__ == "__main__":
    raise SystemExit(main())
