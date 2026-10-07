"""Central source-layout paths, independent of individual module nesting."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = PACKAGE_ROOT.parents[1]
LIBERO_WORKER = PACKAGE_ROOT / "environments/libero/worker.py"
MANISKILL_WORKER = PACKAGE_ROOT / "environments/maniskill/worker.py"
