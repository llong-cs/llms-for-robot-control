#!/usr/bin/env python3
"""Start a native model server in its own environment."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import workspace as w  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", help="GPU index or index list to use")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    args.command = "serve"
    try:
        if args.port is not None and not 1 <= args.port <= 65535:
            raise ValueError("port must be within 1..65535")
        command, env = w.command(args, extra)
        if args.dry_run:
            print(json.dumps({"command": command, "display": shlex.join(command)}, indent=2))
            return 0
        return subprocess.call(command, cwd=ROOT, env=env)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
