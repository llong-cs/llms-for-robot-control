#!/usr/bin/env python3
"""Fetch exact official source revisions into project-local third_party checkouts."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

SOURCES = {
    "openpi": ("https://github.com/Physical-Intelligence/openpi.git", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    "molmoact2": ("https://github.com/allenai/molmoact2.git", "66b87e64efd99dfd103241418113955cf64dfa9c"),
}


def fetch(name: str, destination: Path) -> None:
    url, commit = SOURCES[name]
    project = Path(__file__).resolve().parents[3]
    destination = destination.expanduser()
    resource_root = Path(os.environ.get("AGENTIC_PROJECT_ROOT") or project).expanduser().absolute()
    destination = (destination if destination.is_absolute() else resource_root / destination).resolve()
    code_directories = [project / name for name in (
        "agentic-framework", "ditto-bench", "scripts", "tools", "docs", "examples", "configs"
    )]
    if destination == project or destination in project.parents or any(
        destination == path or path in destination.parents for path in code_directories
    ):
        raise ValueError("Source checkout destination would replace or modify project source files")
    if destination.exists() and (not destination.is_dir() or not (destination / ".git").exists()):
        raise ValueError("Existing source destination is not a Git checkout; select another destination")
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(destination)], env=environment, check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit], check=True)
    current = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if current != commit:
        raise RuntimeError(f"Existing {name} checkout is {current}; expected {commit}. Select another destination.")
    subprocess.run(["git", "-C", str(destination), "diff", "--exit-code", "HEAD", "--", "."], check=True)
    if name == "openpi":
        subprocess.run(["git", "-C", str(destination), "submodule", "update", "--init", "third_party/libero"], env=environment, check=True)
    print(f"{name}: {destination} @ {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    fetch(args.source, args.destination or Path("third_party") / args.source)


if __name__ == "__main__":
    main()
