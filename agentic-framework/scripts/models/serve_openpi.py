#!/usr/bin/env python3
"""Serve an OpenPi-family JAX checkpoint selected by a verified model profile.

Run with <project>/envs/agentic-framework-openpi/bin/python. --check-only performs source
and full checkpoint verification without importing JAX or allocating a GPU.
Model transforms, denoising count, RNG, and action horizon remain official.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from server_profile import (  # noqa: E402
    configure_project_environment,
    load_server_profile,
    profile_parser,
    profile_path,
    server_metadata,
)

from agentic_framework.models.seed_protocol import SEED_KEY, SEED_PROTOCOL, validate_sampling

DEFAULT_PROFILE = "pi05-libero"


class SeededPolicy:
    """Request-local noise passed through the official public inference interface."""

    def __init__(self, policy, horizon, action_dim):
        self.policy, self.horizon, self.action_dim = policy, horizon, action_dim

    def infer(self, observation):
        import numpy as np

        payload = dict(observation)
        sampling = validate_sampling(payload.pop(SEED_KEY, None))
        if sampling is None:
            return self.policy.infer(payload)
        noise = np.random.Generator(np.random.PCG64(sampling["seed"])).standard_normal(
            (self.horizon, self.action_dim), dtype=np.float32
        )
        result = dict(self.policy.infer(payload, noise=noise))
        result[SEED_KEY] = sampling
        return result

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_source(root: Path, profile=None) -> dict[str, Any]:
    profile = load_server_profile(profile, DEFAULT_PROFILE)
    expected_commit = profile["model"]["expected_metadata"]["openpi_commit"]
    root = root.resolve(strict=True)
    commit = _git(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"OpenPI commit mismatch: {commit}; expected {expected_commit}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"Official OpenPI checkout must be clean: {dirty}")
    return {
        "openpi_root": str(root),
        "openpi_commit": commit,
        "openpi_worktree_clean": True,
        "source_files": {
            rel: {"path": str(root / rel), "sha256": sha256_file(root / rel)}
            for rel in profile["server"]["source_files"]
        },
    }


def verify_checkpoint(checkpoint: Path, profile=None) -> dict[str, Any]:
    profile = load_server_profile(profile, DEFAULT_PROFILE)
    checkpoint_uri = profile["model"]["expected_metadata"]["checkpoint"]
    checkpoint = checkpoint.resolve(strict=True)
    manifest_path = checkpoint / "VLA_SOURCE.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_uri") != checkpoint_uri:
        raise RuntimeError(
            "Checkpoint source manifest does not identify the selected model profile"
        )
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("Checkpoint source manifest has no file inventory")
    if (checkpoint / "model.safetensors").exists() or not (checkpoint / "params").is_dir():
        raise RuntimeError("Expected the existing official JAX checkpoint, not converted weights")
    actual_files = {
        str(path.relative_to(checkpoint))
        for path in checkpoint.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != set(entries):
        raise RuntimeError("Checkpoint tree differs from source manifest file inventory")
    aggregate = hashlib.sha256()
    total = 0
    for relative, expected in sorted(entries.items()):
        path = checkpoint / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise RuntimeError(f"Invalid checkpoint manifest path: {relative}")
        if not path.resolve().is_relative_to(checkpoint):
            raise RuntimeError(f"Checkpoint file escapes root: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(f"Checkpoint integrity mismatch: {relative}")
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total += size
    if aggregate.hexdigest() != manifest.get("tree_sha256") or total != manifest.get("total_bytes"):
        raise RuntimeError("Checkpoint aggregate fingerprint differs from source manifest")
    return {
        "checkpoint": checkpoint_uri,
        "checkpoint_path": str(checkpoint),
        "checkpoint_source_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_tree_sha256": aggregate.hexdigest(),
        "checkpoint_files_verified": len(entries),
        "checkpoint_bytes_verified": total,
        "norm_stats_sha256": sha256_file(checkpoint / profile["model"]["norm_stats_path"]),
        "checkpoint_integrity": "full SHA-256 and byte count verified against VLA_SOURCE.json",
    }


def write_manifest(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def arguments(argv: list[str] | None = None):
    parser, profile = profile_parser(__doc__, DEFAULT_PROFILE, argv)
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=profile_path(profile["server"]["source"]),
        help="Verified OpenPi source checkout (defaults to selected profile)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=profile_path(profile["model"]["checkpoint"]),
        help="Local official JAX checkpoint with VLA_SOURCE.json",
    )
    parser.add_argument(
        "--host",
        choices=("127.0.0.1",),
        default=profile["transport"]["host"],
        help="Loopback interface for the native WebSocket server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=profile["transport"]["port"],
        help="WebSocket listening port (defaults to selected profile)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=profile_path("outputs/servers") / profile["id"],
        help="Verification and server manifests (default: <project>/outputs/servers/<profile>)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable GPU autotuning and reject nondeterministic kernels for reproducible restarts",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify source and full checkpoint hashes without JAX or GPU loading",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be within [1, 65535]")
    for key in ("openpi_root", "checkpoint", "output_dir"):
        setattr(args, key, profile_path(getattr(args, key)))
    return args


def configure_determinism(enabled: bool) -> str:
    """Set process-scoped XLA flags before JAX is imported; never alter model code."""
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    if enabled:
        required = {
            "--xla_gpu_autotune_level": "0",
            "--xla_gpu_exclude_nondeterministic_ops": "true",
        }
        for name, value in required.items():
            existing = [flag for flag in flags if flag.split("=", 1)[0] == name]
            desired = f"{name}={value}"
            if any(flag != desired for flag in existing):
                raise ValueError(f"--deterministic conflicts with XLA_FLAGS {name}")
            if not existing:
                flags.append(desired)
        os.environ["XLA_FLAGS"] = shlex.join(flags)
    return os.environ.get("XLA_FLAGS", "")


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    os.umask(0o077)
    configure_project_environment()
    os.environ.setdefault("OPENPI_DATA_HOME", str(profile_path("models/openpi")))
    xla_flags = configure_determinism(args.deterministic)
    profile = args.profile
    expected = profile["model"]["expected_metadata"]
    config_name = profile["model"]["config_name"]
    source = verify_source(args.openpi_root, profile)
    logging.info("Pinned source verified; hashing official checkpoint files")
    provenance = verify_checkpoint(args.checkpoint, profile)
    metadata = {
        **server_metadata(profile),
        **source,
        **provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.executable,
        "python_version": sys.version,
        "host": args.host,
        "port": args.port,
        "pid": os.getpid(),
        "inference": "official create_trained_policy; seeded requests supply public infer(noise=), otherwise official RNG",
        "deterministic_gpu": args.deterministic,
        "xla_flags": xla_flags,
        "seed_protocol": SEED_PROTOCOL,
        "seed_sampling": "NumPy PCG64 standard_normal float32 (native H, model action dim)",
        "status": "verified_not_loaded",
    }
    write_manifest(args.output_dir / "preflight.json", metadata)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "verified_not_loaded",
                    "h": profile["defaults"]["h"],
                    "k": profile["defaults"]["k"],
                    "manifest": str(args.output_dir / "preflight.json"),
                },
                indent=2,
            )
        )
        return

    # Existing official environment supplies all model dependencies. Explicit
    # paths ensure that imported source is the clean checkout just verified.
    repo = Path(source["openpi_root"])
    sys.path.insert(0, str(repo / "packages/openpi-client/src"))
    sys.path.insert(0, str(repo / "src"))
    import jax
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    for module in (policy_config, websocket_policy_server, config):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError(f"Imported source outside verified OpenPI: {module.__file__}")
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("Official OpenPi serving requires an available GPU")
    train_config = config.get_config(config_name)
    if (
        train_config.model.action_horizon != expected["action_horizon"]
        or train_config.model.pi05 != profile["model"]["pi05"]
        or train_config.model.action_dim != metadata["model_action_dim"]
    ):
        raise RuntimeError(f"Loaded official model config does not match profile {profile['id']}")
    policy = policy_config.create_trained_policy(train_config, args.checkpoint.resolve())
    metadata.update(
        {
            "status": "loaded",
            "model_action_dim": train_config.model.action_dim,
            "official_policy_metadata": dict(policy.metadata or {}),
            "devices": [str(device) for device in devices],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpy", "flax", "orbax-checkpoint")
            },
        }
    )
    write_manifest(args.output_dir / "server_metadata.json", metadata)
    logging.info("Loaded profile %s; serving on %s:%d", profile["id"], args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=SeededPolicy(
            policy, train_config.model.action_horizon, train_config.model.action_dim
        ),
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
