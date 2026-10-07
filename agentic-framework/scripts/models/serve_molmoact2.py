#!/usr/bin/env python3
"""Serve pinned MolmoAct2-DROID with the official two-camera simulator protocol.

The checkpoint code is unchanged. Native profile uses raw state8 / actions15x8,
continuous inference, ten flow steps, and no depth reasoning. Model-card fp32
is the default; bf16 uses its documented autocast. --check-only loads no model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

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

DEFAULT_PROFILE = "molmoact2-droid"
_DEFAULT = load_server_profile(None, DEFAULT_PROFILE)
MOLMOACT2_COMMIT = _DEFAULT["model"]["expected_metadata"]["molmoact2_commit"]
REPO_ID = _DEFAULT["model"]["expected_metadata"]["repo_id"]
REVISION = _DEFAULT["model"]["expected_metadata"]["revision"]
SOURCE_URI = "https://huggingface.co/" + REPO_ID
SOURCE_FILES = (
    "examples/droid/host_server_droid.py",
    "sim_eval/inference/common.py",
    "sim_eval/inference/client.py",
    "sim_eval/run_eval.py",
    "sim_eval/robots/franka_droid.py",
    "sim_eval/tasks/droid_tasks/droid_put_everything_in_box.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def verify_source(root, profile=None):
    profile = load_server_profile(profile, DEFAULT_PROFILE)
    expected = profile["model"]["expected_metadata"]
    root = root.resolve(strict=True)
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if commit != expected["molmoact2_commit"]:
        raise RuntimeError("MolmoAct2 source commit mismatch")
    inventory = {}
    for relative in profile["server"].get("source_files", SOURCE_FILES):
        expected = hashlib.sha256(_git(root, "show", f"HEAD:{relative}")).hexdigest()
        actual = sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(f"MolmoAct2 inference source differs from pinned commit: {relative}")
        inventory[relative] = {"sha256": actual, "matches_commit": True}
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    return {
        "molmoact2_root": str(root),
        "molmoact2_commit": commit,
        "molmoact2_worktree_clean": not bool(status.strip()),
        "worktree_status_sha256": hashlib.sha256(status).hexdigest(),
        "worktree_status_lines": len(status.splitlines()),
        "relevant_sources_verified_against_commit": True,
        "source_files": inventory,
    }


def verify_checkpoint(checkpoint, profile=None):
    profile = load_server_profile(profile, DEFAULT_PROFILE)
    expected_model = profile["model"]["expected_metadata"]
    source_uri = "https://huggingface.co/" + expected_model["repo_id"]
    checkpoint = checkpoint.resolve(strict=True)
    manifest_path = checkpoint / "VLA_SOURCE.json"
    manifest = json.loads(manifest_path.read_text())
    for key, value in {
        "source_uri": source_uri,
        "repo_id": expected_model["repo_id"],
        "revision": expected_model["revision"],
    }.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"MolmoAct2 checkpoint provenance mismatch: {key}")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("Checkpoint source manifest has no file inventory")

    # HF local-dir bookkeeping is not checkpoint content. Exclude only its
    # documented download cache, never other hidden files or model source files.
    def is_download_cache(path):
        parts = path.relative_to(checkpoint).parts
        return parts[:3] == (".cache", "huggingface", "download") or parts == (
            ".cache",
            "huggingface",
            ".gitignore",
        )

    actual_files = {
        str(path.relative_to(checkpoint))
        for path in checkpoint.rglob("*")
        if path.is_file() and path != manifest_path and not is_download_cache(path)
    }
    if actual_files != set(entries):
        raise RuntimeError("Checkpoint tree differs from source manifest file inventory")
    aggregate, total = hashlib.sha256(), 0
    for relative, expected in sorted(entries.items()):
        path = checkpoint / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise RuntimeError(f"Invalid checkpoint path: {relative}")
        if not path.resolve().is_relative_to(checkpoint):
            raise RuntimeError(f"Checkpoint file escapes root: {relative}")
        size, digest = path.stat().st_size, sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(f"Checkpoint integrity mismatch: {relative}")
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())
        total += size
    if aggregate.hexdigest() != manifest.get("tree_sha256") or total != manifest.get("total_bytes"):
        raise RuntimeError("Checkpoint aggregate fingerprint differs from source manifest")
    config = json.loads((checkpoint / "config.json").read_text())
    norms = json.loads((checkpoint / "norm_stats.json").read_text())
    tag = norms.get("metadata_by_tag", {}).get(expected_model["norm_tag"], {})
    if (
        config.get("max_action_horizon") != expected_model["action_horizon"]
        or config.get("max_action_dim") != profile["server"]["metadata"]["model_action_dim"]
    ):
        raise RuntimeError("Checkpoint architecture does not match the selected MolmoAct2 profile")
    if (
        config.get("flow_matching_num_steps") != expected_model["flow_steps"]
        or config.get("enable_depth_reasoning") is not expected_model["enable_depth_reasoning"]
    ):
        raise RuntimeError("Checkpoint flow/depth defaults differ from native profile")
    if any(
        tag.get(key) != value
        for key, value in {
            "action_horizon": expected_model["action_horizon"],
            "n_action_steps": expected_model["action_horizon"],
            "control_mode": "delta end-effector pose"
            if expected_model["norm_tag"] == "libero"
            else "absolute joint pose",
            "normalize_gripper": False,
        }.items()
    ):
        raise RuntimeError("Checkpoint normalization metadata differs from selected native profile")
    if len(tag.get("action_stats", {}).get("min", [])) != expected_model["output_action_dim"]:
        raise RuntimeError("Normalization action dimension differs from selected native profile")
    return {
        "repo_id": expected_model["repo_id"],
        "revision": expected_model["revision"],
        "checkpoint": source_uri,
        "checkpoint_path": str(checkpoint),
        "checkpoint_source_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_tree_sha256": aggregate.hexdigest(),
        "checkpoint_files_verified": len(entries),
        "checkpoint_bytes_verified": total,
        "checkpoint_integrity": "full SHA-256 and byte count verified against VLA_SOURCE.json",
        "norm_stats_sha256": sha256_file(checkpoint / "norm_stats.json"),
        "model_code_sha256": sha256_file(checkpoint / "modeling_molmoact2.py"),
    }


def write_manifest(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def arguments(argv=None):
    parser, profile = profile_parser(__doc__, DEFAULT_PROFILE, argv)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=profile_path(profile["server"]["source"]),
        help="Verified upstream source checkout",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=profile_path(profile["model"]["checkpoint"]),
        help="Checkpoint directory with VLA_SOURCE.json",
    )
    parser.add_argument(
        "--host",
        choices=("127.0.0.1",),
        default=profile["transport"]["host"],
        help="Loopback HTTP bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=profile["transport"]["port"],
        help="HTTP listening port from selected profile",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=profile_path("outputs/servers") / profile["id"],
        help="Verification and server manifests (default: <project>/outputs/servers/<profile>)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Official model precision; bfloat16 enables documented autocast",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Enable the official CUDA graph inference option",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Hash source and weights without loading the model or GPU",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be within [1, 65535]")
    for key in ("source_root", "checkpoint", "output_dir"):
        setattr(args, key, profile_path(getattr(args, key)))
    return args


class Policy:
    def __init__(self, checkpoint, dtype, enable_cuda_graph=False, profile=None):
        self.profile = load_server_profile(profile, DEFAULT_PROFILE)
        self.expected = self.profile["model"]["expected_metadata"]
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("MolmoAct2 serving requires an available CUDA GPU")
        self.dtype = getattr(torch, dtype)
        self.autocast = dtype == "bfloat16"
        self.enable_cuda_graph = enable_cuda_graph
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint), trust_remote_code=True, local_files_only=True, extra_special_tokens={}
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                str(checkpoint), trust_remote_code=True, local_files_only=True, dtype=self.dtype
            )
            .to("cuda")
            .eval()
        )
        # Never patch checkpoint or cached model code. The pinned HF revision
        # already includes dtype fixes; bf16 follows its model-card autocast.
        self.lock = threading.Lock()

    def predict(self, payload, *, timing=None):
        timing = {} if timing is None else timing
        prepare_started = time.perf_counter()
        timing.update({
            "schema_version": 1,
            "unit": "seconds",
            "clock": "perf_counter",
            "scope": "host wall time; no CUDA events or global device synchronization",
            "model_elapsed_scope": (
                "official predict_action including its internal preprocessing/denoising, "
                "plus action transfer to CPU; not GPU kernel time"
            ),
        })
        try:
            return self._predict_timed(payload, timing, prepare_started)
        finally:
            timing.setdefault("preprocessing_s", time.perf_counter() - prepare_started)
            timing["policy_total_s"] = time.perf_counter() - prepare_started

    def _predict_timed(self, payload, timing, prepare_started):
        import numpy as np
        import torch
        from PIL import Image

        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        images = []
        for key in ("external_cam", "wrist_cam"):
            image = np.asarray(payload.get(key))
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
                or 0 in image.shape
            ):
                raise ValueError(f"{key} must be nonempty HWC uint8 RGB")
            images.append(Image.fromarray(image))
        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state must be finite float32 shape (8,)")
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty text")
        sampling = validate_sampling(payload.get(SEED_KEY))
        generator = (
            None
            if sampling is None
            else torch.Generator(device="cuda").manual_seed(sampling["seed"])
        )
        wait_started = time.perf_counter()
        timing["preprocessing_s"] = wait_started - prepare_started
        with self.lock, torch.inference_mode():
            timing["lock_wait_s"] = time.perf_counter() - wait_started
            model_started = time.perf_counter()
            try:
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else nullcontext()
                with ctx:
                    result = self.model.predict_action(
                        processor=self.processor,
                        generator=generator,
                        images=images,
                        task=instruction,
                        state=state,
                        norm_tag=self.expected["norm_tag"],
                        inference_action_mode="continuous",
                        enable_depth_reasoning=self.expected["enable_depth_reasoning"],
                        num_steps=self.expected["flow_steps"],
                        normalize_language=True,
                        enable_cuda_graph=self.enable_cuda_graph,
                    )
                actions = result.actions
                if torch.is_tensor(actions):
                    actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
            finally:
                timing["model_elapsed_s"] = time.perf_counter() - model_started
        output_started = time.perf_counter()
        try:
            actions = np.asarray(actions, dtype=np.float32)
            # Official predict_action may retain its singleton batch dimension.
            shape = (self.expected["action_horizon"], self.expected["output_action_dim"])
            if actions.shape == (1, *shape):
                actions = actions[0]
            if actions.shape != shape or not np.isfinite(actions).all():
                raise ValueError(f"model returned invalid native action chunk: {actions.shape}")
            return actions
        finally:
            timing["postprocessing_s"] = time.perf_counter() - output_started


def build_app(policy, metadata):
    import json_numpy
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Pinned MolmoAct2 native simulator server")

    @app.get("/act")
    @app.get("/healthz")
    async def health():
        return JSONResponse(metadata)

    # Request is imported lazily; explicit annotation avoids a global import of
    # model-runtime dependencies during check-only and unit-test inspection.
    async def act(request):
        started = time.perf_counter()
        timing = {
            "schema_version": 1,
            "unit": "seconds",
            "request_started_at": datetime.now(timezone.utc).isoformat(),
            "handler_scope": "request body read to response serialization start; excludes network send",
        }
        try:
            body = await request.body()
            payload = json_numpy.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
            sampling = validate_sampling(payload.get(SEED_KEY))
            timing["request_bytes"] = len(body)
            timing["request_decode_s"] = time.perf_counter() - started
            dispatch_started = time.perf_counter()

            def predict():
                timing["threadpool_wait_s"] = time.perf_counter() - dispatch_started
                return policy.predict(payload, timing=timing)

            actions = await run_in_threadpool(predict)
            timing["handler_elapsed_s"] = time.perf_counter() - started
            timing["response_started_at"] = datetime.now(timezone.utc).isoformat()
            return Response(
                json_numpy.dumps(
                    {
                        "actions": actions,
                        "dt_ms": timing["handler_elapsed_s"] * 1000,
                        "server_timing": timing,
                        **({SEED_KEY: sampling} if sampling is not None else {}),
                    }
                ),
                media_type="application/json",
            )
        except Exception as exc:
            timing["handler_elapsed_s"] = time.perf_counter() - started
            timing["response_started_at"] = datetime.now(timezone.utc).isoformat()
            status_code = 422 if isinstance(exc, (ValueError, TypeError, KeyError)) else 500
            return JSONResponse(
                {"error": str(exc), "type": type(exc).__name__, "server_timing": timing},
                status_code=status_code,
            )

    act.__annotations__["request"] = Request
    app.post("/act")(act)
    return app


def main(argv=None):
    args = arguments(argv)
    os.umask(0o077)
    configure_project_environment()
    profile = args.profile
    source = verify_source(args.source_root, profile)
    checkpoint = verify_checkpoint(args.checkpoint, profile)
    metadata = {
        **server_metadata(profile),
        **source,
        **checkpoint,
        "inference_action_mode": "continuous",
        "normalize_language": True,
        "dtype": args.dtype,
        "enable_cuda_graph": args.enable_cuda_graph,
        "seed_protocol": SEED_PROTOCOL,
        "seed_sampling": "request-local torch.Generator(device=cuda)",
        "raw_action_semantics": "native normalized OSC7"
        if profile["adapter"] == "libero_osc"
        else "absolute joint positions7 plus native knuckle position1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256_file(__file__),
        "python": sys.executable,
        "python_version": sys.version,
        "host": args.host,
        "port": args.port,
        "pid": os.getpid(),
        "status": "verified_not_loaded",
        "checkpoint_code_modified": False,
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
    import torch
    import uvicorn

    policy = Policy(args.checkpoint.resolve(), args.dtype, args.enable_cuda_graph, profile)
    metadata.update(
        {
            "status": "loaded",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": torch.cuda.get_device_name(0),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy", "json-numpy", "fastapi", "uvicorn")
            },
        }
    )
    write_manifest(args.output_dir / "server_metadata.json", metadata)
    uvicorn.run(build_app(policy, metadata), host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
