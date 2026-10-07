#!/usr/bin/env python3
"""Download authorized official MolmoAct2-LIBERO resources and verify provenance."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path

from server_profile import configure_project_environment, profile_path

REPO_ID = "allenai/MolmoAct2-LIBERO"
REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
PATTERNS = ["*"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=profile_path("models/molmoact2-libero"),
        help="Destination for pinned MolmoAct2-LIBERO resources (default: <project>/models/molmoact2-libero)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file containing Hugging Face credentials; otherwise use environment or Hugging Face login",
    )
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="Check Hugging Face model access without downloading weights",
    )
    args = parser.parse_args(argv)
    root = profile_path(args.output_dir)
    if args.env_file is not None:
        args.env_file = profile_path(args.env_file)
    os.umask(0o077)
    configure_project_environment()
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, get_token, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    credentials = dotenv_values(args.env_file) if args.env_file is not None and args.env_file.is_file() else {}
    token = (
        next(
            (
                credentials.get(key)
                for key in (
                    "HF_TOKEN",
                    "HUGGING_FACE_HUB_TOKEN",
                    "HUGGINGFACE_HUB_TOKEN",
                    "HUGGINGFACE_TOKEN",
                )
                if credentials.get(key)
            ),
            None,
        )
        or get_token()
    )
    api = HfApi(token=token)
    try:
        api.auth_check(repo_id=REPO_ID, repo_type="model")
    except GatedRepoError:
        raise SystemExit(
            "Official MolmoAct2 weight access has not been granted to this Hugging Face account. "
            "Authorize access at https://huggingface.co/allenai/MolmoAct2-LIBERO, then retry. "
            "No model inference or evaluation has run."
        ) from None
    if args.check_access:
        print(json.dumps({"repo_id": REPO_ID, "status": "access_granted"}))
        return
    info = api.model_info(REPO_ID, revision=REVISION, files_metadata=True)
    selected = {
        entry.rfilename: entry
        for entry in info.siblings
        if any(fnmatch.fnmatch(entry.rfilename, pattern) for pattern in PATTERNS)
    }
    if not selected:
        raise RuntimeError("Official repository has no matching LIBERO resources")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        allow_patterns=PATTERNS,
        local_dir=root,
        token=token,
        max_workers=4,
    )
    files, aggregate = {}, hashlib.sha256()
    for relative, entry in sorted(selected.items()):
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise RuntimeError(f"Invalid downloaded path: {relative}")
        size = path.stat().st_size
        digest = hashlib.sha256()
        git_digest = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                git_digest.update(block)
        checksum = digest.hexdigest()
        if entry.size is not None and size != entry.size:
            raise RuntimeError(f"Size mismatch: {relative}")
        if entry.lfs is not None and checksum != entry.lfs.sha256:
            raise RuntimeError(f"Official LFS SHA256 mismatch: {relative}")
        if entry.lfs is None and entry.blob_id and git_digest.hexdigest() != entry.blob_id:
            raise RuntimeError(f"Official Git blob mismatch: {relative}")
        files[relative] = {"bytes": size, "sha256": checksum}
        aggregate.update(f"{relative}\0{size}\0{checksum}\n".encode())
    manifest = {
        "source_uri": "https://huggingface.co/" + REPO_ID,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "tree_sha256": aggregate.hexdigest(),
    }
    destination = root / "VLA_SOURCE.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "downloaded_files": len(files),
                "total_bytes": manifest["total_bytes"],
                "tree_sha256": manifest["tree_sha256"],
                "manifest": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
