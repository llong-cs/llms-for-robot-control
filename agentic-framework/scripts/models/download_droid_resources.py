#!/usr/bin/env python3
"""Fetch official robot assets and checkpoints into external data/model locations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from server_profile import configure_project_environment, profile_path

MOLMO_REVISION = "d8c1abd8a27d8e859455bbe514df2bcc617db0fb"
ASSET_REVISION = "9332a64224ff0a813d9f77bd377b845270232513"


def manifest(root, source, **extra):
    entries = {}
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if not path.is_file() or relative == "VLA_SOURCE.json" or ".cache" in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        size, sha = path.stat().st_size, digest.hexdigest()
        entries[relative] = {"bytes": size, "sha256": sha}
        aggregate.update(f"{relative}\0{size}\0{sha}\n".encode())
    value = dict(
        source_uri=source,
        checkpoint=str(root),
        files=entries,
        tree_sha256=aggregate.hexdigest(),
        total_bytes=sum(item["bytes"] for item in entries.values()),
        created_at=datetime.now(timezone.utc).isoformat(),
        **extra,
    )
    temporary = root / "VLA_SOURCE.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(root / "VLA_SOURCE.json")
    print(
        json.dumps(
            {
                "path": str(root),
                "files": len(entries),
                "bytes": value["total_bytes"],
                "tree_sha256": value["tree_sha256"],
            }
        ),
        flush=True,
    )


def hf(kind, output_dir=None):
    from huggingface_hub import snapshot_download

    if kind == "assets":
        repo, revision, repo_type = (
            "TreeePlanter/molmoact2-sim-eval-assets",
            ASSET_REVISION,
            "dataset",
        )
        root = profile_path(output_dir or "data/molmoact2-sim-eval-assets")
    else:
        repo, revision, repo_type = "allenai/MolmoAct2-DROID", MOLMO_REVISION, "model"
        root = profile_path(output_dir or "models/molmoact2-droid")
    snapshot_download(
        repo_id=repo, revision=revision, repo_type=repo_type, local_dir=str(root), max_workers=4
    )
    manifest(
        root,
        "https://huggingface.co/" + ("datasets/" if repo_type == "dataset" else "") + repo,
        repo_id=repo,
        revision=revision,
        repo_type=repo_type,
    )


def pi05(benchmark="droid", output_dir=None):
    name = "pi05_" + benchmark
    root = profile_path(output_dir or ("models/pi05-droid" if benchmark == "droid" else
                                      "models/openpi/openpi-assets/checkpoints/pi05_libero"))
    prefix = "checkpoints/" + name + "/"
    records, token = [], None
    while True:
        params = {"prefix": prefix}
        if token:
            params["pageToken"] = token
        with urlopen(
            "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urlencode(params),
            timeout=60,
        ) as response:
            data = json.load(response)
        records.extend(row for row in data.get("items", []) if not row["name"].endswith("/"))
        token = data.get("nextPageToken")
        if not token:
            break
    if not records:
        raise RuntimeError("Official checkpoint listing is empty")
    print(
        f"Downloading {len(records)} files, {sum(int(x['size']) for x in records)} bytes",
        flush=True,
    )

    def fetch(row):
        relative = row["name"][len(prefix) :]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe asset path")
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + ".partial")
        md5 = hashlib.md5()
        url = (
            "https://storage.googleapis.com/openpi-assets/"
            + quote(row["name"], safe="/")
            + "?generation="
            + row["generation"]
        )
        with urlopen(url, timeout=180) as response, temporary.open("wb") as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
                md5.update(block)
        if temporary.stat().st_size != int(row["size"]):
            raise RuntimeError("download size mismatch: " + relative)
        if row.get("md5Hash") and base64.b64encode(md5.digest()).decode() != row["md5Hash"]:
            raise RuntimeError("download MD5 mismatch: " + relative)
        temporary.replace(dest)
        print("Downloaded " + relative, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, records))
    manifest(
        root,
        "gs://openpi-assets/checkpoints/" + name,
        objects={
            row["name"]: {key: row[key] for key in ("generation", "size", "md5Hash") if key in row}
            for row in records
        },
    )


def tokenizer(output_dir=None):
    root = profile_path(output_dir or "models/openpi/big_vision")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "paligemma_tokenizer.model"
    temporary = destination.with_name(destination.name + ".partial")
    with urlopen("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", timeout=180) as response:
        with temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
    temporary.replace(destination)
    print(json.dumps({"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", choices=("assets", "molmo", "pi05", "pi05-libero", "tokenizer", "ycb"), help="Download robot assets, MolmoAct2-DROID, pi05-DROID, pi05-LIBERO, the PaliGemma tokenizer, or YCB assets")
    parser.add_argument("--output-dir", type=Path, help="Resource directory; relative paths resolve under the project root (YCB uses this as MS_ASSET_DIR)")
    args = parser.parse_args()
    os.umask(0o077)
    configure_project_environment()
    if args.resource == "pi05":
        pi05(output_dir=args.output_dir)
    elif args.resource == "pi05-libero":
        pi05("libero", args.output_dir)
    elif args.resource == "tokenizer":
        tokenizer(args.output_dir)
    elif args.resource == "ycb":
        asset_directory = profile_path(args.output_dir or os.environ.get("MS_ASSET_DIR") or "data/maniskill")
        env = dict(os.environ, MS_ASSET_DIR=str(asset_directory))
        subprocess.run(
            [sys.executable, "-m", "mani_skill.utils.download_asset", "ycb", "-y"],
            env=env,
            check=True,
        )
        root = asset_directory / "data/assets/mani_skill2_ycb"
        manifest(
            root,
            "https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip",
            download_tool="mani_skill.utils.download_asset ycb",
            mani_skill_version="3.0.1",
        )
        if (
            json.loads((root / "VLA_SOURCE.json").read_text())["tree_sha256"]
            != "70ef4590a5b759cfb7c7eaf3fcf50a45a78accc33b3a890de1112b7d3e089bba"
        ):
            raise RuntimeError(
                "Downloaded YCB assets differ from the pinned resource inventory"
            )
    else:
        hf(args.resource, args.output_dir)


if __name__ == "__main__":
    main()
