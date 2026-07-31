#!/usr/bin/env python3
"""Prefetch and attest one immutable Hugging Face embedding model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

PROVENANCE_FILENAME = "MODEL_PROVENANCE.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--model-file", default="model.onnx")
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--license", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if len(args.revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.revision
    ):
        raise ValueError("--revision must be a lowercase 40-character commit SHA")
    if len(args.model_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.model_sha256
    ):
        raise ValueError("--model-sha256 must be a lowercase SHA-256")
    if len(args.provenance_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.provenance_sha256
    ):
        raise ValueError("--provenance-sha256 must be a lowercase SHA-256")

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repository,
        revision=args.revision,
        local_dir=destination,
    )

    model_file = (destination / args.model_file).resolve(strict=True)
    if not model_file.is_relative_to(destination):
        raise ValueError("--model-file escapes --destination")
    actual_model_sha256 = _sha256(model_file)
    if actual_model_sha256 != args.model_sha256:
        raise ValueError(
            f"{args.model_file} SHA-256 mismatch: "
            f"expected {args.model_sha256}, got {actual_model_sha256}"
        )

    files = {}
    for path in sorted(destination.rglob("*")):
        if (
            not path.is_file()
            or PROVENANCE_FILENAME in path.parts
            or ".cache" in path.parts
        ):
            continue
        relative = path.relative_to(destination).as_posix()
        files[relative] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }

    payload = {
        "schema_version": 1,
        "repository": args.repository,
        "revision": args.revision,
        "license": args.license,
        "files": files,
    }
    target = destination / PROVENANCE_FILENAME
    temporary = destination / f".{PROVENANCE_FILENAME}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    actual_provenance_sha256 = _sha256(target)
    if actual_provenance_sha256 != args.provenance_sha256:
        raise ValueError(
            f"{PROVENANCE_FILENAME} SHA-256 mismatch: "
            f"expected {args.provenance_sha256}, "
            f"got {actual_provenance_sha256}"
        )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "repository": args.repository,
                "revision": args.revision,
                "model_sha256": actual_model_sha256,
                "provenance_sha256": actual_provenance_sha256,
                "files": len(files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
