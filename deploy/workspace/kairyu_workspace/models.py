"""Retrieval models for an L4 workspace, fetched at a pinned revision and tree hash.

``fetch`` runs on the host and starts this file as a program in the pinned
Open WebUI image (it carries ``huggingface_hub``) on Docker's default network:
the Chat UI runs with ``OFFLINE_MODE`` and loads the models from a read-only
mount, so each pinned, hashed copy must exist before it starts. A copy whose
attestation matches is not re-hashed unless ``VERIFY_MODEL=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ATTESTATION = ".kairyu-model-attestation.json"


def fetch(workspace, models: Path) -> None:
    """Fetch and attest every retrieval model of ``workspace``'s profile into ``models``."""
    from .spec import OPEN_WEBUI_IMAGE

    models.mkdir(parents=True, exist_ok=True)
    for role in ("embedding", "reranking"):
        model = workspace.profile["retrieval"][role]
        command = ["docker", "run", "--rm", "--entrypoint", "python3"]
        command += ["--volume", f"{Path(__file__).resolve()}:/opt/models.py:ro"]
        command += ["--volume", f"{models}:/models"]
        for name in ("HF_TOKEN", "VERIFY_MODEL"):
            if os.environ.get(name):
                command += ["--env", name]
        command += [OPEN_WEBUI_IMAGE, "/opt/models.py", json.dumps(model)]
        print(f"+ fetch {model['repo']}@{model['revision'][:12]}", flush=True)
        subprocess.run(command, check=True)


def _inventory(target: Path) -> tuple[list[dict], str]:
    rows = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.name == ATTESTATION or ".cache" in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": str(path.relative_to(target)),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    tree = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(tree).hexdigest()


def main(model: dict) -> None:
    target = Path("/models") / model["slug"]
    attestation = target / ATTESTATION
    pinned = (model["repo"], model["revision"], model["tree_sha256"])
    if attestation.exists():
        current = json.loads(attestation.read_text())
        if (current.get("repo"), current.get("revision"), current.get("tree_sha256")) != pinned:
            raise SystemExit(f"{target} is attested for another checkpoint")
        if os.environ.get("VERIFY_MODEL") == "1":
            rows, tree = _inventory(target)
            if tree != model["tree_sha256"] or current.get("files") != rows:
                raise SystemExit(f"{target} differs from its attestation")
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        model["repo"],
        revision=model["revision"],
        local_dir=target,
        allow_patterns=model["allow_patterns"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    rows, tree = _inventory(target)
    if tree != model["tree_sha256"]:
        raise SystemExit(f"{model['repo']}: downloaded tree {tree} differs from the pin")
    record = {
        "schema_version": 1,
        "repo": model["repo"],
        "revision": model["revision"],
        "tree_sha256": tree,
        "files": rows,
    }
    attestation.write_text(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
