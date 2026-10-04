"""Weekly drift check of the M20 reference pins (WP-05, D1).

Compares the vendored OpenAPI closure (``tests/contracts/openai/``) with the
same closure built from ``openai/openai-openapi``'s default branch, and the
pinned Codex release with npm's ``latest`` and ``alpha`` dist-tags. When
anything drifted it writes an issue title to ``--title`` and a Markdown body
to ``--report`` (both empty otherwise); the exit status is 0 either way. The
title names the drifted state, so the workflow opens one issue per state.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import vendor_openai_schema as vendor
from scripts.codex_model_catalog import CODEX_TAG

VENDORED_DIR = Path(__file__).resolve().parents[2] / "tests" / "contracts" / "openai"
# A dist-tag value lands in an issue title; accept only plain versions.
_VERSION = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")
CODEX_TAGS = ("latest", "alpha")
PROCEDURE = (
    "Release procedure (m20 D1): within 7 days of a new Codex stable, re-record the "
    "fixtures through `scripts/codex_gate/record_proxy.py` and `promote` them, "
    "regenerate `extensions.json` with `scripts/codex_gate/extension_inventory.py`, "
    "run `python -m scripts.codex_gate.run_matrix --codex <version>`, then move the pin "
    "(`CODEX_TAG` in `scripts/codex_model_catalog.py`, which the `codex-gate.yml` PR and "
    "nightly pinned jobs follow). An OpenAPI change is re-vendored "
    "with `scripts/vendor_openai_schema.py --ref <sha>` and its divergences reviewed."
)


def _changed(pinned: Mapping[str, Any], head: Mapping[str, Any], label: str) -> list[str]:
    added = sorted(set(head) - set(pinned))
    removed = sorted(set(pinned) - set(head))
    changed = sorted(name for name in set(pinned) & set(head) if pinned[name] != head[name])
    return [
        f"- {label} {kind}: {', '.join(f'`{name}`' for name in names)}"
        for kind, names in (("added", added), ("removed", removed), ("changed", changed))
        if names
    ]


def schema_drift(pinned: Mapping[str, Any], head: Mapping[str, Any]) -> list[str]:
    lines = _changed(pinned["paths"], head["paths"], "paths")
    for section in sorted(set(pinned["components"]) | set(head["components"])):
        lines += _changed(
            pinned["components"].get(section, {}),
            head["components"].get(section, {}),
            f"components.{section}",
        )
    return lines


def head_document(ref: str) -> dict[str, Any]:
    sha = vendor.resolve_sha(ref)
    url = vendor.RAW_URL.format(repo=vendor.REPO, sha=sha, path=vendor.SPEC_PATH)
    with urllib.request.urlopen(url, timeout=120) as response:
        spec = json.loads(response.read())
    return vendor.build_document(spec, sha=sha, source_url=url, generated_on="drift-check")


def codex_drift(pinned: str) -> dict[str, str]:
    """``{dist-tag: version}`` for every tag that differs from the pin."""

    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm is required to read the Codex dist-tags")
    raw = subprocess.run(
        [npm, "view", "@openai/codex", "dist-tags", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    tags = json.loads(raw)
    drifted = {tag: tags[tag] for tag in CODEX_TAGS if tags.get(tag) and tags[tag] != pinned}
    malformed = sorted(v for v in drifted.values() if not _VERSION.fullmatch(str(v)))
    if malformed:
        raise SystemExit(f"unexpected Codex dist-tag versions {malformed}")
    return drifted


def report(pinned_path: Path, ref: str) -> tuple[str, str]:
    """``(title, body)`` of the drift issue, or two empty strings."""

    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    head = head_document(ref)
    schema = schema_drift(pinned, head)
    pinned_codex = CODEX_TAG.removeprefix("rust-v")
    codex = codex_drift(pinned_codex)
    if not schema and not codex:
        return "", ""
    commit = head["provenance"]["commit"][:10]
    state = [f"openapi {commit}"] if schema else []
    state += [f"codex {tag} {version}" for tag, version in codex.items()]
    sections = ["Automated weekly check of the M20 reference pins.", ""]
    if schema:
        sections += [
            f"### OpenAPI: `{pinned['provenance']['commit'][:10]}` → `{commit}`",
            *schema,
            "",
        ]
    if codex:
        sections += [
            f"### Codex (pinned `{pinned_codex}`)",
            *(f"- `{tag}`: `{version}`" for tag, version in codex.items()),
            "",
        ]
    title = "M20 reference drift: " + ", ".join(state)
    return title, "\n".join([*sections, PROCEDURE, ""])


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="main", help="openai/openai-openapi ref to compare")
    parser.add_argument("--title", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    vendored = sorted(VENDORED_DIR.glob("responses-schema@*.json"))
    if len(vendored) != 1:
        raise SystemExit(f"expected one vendored schema in {VENDORED_DIR}, found {vendored}")
    title, body = report(vendored[0], args.ref)
    args.title.write_text(title, encoding="utf-8")
    args.report.write_text(body, encoding="utf-8")
    print(f"{title}\n\n{body}" if title else "no drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
