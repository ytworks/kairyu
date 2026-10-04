#!/usr/bin/env python3
"""Generate a Codex ``model_catalog_json`` for a Kairyu deployment (M20 WP-05, D22).

Codex picks per-model wire behavior from catalog metadata. A model it does not
know gets fallback metadata: a 272,000-token context window (so a smaller model
overflows before Codex compacts), image input, reasoning summaries and the
bundled base instructions. This script writes the Codex-native
``{"models": [ModelInfo, ...]}`` file that ``model_catalog_json`` loads, as
codex-rs ``protocol/src/openai_models.rs`` (``ModelsResponse``) decodes it at
rust-v0.160.0; 0.147.0 and 0.153.4 decode the same file.

Sources (exactly one):

- ``--base-url URL``: a running deployment's ``GET {URL}/models``; each entry's
  ``max_model_len`` becomes ``context_window`` (entries without one, such as
  embedding models, need ``--context-window`` or are skipped);
- ``--static FILE``: YAML or JSON ``{"models": [{"slug", "context_window",
  "vision", "reasoning_levels", "default_reasoning_level"}]}``.

Per-model options: ``--model`` (restrict), ``--context-window ID=N``,
``--reasoning-levels ID=low,high`` (only levels the model honors; default
none, so Codex sends no effort), ``--default-reasoning-level ID=LEVEL`` and
``--vision ID`` (the deployment serves images for ID).

What Kairyu's ``/v1/responses`` does not execute yet stays off whatever the
deployment declares (``SERVER_CAPABILITIES``; the work package that delivers a
capability flips it, m20 DoD 7): image input (WP-25), the freeform
``apply_patch`` tool (WP-22) and reasoning summaries (WP-21).

A catalog entry must carry Codex's base instructions. They are fetched from
the Codex tag (``--codex-tag``, the same ``models-manager/prompt.md`` Codex
uses for unknown models) unless ``--base-instructions FILE`` supplies them.

Example::

    python -m scripts.codex_model_catalog --base-url http://kairyu:8000/v1 \\
        --output ~/.codex/kairyu-models.json
    # ~/.codex/config.toml: model_catalog_json = "/home/me/.codex/kairyu-models.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from kairyu.entrypoints.server.protocol import normalize_reasoning_effort

CODEX_TAG = "rust-v0.160.0"
BASE_INSTRUCTIONS_URL = (
    "https://raw.githubusercontent.com/openai/codex/{tag}/codex-rs/models-manager/prompt.md"
)
# Codex clamps the auto-compaction limit to 90 % of the context window.
AUTO_COMPACT_NUMERATOR, AUTO_COMPACT_DENOMINATOR = 9, 10
# Codex's own fallback tool-output truncation (models-manager model_info.rs).
TOOL_OUTPUT_TRUNCATION = {"mode": "bytes", "limit": 10_000}
HTTP_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ServerCapabilities:
    """Codex-visible behavior ``/v1/responses`` executes (flipped per WP, DoD 7)."""

    image_input: bool = False  # WP-25: input_image / view_image tool output
    apply_patch_freeform: bool = False  # WP-22: custom tools, custom_tool_call items
    reasoning_summaries: bool = False  # WP-21: reasoning items and summaries


SERVER_CAPABILITIES = ServerCapabilities()


class CatalogError(ValueError):
    """The deployment or the options cannot produce a valid catalog."""


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    context_window: int
    vision: bool = False
    reasoning_levels: tuple[str, ...] = ()
    default_reasoning_level: str | None = None

    def __post_init__(self) -> None:
        if not self.slug:
            raise CatalogError("a model slug must be non-empty")
        if type(self.context_window) is not int or self.context_window < 1:
            raise CatalogError(f"{self.slug}: context_window must be a positive integer")
        for level in self.reasoning_levels:
            _check_effort(self.slug, level)
        if self.default_reasoning_level is not None:
            if self.default_reasoning_level not in self.reasoning_levels:
                raise CatalogError(
                    f"{self.slug}: default reasoning level {self.default_reasoning_level!r} "
                    "is not one of its reasoning levels"
                )


def _check_effort(slug: str, level: str) -> None:
    # Kairyu's own effort vocabulary decides; a level it rejects would turn
    # every Codex request into a 400.
    try:
        normalize_reasoning_effort(level)
    except ValueError as error:
        raise CatalogError(f"{slug}: {error}") from error


def model_info(
    spec: ModelSpec,
    base_instructions: str,
    capabilities: ServerCapabilities = SERVER_CAPABILITIES,
    priority: int = 0,
) -> dict[str, Any]:
    """One Codex ``ModelInfo`` (required keys plus the wire-relevant ones)."""

    modalities = ["text", "image"] if spec.vision and capabilities.image_input else ["text"]
    info: dict[str, Any] = {
        "slug": spec.slug,
        "display_name": spec.slug,
        "description": f"{spec.slug} served by Kairyu",
        "supported_reasoning_levels": [
            {"effort": level, "description": f"Kairyu reasoning effort {level}"}
            for level in spec.reasoning_levels
        ],
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": base_instructions,
        "supports_reasoning_summary_parameter": capabilities.reasoning_summaries,
        "default_reasoning_summary": "auto" if capabilities.reasoning_summaries else "none",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "freeform" if capabilities.apply_patch_freeform else None,
        "truncation_policy": dict(TOOL_OUTPUT_TRUNCATION),
        # Read only by Codex before 0.153 (removed there); false keeps the
        # 0.147 request shape (parallel_tool_calls:false).
        "supports_parallel_tool_calls": False,
        "context_window": spec.context_window,
        "max_context_window": spec.context_window,
        "auto_compact_token_limit": (
            spec.context_window * AUTO_COMPACT_NUMERATOR // AUTO_COMPACT_DENOMINATOR
        ),
        "experimental_supported_tools": [],
        "input_modalities": modalities,
        "use_responses_lite": False,
    }
    if spec.default_reasoning_level is not None:
        info["default_reasoning_level"] = spec.default_reasoning_level
    return info


def build_catalog(
    specs: Sequence[ModelSpec],
    base_instructions: str,
    capabilities: ServerCapabilities = SERVER_CAPABILITIES,
) -> dict[str, list[dict[str, Any]]]:
    if not specs:
        raise CatalogError("Codex requires at least one model in model_catalog_json")
    slugs = [spec.slug for spec in specs]
    if len(set(slugs)) != len(slugs):
        raise CatalogError(f"duplicate model slugs: {sorted(slugs)}")
    if not base_instructions.strip():
        raise CatalogError("base instructions are empty; Codex would send no instructions")
    return {
        "models": [
            model_info(spec, base_instructions, capabilities, priority)
            for priority, spec in enumerate(specs)
        ]
    }


def fetch_models(base_url: str, api_key: str | None) -> Any:
    """The JSON body of ``GET {base_url}/models``."""

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def served_models(payload: Mapping[str, Any]) -> dict[str, int | None]:
    """``{id: max_model_len}`` from an OpenAI ``/v1/models`` list body."""

    data = payload.get("data")
    if not isinstance(data, list):
        raise CatalogError("GET /models did not return an OpenAI model list ('data')")
    models: dict[str, int | None] = {}
    for entry in data:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise CatalogError(f"malformed model entry {entry!r}")
        length = entry.get("max_model_len")
        models[entry["id"]] = length if type(length) is int and length > 0 else None
    return models


def specs_from_served(
    models: Mapping[str, int | None], context_windows: Mapping[str, int]
) -> list[ModelSpec]:
    specs = []
    for slug, length in models.items():
        window = context_windows.get(slug, length)
        if window is None:
            print(
                f"skipping {slug!r}: /models advertises no max_model_len "
                "(pass --context-window ID=N to include it)",
                file=sys.stderr,
            )
            continue
        specs.append(ModelSpec(slug=slug, context_window=window))
    return specs


def specs_from_static(path: Path) -> list[ModelSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), list):
        raise CatalogError(f"{path}: expected a mapping with a 'models' list")
    allowed = {"slug", "context_window", "vision", "reasoning_levels", "default_reasoning_level"}
    specs = []
    for entry in payload["models"]:
        if not isinstance(entry, Mapping) or set(entry) - allowed:
            raise CatalogError(f"{path}: model entries take only {sorted(allowed)}: {entry!r}")
        specs.append(
            ModelSpec(
                slug=str(entry.get("slug", "")),
                context_window=entry.get("context_window"),
                vision=bool(entry.get("vision", False)),
                reasoning_levels=tuple(entry.get("reasoning_levels") or ()),
                default_reasoning_level=entry.get("default_reasoning_level"),
            )
        )
    return specs


def _pairs(values: Iterable[str], flag: str) -> dict[str, str]:
    pairs = {}
    for value in values:
        slug, separator, setting = value.partition("=")
        if not separator or not slug or not setting:
            raise CatalogError(f"{flag} expects ID=VALUE, got {value!r}")
        pairs[slug] = setting
    return pairs


def apply_options(specs: Sequence[ModelSpec], args: argparse.Namespace) -> list[ModelSpec]:
    """Restrict and annotate the specs with the per-model command-line options."""

    levels = {
        slug: tuple(level for level in setting.split(",") if level)
        for slug, setting in _pairs(args.reasoning_levels, "--reasoning-levels").items()
    }
    defaults = _pairs(args.default_reasoning_level, "--default-reasoning-level")
    vision = set(args.vision)
    known = {spec.slug for spec in specs}
    unknown = (set(args.model) | set(levels) | set(defaults) | vision) - known
    if unknown:
        raise CatalogError(f"options name models the source does not list: {sorted(unknown)}")
    selected = [spec for spec in specs if not args.model or spec.slug in args.model]
    return [
        replace(
            spec,
            vision=spec.vision or spec.slug in vision,
            reasoning_levels=levels.get(spec.slug, spec.reasoning_levels),
            default_reasoning_level=defaults.get(spec.slug, spec.default_reasoning_level),
        )
        for spec in selected
    ]


def load_base_instructions(path: Path | None, codex_tag: str) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    url = BASE_INSTRUCTIONS_URL.format(tag=codex_tag)
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:
            return response.read().decode("utf-8")
    except OSError as error:
        raise CatalogError(
            f"could not fetch Codex base instructions from {url} ({error}); "
            "pass --base-instructions FILE"
        ) from error


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url", help="Kairyu OpenAI base URL, e.g. http://host:8000/v1")
    source.add_argument("--static", type=Path, help="YAML/JSON model list (no server needed)")
    parser.add_argument(
        "--api-key-env",
        default="KAIRYU_API_KEY",
        help="environment variable with the Kairyu API key for GET /models (if set)",
    )
    parser.add_argument("--model", action="append", default=[], metavar="ID")
    parser.add_argument("--context-window", action="append", default=[], metavar="ID=N")
    parser.add_argument("--reasoning-levels", action="append", default=[], metavar="ID=L1,L2")
    parser.add_argument(
        "--default-reasoning-level", action="append", default=[], metavar="ID=LEVEL"
    )
    parser.add_argument("--vision", action="append", default=[], metavar="ID")
    parser.add_argument("--base-instructions", type=Path, metavar="FILE")
    parser.add_argument("--codex-tag", default=CODEX_TAG)
    parser.add_argument("--output", type=Path, help="write here instead of stdout")
    return parser.parse_args(argv)


def generate(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[ModelSpec]]:
    windows = {
        slug: int(value) for slug, value in _pairs(args.context_window, "--context-window").items()
    }
    if args.base_url:
        payload = fetch_models(args.base_url, os.environ.get(args.api_key_env) or None)
        served = served_models(payload)
        unlisted = set(windows) - set(served)
        specs = specs_from_served(served, windows)
    else:
        static = specs_from_static(args.static)
        unlisted = set(windows) - {spec.slug for spec in static}
        specs = [
            replace(spec, context_window=windows.get(spec.slug, spec.context_window))
            for spec in static
        ]
    if unlisted:
        raise CatalogError(f"--context-window names unlisted models: {sorted(unlisted)}")
    selected = apply_options(specs, args)
    instructions = load_base_instructions(args.base_instructions, args.codex_tag)
    return build_catalog(selected, instructions), selected


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        catalog, specs = generate(args)
    except (CatalogError, OSError, ValueError) as error:
        print(f"codex_model_catalog: {error}", file=sys.stderr)
        return 2
    text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if any(spec.vision for spec in specs) and not SERVER_CAPABILITIES.image_input:
        print(
            "note: declared vision stays text-only until /v1/responses accepts images (WP-25)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
