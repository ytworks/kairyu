#!/usr/bin/env python3
"""Fail unless an OpenAI-compatible server exposes the requested model ID."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import httpx


class ModelPreflightError(RuntimeError):
    """A safe-to-print production model preflight failure."""


def _served_model_ids(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        raise ModelPreflightError("model list response must be a JSON object")

    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelPreflightError("model list response must contain a data list")

    served_ids: list[str] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ModelPreflightError(
                "model list response data entries must contain string IDs"
            )
        served_ids.append(entry["id"])
    return tuple(served_ids)


def check_served_model(
    base_url: str,
    requested_model: str,
    *,
    client: httpx.Client,
) -> tuple[str, ...]:
    """Return served IDs when ``requested_model`` is present by exact equality."""
    try:
        response = client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as error:
        raise ModelPreflightError(
            f"model list request failed ({type(error).__name__})"
        ) from None

    if not 200 <= response.status_code < 300:
        raise ModelPreflightError(
            f"model list request returned HTTP {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError:
        raise ModelPreflightError("model list response is not valid JSON") from None

    served_ids = _served_model_ids(payload)
    if requested_model not in served_ids:
        raise ModelPreflightError(
            f"requested model {requested_model!r} is not served; "
            f"served model IDs: {list(served_ids)!r}"
        )
    return served_ids


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that an OpenAI-compatible server exposes a model ID."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        with httpx.Client(timeout=10.0) as client:
            check_served_model(args.base_url, args.model, client=client)
    except ModelPreflightError as error:
        print(f"model preflight failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
