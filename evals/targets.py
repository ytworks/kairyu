"""Shared target parsing, URL normalization, and credential resolution.

Benchmark configuration records environment-variable *names*, never resolved
secret values.  The same target grammar is used by the installed benchmark
suites and the repository-only comparison harnesses:

``name=base_url=model[=api_key_env]``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from evals.types import BenchTarget

TARGET_SPEC_FORMAT = "name=base_url=model[=api_key_env]"
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_base_url(base_url: str) -> str:
    """Return the canonical OpenAI-compatible API root ending in ``/v1``."""
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    root = base_url.strip().rstrip("/")
    if not root:
        raise ValueError("base_url must not be empty")
    parsed = urlsplit(root)
    if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain userinfo credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    return root if root.endswith("/v1") else f"{root}/v1"


def validate_api_key_env(api_key_env: str | None) -> str | None:
    """Validate and canonicalize a credential environment-variable name."""

    if api_key_env is None:
        return None
    if not isinstance(api_key_env, str):
        raise ValueError("api_key_env must be an environment-variable name")
    name = api_key_env.strip()
    if _ENV_NAME_RE.fullmatch(name) is None:
        raise ValueError("api_key_env must be an environment-variable name")
    return name


def parse_target_spec(spec: str, **sampling: object) -> BenchTarget:
    """Parse ``name=base_url=model[=api_key_env]`` into a :class:`BenchTarget`."""
    from evals.types import BenchTarget

    if not isinstance(spec, str):
        raise ValueError(f"--target: expected {TARGET_SPEC_FORMAT}")
    parts = spec.split("=")
    if len(parts) not in (3, 4):
        raise ValueError(f"--target: expected {TARGET_SPEC_FORMAT}")

    name, base_url, model = (part.strip() for part in parts[:3])
    api_key_env = parts[3].strip() if len(parts) == 4 else None
    if not name or not base_url or not model:
        raise ValueError("--target: name, base_url, and model must be non-empty")
    try:
        api_key_env = validate_api_key_env(api_key_env)
    except ValueError as error:
        raise ValueError(f"--target: {error}") from error

    return BenchTarget(
        name=name,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        **sampling,
    )


def resolve_api_key_env(
    api_key_env: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    required: bool = False,
) -> str | None:
    """Resolve one API-key environment variable without exposing its value.

    ``required`` is useful for executable wrappers: once a user explicitly
    names a credential variable, silently issuing an unauthenticated request is
    almost certainly a configuration error.  Library callers retain the former
    optional behavior by leaving it false.
    """
    api_key_env = validate_api_key_env(api_key_env)
    if api_key_env is None:
        return None
    source = os.environ if environ is None else environ
    value = source.get(api_key_env)
    if value:
        return value
    if required:
        raise ValueError(f"API key environment variable {api_key_env!r} is not set or is empty")
    return None


def target_api_key(
    target: BenchTarget,
    *,
    environ: Mapping[str, str] | None = None,
    required: bool | None = None,
) -> str | None:
    """Resolve ``target.api_key_env`` while keeping the secret out of the target.

    An explicitly configured variable fails closed by default. Callers must
    pass ``required=False`` to opt into unauthenticated fallback.
    """
    if required is None:
        required = target.api_key_env is not None
    return resolve_api_key_env(
        target.api_key_env,
        environ=environ,
        required=required,
    )


__all__ = [
    "TARGET_SPEC_FORMAT",
    "normalize_base_url",
    "parse_target_spec",
    "resolve_api_key_env",
    "target_api_key",
    "validate_api_key_env",
]
