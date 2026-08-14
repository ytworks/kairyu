"""Checkout-only endpoint targets shared by verification workloads."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

TARGET_SPEC_FORMAT = "name=base_url=model[=api_key_env]"
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_base_url(base_url: str) -> str:
    """Return a canonical OpenAI-compatible API root ending in ``/v1``."""

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
    if api_key_env is None:
        return None
    if not isinstance(api_key_env, str):
        raise ValueError("api_key_env must be an environment-variable name")
    name = api_key_env.strip()
    if _ENV_NAME_RE.fullmatch(name) is None:
        raise ValueError("api_key_env must be an environment-variable name")
    return name


@dataclass(frozen=True, slots=True)
class BenchTarget:
    """One endpoint/model pair used by a verification workload."""

    base_url: str
    model: str
    name: str = ""
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "api_key_env", validate_api_key_env(self.api_key_env))

    def label(self) -> str:
        return self.name or self.model


def parse_target_spec(spec: str, **sampling: object) -> BenchTarget:
    """Parse ``name=base_url=model[=api_key_env]``."""

    if sampling:
        unsupported = ", ".join(sorted(sampling))
        raise ValueError(f"verification targets do not support sampling fields: {unsupported}")
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
        return BenchTarget(
            name=name,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
        )
    except ValueError as error:
        raise ValueError(f"--target: {error}") from error


def resolve_api_key_env(
    api_key_env: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    required: bool = False,
) -> str | None:
    name = validate_api_key_env(api_key_env)
    if name is None:
        return None
    source = os.environ if environ is None else environ
    value = source.get(name)
    if value:
        return value
    if required:
        raise ValueError(f"API key environment variable {name!r} is not set or is empty")
    return None


def target_api_key(
    target: BenchTarget,
    *,
    environ: Mapping[str, str] | None = None,
    required: bool | None = None,
) -> str | None:
    if required is None:
        required = target.api_key_env is not None
    return resolve_api_key_env(
        target.api_key_env,
        environ=environ,
        required=required,
    )


__all__ = [
    "BenchTarget",
    "TARGET_SPEC_FORMAT",
    "normalize_base_url",
    "parse_target_spec",
    "resolve_api_key_env",
    "target_api_key",
    "validate_api_key_env",
]
