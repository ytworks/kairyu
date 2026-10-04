"""Startup check of tenant token budgets against remaining-context reservations."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from kairyu.engine.backend import backend_max_model_len
from kairyu.entrypoints.server.tenancy import TenantConfig

logger = logging.getLogger(__name__)


def unreachable_output_budgets(
    config: TenantConfig,
    engines: Mapping[str, object],
) -> tuple[str, ...]:
    """Describe tenant/model pairs whose omitted-output reservation never fits.

    Chat and Responses requests without an output cap reserve the model's whole
    ``max_model_len`` for decode (OpenAI remaining-context semantics, #496) on
    top of the prompt. A token bucket that cannot hold that reservation rejects
    every such request: HTTP callers get a 429 with a retry hint, async
    submissions ``token_request_too_large``. The check is a lower bound, since
    the prompt's own reservation is unknown at startup.
    """

    tenants = sorted({config.default_tenant, *config.key_tenants.values(), *config.limits})
    lengths = sorted(
        (model, length)
        for model, engine in engines.items()
        if (length := backend_max_model_len(engine)) is not None
    )
    return tuple(
        f"tenant {tenant!r} token budget ({capacity}) cannot hold model {model!r} "
        f"max_model_len ({length}): requests that omit max_tokens/max_output_tokens "
        "reserve at least max_model_len and are always rejected with 429; raise the "
        "tenant's tokens_per_minute/token_burst or have clients send an output cap"
        for tenant in tenants
        for capacity in (_token_capacity(config, tenant),)
        for model, length in lengths
        if capacity <= length
    )


def warn_unreachable_output_budgets(
    config: TenantConfig,
    engines: Mapping[str, object],
) -> None:
    for message in unreachable_output_budgets(config, engines):
        logger.warning(message)


def _token_capacity(config: TenantConfig, tenant: str) -> int:
    limits = config.limits_for(tenant)
    return limits.tokens_per_minute if limits.token_burst is None else limits.token_burst
