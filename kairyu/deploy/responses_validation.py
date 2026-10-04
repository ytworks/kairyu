"""Responses API deployment preflight (m20 C22, D6).

Every gateway seals Kairyu-issued Responses items (compaction tokens; reasoning
from WP-21) with the secret in ``server.responses_compaction_secret_env``.
Without one each process uses its own ephemeral key, so a restart, a rolling
update or a hop to another gateway breaks every compacted Codex session, at
any replica count.

Staged rule: this release warns (``kairyu validate`` and startup) unless the
deployment sets a secret or opts in with ``server.sealing.ephemeral: true``;
the next release refuses to start instead. Direct ``create_app`` callers keep
the ephemeral key without a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kairyu.deploy.spec import DeploymentSpec

logger = logging.getLogger(__name__)

SEALING_SECRET_MISSING = "schema.sealing_secret_missing"
_SEALING_SECRET_FIELD = "server.responses_compaction_secret_env"
_SEALING_SECRET_MESSAGE = (
    "no Responses sealing secret: each process seals compaction tokens with its "
    "own ephemeral key, so restarts, rolling updates and gateway hops break "
    "compacted Codex sessions; give every gateway the same secret or opt in "
    "with server.sealing.ephemeral: true (the next release refuses to start "
    "without either)"
)


@dataclass(frozen=True)
class PreflightWarning:
    """One privacy-safe deployment warning; never carries configured values."""

    field: str
    code: str
    message: str


def responses_preflight_warnings(spec: DeploymentSpec) -> tuple[PreflightWarning, ...]:
    server = spec.server
    if server.responses_compaction_secret_env is not None or server.sealing.ephemeral:
        return ()
    return (
        PreflightWarning(
            field=_SEALING_SECRET_FIELD,
            code=SEALING_SECRET_MISSING,
            message=_SEALING_SECRET_MESSAGE,
        ),
    )


def log_responses_preflight_warnings(spec: DeploymentSpec) -> None:
    """Startup half of the staged rule; ``kairyu validate`` reports the same."""

    for warning in responses_preflight_warnings(spec):
        logger.warning("%s (%s): %s", warning.field, warning.code, warning.message)
