"""Fail-closed selection and full-run guards shared by every entry point."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from kairyu.evaluation.schemas import RunMode


class RunGuardError(ValueError):
    """A run selection is unsafe, ambiguous, or forbidden in this environment."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RunGuardDecision:
    mode: RunMode
    limit: int | None
    sample_ids: tuple[str, ...]
    official_eligible: bool


def validate_run_guard(
    mode: RunMode | str,
    *,
    confirm_full_run: bool = False,
    limit: int | None = None,
    sample_ids: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> RunGuardDecision:
    """Validate run scope before preparation, model calls, or executor startup."""
    try:
        selected_mode = RunMode(mode)
    except ValueError as exc:
        raise RunGuardError("invalid_mode", f"unsupported benchmark mode: {mode!r}") from exc

    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise RunGuardError("invalid_limit", "limit must be a positive integer")
    selected_ids = tuple(sample_ids)
    if any(not isinstance(item, str) or not item.strip() for item in selected_ids):
        raise RunGuardError("invalid_sample_ids", "sample IDs must be non-blank strings")
    if len(set(selected_ids)) != len(selected_ids):
        raise RunGuardError("duplicate_sample_ids", "sample IDs must be unique")

    environment = os.environ if environ is None else environ
    if selected_mode is RunMode.FULL:
        if _truthy(environment.get("CI")):
            raise RunGuardError("full_forbidden_in_ci", "full benchmark runs are forbidden in CI")
        if limit is not None or selected_ids:
            raise RunGuardError(
                "full_with_subset",
                "full mode cannot be combined with a limit or explicit sample IDs",
            )
        if confirm_full_run is not True:
            raise RunGuardError(
                "full_confirmation_required",
                "full mode requires the explicit confirmation flag",
            )
        if environment.get("BENCHMARK_ALLOW_FULL_RUN") != "1":
            raise RunGuardError(
                "full_environment_opt_in_required",
                "full mode requires BENCHMARK_ALLOW_FULL_RUN=1",
            )
        return RunGuardDecision(
            mode=selected_mode,
            limit=None,
            sample_ids=(),
            official_eligible=True,
        )

    if selected_mode is RunMode.SAMPLE and limit is None and not selected_ids:
        raise RunGuardError(
            "sample_scope_required",
            "sample mode requires a positive limit or explicit sample IDs",
        )
    return RunGuardDecision(
        mode=selected_mode,
        limit=limit,
        sample_ids=selected_ids,
        official_eligible=False,
    )


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}
