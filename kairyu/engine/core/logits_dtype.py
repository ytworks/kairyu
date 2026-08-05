"""Public logits-output dtype validation and startup resolution."""

from __future__ import annotations

SUPPORTED_LOGITS_DTYPES = ("model", "float32")
_SUPPORTED_RESOLVED_LOGITS_DTYPES = frozenset(
    {"float16", "bfloat16", "float32"}
)


def validate_logits_dtype(requested: object) -> str:
    """Return one canonical public request without importing torch."""

    if not isinstance(requested, str) or requested not in SUPPORTED_LOGITS_DTYPES:
        choices = ", ".join(SUPPORTED_LOGITS_DTYPES)
        raise ValueError(
            f"logits_dtype must be one of: {choices}; got {requested!r}"
        )
    return requested


def resolve_logits_dtype(requested: object, model_dtype: object) -> str:
    """Resolve a request to the dtype the model's logits call will return."""

    requested = validate_logits_dtype(requested)
    if requested == "float32":
        return "float32"
    resolved = str(model_dtype).removeprefix("torch.")
    if resolved not in _SUPPORTED_RESOLVED_LOGITS_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_RESOLVED_LOGITS_DTYPES))
        raise ValueError(
            "model logits dtype must resolve to one of: "
            f"{supported}; got {model_dtype!r}"
        )
    return resolved


__all__ = [
    "SUPPORTED_LOGITS_DTYPES",
    "resolve_logits_dtype",
    "validate_logits_dtype",
]
