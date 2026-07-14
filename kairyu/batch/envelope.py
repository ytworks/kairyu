"""Typed OpenAI batch-line envelope validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class BatchLineEnvelope(BaseModel):
    """One immutable request line, validated against its owning batch job."""

    model_config = ConfigDict(frozen=True)

    custom_id: str
    method: Literal["POST"]
    url: str
    body: dict

    @field_validator("custom_id")
    @classmethod
    def custom_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("custom_id must be a non-empty string")
        return value

    @field_validator("url")
    @classmethod
    def url_must_match_batch_endpoint(
        cls, value: str, info: ValidationInfo
    ) -> str:
        context = info.context if isinstance(info.context, dict) else {}
        endpoint = context.get("endpoint")
        if endpoint is None:
            raise ValueError("batch endpoint context is required")
        if value != endpoint:
            raise ValueError(f"url must match batch endpoint {endpoint!r}")
        return value
