"""Chat Completions error currency shared by the HTTP and worker surfaces.

``ChatRequestError`` is the tenant-safe failure the Chat handler, the batch
and async workers, and the Messages/Responses adapters exchange. Its
constructors consult ``error_classifier`` first, so a prompt overflow carries
``context_length_exceeded`` and ``param: "messages"`` wherever it is raised.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kairyu.engine.backend import UpstreamClientError
from kairyu.entrypoints.server.error_classifier import (
    ClassifiedError,
    classify_request_error,
    pre_stream_error,
)

if TYPE_CHECKING:
    from kairyu.entrypoints.server.chat_service import ExecutedChat

logger = logging.getLogger(__name__)


class ChatRequestError(Exception):
    """A controlled request-boundary failure safe to return to a tenant."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "invalid_request",
        error_type: str = "invalid_request_error",
        param: str | None = None,
        execution: ExecutedChat | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.param = param
        self.execution = execution

    @classmethod
    def from_classified(cls, classified: ClassifiedError) -> ChatRequestError:
        return cls(
            classified.message,
            status_code=classified.status,
            code=classified.code,
            error_type=classified.type,
            param=classified.param,
        )

    def classified(self) -> ClassifiedError:
        """This failure for the shared renderers (WP-07)."""

        return pre_stream_error(
            self.status_code, self.error_type, self.code, str(self), param=self.param
        )

    def payload(self) -> dict:
        """The OpenAI error member; ``param`` is always present (nullable)."""

        return self.classified().openai_payload()


def chat_error_from_upstream_client_error(
    error: UpstreamClientError,
) -> ChatRequestError:
    """Translate a backend 4xx without exposing arbitrary upstream text."""

    classified = classify_request_error(error, "chat")
    if classified is not None:
        return ChatRequestError.from_classified(classified)
    if error.public_message is None:
        logger.warning(
            "OpenAI-compatible upstream rejected a request",
            exc_info=error,
        )
        return ChatRequestError(
            "upstream backend rejected the request",
            status_code=502,
            code="backend_error",
            error_type="upstream_error",
        )
    return ChatRequestError(
        error.public_message,
        status_code=error.status_code,
        code=getattr(error, "code", "invalid_request"),
    )


def chat_error_from_value_error(error: ValueError) -> ChatRequestError:
    """Translate a request-validation failure, classifying prompt overflow."""

    classified = classify_request_error(error, "chat")
    if classified is not None:
        return ChatRequestError.from_classified(classified)
    return ChatRequestError(str(error), code=getattr(error, "code", "invalid_request"))


def chat_stream_error_payload(error: BaseException) -> dict:
    """The in-band SSE error frame after Chat stream headers were sent."""

    classified = classify_request_error(error, "chat")
    if classified is not None:
        return {"error": classified.openai_payload()}
    # M3: only the class name, no raw backend message.
    return {
        "error": {
            "message": f"upstream backend error ({type(error).__name__})",
            "type": "upstream_error",
        }
    }
