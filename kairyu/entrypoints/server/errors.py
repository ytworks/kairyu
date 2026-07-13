"""Shared OpenAI-style HTTP error responses."""

from fastapi.responses import JSONResponse


def invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        },
    )
