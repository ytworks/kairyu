"""Framework errors in the Responses dialect (M20 WP-07, Principle 3).

FastAPI answers a schema error with 422 ``{"detail": [...]}`` and Starlette an
unknown path, a wrong method or an undecodable body with ``{"detail": ...}``.
On the Responses-dialect paths (``/v1/responses*``, ``/v1/conversations*``
and Codex's ``/v1/alpha/search``) they are the OpenAI envelope instead:

* 422 -> 400 with ``param`` joined from the first failing field's deepest
  error location (``input[0]`` for a list sent to ``str | list[...]``, not the
  ``str`` branch's error);
* malformed or undecodable JSON -> 400 ``invalid_json``;
* 404 -> ``Invalid URL (METHOD /path)`` and 405 likewise, keeping ``Allow``.

The handlers are app-wide (FastAPI has no per-router handlers) and every other
path keeps FastAPI's default rendering byte for byte.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException

from kairyu.entrypoints.server.error_classifier import (
    ClassifiedError,
    pre_stream_error,
    speaks_responses_dialect,
)
from kairyu.entrypoints.server.errors import classified_response

_INVALID = "invalid_request_error"
_INVALID_JSON = "We could not parse the JSON body of your request."
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Pydantic names the union member it tried ("str", "list[dict[any,any]]") in
# the error location; the member is not part of the request's parameter path.
_SCALAR_UNION_MEMBERS = frozenset({"str", "int", "float", "bool", "bytes", "none"})


def install_scoped_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(HTTPException, _http_error)


def renders_default_validation(handler: Callable | None, path: str) -> bool:
    """Whether ``handler`` answers a validation error on ``path`` with FastAPI's body."""

    return handler is request_validation_exception_handler or (
        handler is _validation_error and not speaks_responses_dialect(path)
    )


async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
    if not speaks_responses_dialect(request.url.path):
        return await request_validation_exception_handler(request, exc)
    return classified_response(_schema_error(exc.errors()), responses=True)


async def _http_error(request: Request, exc: HTTPException) -> Response:
    if not speaks_responses_dialect(request.url.path):
        return await http_exception_handler(request, exc)
    response = classified_response(_framework_error(request, exc), responses=True)
    response.headers.update(exc.headers or {})
    return response


def _framework_error(request: Request, exc: HTTPException) -> ClassifiedError:
    target = f"{request.method} {request.url.path}"
    if exc.status_code == 404:
        return pre_stream_error(404, _INVALID, None, f"Invalid URL ({target})")
    if exc.status_code == 405:
        return pre_stream_error(405, _INVALID, None, f"Method not allowed ({target})")
    if exc.status_code == 400:
        # FastAPI's "There was an error parsing the body" (e.g. undecoded zstd).
        return pre_stream_error(400, _INVALID, "invalid_json", _INVALID_JSON)
    error_type = _INVALID if exc.status_code < 500 else "server_error"
    return pre_stream_error(exc.status_code, error_type, None, str(exc.detail))


def _schema_error(errors: Sequence[Mapping]) -> ClassifiedError:
    error = _most_specific(errors)
    kind = str(error.get("type", ""))
    if kind == "json_invalid":
        return pre_stream_error(400, _INVALID, "invalid_json", _INVALID_JSON)
    param = _param_from_loc(error.get("loc", ()))
    if kind == "missing":
        message = f"Missing required parameter: '{param}'." if param else "Missing request body."
        return pre_stream_error(400, _INVALID, "missing_required_parameter", message, param=param)
    code = "invalid_type" if kind.endswith(("_type", "_parsing")) else "invalid_value"
    subject = f"value for '{param}'" if param else "request body"
    message = f"Invalid {subject}: {error.get('msg', 'invalid')}."
    return pre_stream_error(400, _INVALID, code, message, param=param)


def _most_specific(errors: Sequence[Mapping]) -> Mapping:
    """The deepest error of the first failing field.

    Pydantic reports every branch of a union field; the deepest one is the
    branch that matched the JSON type and failed inside it.
    """

    if not errors:
        return {}
    field = _param_segments(errors[0].get("loc", ()))[:1]
    candidates = [e for e in errors if _param_segments(e.get("loc", ()))[:1] == field]
    return max(candidates, key=lambda e: len(_param_segments(e.get("loc", ()))))


def _param_from_loc(loc: Sequence[str | int]) -> str | None:
    """``("body", "tools", 0, "name")`` -> ``"tools[0].name"``."""

    param = ""
    for segment in _param_segments(loc):
        if isinstance(segment, int):
            param += f"[{segment}]"
        else:
            param += f".{segment}" if param else segment
    return param or None


def _param_segments(loc: Sequence[str | int]) -> tuple[str | int, ...]:
    """The request-parameter path of a location: no ``body``, no union labels."""

    segments = tuple(loc)
    if segments[:1] == ("body",):
        segments = segments[1:]
    return tuple(
        segment
        for segment in segments
        if isinstance(segment, int)
        or (segment not in _SCALAR_UNION_MEMBERS and _FIELD.fullmatch(segment))
    )
