"""Validate Responses and Conversations wire output against the pinned OpenAI spec.

``ContractValidator`` maps a recorded HTTP exchange on ``/v1/responses*`` or
``/v1/conversations*`` to its operation in the vendored OpenAPI closure
(``scripts/vendor_openai_schema.py``) and checks what Kairyu sent:

- a JSON success body against the operation's declared schema;
- each SSE event against the stream-event variant named by its ``type``
  (an unknown ``type`` is a violation, and so is an ``event:`` name that
  disagrees with it);
- every error body (status >= 400) against ``ErrorResponse``.

A route under those prefixes with no operation in the spec is a violation too.
Violations are reported per leaf (one per missing or unexpected property), so
``divergences.toml`` can allowlist them by schema, JSON pointer and keyword.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry
from referencing.jsonschema import DRAFT202012

SCHEMA_FILE = Path(__file__).parent / "openai" / "responses-schema@13fa6e7ab9.json"
SCHEMA_URI = "urn:kairyu:contracts:openai-responses"
API_PREFIX = "/v1"
GATED_PREFIXES = ("/v1/responses", "/v1/conversations")
STREAM_ROOT = "ResponseStreamEvent"
ERROR_ROOT = "ErrorResponse"
SSE_DONE = "[DONE]"
JSON_MEDIA = "application/json"
SSE_MEDIA = "text/event-stream"
# Gate checks that are not JSON Schema keywords (``Violation.keyword``).
ROUTE = "route"
EVENT_TYPE = "event-type"
EVENT_NAME = "sse-event-name"
CONTENT_TYPE = "content-type"
JSON_SYNTAX = "json"
_MESSAGE_LIMIT = 240
_VARIANT_REJECTION = frozenset({"const", "enum", "type"})


@dataclass(frozen=True)
class Exchange:
    """One recorded HTTP response, as the client received it."""

    method: str
    path: str
    status: int
    content_type: str
    body: bytes

    @property
    def route(self) -> str:
        return f"{self.method} {self.path} -> {self.status}"


@dataclass(frozen=True)
class Violation:
    """One contract failure.

    ``schema`` is the component (or operation) validated, or ``"route"``;
    ``pointer`` is a JSON pointer into the instance, or ``"METHOD /path"`` for
    a route failure; ``keyword`` is the failing JSON Schema keyword or one of
    the gate checks defined above.
    """

    schema: str
    pointer: str
    keyword: str
    message: str
    route: str


@dataclass(frozen=True)
class ExchangeReport:
    validated: int
    violations: tuple[Violation, ...]


@dataclass(frozen=True)
class _Operation:
    template: str
    method: str
    pattern: re.Pattern[str]
    literal_segments: int
    # status -> media type -> (schema label, JSON pointer of the schema)
    responses: Mapping[str, Mapping[str, tuple[str, str]]]


def is_gated(path: str) -> bool:
    return path.startswith(GATED_PREFIXES)


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer(path: Any) -> str:
    return "".join(f"/{_escape(str(part))}" for part in path)


def _ref_name(schema: Mapping[str, Any]) -> str | None:
    ref = schema.get("$ref")
    return ref.rsplit("/", 1)[-1] if isinstance(ref, str) and len(schema) == 1 else None


def _compile_operations(paths: Mapping[str, Any]) -> tuple[_Operation, ...]:
    operations = []
    for template, methods in paths.items():
        segments = template.strip("/").split("/")
        regex = "/" + "/".join(
            "[^/]+" if part.startswith("{") else re.escape(part) for part in segments
        )
        for method, operation in methods.items():
            responses = {
                status: {
                    media: (
                        _ref_name(schema) or operation["operationId"],
                        "/paths/"
                        + "/".join(
                            _escape(part) for part in (template, method, "responses", status, media)
                        ),
                    )
                    for media, schema in content.items()
                }
                for status, content in operation["responses"].items()
            }
            operations.append(
                _Operation(
                    template=template,
                    method=method.upper(),
                    pattern=re.compile(regex),
                    literal_segments=sum(not part.startswith("{") for part in segments),
                    responses=responses,
                )
            )
    # A literal segment beats a parameter (``/responses/compact`` vs ``/{response_id}``).
    return tuple(sorted(operations, key=lambda op: -op.literal_segments))


def _variant_rejected(error: ValidationError) -> bool:
    """True when a union branch fails on the instance's kind, not its contents."""

    relative = list(error.relative_path)
    return error.validator in _VARIANT_REJECTION and relative in ([], ["type"])


def _leaf_errors(error: ValidationError) -> Iterator[ValidationError]:
    """Descend ``anyOf``/``oneOf`` into the branch that matches the instance's kind."""

    if error.validator in ("anyOf", "oneOf") and error.context:
        branches: dict[int, list[ValidationError]] = {}
        for sub in error.context:
            branches.setdefault(sub.relative_schema_path[0], []).append(sub)
        candidates = [
            errors
            for _, errors in sorted(branches.items())
            if not any(_variant_rejected(sub) for sub in errors)
        ]
        if candidates:
            for sub in min(candidates, key=len):
                yield from _leaf_errors(sub)
            return
    yield error


def _unexpected_properties(error: ValidationError) -> list[str]:
    declared = error.schema.get("properties", {}) if isinstance(error.schema, dict) else {}
    return [name for name in error.instance if name not in declared]


def _violations(label: str, error: ValidationError, route: str) -> Iterator[Violation]:
    pointer = _pointer(error.absolute_path)
    if error.validator == "required":
        # jsonschema raises one error per missing property, named only in the message.
        for name in error.validator_value:
            if error.message == f"{name!r} is a required property":
                yield Violation(label, f"{pointer}/{_escape(name)}", "required", "missing", route)
                return
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        for name in _unexpected_properties(error):
            yield Violation(
                label,
                f"{pointer}/{_escape(name)}",
                "additionalProperties",
                "not declared",
                route,
            )
        return
    yield Violation(label, pointer, str(error.validator), error.message[:_MESSAGE_LIMIT], route)


def _sse_events(text: str) -> Iterator[tuple[str | None, str]]:
    """Yield ``(event name, data)`` per SSE frame that carries data."""

    for frame in re.split(r"\r?\n\r?\n", text):
        name: str | None = None
        data: list[str] = []
        for line in frame.splitlines():
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
        if data:
            yield name, "\n".join(data)


class ContractValidator:
    """Route mapping and schema validation over one vendored spec closure."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        resource = DRAFT202012.create_resource(document)
        self._registry = Registry().with_resource(SCHEMA_URI, resource)
        schemas = document["components"]["schemas"]
        variants = {}
        for ref in schemas[STREAM_ROOT]["anyOf"]:
            name = ref["$ref"].rsplit("/", 1)[-1]
            (event_type,) = schemas[name]["properties"]["type"]["enum"]
            variants[event_type] = name
        self._variants: Mapping[str, str] = variants
        self._operations = _compile_operations(document["paths"])
        self._validators: dict[str, Draft202012Validator] = {}

    def _validator(self, pointer: str) -> Draft202012Validator:
        validator = self._validators.get(pointer)
        if validator is None:
            ref = f"{SCHEMA_URI}#{quote(pointer, safe='/~')}"
            validator = Draft202012Validator({"$ref": ref}, registry=self._registry)
            self._validators[pointer] = validator
        return validator

    def validate_instance(
        self, label: str, pointer: str, instance: Any, route: str
    ) -> tuple[Violation, ...]:
        found = []
        for error in self._validator(pointer).iter_errors(instance):
            for leaf in _leaf_errors(error):
                found.extend(_violations(label, leaf, route))
        return tuple(found)

    def validate_component(self, name: str, instance: Any, route: str) -> tuple[Violation, ...]:
        return self.validate_instance(name, f"/components/schemas/{name}", instance, route)

    def _operation(self, method: str, path: str) -> _Operation | None:
        api_path = path.removeprefix(API_PREFIX)
        for operation in self._operations:
            if operation.method == method and operation.pattern.fullmatch(api_path):
                return operation
        return None

    def validate_event(
        self, name: str | None, data: str, route: str
    ) -> tuple[int, tuple[Violation, ...]]:
        if data == SSE_DONE:
            return 0, ()
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            return 1, (Violation(STREAM_ROOT, "", JSON_SYNTAX, str(error), route),)
        event_type = event.get("type") if isinstance(event, dict) else None
        variant = self._variants.get(event_type) if isinstance(event_type, str) else None
        if variant is None:
            message = f"unknown stream event type {event_type!r}"
            return 1, (Violation(STREAM_ROOT, "/type", EVENT_TYPE, message, route),)
        found = self.validate_component(variant, event, route)
        if name is not None and name != event_type:
            message = f"SSE event name {name!r} differs from type {event_type!r}"
            found = (*found, Violation(variant, "", EVENT_NAME, message, route))
        return 1, found

    def _validate_stream(self, exchange: Exchange) -> ExchangeReport:
        validated = 0
        found: list[Violation] = []
        try:
            text = exchange.body.decode("utf-8")
        except UnicodeDecodeError as error:
            violation = Violation(STREAM_ROOT, "", JSON_SYNTAX, str(error), exchange.route)
            return ExchangeReport(0, (violation,))
        for name, data in _sse_events(text):
            count, violations = self.validate_event(name, data, exchange.route)
            validated += count
            found.extend(violations)
        return ExchangeReport(validated, tuple(found))

    def _validate_json(self, label: str, pointer: str, exchange: Exchange) -> ExchangeReport:
        try:
            body = json.loads(exchange.body)
        except json.JSONDecodeError as error:
            violation = Violation(label, "", JSON_SYNTAX, str(error), exchange.route)
            return ExchangeReport(1, (violation,))
        return ExchangeReport(1, self.validate_instance(label, pointer, body, exchange.route))

    def validate(self, exchange: Exchange) -> ExchangeReport:
        """Check one exchange on a gated route; see the module docstring."""

        media = exchange.content_type.split(";", 1)[0].strip().lower()
        operation = self._operation(exchange.method, exchange.path)
        route_violations: tuple[Violation, ...] = ()
        if operation is None:
            locator = f"{exchange.method} {exchange.path}"
            message = "no operation for this method and path in the pinned spec"
            route_violations = (Violation(ROUTE, locator, ROUTE, message, exchange.route),)
        if exchange.status >= 400:
            report = self._check_media(ERROR_ROOT, media, JSON_MEDIA, exchange) or (
                self._validate_json(ERROR_ROOT, f"/components/schemas/{ERROR_ROOT}", exchange)
            )
        elif operation is None:
            report = ExchangeReport(0, ())
        else:
            report = self._validate_success(operation, media, exchange)
        return ExchangeReport(report.validated, route_violations + report.violations)

    def _validate_success(
        self, operation: _Operation, media: str, exchange: Exchange
    ) -> ExchangeReport:
        declared = operation.responses.get(str(exchange.status), {})
        if media == SSE_MEDIA and SSE_MEDIA in declared:
            return self._validate_stream(exchange)
        if media == JSON_MEDIA and JSON_MEDIA in declared:
            label, pointer = declared[JSON_MEDIA]
            return self._validate_json(label, pointer, exchange)
        message = f"{media!r} is not declared for status {exchange.status}"
        violation = Violation(operation.template, "", CONTENT_TYPE, message, exchange.route)
        return ExchangeReport(0, (violation,))

    @staticmethod
    def _check_media(
        label: str, media: str, expected: str, exchange: Exchange
    ) -> ExchangeReport | None:
        if media == expected:
            return None
        message = f"expected {expected!r}, got {media!r}"
        return ExchangeReport(0, (Violation(label, "", CONTENT_TYPE, message, exchange.route),))


@cache
def contract_validator() -> ContractValidator:
    """The process-wide validator over the vendored closure (built on first use)."""

    return ContractValidator(json.loads(SCHEMA_FILE.read_text(encoding="utf-8")))
