"""ASGI recording for the schema gate and the opt-in wire capture.

``install_recorders`` wraps the app of every in-process test transport
(``starlette.testclient.TestClient`` and ``httpx.ASGITransport``), so every
app a test builds -- ``create_legacy_app``, a direct ``create_app`` or any
other builder -- is recorded without editing the test. The wrapper only
observes ASGI messages and forwards them unchanged.

``ContractGate`` owns the recording of the running test:

- exchanges on ``/v1/responses*`` and ``/v1/conversations*`` are validated by
  ``ContractValidator`` when the test ends, and violations that no
  ``divergences.toml`` entry allows fail the test;
- with ``KAIRYU_WIRE_CAPTURE=<dir>``, every exchange on every route (Chat,
  Messages, Responses, ...) is also written to ``<dir>`` for the byte-identity
  gate of refactor-only work packages (``tests/contracts/wire_capture.py``).
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

import httpx

from tests.contracts.divergences import Divergence, triage
from tests.contracts.openai_contract import Violation, contract_validator, is_gated
from tests.contracts.wire_capture import WireCapture, WireRecord

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


@dataclass(frozen=True)
class GateOutcome:
    """What the gate observed during one test."""

    validated: int
    unallowed: tuple[Violation, ...]
    matched: frozenset[str]


class ContractGate:
    """Single-owner recorder for the running test plus session bookkeeping.

    Requests may be served on a TestClient portal thread, so ``record`` is
    thread-safe; ``begin``/``finish`` run on the pytest thread.
    """

    def __init__(self, divergences: tuple[Divergence, ...], capture: WireCapture | None) -> None:
        self.divergences = divergences
        self._capture = capture
        self._lock = threading.Lock()
        self._records: list[WireRecord] | None = None
        self.validated = 0
        self.exchanges = 0
        self.matched: set[str] = set()

    @property
    def captures_all_routes(self) -> bool:
        return self._capture is not None

    def wants(self, path: str) -> bool:
        return self._records is not None and (self.captures_all_routes or is_gated(path))

    def begin(self) -> None:
        with self._lock:
            self._records = []

    def record(self, record: WireRecord) -> None:
        with self._lock:
            if self._records is not None:
                self._records.append(record)

    def finish(self, test_id: str) -> GateOutcome:
        with self._lock:
            records, self._records = tuple(self._records or ()), None
        if self._capture is not None:
            self._capture.write(test_id, records)
        violations: list[Violation] = []
        validated = 0
        for record in records:
            if not is_gated(record.path):
                continue
            report = contract_validator().validate(record.exchange())
            validated += report.validated
            violations.extend(report.violations)
            self.exchanges += 1
        unallowed, matched = triage(violations, self.divergences)
        self.validated += validated
        self.matched.update(matched)
        return GateOutcome(validated, unallowed, matched)

    def summary(self) -> dict[str, Any]:
        """Session counters in a form an xdist worker can send to the controller."""

        return {
            "validated": self.validated,
            "exchanges": self.exchanges,
            "matched": sorted(self.matched),
        }

    def merge(self, summary: dict[str, Any]) -> None:
        self.validated += summary["validated"]
        self.exchanges += summary["exchanges"]
        self.matched.update(summary["matched"])

    def strict_problems(self) -> list[str]:
        """Why a full ``CONTRACT_STRICT=1`` run fails: nothing validated, or stale entries."""

        problems = []
        if self.validated == 0:
            problems.append("no Responses/Conversations body or event was validated")
        problems.extend(
            f"stale divergence {entry.id} ({entry.gap_id}, {entry.owner_wp}): matched nothing"
            for entry in self.divergences
            if entry.id not in self.matched
        )
        return problems


class RecordingApp:
    """Transparent ASGI wrapper that reports completed HTTP exchanges to the gate."""

    def __init__(self, app: ASGIApp, gate: ContractGate) -> None:
        self.app = app
        self._gate = gate

    def __getattr__(self, name: str) -> Any:
        if name in {"app", "_gate"}:  # not yet set (copy/pickle)
            raise AttributeError(name)
        return getattr(self.app, name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._gate.wants(scope["path"]):
            await self.app(scope, receive, send)
            return
        request_body = bytearray()
        start: dict[str, Any] = {}
        body = bytearray()
        complete = False

        async def observe_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
            return message

        async def observe_send(message: Message) -> None:
            nonlocal complete
            if message["type"] == "http.response.start":
                start.update(status=message["status"], headers=message.get("headers", []))
            elif message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
                complete = not message.get("more_body", False)
            await send(message)

        await self.app(scope, observe_receive, observe_send)
        if complete:
            self._gate.record(WireRecord.from_asgi(scope, bytes(request_body), start, bytes(body)))


def install_recorders(gate: ContractGate) -> Callable[[], None]:
    """Wrap the in-process test transports' apps; return the uninstaller."""

    from starlette.testclient import TestClient  # import-time deprecation warning

    original_client_init = TestClient.__init__
    original_transport_init = httpx.ASGITransport.__init__

    def client_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_client_init(self, *args, **kwargs)
        transport = self._transport
        if not hasattr(transport, "app"):
            raise RuntimeError("starlette TestClient transport no longer exposes `app`")
        if not isinstance(transport.app, RecordingApp):
            transport.app = RecordingApp(transport.app, gate)

    def transport_init(self: httpx.ASGITransport, app: ASGIApp, *args: Any, **kwargs: Any) -> None:
        wrapped = app if isinstance(app, RecordingApp) else RecordingApp(app, gate)
        original_transport_init(self, wrapped, *args, **kwargs)

    TestClient.__init__ = client_init  # type: ignore[method-assign]
    httpx.ASGITransport.__init__ = transport_init  # type: ignore[method-assign]

    def uninstall() -> None:
        TestClient.__init__ = original_client_init  # type: ignore[method-assign]
        httpx.ASGITransport.__init__ = original_transport_init  # type: ignore[method-assign]

    return uninstall


def describe(outcome: GateOutcome, limit: int = 20) -> str:
    """Render unallowed violations, folding repeats (e.g. heartbeat events)."""

    counts = Counter(outcome.unallowed)
    lines = [
        f"{len(outcome.unallowed)} OpenAI contract violation(s) not allowed by "
        "tests/contracts/divergences.toml (allowlist only with a gap ID and owner WP):"
    ]
    for violation, count in list(counts.items())[:limit]:
        repeat = f" (x{count})" if count > 1 else ""
        lines.append(
            f"  {violation.route}: schema={violation.schema} "
            f"pointer={violation.pointer or '/'} keyword={violation.keyword}{repeat} "
            f"-- {violation.message}"
        )
    if len(counts) > limit:
        lines.append(f"  ... and {len(counts) - limit} more distinct violation(s)")
    return "\n".join(lines)
