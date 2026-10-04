"""Pytest wiring of the M20 OpenAI schema gate (``-p`` in pyproject ``addopts``).

- Every test's ``/v1/responses*`` and ``/v1/conversations*`` exchanges are
  validated at teardown; an unallowed violation fails the test.
- ``CONTRACT_STRICT=1`` also fails the session when a ``divergences.toml``
  entry matched nothing (stale) or when nothing was validated. Use it on full
  gate runs only: a partial selection legitimately leaves entries unmatched.
- ``KAIRYU_WIRE_CAPTURE=<dir>`` captures every route's wire bytes per test.

Under pytest-xdist each worker reports its counts and matched entries to the
controller, which applies the strict checks to the union.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.contracts.asgi_contract import ContractGate, describe, install_recorders
from tests.contracts.divergences import load_divergences
from tests.contracts.wire_capture import WireCapture

STRICT_ENV = "CONTRACT_STRICT"
CAPTURE_ENV = "KAIRYU_WIRE_CAPTURE"
_GATE = pytest.StashKey[ContractGate]()
_UNINSTALL = pytest.StashKey[Callable[[], None]]()
_WORKER_OUTPUT = "kairyu_contract_gate"


def pytest_configure(config: pytest.Config) -> None:
    capture_dir = os.environ.get(CAPTURE_ENV)
    capture = WireCapture(Path(capture_dir)) if capture_dir else None
    config.stash[_GATE] = ContractGate(load_divergences(), capture)


def pytest_unconfigure(config: pytest.Config) -> None:
    uninstall = config.stash.get(_UNINSTALL, None)
    if uninstall is not None:
        uninstall()


@pytest.hookimpl(wrapper=True)
def pytest_collection(session: pytest.Session) -> Generator[None, object, object]:
    # Before any test module is imported or any fixture runs, so a client built
    # at import time or by a module/session-scoped fixture is recorded no matter
    # which test runs first. Inside the warnings plugin's (tryfirst) collection
    # wrapper, so the transports' import-time warning is captured as usual.
    config = session.config
    config.stash[_UNINSTALL] = install_recorders(config.stash[_GATE])
    return (yield)


@pytest.fixture(autouse=True)
def _openai_contract_gate(request: pytest.FixtureRequest) -> Iterator[None]:
    config = request.config
    gate = config.stash[_GATE]
    if _UNINSTALL not in config.stash:
        raise RuntimeError("schema gate recorders were not installed at collection")
    gate.begin()
    yield
    outcome = gate.finish(request.node.nodeid)
    if outcome.unallowed:
        pytest.fail(describe(outcome), pytrace=False)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    gate = config.stash.get(_GATE, None)
    if gate is None:
        return
    worker_output = getattr(config, "workeroutput", None)
    if worker_output is not None:
        worker_output[_WORKER_OUTPUT] = gate.summary()
        return
    if os.environ.get(STRICT_ENV) != "1" or config.option.collectonly:
        return
    problems = gate.strict_problems()
    if problems:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.ensure_newline()
            reporter.write_sep("=", f"{STRICT_ENV}=1 schema gate failed")
            for problem in problems:
                reporter.write_line(problem)
        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    summary = getattr(node, "workeroutput", {}).get(_WORKER_OUTPUT)
    gate = node.config.stash.get(_GATE, None)
    if summary is None or gate is None:
        return
    gate.merge(summary)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    gate = config.stash.get(_GATE, None)
    if gate is None or getattr(config, "workeroutput", None) is not None:
        return
    if gate.exchanges:
        terminalreporter.write_line(
            f"OpenAI schema gate: {gate.validated} item(s) validated in "
            f"{gate.exchanges} exchange(s); {len(gate.matched)} of "
            f"{len(gate.divergences)} divergence entries matched"
        )
