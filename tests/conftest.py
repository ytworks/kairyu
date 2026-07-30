"""Repository-wide pytest outcome controls."""

from __future__ import annotations

import pytest

_selected_skip_count = 0


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("kairyu")
    group.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="fail the pytest session if any selected test is skipped",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _selected_skip_count
    _selected_skip_count = 0


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    global _selected_skip_count
    if report.skipped and not hasattr(report, "wasxfail"):
        _selected_skip_count += 1


def pytest_collectreport(report: pytest.CollectReport) -> None:
    global _selected_skip_count
    if report.skipped:
        _selected_skip_count += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("fail_on_skip"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if not _selected_skip_count:
        return
    if reporter is not None:
        reporter.write_sep(
            "=",
            f"--fail-on-skip rejected {_selected_skip_count} skipped selected test(s)",
        )
    if exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
