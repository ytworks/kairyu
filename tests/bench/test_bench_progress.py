"""Live progress: bars on a TTY, throttled lines in a log, silence when off.

A full run is thousands of judged items over hours, so "no output" and "hung"
were indistinguishable. The reporter is an observer: it must never change what a
run produces, and a 2,500-item slot must not emit 2,500 log lines.
"""

import io
import json

from conftest import make_config

from kairyu.bench.progress import (
    LineProgress,
    NullProgress,
    SafeReporter,
    TqdmProgress,
    make_reporter,
)
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakeBar:
    """Minimal tqdm stand-in recording what the reporter drives.

    `write` is a *classmethod* honouring `file=`, like real tqdm: tqdm.write
    defaults to stdout regardless of the bar's own stream, so the reporter must
    pass its stream explicitly and the fake must be able to catch that.
    """

    instances: list["_FakeBar"] = []
    written: list[tuple[str, object]] = []

    def __init__(self, total=None, desc="", unit="", position=0, leave=True, file=None):
        self.total = total
        self.desc = desc
        self.file = file
        self.updates = 0
        self.closed = False
        self.refreshed = 0
        _FakeBar.instances.append(self)

    def update(self, n):
        self.updates += n

    def refresh(self):
        self.refreshed += 1

    @classmethod
    def write(cls, text, file=None):
        cls.written.append((text, file))
        if file is not None:
            print(text, file=file)

    def close(self):
        self.closed = True


# -- reporter selection --------------------------------------------------------


def _inner(reporter):
    """Every built reporter is wrapped so a display failure cannot end a run."""
    assert isinstance(reporter, SafeReporter)
    return reporter._inner


def test_make_reporter_uses_lines_for_non_tty_streams():
    assert isinstance(_inner(make_reporter(stream=io.StringIO())), LineProgress)


def test_make_reporter_is_silent_when_disabled():
    assert isinstance(make_reporter(enabled=False, stream=_Tty()), NullProgress)


def test_make_reporter_falls_back_to_lines_without_tqdm(monkeypatch):
    """tqdm is an optional extra; its absence must not be an import error."""
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tqdm)
    assert isinstance(_inner(make_reporter(stream=_Tty())), LineProgress)


# -- line reporter -------------------------------------------------------------


def test_line_reporter_names_the_slot_and_counts_items():
    stream = io.StringIO()
    reporter = LineProgress(stream, interval_s=0.0)
    reporter.suite_start(2)
    reporter.pair_start("gpqa-diamond", "qwen3-32b")
    reporter.items_total(3)
    for _ in range(3):
        reporter.item_done()
    reporter.pair_done("completed", 0.42)
    output = stream.getvalue()

    assert "2 benchmark×target pairs" in output
    assert "[bench 1/2] gpqa-diamond × qwen3-32b" in output
    assert "3 items" in output
    assert "3/3 items" in output
    assert "completed (score=42.0)" in output


def test_line_reporter_throttles_but_always_emits_the_last_item():
    """2,500 items must not become 2,500 lines — yet completion is never lost."""
    stream = io.StringIO()
    reporter = LineProgress(stream, interval_s=3600.0)
    reporter.suite_start(1)
    reporter.pair_start("hle", "m")
    reporter.items_total(50)
    for _ in range(50):
        reporter.item_done()

    counters = [line for line in stream.getvalue().splitlines() if "items (" in line]
    assert len(counters) == 1
    assert "50/50 items" in counters[0]


def test_line_reporter_handles_unknown_totals():
    stream = io.StringIO()
    reporter = LineProgress(stream, interval_s=0.0)
    reporter.suite_start(1)
    reporter.pair_start("terminal-bench", "m", note="agentic harness")
    reporter.item_done()
    assert "agentic harness" in stream.getvalue()
    assert "1/? items" in stream.getvalue()


def test_line_reporter_marks_cached_pairs():
    stream = io.StringIO()
    reporter = LineProgress(stream, interval_s=0.0)
    reporter.suite_start(1)
    reporter.pair_start("gpqa-diamond", "m")
    reporter.pair_done("completed", None, cached=True)
    assert "cached — completed (score=n/a)" in stream.getvalue()


# -- tqdm reporter -------------------------------------------------------------


def test_tqdm_reporter_drives_suite_and_pair_bars():
    _FakeBar.instances.clear()
    _FakeBar.written.clear()
    stream = io.StringIO()
    reporter = TqdmProgress(_FakeBar, stream)
    reporter.suite_start(2)
    reporter.pair_start("scicode", "m")
    reporter.items_total(4)
    reporter.item_done()
    reporter.item_done()
    reporter.pair_done("partial", 0.5)
    reporter.close()

    suite, pair = _FakeBar.instances[0], _FakeBar.instances[1]
    assert suite.total == 2 and suite.updates == 1 and suite.closed
    assert pair.desc == "scicode × m"
    assert pair.total == 4 and pair.updates == 2 and pair.closed
    assert any("partial" in text for text, _ in _FakeBar.written)


def test_tqdm_completion_lines_go_to_the_reporter_stream(capsys):
    """tqdm.write defaults to stdout even for a stderr bar; that would leak."""
    _FakeBar.instances.clear()
    _FakeBar.written.clear()
    stream = io.StringIO()
    reporter = TqdmProgress(_FakeBar, stream)
    reporter.suite_start(1)
    reporter.pair_start("gpqa-diamond", "m")
    reporter.pair_done("completed", 0.42)
    reporter.close()

    assert all(file is stream for _, file in _FakeBar.written)
    assert "completed" in stream.getvalue()
    assert "completed" not in capsys.readouterr().out


def test_tqdm_heartbeat_refreshes_the_pair_bar():
    _FakeBar.instances.clear()
    reporter = TqdmProgress(_FakeBar, io.StringIO())
    reporter.suite_start(1)
    reporter.pair_start("terminal-bench", "m", note="agentic harness")
    reporter.pair_heartbeat()
    assert _FakeBar.instances[1].refreshed >= 1
    reporter.close()


def test_tqdm_reporter_closes_the_previous_pair_bar():
    _FakeBar.instances.clear()
    reporter = TqdmProgress(_FakeBar, io.StringIO())
    reporter.suite_start(2)
    reporter.pair_start("a", "m")
    reporter.pair_start("b", "m")
    assert _FakeBar.instances[1].closed  # no leaked bar
    reporter.close()


# -- runner integration --------------------------------------------------------


async def test_runner_reports_every_slot_and_item(tmp_path, http_factory):
    stream = io.StringIO()
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond", "mrcr-v2"))
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (False, "t"),
        progress=LineProgress(stream, interval_s=0.0),
    )
    assert await runner.run() == 0
    output = stream.getvalue()

    assert "2 benchmark×target pairs" in output
    assert "[bench 1/2] gpqa-diamond × m" in output
    assert "[bench 2/2] mrcr-v2 × m" in output
    # fixture sizes reported, and each item advanced the counter
    assert "gpqa-diamond × m: 3 items" in output
    assert "3/3 items" in output


async def test_progress_does_not_change_results(tmp_path, http_factory):
    """The scoreboard is byte-identical with the reporter on and off."""

    async def scoreboard(progress, run_id):
        config = make_config(
            tmp_path,
            models=("m",),
            only=("gpqa-diamond",),
            run_id=run_id,
            results_dir=str(tmp_path / run_id),
        )
        runner = SuiteRunner(
            config,
            http_factory=http_factory,
            probe_docker=lambda: (False, "t"),
            progress=progress,
        )
        await runner.run()
        pair = ResultStore(tmp_path / run_id, run_id).load_pair("gpqa-diamond", "m")
        return pair.model_dump(exclude={"started_at", "finished_at", "items"})

    quiet = await scoreboard(NullProgress(), "quiet")
    loud = await scoreboard(LineProgress(io.StringIO(), interval_s=0.0), "loud")
    assert quiet == loud


async def test_pair_play_by_play_is_not_duplicated_on_stdout(tmp_path, http_factory, capsys):
    """Live output belongs to the reporter; stdout keeps the artifacts."""
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (False, "t"),
        progress=LineProgress(io.StringIO(), interval_s=0.0),
    )
    await runner.run()
    out = capsys.readouterr().out
    assert "[run] gpqa-diamond" not in out
    assert "# Fugu benchmark scoreboard" in out  # the artifact still prints


async def test_cached_pairs_are_reported_as_cached(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    for _ in range(2):
        stream = io.StringIO()
        runner = SuiteRunner(
            config,
            http_factory=http_factory,
            probe_docker=lambda: (False, "t"),
            progress=LineProgress(stream, interval_s=0.0),
        )
        await runner.run()
    assert "cached — completed" in stream.getvalue()


async def test_progress_is_excluded_from_the_run_fingerprint(tmp_path, http_factory):
    """Turning the display off must not invalidate stored evidence."""
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    await runner.run()
    first = json.loads(
        (tmp_path / "results" / "test-run" / "run.json").read_text(encoding="utf-8")
    )["fingerprint"]

    quiet = config.model_copy(update={"progress": False})
    runner = SuiteRunner(quiet, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0  # same run id resumes rather than being refused
    second = json.loads(
        (tmp_path / "results" / "test-run" / "run.json").read_text(encoding="utf-8")
    )["fingerprint"]
    assert first == second


class _ExplodingReporter:
    """Fails at every callback, the way a closed stderr or a tqdm bug would."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _boom(self, name: str):
        self.calls.append(name)
        raise RuntimeError(f"reporter failed in {name}")

    def suite_start(self, pairs):
        self._boom("suite_start")

    def pair_start(self, benchmark, target, note=""):
        self._boom("pair_start")

    def items_total(self, total):
        self._boom("items_total")

    def item_done(self):
        self._boom("item_done")

    def pair_heartbeat(self):
        self._boom("pair_heartbeat")

    def pair_done(self, status, score, cached=False):
        self._boom("pair_done")

    def close(self):
        self._boom("close")


async def test_a_failing_reporter_cannot_change_the_run(tmp_path, http_factory):
    """Display is not evidence: every callback may fail without cost."""
    reporter = _ExplodingReporter()
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (False, "t"),
        progress=reporter,
    )
    assert await runner.run() == 0

    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "completed"
    assert pair.metrics["n_total"] == 3  # fixture size: no item lost to the reporter
    assert all(item.status == "completed" for item in pair.items)
    # and it really was exercised at every callback
    assert {"suite_start", "pair_start", "items_total", "item_done", "pair_done"} <= set(
        reporter.calls
    )


async def test_reporter_is_closed_even_when_the_run_raises(tmp_path, http_factory):
    """close() belongs in a finally, or cancellation leaks the bar."""
    closed: list[str] = []

    class Recording(NullProgress):
        def close(self) -> None:
            closed.append("closed")

    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
        progress=Recording(),
    )
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="probe failed"):
        await runner.run()
    assert closed == ["closed"]


async def test_agentic_pairs_emit_a_heartbeat_while_they_run(tmp_path, monkeypatch):
    """An 8-hour harness must not look identical to a hang in a log."""
    import asyncio

    import httpx

    import kairyu.bench.runner as runner_mod
    from kairyu.bench.adapters.base import AdapterInfo, DownloadContext
    from kairyu.bench.types import DownloadReport, PairResult

    monkeypatch.setattr(runner_mod, "_HEARTBEAT_INTERVAL_S", 0.01)

    class SlowHarness:
        info = AdapterInfo(
            name="terminal-bench",
            display_name="Terminal-Bench 2.1",
            metric="accuracy",
            agentic=True,
        )

        def download(self, ctx: DownloadContext) -> DownloadReport:
            return DownloadReport(adapter=self.info.name, status="ok")

        async def run(self, target, ctx) -> PairResult:
            await asyncio.sleep(0.1)  # stands in for the subprocess
            now = "2026-07-25T00:00:00+00:00"
            return PairResult(
                benchmark=self.info.name,
                target=target.label(),
                status="completed",
                metrics={"score": 1.0, "n_total": 1},
                started_at=now,
                finished_at=now,
            )

    monkeypatch.setattr(
        runner_mod, "suite_adapters", lambda *a, **k: [SlowHarness()]
    )
    stream = io.StringIO()
    config = make_config(tmp_path, models=("m",))
    runner = SuiteRunner(
        config,
        http_factory=lambda: httpx.AsyncClient(),
        probe_docker=lambda: (True, "docker available"),
        progress=LineProgress(stream, interval_s=0.0),
    )
    assert await runner.run() == 0

    heartbeats = [line for line in stream.getvalue().splitlines() if "items (" in line]
    assert heartbeats, stream.getvalue()
    assert "terminal-bench × m" in heartbeats[0]
