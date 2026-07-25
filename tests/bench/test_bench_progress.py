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
    TqdmProgress,
    make_reporter,
)
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakeBar:
    """Minimal tqdm stand-in recording what the reporter drives."""

    instances: list["_FakeBar"] = []

    def __init__(self, total=None, desc="", unit="", position=0, leave=True, file=None):
        self.total = total
        self.desc = desc
        self.updates = 0
        self.closed = False
        self.written: list[str] = []
        self.refreshed = 0
        _FakeBar.instances.append(self)

    def update(self, n):
        self.updates += n

    def refresh(self):
        self.refreshed += 1

    def write(self, text):
        self.written.append(text)

    def close(self):
        self.closed = True


# -- reporter selection --------------------------------------------------------


def test_make_reporter_uses_lines_for_non_tty_streams():
    assert isinstance(make_reporter(stream=io.StringIO()), LineProgress)


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
    assert isinstance(make_reporter(stream=_Tty()), LineProgress)


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
    reporter = TqdmProgress(_FakeBar, io.StringIO())
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
    assert any("partial" in line for line in suite.written)


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
