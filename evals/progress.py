"""Live progress for a suite run: which slot is running, and how far in.

A full run can be thousands of judged items and take hours, so "no output for
40 minutes" is indistinguishable from a hang. Three reporters cover the ways a
run is watched:

- `TqdmProgress` for an interactive terminal (a bar per pair plus an overall
  bar),
- `LineProgress` for a log — CI, `docker compose logs`, nohup — where a
  redrawing bar is noise, so it emits one throttled line instead,
- `NullProgress` for tests and `--no-progress`.

The reporter never changes what a run does; every method is best-effort and a
failure inside one must not take down the run that is producing evidence.
"""

from __future__ import annotations

import sys
import time
from typing import Protocol, TextIO


class ProgressReporter(Protocol):
    """Observer for a suite run. All methods are optional no-ops."""

    def suite_start(self, pairs: int) -> None: ...

    def pair_start(self, benchmark: str, target: str, note: str = "") -> None: ...

    def items_total(self, total: int) -> None: ...

    def item_done(self) -> None: ...

    def pair_heartbeat(self) -> None: ...

    def pair_done(self, status: str, score: float | None, cached: bool = False) -> None: ...

    def close(self) -> None: ...


_METHODS = (
    "suite_start",
    "pair_start",
    "items_total",
    "item_done",
    "pair_heartbeat",
    "pair_done",
    "close",
)


class SafeReporter:
    """A reporter that cannot affect the run it is watching.

    Display is not evidence. A closed stderr, a broken pipe, or a tqdm bug must
    not abort a run before any pair executes — and an exception from
    `item_done()` in a `finally` block would otherwise replace a valid item
    result and turn the whole cell into "adapter crashed".
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def _call(self, name: str, *args, **kwargs) -> None:
        method = getattr(self._inner, name, None)
        if method is None:
            return
        try:
            method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - the run outranks its own progress bar
            pass

    def suite_start(self, pairs: int) -> None:
        self._call("suite_start", pairs)

    def pair_start(self, benchmark: str, target: str, note: str = "") -> None:
        self._call("pair_start", benchmark, target, note)

    def items_total(self, total: int) -> None:
        self._call("items_total", total)

    def item_done(self) -> None:
        self._call("item_done")

    def pair_heartbeat(self) -> None:
        self._call("pair_heartbeat")

    def pair_done(self, status: str, score: float | None, cached: bool = False) -> None:
        self._call("pair_done", status, score, cached)

    def close(self) -> None:
        self._call("close")


def _score_text(score: float | None) -> str:
    return "n/a" if score is None else f"{score * 100:.1f}"


class NullProgress:
    """Silent reporter: the run still writes its scoreboard and evidence."""

    def suite_start(self, pairs: int) -> None:
        return None

    def pair_start(self, benchmark: str, target: str, note: str = "") -> None:
        return None

    def items_total(self, total: int) -> None:
        return None

    def item_done(self) -> None:
        return None

    def pair_heartbeat(self) -> None:
        return None

    def pair_done(self, status: str, score: float | None, cached: bool = False) -> None:
        return None

    def close(self) -> None:
        return None


class LineProgress:
    """One line per event, plus a throttled item counter.

    Written for logs that are read after the fact, so every line stands on its
    own: no carriage returns, no assumption that the previous line is visible.
    """

    def __init__(self, stream: TextIO | None = None, interval_s: float = 15.0) -> None:
        self._stream = stream or sys.stderr
        self._interval_s = interval_s
        self._pairs = 0
        self._pair_index = 0
        self._label = ""
        self._total: int | None = None
        self._done = 0
        self._last_emit = 0.0
        self._started = 0.0

    def _write(self, text: str) -> None:
        print(text, file=self._stream, flush=True)

    def suite_start(self, pairs: int) -> None:
        self._pairs = pairs
        self._write(f"[bench] {pairs} benchmark×target pairs to run")

    def pair_start(self, benchmark: str, target: str, note: str = "") -> None:
        self._pair_index += 1
        self._label = f"{benchmark} × {target}"
        self._total = None
        self._done = 0
        self._started = self._last_emit = time.monotonic()
        suffix = f" — {note}" if note else ""
        self._write(f"[bench {self._pair_index}/{self._pairs}] {self._label}{suffix}")

    def items_total(self, total: int) -> None:
        self._total = total
        self._write(f"[bench {self._pair_index}/{self._pairs}] {self._label}: {total} items")

    def item_done(self) -> None:
        self._done += 1
        now = time.monotonic()
        # throttle: a 2,500-item slot must not emit 2,500 lines
        if now - self._last_emit < self._interval_s and self._done != self._total:
            return
        self._last_emit = now
        self._write_progress(now)

    def pair_heartbeat(self) -> None:
        """Keep a log moving while an external harness runs.

        The agentic slots never call `item_done()` and their subprocess output is
        captured, so without this an 8-hour SWE or Terminal-Bench run prints one
        line and then goes silent — exactly the "working vs hung" ambiguity the
        reporter exists to remove.
        """
        now = time.monotonic()
        if now - self._last_emit < self._interval_s:
            return
        self._last_emit = now
        self._write_progress(now)

    def _write_progress(self, now: float) -> None:
        total = "?" if self._total is None else str(self._total)
        elapsed = int(now - self._started)
        self._write(
            f"[bench {self._pair_index}/{self._pairs}] {self._label}: "
            f"{self._done}/{total} items ({elapsed}s)"
        )

    def pair_done(self, status: str, score: float | None, cached: bool = False) -> None:
        prefix = "cached" if cached else "done"
        self._write(
            f"[bench {self._pair_index}/{self._pairs}] {self._label}: "
            f"{prefix} — {status} (score={_score_text(score)})"
        )

    def close(self) -> None:
        return None


class TqdmProgress:
    """Interactive bars: one for the suite, one for the current pair."""

    def __init__(self, tqdm_factory, stream: TextIO | None = None) -> None:
        self._tqdm = tqdm_factory
        self._stream = stream or sys.stderr
        self._suite = None
        self._pair = None
        self._label = ""

    def _log(self, text: str) -> None:
        """Write above the bars, on the reporter's own stream.

        `tqdm.write` defaults to **stdout** even when the bar was created with
        stderr, which would put the play-by-play back into a piped scoreboard.
        """
        writer = getattr(self._tqdm, "write", None)
        if writer is not None:
            writer(text, file=self._stream)
            return
        print(text, file=self._stream, flush=True)

    def suite_start(self, pairs: int) -> None:
        self._suite = self._tqdm(
            total=pairs, desc="benchmark suite", unit="pair", position=0, leave=True,
            file=self._stream,
        )

    def pair_start(self, benchmark: str, target: str, note: str = "") -> None:
        self._label = f"{benchmark} × {target}"
        self._close_pair()
        self._pair = self._tqdm(
            total=None,
            desc=self._label if not note else f"{self._label} ({note})",
            unit="item",
            position=1,
            leave=False,
            file=self._stream,
        )

    def items_total(self, total: int) -> None:
        if self._pair is not None:
            self._pair.total = total
            self._pair.refresh()

    def item_done(self) -> None:
        if self._pair is not None:
            self._pair.update(1)

    def pair_heartbeat(self) -> None:
        if self._pair is not None:
            self._pair.refresh()

    def pair_done(self, status: str, score: float | None, cached: bool = False) -> None:
        self._close_pair()
        prefix = "cached" if cached else "done"
        self._log(f"{self._label}: {prefix} — {status} (score={_score_text(score)})")
        if self._suite is not None:
            self._suite.update(1)

    def _close_pair(self) -> None:
        if self._pair is not None:
            self._pair.close()
            self._pair = None

    def close(self) -> None:
        self._close_pair()
        if self._suite is not None:
            self._suite.close()
            self._suite = None


def make_reporter(
    *, enabled: bool = True, stream: TextIO | None = None
) -> ProgressReporter:
    """Pick a reporter: bars on a TTY, lines in a log, silence when disabled.

    Always wrapped in `SafeReporter`: a display failure must not be able to end
    a run that is producing evidence.

    tqdm is optional (`kairyu[bench]`); without it an interactive run simply
    gets the line reporter rather than an import error.
    """
    if not enabled:
        return NullProgress()
    target = stream or sys.stderr
    if getattr(target, "isatty", lambda: False)():
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return SafeReporter(LineProgress(target))
        return SafeReporter(TqdmProgress(tqdm, target))
    return SafeReporter(LineProgress(target))
