"""Incremental stop matching parity and bounded-work coverage (issue #217)."""

from __future__ import annotations

import random

import pytest

from kairyu.engine.engine_loop import _IncrementalStopMatcher


class _CountingText(str):
    searched_characters = 0

    def find(self, sub, start=0, end=None):
        limit = len(self) if end is None else end
        type(self).searched_characters += max(0, limit - start)
        if end is None:
            return super().find(sub, start)
        return super().find(sub, start, end)


def _full_scan(text: str, stops: tuple[str, ...]) -> int | None:
    matches = [
        index for stop in stops if (index := text.find(stop)) != -1
    ]
    return min(matches, default=None)


def test_stop_matcher_finds_earliest_cross_chunk_and_overlapping_match() -> None:
    matcher = _IncrementalStopMatcher(("END", "ND", "D!", "END!"))

    assert matcher.find("prefix E") is None
    assert matcher.find("prefix EN") is None
    assert matcher.find("prefix END") == len("prefix ")
    # Once detected, later overlapping patterns cannot change the finish point.
    assert matcher.find("prefix END!") == len("prefix ")


def test_stop_matcher_searches_final_utf8_tail() -> None:
    matcher = _IncrementalStopMatcher(("界!",))

    assert matcher.find("こんにちは世") is None
    assert matcher.find("こんにちは世界") is None
    assert matcher.find("こんにちは世界!") == len("こんにちは世")


def test_stop_matcher_repeated_text_does_no_duplicate_work() -> None:
    matcher = _IncrementalStopMatcher(("never", "stop"))
    text = _CountingText("some text")
    _CountingText.searched_characters = 0

    assert matcher.find(text) is None
    work = _CountingText.searched_characters
    assert matcher.find(text) is None
    assert _CountingText.searched_characters == work


def test_stop_matcher_rejects_retracted_text() -> None:
    matcher = _IncrementalStopMatcher(("stop",))
    assert matcher.find("growing text") is None

    with pytest.raises(ValueError, match="never retract"):
        matcher.find("short")


def test_randomized_incremental_match_is_equivalent_to_full_scan() -> None:
    rng = random.Random(217)
    alphabet = "abca日本語🙂"

    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 300)))
        stops = tuple(
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 14)))
            for _ in range(rng.randrange(1, 20))
        )
        matcher = _IncrementalStopMatcher(stops)
        end = 0
        while end < len(text):
            end = min(len(text), end + rng.randrange(1, 18))
            prefix = text[:end]
            actual = matcher.find(prefix)
            assert actual == _full_scan(prefix, stops)
            # Production finishes on first detection, which fixes the same
            # earliest match the historical per-update full scan observed.
            if actual is not None:
                break


def test_long_output_many_stops_has_bounded_linear_search_work() -> None:
    stops = tuple(f"!never-{index:03d}-stop!" for index in range(64))
    text = ("abcdefghijklmnopqrstuvwxyz日本語" * 1200)[:32768]
    matcher = _IncrementalStopMatcher(stops)
    chunk_size = 32
    legacy_full_scan_work = 0
    calls = 0
    _CountingText.searched_characters = 0

    for end in range(chunk_size, len(text) + chunk_size, chunk_size):
        prefix = _CountingText(text[:end])
        assert matcher.find(prefix) is None
        legacy_full_scan_work += len(prefix) * len(stops)
        calls += 1

    bounded_work = len(stops) * (
        len(text) + max(0, calls - 1) * matcher.overlap
    )
    assert _CountingText.searched_characters <= bounded_work
    assert _CountingText.searched_characters < legacy_full_scan_work // 100
