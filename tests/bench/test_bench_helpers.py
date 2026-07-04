"""Shared adapter helpers: shuffling, extraction, sampling, URL normalization."""

from kairyu.bench.adapters.base import (
    estimate_tokens,
    extract_choice_letter,
    extract_code_block,
    mcq_prompt,
    normalize_base_url,
    select_items,
    shuffle_choices,
)
from kairyu.bench.types import BenchItem


def test_shuffle_choices_deterministic_per_seed_and_item():
    first = shuffle_choices(0, "item-1", "right", ["a", "b", "c"])
    again = shuffle_choices(0, "item-1", "right", ["a", "b", "c"])
    assert first == again
    choices, letter = first
    assert set(choices) == {"right", "a", "b", "c"}
    assert choices[ord(letter) - ord("A")] == "right"


def test_shuffle_choices_varies_across_items():
    orders = {
        tuple(shuffle_choices(0, f"item-{i}", "right", ["a", "b", "c"])[0])
        for i in range(20)
    }
    assert len(orders) > 1  # not a fixed permutation


def test_extract_choice_letter_answer_marker_wins():
    assert extract_choice_letter("I think B... Answer: C") == "C"
    assert extract_choice_letter("**Answer: (D)**") == "D"
    assert extract_choice_letter("the answer is a") == "A"


def test_extract_choice_letter_fallback_and_none():
    assert extract_choice_letter("Definitely option B here") == "B"
    assert extract_choice_letter("no letters at all") is None
    assert extract_choice_letter("E is out of range") is None


def test_mcq_prompt_layout():
    prompt = mcq_prompt("Q?", ["one", "two"])
    assert "A) one" in prompt and "B) two" in prompt
    assert 'End your reply with "Answer: <letter>"' in prompt


def test_extract_code_block_takes_last():
    text = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
    assert extract_code_block(text) == "second"
    assert extract_code_block("no code") is None


def test_select_items_deterministic_subset():
    items = [BenchItem(id=str(i), payload={}) for i in range(100)]
    first = select_items(items, 10, seed=1)
    again = select_items(items, 10, seed=1)
    other = select_items(items, 10, seed=2)
    assert [i.id for i in first] == [i.id for i in again]
    assert len(first) == 10
    assert [i.id for i in first] != [i.id for i in other]
    assert select_items(items, None, seed=1) == items
    assert select_items(items, 200, seed=1) == items


def test_normalize_base_url():
    assert normalize_base_url("http://gw:8000") == "http://gw:8000/v1"
    assert normalize_base_url("http://gw:8000/") == "http://gw:8000/v1"
    assert normalize_base_url("http://gw:8000/v1") == "http://gw:8000/v1"
    assert normalize_base_url("http://gw:8000/v1/") == "http://gw:8000/v1"


def test_estimate_tokens_scales_with_chars():
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 101
