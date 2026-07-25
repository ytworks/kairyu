"""MRCRv2 (OpenAI multi-round co-reference): string-similarity scoring, no judge.

Official metric: the response must start with the per-item random prepend
string; the score is SequenceMatcher.ratio() between response and answer
(both with the prepend stripped), else 0.

The published `openai/mrcr` split mixes 2-, 4- and 8-needle items across
context lengths up to 1M tokens. Fugu reports the **8-needle** subset at up to
**128K** context, so this adapter selects exactly that slice: averaging over the
whole 2,400-row split would compare a different (easier, shorter) population
against Fugu's number.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    estimate_tokens,
    excerpt,
)
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    ItemResult,
    SkipItem,
)

# Fugu's reported conditions for this row.
_NEEDLES = 8
_MAX_CONTEXT_TOKENS = 131_072  # "up to 128K context"

# The dataset card defines its bins by the tokens used by **prompt + answer**
# under `o200k_base`, with 100 samples per (needle count, bin):
#   [4096, 8192] (8192, 16384] (16384, 32768] (32768, 65536] (65536, 131072]
#   (131072, 262144] (262144, 524288] (524288, 1048576]
# Five of those bins sit at or below 128K, so Fugu's 8-needle slice is exactly
# 500 rows. A chars/4 approximation over the prompt alone cannot land on those
# boundaries, so it cannot select the same population.
_BIN_UPPER_BOUNDS = (8192, 16384, 32768, 65536, 131_072, 262_144, 524_288, 1_048_576)
_SAMPLES_PER_BIN = 100
_ENCODING = "o200k_base"


def _encoder():
    """The official `o200k_base` encoder, or a typed failure.

    Approximating here would silently score a different population than the one
    the published bins define, so a missing tokenizer is a skipped cell.
    """
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise DatasetUnavailable(
            "MRCRv2 selects the official token bins, which needs tiktoken: "
            "pip install 'kairyu[bench]'"
        ) from error
    return tiktoken.get_encoding(_ENCODING)


def selected_bins() -> tuple[int, ...]:
    """Official bins at or below Fugu's 128K ceiling."""
    return tuple(bound for bound in _BIN_UPPER_BOUNDS if bound <= _MAX_CONTEXT_TOKENS)


def expected_rows() -> int:
    """100 samples per selected bin, per the dataset card."""
    return _SAMPLES_PER_BIN * len(selected_bins())


def token_bin(total_tokens: int) -> int | None:
    """Upper bound of the official bin holding `total_tokens`, else None."""
    if total_tokens < 4096:
        return None
    for bound in _BIN_UPPER_BOUNDS:
        if total_tokens <= bound:
            return bound
    return None


def count_tokens(encoder, messages: list[dict], answer: str = "") -> int:
    """Exact `o200k_base` tokens, matching the card's `n_tokens` helper."""
    total = sum(len(encoder.encode(str(message.get("content", "")))) for message in messages)
    return total + (len(encoder.encode(answer)) if answer else 0)


def prompt_tokens(messages: list[dict], n_chars: object = None) -> int:
    """Approximate prompt tokens, for the target's context gate only.

    Selection uses exact counts (see `count_tokens`); this heuristic only decides
    whether an item fits a *target's* declared window, where the dataset's own
    `n_chars` is the better signal when present.
    """
    if isinstance(n_chars, int | float) and n_chars > 0:
        return int(n_chars) // 4 + 1
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


def mrcr_grade(response: str, answer: str, prepend: str) -> float:
    if not response.startswith(prepend):
        return 0.0
    response = response.removeprefix(prepend)
    answer = answer.removeprefix(prepend)
    return SequenceMatcher(None, response, answer).ratio()


class MrcrAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="mrcr-v2",
        display_name="MRCRv2",
        metric="sequence-match ratio",
        hf_dataset="openai/mrcr",
        annotations=(
            f"{_NEEDLES}-needle items in the official o200k_base bins at or below "
            f"{_MAX_CONTEXT_TOKENS // 1024}K prompt+answer tokens (Fugu's reported "
            "slice); the published split also contains 2/4-needle and "
            "longer-context bins, which are excluded",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(
            self.info.hf_dataset, split="train", revision=self.info.hf_revision
        )
        encoder = _encoder()
        normalized: list[dict] = []
        wrong_needles = out_of_range = 0
        per_bin: dict[int, int] = {}
        for index, row in enumerate(rows):
            if row.get("n_needles") != _NEEDLES:
                wrong_needles += 1
                continue
            messages = json.loads(row["prompt"])
            answer = row["answer"]
            # the card bins on prompt + answer, so selection must too
            total = count_tokens(encoder, messages, answer)
            bin_bound = token_bin(total)
            if bin_bound is None or bin_bound > _MAX_CONTEXT_TOKENS:
                out_of_range += 1
                continue
            per_bin[bin_bound] = per_bin.get(bin_bound, 0) + 1
            normalized.append(
                {
                    "id": f"mrcr-{index:05d}",
                    "messages": messages,
                    "answer": answer,
                    "prepend": row["random_string_to_prepend"],
                    "n_needles": row.get("n_needles"),
                    "token_bin": bin_bound,
                    "total_tokens": total,
                    "prompt_tokens": count_tokens(encoder, messages),
                    "est_prompt_tokens": prompt_tokens(messages, row.get("n_chars")),
                }
            )
        # Per bin, not just in total: 500 rows weighted 99/101/100/100/100 is a
        # different population than the official 100-per-bin slice, and it would
        # otherwise be cached while the report claims the official one.
        expected_per_bin = dict.fromkeys(selected_bins(), _SAMPLES_PER_BIN)
        if per_bin != expected_per_bin:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} did not yield the "
                f"official {_NEEDLES}-needle slice: expected {_SAMPLES_PER_BIN} rows "
                f"in each of {list(selected_bins())}, got "
                f"{dict(sorted(per_bin.items()))} ({len(normalized)} rows total)"
            )
        # The excluded counts are the denominator story; never silent.
        print(
            f"[mrcr-v2] kept {len(normalized)}/{len(rows)} rows "
            f"({wrong_needles} not {_NEEDLES}-needle, {out_of_range} outside the "
            f"bins at or below {_MAX_CONTEXT_TOKENS} prompt+answer tokens); "
            f"per-bin counts {dict(sorted(per_bin.items()))}"
        )
        return normalized



        rows = load_hf_rows(
            self.info.hf_dataset, split="train", revision=self.info.hf_revision
        )
        return [
            {
                "id": f"mrcr-{index:05d}",
                "messages": json.loads(row["prompt"]),
                "answer": row["answer"],
                "prepend": row["random_string_to_prepend"],
                "n_needles": row.get("n_needles"),
            }
            for index, row in enumerate(rows)
        ]

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        messages = item.payload["messages"]
        # Exact o200k_base message-content tokens when normalization recorded
        # them, matching the official runner's own
        # `n_tokens(messages) > MAX_CONTEXT_WINDOW` gate. The chars/4 heuristic is
        # only a fallback for rows normalized before that field existed; near a
        # target's limit the two disagree and would skip a fitting row or send an
        # oversized one.
        est = item.payload.get("prompt_tokens")
        if not isinstance(est, int):
            est = item.payload.get("est_prompt_tokens")
        if not isinstance(est, int):
            est = prompt_tokens(messages)
        if target.max_context_tokens is not None and est > target.max_context_tokens:
            return SkipItem(
                reason=f"est. {est} prompt tokens > target limit {target.max_context_tokens}"
            )
        return ChatRequestSpec(
            messages=tuple(messages),
            max_tokens=target.max_output_tokens,
            est_prompt_tokens=est,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        score = mrcr_grade(response_text, item.payload["answer"], item.payload["prepend"])
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=score,
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base["needles"] = _NEEDLES
        base["max_context_tokens"] = _MAX_CONTEXT_TOKENS
        base["tokenizer"] = _ENCODING
        base["selected_bins"] = list(selected_bins())
        base["expected_rows"] = expected_rows()
        base["selection"] = (
            f"only n_needles == {_NEEDLES} rows whose exact {_ENCODING} "
            "prompt+answer token count falls in an official bin at or below "
            f"{_MAX_CONTEXT_TOKENS} ({_SAMPLES_PER_BIN} samples per bin)"
        )
        base["truncation_policy"] = (
            f"items whose exact {_ENCODING} prompt tokens exceed the target's "
            "max_context_tokens are skipped, never truncated (the official "
            "runner's own gate; chars/4 only for legacy rows)"
        )
        return base
