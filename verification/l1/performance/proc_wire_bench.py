"""Deterministic byte-volume gate for the kairyu-proc result wire.

This benchmark feeds cumulative ``StreamUpdate`` values through both production
encoders and retains every msgpack frame size. The legacy protocol retransmits
the complete prefix after each token; wire v2 sends one snapshot and sequenced
deltas. Byte volume, rather than wall time, is the binding metric, so host load
and OS jitter cannot change the verdict.

Run:

    uv run python verification/l1/performance/proc_wire_bench.py --assert-gate \
        --output bench/results/proc-wire-delta-YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

import msgpack

from kairyu.engine.core.engine_service import (
    LEGACY_WIRE_VERSION,
    WIRE_VERSION,
    WireEventCursor,
    event_from_update,
)
from kairyu.engine.engine_loop import StreamUpdate
from kairyu.outputs import TokenLogprob

DEFAULT_LENGTHS = (128, 256, 512, 1024)
_MAX_V2_GROWTH = 2.15
_MIN_LEGACY_GROWTH = 3.50
_SNAPSHOT_KEYS = frozenset(
    {
        "wire_version",
        "request_id",
        "stream_id",
        "sequence",
        "event",
        "outputs",
        "text",
        "logprobs",
        "logprob_content",
        "num_prompt_tokens",
        "num_cached_tokens",
        "cumulative_logprob",
        "finished",
        "finish_reason",
    }
)
_DELTA_KEYS = frozenset(
    {
        "wire_version",
        "request_id",
        "stream_id",
        "sequence",
        "event",
        "output_offset",
        "new_token_ids",
        "text_offset",
        "text_delta",
        "logprob_offset",
        "new_logprobs",
        "logprob_content_offset",
        "new_logprob_content",
        "num_cached_tokens",
        "cumulative_logprob",
        "finished",
        "finish_reason",
    }
)


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(result: dict[str, object]) -> str:
    payload = {key: value for key, value in result.items() if key != "artifact_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _update(length: int, total_length: int) -> StreamUpdate:
    token_ids = tuple(index % 100 for index in range(length))
    logprobs = tuple(
        {token_id: -0.25, (token_id + 1) % 100: -1.25}
        for token_id in token_ids
    )
    content = tuple(
        TokenLogprob(
            token="x",
            token_id=token_id,
            logprob=-0.25,
            bytes_=(120,),
            top=(
                TokenLogprob(
                    token="y",
                    token_id=(token_id + 1) % 100,
                    logprob=-1.25,
                    bytes_=(121,),
                ),
            ),
        )
        for token_id in token_ids
    )
    return StreamUpdate(
        outputs=token_ids,
        text="x" * length,
        finished=length == total_length,
        finish_reason="length" if length == total_length else None,
        logprobs=logprobs,
        cumulative_logprob=-0.25 * length,
        num_prompt_tokens=64,
        num_cached_tokens=48,
        logprob_content=content,
    )


def measure_wire_bytes(length: int) -> dict[str, object]:
    """Return raw per-event and total byte counts for one output length."""

    if length < 2:
        raise ValueError("length must be >= 2")
    cursor = WireEventCursor()
    legacy_sizes: list[int] = []
    v2_sizes: list[int] = []
    v2_new_token_items = 0
    v2_new_logprob_items = 0
    v2_new_logprob_content_items = 0
    v2_text_delta_chars = 0
    v2_structure_valid = True
    previous_outputs = 0
    previous_text = ""
    previous_logprobs = 0
    previous_content = 0
    for step in range(1, length + 1):
        update = _update(step, length)
        legacy = event_from_update(
            "bench",
            update,
            wire_version=LEGACY_WIRE_VERSION,
        )
        v2 = event_from_update(
            "bench",
            update,
            wire_version=WIRE_VERSION,
            cursor=cursor,
        )
        if v2 is None:
            raise AssertionError("a token-producing benchmark step was suppressed")
        v2["stream_id"] = "bench-stream"
        common_valid = (
            v2.get("wire_version") == WIRE_VERSION
            and v2.get("request_id") == legacy["request_id"] == "bench"
            and v2.get("stream_id") == "bench-stream"
            and v2.get("sequence") == step - 1
            and v2.get("finished") == legacy["finished"] == (step == length)
            and v2.get("finish_reason") == legacy["finish_reason"]
            and v2.get("num_cached_tokens") == legacy["num_cached_tokens"]
            and v2.get("cumulative_logprob") == legacy["cumulative_logprob"]
        )
        if step == 1:
            outputs = v2.get("outputs")
            text = v2.get("text")
            logprobs = v2.get("logprobs")
            logprob_content = v2.get("logprob_content")
            fields_valid = (
                isinstance(outputs, list)
                and isinstance(text, str)
                and isinstance(logprobs, list)
                and isinstance(logprob_content, list)
            )
            if fields_valid:
                v2_new_token_items += len(outputs)
                v2_new_logprob_items += len(logprobs)
                v2_new_logprob_content_items += len(logprob_content)
                v2_text_delta_chars += len(text)
            v2_structure_valid &= (
                common_valid
                and fields_valid
                and set(v2) == _SNAPSHOT_KEYS
                and v2.get("event") == "snapshot"
                and outputs == legacy["outputs"]
                and text == legacy["text"]
                and logprobs == legacy["logprobs"]
                and logprob_content == legacy["logprob_content"]
                and v2.get("num_prompt_tokens") == legacy["num_prompt_tokens"]
            )
        else:
            new_token_ids = v2.get("new_token_ids")
            text_delta = v2.get("text_delta")
            new_logprobs = v2.get("new_logprobs")
            new_content = v2.get("new_logprob_content")
            fields_valid = (
                isinstance(new_token_ids, list)
                and isinstance(text_delta, str)
                and isinstance(new_logprobs, list)
                and isinstance(new_content, list)
            )
            if fields_valid:
                v2_new_token_items += len(new_token_ids)
                v2_new_logprob_items += len(new_logprobs)
                v2_new_logprob_content_items += len(new_content)
                v2_text_delta_chars += len(text_delta)
            if step == length:
                text_offset = 0
                for previous, current in zip(
                    previous_text,
                    legacy["text"],
                    strict=False,
                ):
                    if previous != current:
                        break
                    text_offset += 1
            else:
                text_offset = len(previous_text)
            v2_structure_valid &= (
                common_valid
                and fields_valid
                and set(v2) == _DELTA_KEYS
                and v2.get("event") == "delta"
                and v2.get("output_offset") == previous_outputs
                and new_token_ids == legacy["outputs"][previous_outputs:]
                and v2.get("text_offset") == text_offset
                and text_delta == legacy["text"][text_offset:]
                and v2.get("logprob_offset") == previous_logprobs
                and new_logprobs == legacy["logprobs"][previous_logprobs:]
                and v2.get("logprob_content_offset") == previous_content
                and new_content == legacy["logprob_content"][previous_content:]
            )
        legacy_sizes.append(len(msgpack.packb(legacy, use_bin_type=True)))
        v2_sizes.append(len(msgpack.packb(v2, use_bin_type=True)))
        previous_outputs = len(legacy["outputs"])
        previous_text = legacy["text"]
        previous_logprobs = len(legacy["logprobs"])
        previous_content = len(legacy["logprob_content"])
    return {
        "output_tokens": length,
        "legacy_event_bytes": legacy_sizes,
        "v2_event_bytes": v2_sizes,
        "legacy_total_bytes": sum(legacy_sizes),
        "v2_total_bytes": sum(v2_sizes),
        "v2_new_token_items": v2_new_token_items,
        "v2_new_logprob_items": v2_new_logprob_items,
        "v2_new_logprob_content_items": v2_new_logprob_content_items,
        "v2_text_delta_chars": v2_text_delta_chars,
        "v2_structure_valid": bool(
            v2_structure_valid
            and v2_new_token_items == length
            and v2_new_logprob_items == length
            and v2_new_logprob_content_items == length
            and v2_text_delta_chars == length
        ),
    }


def evaluate(lengths: Sequence[int]) -> dict[str, object]:
    ordered = tuple(sorted(set(lengths)))
    if len(ordered) < 2 or ordered[0] < 2:
        raise ValueError("provide at least two unique lengths >= 2")
    measurements = [measure_wire_bytes(length) for length in ordered]
    growth = []
    checks = []
    for previous, current in zip(measurements, measurements[1:], strict=False):
        length_ratio = current["output_tokens"] / previous["output_tokens"]
        legacy_ratio = (
            current["legacy_total_bytes"] / previous["legacy_total_bytes"]
        )
        v2_ratio = current["v2_total_bytes"] / previous["v2_total_bytes"]
        legacy_exponent = math.log(legacy_ratio, length_ratio)
        v2_exponent = math.log(v2_ratio, length_ratio)
        growth.append(
            {
                "from_tokens": previous["output_tokens"],
                "to_tokens": current["output_tokens"],
                "legacy_ratio": legacy_ratio,
                "v2_ratio": v2_ratio,
                "legacy_empirical_exponent": legacy_exponent,
                "v2_empirical_exponent": v2_exponent,
            }
        )
        checks.extend(
            (
                {
                    "name": (
                        f"legacy_growth_{previous['output_tokens']}_"
                        f"{current['output_tokens']}"
                    ),
                    "passed": legacy_ratio >= _MIN_LEGACY_GROWTH,
                    "actual": legacy_ratio,
                    "required": f">={_MIN_LEGACY_GROWTH}",
                },
                {
                    "name": (
                        f"v2_growth_{previous['output_tokens']}_"
                        f"{current['output_tokens']}"
                    ),
                    "passed": v2_ratio <= _MAX_V2_GROWTH,
                    "actual": v2_ratio,
                    "required": f"<={_MAX_V2_GROWTH}",
                },
            )
        )
    checks.extend(
        {
            "name": f"v2_smaller_at_{measurement['output_tokens']}",
            "passed": measurement["v2_total_bytes"]
            < measurement["legacy_total_bytes"],
            "actual": measurement["v2_total_bytes"],
            "required": f"<{measurement['legacy_total_bytes']}",
        }
        for measurement in measurements
    )
    checks.extend(
        {
            "name": f"v2_structure_at_{measurement['output_tokens']}",
            "passed": measurement["v2_structure_valid"],
            "actual": {
                "new_token_items": measurement["v2_new_token_items"],
                "new_logprob_items": measurement["v2_new_logprob_items"],
                "new_logprob_content_items": measurement[
                    "v2_new_logprob_content_items"
                ],
                "text_delta_chars": measurement["v2_text_delta_chars"],
            },
            "required": (
                "each cumulative item appears once and deltas contain no "
                "cumulative output/text/logprob fields"
            ),
        }
        for measurement in measurements
    )
    return {
        "schema_version": 1,
        "metric": "msgpack_frame_bytes",
        "binding_rationale": (
            "exact serialized byte counts are deterministic and independent "
            "of wall-clock scheduling or OS jitter"
        ),
        "protocols": {
            "legacy": LEGACY_WIRE_VERSION,
            "delta": WIRE_VERSION,
        },
        "thresholds": {
            "max_v2_growth_per_doubling": _MAX_V2_GROWTH,
            "min_legacy_growth_per_doubling": _MIN_LEGACY_GROWTH,
        },
        "measurements": measurements,
        "growth": growth,
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }


def add_provenance(result: dict[str, object]) -> dict[str, object]:
    script = Path(__file__).resolve()
    tracked_dirty = bool(
        _git_output("status", "--porcelain", "--untracked-files=no")
    )
    clean_check = {
        "name": "clean_tracked_source",
        "passed": not tracked_dirty,
        "actual": tracked_dirty,
        "required": "false",
    }
    complete = {
        **result,
        "checks": [*result["checks"], clean_check],
        "all_passed": result["all_passed"] and clean_check["passed"],
        "source": {
            "commit": _git_output("rev-parse", "HEAD"),
            "tracked_dirty": tracked_dirty,
            "script": "verification/l1/performance/proc_wire_bench.py",
            "script_sha256": _sha256_file(script),
        },
        "environment": {
            "python": platform.python_version(),
            "msgpack": msgpack.version,
            "platform": platform.platform(),
        },
    }
    complete["artifact_sha256"] = artifact_sha256(complete)
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        default=",".join(str(value) for value in DEFAULT_LENGTHS),
        help="comma-separated output lengths",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    result = add_provenance(evaluate(lengths))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.assert_gate and not result["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
