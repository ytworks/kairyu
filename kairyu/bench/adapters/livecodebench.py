"""LiveCodeBench (code_generation_lite): pass@1 via sandboxed test execution."""

from __future__ import annotations

import asyncio
import base64
import json
import pickle
import zlib

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    excerpt,
    extract_code_block,
)
from kairyu.bench.sandbox import run_python
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    ItemResult,
    SkipItem,
)

_TEST_TIMEOUT_S = 6.0
_MEMORY_MB = 4096

# `livecodebench/code_generation_lite` ships raw JSONL plus a loading script;
# its release_vN "versions" are CONFIG names, not git refs, and the repo has a
# single `main` branch with no tags. Passing "release_v6" as `revision` fails,
# and the script path needs trust_remote_code (removed in datasets 4.x), so the
# files are read directly and the revision pins a real commit.
_LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
# release_vN is the union of the first N shards (v6 == the full 1,055 problems
# from 2023-05 to 2025-04, which is the set Fugu reports).
_RELEASE_FILES: dict[str, tuple[str, ...]] = {
    "release_v6": (
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ),
}
_RELEASE_ROWS = {"release_v6": 1055}
_RELEASE = "release_v6"

# LCB starter code leans on these being in scope (official harness does the same)
_IMPORT_HEADER = """\
import collections
import heapq
import bisect
import itertools
import functools
import math
import re
import sys
from typing import Optional, List, Dict, Tuple, Set, Any
"""

_FUNCTIONAL_DRIVER = """

def _kairyu_norm(value):
    import json as _json
    return _json.loads(_json.dumps(value))

def _kairyu_main():
    import json as _json
    import sys as _sys
    spec = _json.loads(_sys.stdin.read())
    args = [_json.loads(line) for line in spec["input"].split("\\n") if line.strip()]
    expected = _json.loads(spec["expected"])
    result = getattr(Solution(), spec["fn"])(*args)
    _sys.exit(0 if _kairyu_norm(result) == _kairyu_norm(expected) else 1)

_kairyu_main()
"""

_PROMPT = """You will be given a competitive programming problem. Write a correct, \
efficient Python solution.

{io_instruction}

### Question:
{question}
{starter_section}
Write your full solution in a single ```python code block```."""

_STDIN_INSTRUCTION = (
    "Read input from standard input and write the answer to standard output."
)
_FUNCTIONAL_INSTRUCTION = (
    "Complete the starter code; keep the class and method signature unchanged."
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Blocks class/global loading so a hostile private-test blob cannot execute
    arbitrary code at download time (M7). The payload is a JSON string, so no
    globals are legitimately needed — any attempt fails loudly."""

    def find_class(self, module, name):  # pragma: no cover - security guard
        raise pickle.UnpicklingError(
            f"blocked global {module}.{name} while decoding private tests"
        )


def _compose_solution(code: str, driver: str = "") -> str:
    """Prepend the import header, hoisting any leading `from __future__` imports
    the model wrote so they stay the first statements (M8 — otherwise the header
    displaces them and every such solution is a SyntaxError)."""
    future, body = [], []
    for line in code.splitlines(keepends=True):
        target = future if line.lstrip().startswith("from __future__ import") else body
        target.append(line)
    return "".join(future) + _IMPORT_HEADER + "".join(body) + driver


def decode_private_tests(blob: str) -> list[dict]:
    """LCB lite stores private tests either as JSON or zlib+pickle+base64 JSON."""
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        import io

        raw = zlib.decompress(base64.b64decode(blob.encode()))
        return json.loads(_RestrictedUnpickler(io.BytesIO(raw)).load())


def normalize_output(text: str) -> list[str]:
    """Judge-style normalization: strip each line, drop trailing blank lines.

    Whole-output `.strip()` alone fails a correct solution that emits trailing
    spaces or CRLF — a false negative every competitive-programming judge
    avoids. Line structure is still significant.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def grade_code(code: str, tests: list[dict], fn_name: str | None) -> tuple[bool, str]:
    """Run every test in the sandbox; (passed, first failure detail)."""
    for index, test in enumerate(tests):
        if test.get("testtype") == "functional":
            spec = json.dumps(
                {"input": test["input"], "expected": test["output"], "fn": fn_name}
            )
            result = run_python(
                _compose_solution(code, _FUNCTIONAL_DRIVER),
                stdin=spec,
                timeout_s=_TEST_TIMEOUT_S,
                memory_mb=_MEMORY_MB,
            )
            if not result.ok:
                detail = "timeout" if result.timed_out else result.stderr[-300:]
                return False, f"functional test {index}: {detail or 'wrong answer'}"
        else:
            result = run_python(
                _compose_solution(code),
                stdin=test["input"],
                timeout_s=_TEST_TIMEOUT_S,
                memory_mb=_MEMORY_MB,
            )
            if result.timed_out or result.returncode != 0:
                detail = "timeout" if result.timed_out else result.stderr[-300:]
                return False, f"stdin test {index}: {detail}"
            if normalize_output(result.stdout) != normalize_output(test["output"]):
                return False, f"stdin test {index}: wrong answer"
    return True, ""


class LiveCodeBenchAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="livecodebench",
        display_name="LiveCodeBench",
        metric="pass@1",
        hf_dataset="livecodebench/code_generation_lite",
        hf_revision=_LCB_REVISION,
        needs_execution=True,
        annotations=(
            f"LiveCodeBench {_RELEASE} — the full "
            f"{_RELEASE_ROWS[_RELEASE]}-problem set (2023-05..2025-04), one sample per "
            "problem (pass@1)",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_jsonl_files

        rows = load_jsonl_files(
            self.info.hf_dataset,
            _RELEASE_FILES[_RELEASE],
            revision=self.info.hf_revision,
        )
        expected = _RELEASE_ROWS[_RELEASE]
        if len(rows) != expected:
            # The revision is pinned, so a count change means a partial download
            # or an upstream reshuffle — fail closed rather than score a subset
            # of the benchmark as if it were the whole thing.
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} {_RELEASE} yielded "
                f"{len(rows)} problems, expected {expected}"
            )
        normalized = []
        for row in rows:
            tests = json.loads(row["public_test_cases"])
            tests += decode_private_tests(row["private_test_cases"])
            metadata = json.loads(row.get("metadata") or "{}")
            normalized.append(
                {
                    "id": row["question_id"],
                    "question": row["question_content"],
                    "starter_code": row.get("starter_code") or "",
                    "fn_name": metadata.get("func_name"),
                    "tests": tests,
                    "contest_date": str(row.get("contest_date", "")),
                }
            )
        return normalized

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        payload = item.payload
        starter = payload.get("starter_code") or ""
        functional = any(t.get("testtype") == "functional" for t in payload["tests"])
        prompt = _PROMPT.format(
            io_instruction=_FUNCTIONAL_INSTRUCTION if functional else _STDIN_INSTRUCTION,
            question=payload["question"],
            starter_section=f"\n### Starter code:\n```python\n{starter}\n```\n"
            if starter
            else "",
        )
        return ChatRequestSpec(
            messages=({"role": "user", "content": prompt},),
            max_tokens=target.max_output_tokens,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        code = extract_code_block(response_text)
        if code is None:
            return ItemResult(
                item_id=item.id,
                status="completed",
                score=0.0,
                error="no code block in response",
                response_excerpt=excerpt(response_text),
            )
        async with ctx.exec_semaphore:
            passed, detail = await asyncio.to_thread(
                grade_code, code, item.payload["tests"], item.payload.get("fn_name")
            )
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if passed else 0.0,
            error=detail or None,
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base["release"] = _RELEASE
        base["release_problems"] = _RELEASE_ROWS[_RELEASE]
        base["execution"] = (
            f"local subprocess sandbox, {_TEST_TIMEOUT_S}s/test, {_MEMORY_MB}MB rlimit; "
            "pass@1 = all public+private tests pass"
        )
        return base
