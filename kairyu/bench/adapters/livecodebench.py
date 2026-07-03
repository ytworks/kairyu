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
from kairyu.bench.types import BenchItem, BenchTarget, ChatRequestSpec, ItemResult, SkipItem

_TEST_TIMEOUT_S = 6.0
_MEMORY_MB = 4096

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


def decode_private_tests(blob: str) -> list[dict]:
    """LCB lite stores private tests either as JSON or zlib+pickle+base64 JSON."""
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode()))))


def grade_code(code: str, tests: list[dict], fn_name: str | None) -> tuple[bool, str]:
    """Run every test in the sandbox; (passed, first failure detail)."""
    for index, test in enumerate(tests):
        if test.get("testtype") == "functional":
            spec = json.dumps(
                {"input": test["input"], "expected": test["output"], "fn": fn_name}
            )
            result = run_python(
                _IMPORT_HEADER + code + _FUNCTIONAL_DRIVER,
                stdin=spec,
                timeout_s=_TEST_TIMEOUT_S,
                memory_mb=_MEMORY_MB,
            )
            if not result.ok:
                detail = "timeout" if result.timed_out else result.stderr[-300:]
                return False, f"functional test {index}: {detail or 'wrong answer'}"
        else:
            result = run_python(
                _IMPORT_HEADER + code,
                stdin=test["input"],
                timeout_s=_TEST_TIMEOUT_S,
                memory_mb=_MEMORY_MB,
            )
            if result.timed_out or result.returncode != 0:
                detail = "timeout" if result.timed_out else result.stderr[-300:]
                return False, f"stdin test {index}: {detail}"
            if result.stdout.strip() != test["output"].strip():
                return False, f"stdin test {index}: wrong answer"
    return True, ""


class LiveCodeBenchAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="livecodebench",
        display_name="LiveCodeBench",
        metric="pass@1",
        hf_dataset="livecodebench/code_generation_lite",
        hf_revision="release_v6",
        needs_execution=True,
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(
            self.info.hf_dataset,
            split="test",
            revision=self.info.hf_revision,
            name="release_v6",
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
        base["execution"] = (
            f"local subprocess sandbox, {_TEST_TIMEOUT_S}s/test, {_MEMORY_MB}MB rlimit; "
            "pass@1 = all public+private tests pass"
        )
        return base
