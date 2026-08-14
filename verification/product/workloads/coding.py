"""Pinned competitive-programming prompt and local pass/fail grader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from verification.product.workloads.sandbox import run_python

_TEST_TIMEOUT_S = 6.0
_MEMORY_MB = 4096
_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

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

_STDIN_INSTRUCTION = "Read input from standard input and write the answer to standard output."
_FUNCTIONAL_INSTRUCTION = (
    "Complete the starter code; keep the class and method signature unchanged."
)


@dataclass(frozen=True, slots=True)
class CodeItem:
    id: str
    payload: dict


@dataclass(frozen=True, slots=True)
class CodeRequest:
    messages: tuple[dict, ...]
    max_tokens: int


def build_request(item: CodeItem, *, max_tokens: int) -> CodeRequest:
    payload = item.payload
    starter = payload.get("starter_code") or ""
    functional = any(test.get("testtype") == "functional" for test in payload["tests"])
    prompt = _PROMPT.format(
        io_instruction=_FUNCTIONAL_INSTRUCTION if functional else _STDIN_INSTRUCTION,
        question=payload["question"],
        starter_section=(f"\n### Starter code:\n```python\n{starter}\n```\n" if starter else ""),
    )
    return CodeRequest(
        messages=({"role": "user", "content": prompt},),
        max_tokens=max_tokens,
    )


def extract_code_block(text: str) -> str | None:
    blocks = _CODE_BLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def _compose_solution(code: str, driver: str = "") -> str:
    future: list[str] = []
    body: list[str] = []
    for line in code.splitlines(keepends=True):
        target = future if line.lstrip().startswith("from __future__ import") else body
        target.append(line)
    return "".join(future) + _IMPORT_HEADER + "".join(body) + driver


def _normalize_output(text: str) -> list[str]:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def grade_code(
    code: str,
    tests: list[dict],
    fn_name: str | None,
) -> tuple[bool, str]:
    for index, test in enumerate(tests):
        if test.get("testtype") == "functional":
            spec = json.dumps({"input": test["input"], "expected": test["output"], "fn": fn_name})
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
            if _normalize_output(result.stdout) != _normalize_output(test["output"]):
                return False, f"stdin test {index}: wrong answer"
    return True, ""


__all__ = [
    "CodeItem",
    "CodeRequest",
    "build_request",
    "extract_code_block",
    "grade_code",
]
