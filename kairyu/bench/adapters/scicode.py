"""SciCode: per-sub-step scientific coding, scored by executing the dataset's tests.

One item per sub-step; the prompt provides the problem plus the GOLD code of
previous steps (so steps stay independent and parallelizable). Test cases
that compare against the official golden data (`target`) need the dataset's
test_data.h5 — fetched at download time when available; otherwise those
sub-steps are recorded "unjudged", never guessed.
"""

from __future__ import annotations

import asyncio

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

_TEST_TIMEOUT_S = 60.0
_MEMORY_MB = 4096
_H5_NAME = "test_data.h5"

_PROMPT = """You are implementing one step of a scientific computing problem.

PROBLEM:
{main_description}

Already-implemented helper code (available in scope, do not repeat it):
```python
{context_code}
```

Now implement the next step.

{step_description}

Function header (keep it exactly):
```python
{function_header}
```
{return_line}
Write ONLY this function's implementation (plus any imports you need) in a
single ```python code block```."""

# Compact reconstruction of the official process_hdf5_to_tuple loader:
# dataset -> array/scalar, group -> tuple of children in key order.
_H5_LOADER = """

def process_hdf5_to_tuple(step_id, test_num, h5file="test_data.h5"):
    import h5py
    import numpy as np

    def load_node(node):
        if isinstance(node, h5py.Dataset):
            value = node[()]
            if isinstance(value, bytes):
                return value.decode()
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else arr
        keys = list(node.keys())
        return tuple(load_node(node[key]) for key in keys)

    results = []
    with h5py.File(h5file, "r") as handle:
        for test_id in range(1, test_num + 1):
            node = handle[f"{step_id}/test{test_id}"]
            loaded = load_node(node)
            if isinstance(loaded, tuple) and len(loaded) == 1:
                loaded = loaded[0]
            results.append(loaded)
    return results
"""


class SciCodeAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="scicode",
        display_name="SciCode",
        metric="sub-problem pass rate",
        hf_dataset="SciCode1/SciCode",
        needs_execution=True,
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import download_file, load_hf_rows

        rows = load_hf_rows(self.info.hf_dataset, split="test")
        golden = download_file(
            self.info.hf_dataset,
            _H5_NAME,
            ctx.cache.assets_dir(self.info.name) / _H5_NAME,
        )
        normalized = []
        for row in rows:
            prior_gold: list[str] = []
            deps = row.get("required_dependencies") or ""
            for step in row.get("sub_steps") or []:
                step_id = f"{row['problem_id']}.{step['step_number'].split('.')[-1]}"
                normalized.append(
                    {
                        "id": f"scicode-{step_id}",
                        "step_id": step["step_number"],
                        "main_description": row.get("problem_description_main") or "",
                        "dependencies": deps,
                        "prior_code": "\n\n".join(prior_gold),
                        "step_description": step.get("step_description_prompt") or "",
                        "function_header": step.get("function_header") or "",
                        "return_line": step.get("return_line") or "",
                        "test_cases": list(step.get("test_cases") or []),
                        "has_golden_data": golden is not None,
                    }
                )
                gold_code = step.get("ground_truth_code")
                if gold_code:
                    prior_gold.append(gold_code)
        return normalized

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        reason = super().check_preconditions(target, ctx)
        if reason is not None:
            return reason
        from kairyu.bench.sandbox import has_module

        if not has_module("numpy"):
            return "sandbox interpreter lacks numpy (pip install numpy)"
        return None

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        payload = item.payload
        context_code = f"{payload['dependencies']}\n\n{payload['prior_code']}".strip()
        return_line = payload["return_line"]
        prompt = _PROMPT.format(
            main_description=payload["main_description"],
            context_code=context_code or "# (none)",
            step_description=payload["step_description"],
            function_header=payload["function_header"],
            return_line=f"Return: {return_line}\n" if return_line else "",
        )
        return ChatRequestSpec(
            messages=({"role": "user", "content": prompt},),
            max_tokens=target.max_output_tokens,
        )

    def _needs_golden(self, tests: list[str]) -> bool:
        return any("target" in test for test in tests)

    def _h5_bytes(self, ctx: RunContext) -> bytes | None:
        path = ctx.cache.assets_dir(self.info.name) / _H5_NAME
        return path.read_bytes() if path.exists() else None

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        payload = item.payload
        code = extract_code_block(response_text)
        if code is None:
            return ItemResult(
                item_id=item.id,
                status="completed",
                score=0.0,
                error="no code block in response",
                response_excerpt=excerpt(response_text),
            )
        tests = payload["test_cases"]
        files: dict[str, bytes] = {}
        needs_golden = self._needs_golden(tests)
        harness = ""
        if needs_golden:
            h5 = self._h5_bytes(ctx) if not ctx.offline_fixtures else None
            if h5 is None:
                return ItemResult(
                    item_id=item.id,
                    status="unjudged",
                    error="official golden data (test_data.h5) not available",
                    response_excerpt=excerpt(response_text),
                )
            files[_H5_NAME] = h5
            harness = (
                _H5_LOADER
                + f"\ntargets = process_hdf5_to_tuple({payload['step_id']!r}, {len(tests)})\n"
            )
        blocks = [payload["dependencies"], payload["prior_code"], code, harness]
        for index, test in enumerate(tests):
            if needs_golden:
                blocks.append(f"target = targets[{index}]")
            blocks.append(test)
        program = "\n\n".join(block for block in blocks if block)
        async with ctx.exec_semaphore:
            result = await asyncio.to_thread(
                run_python,
                program,
                timeout_s=_TEST_TIMEOUT_S,
                memory_mb=_MEMORY_MB,
                files=files,
            )
        detail = None
        if not result.ok:
            detail = "timeout" if result.timed_out else result.stderr[-300:]
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if result.ok else 0.0,
            error=detail,
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base["execution"] = (
            "per-sub-step: gold prior-step code in prompt and in scope; dataset "
            "test_cases executed in the local sandbox; target-based tests use "
            "test_data.h5 via a compact reimplementation of the official loader"
        )
        return base
