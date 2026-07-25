"""SciCode: per-sub-step scientific coding, scored by executing the dataset's tests.

Three properties of the published dataset shape this adapter:

1. `SciCode1/SciCode` ships **no reference code at all** — every sub-step's
   `ground_truth_code` and every problem's `general_solution` is null. There is
   therefore no "gold previous steps" setting available, so sub-steps run
   SEQUENTIALLY per problem and each step sees the model's OWN earlier code.
   That is SciCode's main setting, and it is the only faithful one here:
   grading a later step in isolation just raises NameError on the helper the
   earlier step was supposed to define.
2. 288 of the 291 test-split sub-steps compare against golden data (`target`),
   which lives in `test_data.h5` — a file the HF export does not contain. It is
   fetched from the sources below and verified by HDF5 magic bytes; without it
   those sub-steps are recorded "unjudged", never guessed.
3. Fugu evaluates **with background information**, so both the problem-level
   and step-level background are part of the prompt.
"""

from __future__ import annotations

import asyncio
import random

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    attempt_item,
    excerpt,
    extract_code_block,
    skipped_pair,
    summarize_items,
    target_api_key,
    utc_now,
)
from kairyu.bench.sandbox import run_python
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    ItemResult,
    PairResult,
    SkipItem,
)

_TEST_TIMEOUT_S = 60.0
_MEMORY_MB = 4096
_H5_NAME = "test_data.h5"
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
# The golden data is not in the HF export; the second entry is a public,
# ungated third-party mirror pinned to a commit. Provenance is recorded in the
# methodology so a number is never traceable to an unknown file.
_H5_SOURCES: tuple[tuple[str, str | None], ...] = (
    ("SciCode1/SciCode", None),
    ("Srimadh/Scicode-test-data-h5", "72c247d3a8410921b2e848e046d71ed63d9a0ddb"),
)
_TEST_SPLIT_SUBSTEPS = 291
_GOLDEN_SUBSTEPS = 288  # sub-steps whose tests use `target` (Fugu reports 288)

_PROMPT = """You are implementing one step of a scientific computing problem.

PROBLEM:
{main_description}
{main_background}
Already-implemented steps of this problem (available in scope, do not repeat them):
```python
{context_code}
```

Now implement the next step.

{step_description}
{step_background}
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


def _section(label: str, text: object) -> str:
    body = str(text or "").strip()
    return f"\n{label}:\n{body}\n" if body else ""


def group_by_problem(items: list[BenchItem]) -> list[list[BenchItem]]:
    """Sub-steps grouped per problem, each group in dataset step order."""
    groups: dict[str, list[BenchItem]] = {}
    for item in items:
        key = str(item.payload.get("problem_id", item.id))
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        group.sort(key=lambda item: int(item.payload.get("step_index", 0)))
    return list(groups.values())


def select_problem_groups(
    groups: list[list[BenchItem]], limit: int | None, seed: int
) -> list[list[BenchItem]]:
    """Deterministic subset that keeps WHOLE problems.

    A sub-step limit that cut a problem in half would evaluate steps whose
    dependencies were never generated, so the limit is applied at problem
    granularity (at least one problem always survives).
    """
    total = sum(len(group) for group in groups)
    if limit is None or total <= limit:
        return groups
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    picked: list[int] = []
    used = 0
    for index in order:
        size = len(groups[index])
        if picked and used + size > limit:
            continue
        picked.append(index)
        used += size
        if used >= limit:
            break
    return [groups[index] for index in sorted(picked)]


class SciCodeAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="scicode",
        display_name="SciCode",
        metric="sub-problem pass rate",
        hf_dataset="SciCode1/SciCode",
        needs_execution=True,
        annotations=(
            "sub-steps run sequentially per problem on the model's OWN earlier "
            "code: the published dataset ships no reference code, so a "
            "gold-previous-steps setting is not available",
            f"{_GOLDEN_SUBSTEPS} of {_TEST_SPLIT_SUBSTEPS} test-split sub-steps "
            "need test_data.h5 golden data; it is absent from the HF export and "
            "fetched from a pinned public mirror, and any sub-step left without "
            "it is unjudged rather than scored",
            "prompts include problem-level and step-level background (Fugu's "
            "with-background condition)",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(self.info.hf_dataset, split="test")
        golden = self._fetch_golden(ctx)
        normalized = []
        for row in rows:
            deps = row.get("required_dependencies") or ""
            for step_index, step in enumerate(row.get("sub_steps") or []):
                step_id = f"{row['problem_id']}.{step['step_number'].split('.')[-1]}"
                normalized.append(
                    {
                        "id": f"scicode-{step_id}",
                        "problem_id": str(row["problem_id"]),
                        "step_index": step_index,
                        "step_id": step["step_number"],
                        "main_description": row.get("problem_description_main") or "",
                        "main_background": row.get("problem_background_main") or "",
                        "dependencies": deps,
                        "step_description": step.get("step_description_prompt") or "",
                        "step_background": step.get("step_background") or "",
                        "function_header": step.get("function_header") or "",
                        "return_line": step.get("return_line") or "",
                        "test_cases": list(step.get("test_cases") or []),
                        "has_golden_data": golden is not None,
                    }
                )
        return normalized

    def _fetch_golden(self, ctx: DownloadContext):
        """First source that yields a real HDF5 file wins; None if none do."""
        from kairyu.bench.hub import download_file

        dest = ctx.cache.assets_dir(self.info.name) / _H5_NAME
        for repo_id, revision in _H5_SOURCES:
            path = download_file(repo_id, _H5_NAME, dest, revision=revision)
            if path is None:
                continue
            with open(path, "rb") as handle:
                if handle.read(len(_HDF5_MAGIC)) == _HDF5_MAGIC:
                    return path
            path.unlink(missing_ok=True)  # not HDF5: an LFS pointer or an error page
        return None

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
        context_code = f"{payload['dependencies']}\n\n{payload.get('prior_code') or ''}".strip()
        return_line = payload["return_line"]
        prompt = _PROMPT.format(
            main_description=payload["main_description"],
            main_background=_section("BACKGROUND", payload.get("main_background")),
            context_code=context_code or "# (none)",
            step_description=payload["step_description"],
            step_background=_section("STEP BACKGROUND", payload.get("step_background")),
            function_header=payload["function_header"],
            return_line=f"Return: {return_line}\n" if return_line else "",
        )
        return ChatRequestSpec(
            messages=({"role": "user", "content": prompt},),
            max_tokens=target.max_output_tokens,
        )

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        """Problems in parallel, sub-steps sequential within a problem."""
        started_at = utc_now()
        skip_reason = self.check_preconditions(target, ctx)
        if skip_reason is not None:
            return skipped_pair(
                self.info.name,
                target.label(),
                skip_reason,
                annotations=self.info.annotations,
            )

        groups = select_problem_groups(
            group_by_problem(self.load_items(ctx)), ctx.limit, ctx.seed
        )
        semaphore = asyncio.Semaphore(ctx.concurrency)
        api_key = target_api_key(target)

        async with ctx.http_factory() as client:

            async def run_problem(steps: list[BenchItem]) -> list[ItemResult]:
                results: list[ItemResult] = []
                prior: list[str] = []
                for item in steps:
                    staged = item.model_copy(
                        update={
                            "payload": {**item.payload, "prior_code": "\n\n".join(prior)}
                        }
                    )
                    request = self.build_request(staged, target, ctx)
                    if isinstance(request, SkipItem):
                        results.append(
                            ItemResult(
                                item_id=item.id, status="skipped", error=request.reason
                            )
                        )
                        continue
                    async with semaphore:
                        attempt = await attempt_item(
                            client,
                            target,
                            request,
                            item_id=item.id,
                            ctx=ctx,
                            api_key=api_key,
                        )
                    if attempt.failure is not None:
                        results.append(attempt.failure)
                        continue
                    text = attempt.text or ""
                    code = extract_code_block(text)
                    if code is not None:
                        # carried forward pass or fail: SciCode measures the
                        # cascade, so a wrong step stays in the context.
                        prior.append(code)
                    result = await self.score(staged, text, ctx)
                    results.append(
                        result.model_copy(update={"latency_s": round(attempt.latency_s, 3)})
                    )
                return results

            gathered = await asyncio.gather(*(run_problem(group) for group in groups))

        return summarize_items(
            self.info.name,
            target.label(),
            [result for group in gathered for result in group],
            methodology=self.methodology(ctx),
            annotations=self.info.annotations,
            started_at=started_at,
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
        blocks = [
            payload["dependencies"],
            payload.get("prior_code") or "",
            code,
            harness,
        ]
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
        base["evaluation"] = (
            "sequential per problem: each sub-step prompt and executed program "
            "carry the model's own accepted code from the earlier steps of that "
            "problem (the dataset ships no reference code)"
        )
        base["background"] = "problem-level and step-level background included"
        base["limit_granularity"] = "whole problems"
        base["golden_data_sources"] = [
            f"{repo}@{revision or 'main'}" for repo, revision in _H5_SOURCES
        ]
        base["test_split_substeps"] = _TEST_SPLIT_SUBSTEPS
        base["golden_backed_substeps"] = _GOLDEN_SUBSTEPS
        base["execution"] = (
            "dataset test_cases executed in the local sandbox; target-based "
            "tests use test_data.h5 via a compact reimplementation of the "
            "official loader"
        )
        return base
