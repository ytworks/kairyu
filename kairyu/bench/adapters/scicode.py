"""SciCode: per-sub-step scientific coding, scored by executing the dataset's tests.

Three properties of the published dataset shape this adapter:

1. `SciCode1/SciCode` ships **no reference code at all** — every sub-step's
   `ground_truth_code` and every problem's `general_solution` is null. There is
   therefore no "gold previous steps" setting available, so sub-steps run
   SEQUENTIALLY per problem and each step sees the model's OWN earlier code.
   That is SciCode's main setting, and it is the only faithful one here:
   grading a later step in isolation just raises NameError on the helper the
   earlier step was supposed to define.
2. The official evaluator skips three sub-steps and supplies their code, so the
   scored population -- and Fugu's denominator -- is 288 of 291. Nearly all of
   those compare against golden data (`target`) from `test_data.h5`, which the HF
   export does not contain; it is fetched from the sources below and accepted
   only at a pinned content hash. Without it those sub-steps are recorded
   "unjudged", never guessed.
3. Fugu evaluates **with background information**, so both the problem-level
   and step-level background are part of the prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from pathlib import Path

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
from kairyu.bench.sampling import combine_attempts, seed_schedule
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    ItemResult,
    PairResult,
    SamplingAttemptResult,
    SkipItem,
)

_TEST_TIMEOUT_S = 60.0
_MEMORY_MB = 4096
_H5_NAME = "test_data.h5"
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
# The golden data is not in the HF export. The mirror below is public and
# ungated, pinned by BOTH its commit and the SHA-256 of its content: magic bytes
# only prove the file format, so a different-but-valid HDF5 would otherwise be
# accepted as every expected value in the benchmark.
#
# NOT independently cross-checked against the official Google Drive artifact —
# this pins WHICH bytes were used, not that they are the upstream ones. The
# methodology says so, and `--h5-source` style overrides are deliberately absent.
_H5_SOURCES: tuple[tuple[str, str | None], ...] = (
    ("SciCode1/SciCode", None),
    ("Srimadh/Scicode-test-data-h5", "72c247d3a8410921b2e848e046d71ed63d9a0ddb"),
)
_H5_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
_H5_BYTES = 1_049_345_865

_TEST_SPLIT_SUBSTEPS = 291
# The official evaluator `continue`s past three sub-steps (0-based indices), so
# its denominator -- and Fugu's -- is 288. They are not scored; instead the
# repository ships their implementation as a text file, which later steps of the
# same problem depend on.
_EXCLUDED_STEPS: frozenset[tuple[str, int]] = frozenset(
    {("13", 5), ("62", 0), ("76", 2)}
)
_GOLDEN_SUBSTEPS = _TEST_SPLIT_SUBSTEPS - len(_EXCLUDED_STEPS)
# Pinned by content: these stand in for steps the model never generates.
_PROVIDED_STEP_URL = (
    "https://raw.githubusercontent.com/scicode-bench/SciCode/main/eval/data/{name}"
)
_PROVIDED_STEPS: dict[tuple[str, int], tuple[str, str]] = {
    ("13", 5): (
        "13.6.txt",
        "795a2b57c2d9bb12ca4eaf16d6b8e1f202015a89a886628858abf42a1b18a94e",
    ),
    ("62", 0): (
        "62.1.txt",
        "bc9931d88a7d5950091b72a996a25b8be6c936fd136b01005e22c3d45b0008a2",
    ),
    ("76", 2): (
        "76.3.txt",
        "4758300d96ea726cdc0fbf749f1bc437030d2a3e8b43bde232ecdeaf636cd367",
    ),
}

_PROMPT = """You are implementing one step of a scientific computing problem.

PROBLEM:
{main_description}
{main_background}
Already-implemented steps of this problem (available in scope, do not repeat them):
{previous_steps}
Dependencies available in scope:
```python
{dependencies}
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

# The official `process_problem_steps()` gives each prior step its description
# (plus background in the with-background condition) alongside its code, with the
# steps separated by "------". Passing only the concatenated code loses the
# statement of what each helper was for.
_PREVIOUS_STEP = """{description}{background}
```python
{code}
```"""
_STEP_SEPARATOR = "\n------\n"

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def golden_data_is_authentic(path: Path) -> bool:
    """The file is HDF5 AND hashes to the pinned digest at the pinned size."""
    try:
        if path.stat().st_size != _H5_BYTES:
            return False
        with path.open("rb") as handle:
            if handle.read(len(_HDF5_MAGIC)) != _HDF5_MAGIC:
                return False
        return sha256_file(path) == _H5_SHA256
    except OSError:
        return False


def _section(label: str, text: object) -> str:
    body = str(text or "").strip()
    return f"\n{label}:\n{body}\n" if body else ""


def render_previous_steps(payload: dict) -> str:
    """Prior steps as the official prompt renders them: description + code.

    Falls back to the concatenated code when no per-step history is available
    (a directly constructed item, or a fixture row).
    """
    history = payload.get("prior_steps")
    if not history:
        code = (payload.get("prior_code") or "").strip()
        return f"```python\n{code}\n```" if code else "# (none)"
    blocks = []
    for step in history:
        blocks.append(
            _PREVIOUS_STEP.format(
                description=str(step.get("step_description") or "").strip(),
                background=_section(
                    "STEP BACKGROUND", step.get("step_background")
                ).rstrip(),
                code=step.get("code") or "",
            )
        )
    return _STEP_SEPARATOR.join(blocks)


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
        binary_outcomes=True,
        paired_cluster_key="item_id_before_final_dot_v1",
        hf_dataset="SciCode1/SciCode",
        needs_execution=True,
        annotations=(
            "sub-steps run sequentially per problem on the model's OWN earlier "
            "code: the published dataset ships no reference code, so a "
            "gold-previous-steps setting is not available",
            f"scored population is {_GOLDEN_SUBSTEPS} of {_TEST_SPLIT_SUBSTEPS} "
            "test-split sub-steps: the official evaluator skips three, whose "
            "implementation it supplies instead (Fugu also reports "
            f"{_GOLDEN_SUBSTEPS})",
            "test_data.h5 golden data is absent from the HF export; it is fetched "
            "from a content-hash-pinned public mirror, NOT cross-checked against "
            "the official artifact, and any sub-step left without it is unjudged",
            "prompts include problem-level and step-level background (Fugu's "
            "with-background condition)",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(
            self.info.hf_dataset, split="test", revision=self.info.hf_revision
        )
        golden = self._fetch_golden(ctx)
        provided = self._fetch_provided_steps()
        normalized = []
        seen_substeps = 0
        for row in rows:
            deps = row.get("required_dependencies") or ""
            problem_id = str(row["problem_id"])
            for step_index, step in enumerate(row.get("sub_steps") or []):
                seen_substeps += 1
                step_id = f"{problem_id}.{step['step_number'].split('.')[-1]}"
                excluded = (problem_id, step_index) in _EXCLUDED_STEPS
                normalized.append(
                    {
                        "id": f"scicode-{step_id}",
                        "problem_id": problem_id,
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
                        # Not scored by the official evaluator; its implementation
                        # is provided so later steps of this problem still have it.
                        "excluded": excluded,
                        "provided_code": provided.get((problem_id, step_index), ""),
                    }
                )
        if seen_substeps != _TEST_SPLIT_SUBSTEPS:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} test split has "
                f"{seen_substeps} sub-steps, expected {_TEST_SPLIT_SUBSTEPS}"
            )
        scoreable = sum(1 for row in normalized if not row["excluded"])
        if scoreable != _GOLDEN_SUBSTEPS:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset} yielded {scoreable} scoreable sub-steps, "
                f"expected {_GOLDEN_SUBSTEPS} (the official evaluator skips "
                f"{sorted(_EXCLUDED_STEPS)})"
            )
        missing = [key for key in _PROVIDED_STEPS if key not in provided]
        if missing:
            # Without these, later steps of problems 13/62/76 call helpers nothing
            # defines and would score zero -- polluting the 288-step denominator
            # with failures that are the harness's fault, not the model's.
            raise DatasetUnavailable(
                "SciCode's official provided implementations are unavailable for "
                f"{[f'problem {p} step {i + 1}' for p, i in sorted(missing)]} "
                "(fetch failed, or the content hash did not match); later steps of "
                "those problems depend on them, so scoring would be wrong"
            )
        return normalized

    def _fetch_provided_steps(self) -> dict[tuple[str, int], str]:
        """The three implementations the official evaluator supplies, by content hash.

        Later steps of problems 13/62/76 call helpers these steps define, so
        without them those steps could only raise NameError.
        """
        import urllib.error
        import urllib.request

        fetched: dict[tuple[str, int], str] = {}
        for key, (name, digest) in _PROVIDED_STEPS.items():
            try:
                with urllib.request.urlopen(  # noqa: S310 - fixed https URL
                    _PROVIDED_STEP_URL.format(name=name), timeout=30
                ) as response:
                    body = response.read()
            except (urllib.error.URLError, OSError, ValueError):
                continue
            if hashlib.sha256(body).hexdigest() != digest:
                continue  # not the pinned artifact
            fetched[key] = body.decode("utf-8", errors="replace")
        return fetched

    def _fetch_golden(self, ctx: DownloadContext):
        """First source whose bytes hash to the pinned digest wins.

        HDF5 magic bytes only prove the format; the content hash is what makes
        "these are the expected values I scored against" checkable.
        """
        from kairyu.bench.hub import download_file

        dest = ctx.cache.assets_dir(self.info.name) / _H5_NAME
        for repo_id, revision in _H5_SOURCES:
            # the upstream repo is the slot's own dataset, so it uses the slot's
            # pin; the mirror carries its own
            if revision is None and repo_id == self.info.hf_dataset:
                revision = self.info.hf_revision
            path = download_file(repo_id, _H5_NAME, dest, revision=revision)
            if path is None:
                continue
            if golden_data_is_authentic(path):
                return path
            # not the pinned artifact: an LFS pointer, an error page, or a
            # different HDF5 file entirely
            path.unlink(missing_ok=True)
        return None

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        reason = super().check_preconditions(target, ctx)
        if reason is not None:
            return reason
        available, detail = ctx.execution_runner.availability()
        if not available:
            return f"selected execution runner unavailable ({detail})"
        if not ctx.execution_runner.has_module("numpy"):
            return "selected execution runner lacks numpy (pip install numpy)"
        return None

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        payload = item.payload
        return_line = payload["return_line"]
        prompt = _PROMPT.format(
            main_description=payload["main_description"],
            main_background=_section("BACKGROUND", payload.get("main_background")),
            previous_steps=render_previous_steps(payload),
            dependencies=payload["dependencies"] or "# (none)",
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

        # one validation of the ~1 GB golden asset per pair, not per sub-step
        self._h5_cache = None
        groups = select_problem_groups(
            group_by_problem(self.load_items(ctx)), ctx.limit, ctx.seed
        )
        seeds = seed_schedule(target.seed, ctx.attempts)
        scored_items = sum(
            not item.payload.get("excluded") for group in groups for item in group
        )
        ctx.progress.items_total(scored_items * ctx.attempts)
        semaphore = asyncio.Semaphore(ctx.concurrency)
        api_key = target_api_key(target)

        async with ctx.http_factory() as client:

            async def run_problem(
                steps: list[BenchItem],
                *,
                sampling_seed: int | None = None,
            ) -> list[ItemResult]:
                results: list[ItemResult] = []
                prior: list[str] = []
                history: list[dict] = []
                for item in steps:
                    if item.payload.get("excluded"):
                        # The official evaluator does not score these; it supplies
                        # their implementation so later steps can build on it.
                        provided = item.payload.get("provided_code") or ""
                        if provided:
                            prior.append(provided)
                            history.append({**item.payload, "code": provided})
                        continue
                    staged = item.model_copy(
                        update={
                            "payload": {
                                **item.payload,
                                "prior_code": "\n\n".join(prior),
                                "prior_steps": list(history),
                            }
                        }
                    )
                    request = self.build_request(staged, target, ctx)
                    if isinstance(request, SkipItem):
                        results.append(
                            ItemResult(
                                item_id=item.id, status="skipped", error=request.reason
                            )
                        )
                        ctx.progress.item_done()
                        continue
                    try:
                        async with semaphore:
                            attempt = await attempt_item(
                                client,
                                target,
                                request,
                                item_id=item.id,
                                ctx=ctx,
                                api_key=api_key,
                                sampling_seed=sampling_seed,
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
                            history.append({**item.payload, "code": code})
                        result = await self.score(staged, text, ctx)
                        results.append(
                            result.model_copy(
                                update={
                                    "item_id": item.id,
                                    "latency_s": round(attempt.latency_s, 3),
                                }
                            )
                        )
                    finally:
                        ctx.progress.item_done()
                return results

            if ctx.attempts == 1:
                gathered = await asyncio.gather(*(run_problem(group) for group in groups))
                results = [result for group in gathered for result in group]
            else:
                gathered = await asyncio.gather(
                    *(
                        run_problem(group, sampling_seed=int(seed))
                        for seed in seeds
                        for group in groups
                    )
                )
                per_attempt: list[list[ItemResult]] = []
                for attempt_index in range(ctx.attempts):
                    start = attempt_index * len(groups)
                    selected = gathered[start : start + len(groups)]
                    per_attempt.append(
                        [result for group_results in selected for result in group_results]
                    )
                if any(len(row) != len(per_attempt[0]) for row in per_attempt):
                    raise RuntimeError("SciCode sampling attempts produced different item sets")
                results = []
                for item_index, first in enumerate(per_attempt[0]):
                    retained = tuple(
                        SamplingAttemptResult(
                            attempt=attempt_index + 1,
                            seed=int(seeds[attempt_index]),
                            result=per_attempt[attempt_index][item_index],
                        )
                        for attempt_index in range(ctx.attempts)
                    )
                    results.append(combine_attempts(first.item_id, retained))

        pair = summarize_items(
            self.info.name,
            target.label(),
            results,
            methodology=self.pair_methodology(target, ctx),
            annotations=self.info.annotations,
            started_at=started_at,
        )
        return self._attach_sampling_metrics(pair, target, ctx)

    def _needs_golden(self, tests: list[str]) -> bool:
        return any("target" in test for test in tests)

    def _h5_bytes(self, ctx: RunContext) -> bytes | None:
        """Cached golden data, revalidated against the pin before it is trusted.

        Checking only at download time would let a replaced or truncated asset
        become the expected-answer source under a manifest that still advertises
        the pinned hash. Cached per pair, not per item: the file is ~1 GB.
        """
        cached = getattr(self, "_h5_cache", None)
        if cached is not None:
            return cached or None
        path = ctx.cache.assets_dir(self.info.name) / _H5_NAME
        data = b""
        if path.exists() and golden_data_is_authentic(path):
            data = path.read_bytes()
        self._h5_cache = data
        return data or None

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
                    error=(
                        "official golden data (test_data.h5) not available or does "
                        "not match its pinned hash"
                    ),
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
                ctx.execution_runner.run_python,
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
        base["golden_data_sha256"] = _H5_SHA256
        base["golden_data_provenance"] = (
            "content-hash pinned; NOT independently cross-checked against the "
            "official Google Drive artifact"
        )
        base["test_split_substeps"] = _TEST_SPLIT_SUBSTEPS
        base["scored_substeps"] = _GOLDEN_SUBSTEPS
        base["excluded_substeps"] = [
            f"problem {problem} step {index + 1}"
            for problem, index in sorted(_EXCLUDED_STEPS)
        ]
        base["previous_step_prompt"] = (
            "description + background + generated code per prior step, separated "
            "by ------ (official process_problem_steps)"
        )
        base["execution"] = {
            "runner": ctx.execution_runner.metadata(),
            "per_test_limits": {
                "timeout_s": _TEST_TIMEOUT_S,
                "memory_mb": _MEMORY_MB,
            },
            "scoring": (
                "dataset test_cases executed by the selected runner; target-based "
                "tests use test_data.h5 via a compact reimplementation of the "
                "official loader"
            ),
        }
        return base
