"""LiveCodeBench Pro slot: competitive-programming problems, sandboxed grading.

Fugu reports the 2025 Q2 slice, so that is the split this adapter pins. Two
upstream facts shape the implementation:

- the problem repo is a GATED parquet dataset keyed by `problem_id`, and
- the testcase repo is one ZIP per problem holding `testdata/<n>.in` /
  `testdata/<n>.ans` plus a per-problem testlib `checker.cpp`.

Kairyu does not compile the checker, so grading is the shared per-line
whitespace-normalized comparison against `.ans`. That is exact for
single-answer problems and a LOWER BOUND for any problem accepting several
valid outputs — annotated on every report rather than presented as the official
Accepted rate.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

from kairyu.bench.adapters.base import AdapterInfo, DownloadContext, RunContext
from kairyu.bench.adapters.livecodebench import LiveCodeBenchAdapter
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    SkipItem,
)

# Fugu's condition: 2025 Q2, text only, no tools.
_SPLIT = "quater_2025_4_6"
_SPLIT_PROBLEMS = 167
_PROBLEM_REVISION = "adebffce047dddb7768a86bace6aea4f7425e3bc"
_TESTCASE_REVISION = "5257736c0a4e30ba0949d41c56a257c323d9c600"
_KNOWN_MISSING_TESTCASES = frozenset({"2086F", "2101F", "2109B"})
_DEFAULT_INPUT_SUFFIX = ".in"
_DEFAULT_OUTPUT_SUFFIX = ".ans"


@dataclass(frozen=True)
class TestcaseArchive:
    """Parsed testcase ZIP: the cases, and how many the archive declares.

    `declared_cases` is `sum(subtasks[].n_cases)` from `config.yaml`. It is what
    makes "did I get the whole archive" answerable: a truncated download or an
    unpaired `.in` would otherwise just shrink the denominator.
    """

    tests: list[dict]
    declared_cases: int | None
    #: input or output halves with no counterpart, either direction
    unpaired: tuple[str, ...] = ()
    #: paired numeric files beyond the config-declared official denominator
    ignored_extra_cases: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Complete only when the archive's OWN declared count is met.

        `sum(subtasks[].n_cases)` is the only denominator evidence there is, so a
        missing, renamed or malformed `config.yaml` cannot be treated as "as many
        cases as happened to arrive": that would let a truncated or
        schema-drifted archive pass as a valid benchmark. Every archive in the
        pinned source declares it.
        """
        if self.unpaired:
            return False
        if self.declared_cases is None:
            return False
        return len(self.tests) == self.declared_cases


def read_testcase_archive(data: bytes) -> TestcaseArchive:
    """ZIP bytes -> cases ordered by numeric name, plus completeness evidence.

    `config.yaml` carries the suffixes; both default to the observed
    `.in`/`.ans`.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        input_suffix, output_suffix = _DEFAULT_INPUT_SUFFIX, _DEFAULT_OUTPUT_SUFFIX
        declared: int | None = None
        if "config.yaml" in names:
            import yaml

            config = yaml.safe_load(archive.read("config.yaml").decode("utf-8")) or {}
            if isinstance(config, dict):
                input_suffix = config.get("input_suffix") or input_suffix
                output_suffix = config.get("output_suffix") or output_suffix
                subtasks = config.get("subtasks")
                if isinstance(subtasks, list):
                    counts = [
                        subtask.get("n_cases")
                        for subtask in subtasks
                        if isinstance(subtask, dict)
                        and isinstance(subtask.get("n_cases"), int)
                    ]
                    if counts:
                        declared = sum(counts)

        stems: list[tuple[float, str, str]] = []
        unpaired: list[str] = []
        for name in names:
            if name.endswith("/"):
                continue
            if name.endswith(input_suffix):
                stem = name[: -len(input_suffix)]
                if f"{stem}{output_suffix}" not in names:
                    unpaired.append(name)
                    continue
                leaf = stem.rsplit("/", maxsplit=1)[-1]
                order = float(leaf) if leaf.isdigit() else float("inf")
                stems.append((order, leaf, stem))
            elif name.endswith(output_suffix):
                # an expected output with no input is drift too, in the other
                # direction: the case exists but can never be run
                stem = name[: -len(output_suffix)]
                if f"{stem}{input_suffix}" not in names:
                    unpaired.append(name)

        ordered_stems = sorted(stems)
        ignored_extra_cases: tuple[str, ...] = ()
        if declared is not None:
            by_leaf = {leaf: stem for _, leaf, stem in ordered_stems}
            expected_leaves = tuple(str(index) for index in range(1, declared + 1))
            selected_stems = [
                (float(leaf), leaf, by_leaf[leaf])
                for leaf in expected_leaves
                if leaf in by_leaf
            ]
            ignored_extra_cases = tuple(
                leaf for _, leaf, _ in ordered_stems if leaf not in expected_leaves
            )
        else:
            selected_stems = ordered_stems

        tests = []
        for _, _, stem in selected_stems:
            tests.append(
                {
                    "input": archive.read(f"{stem}{input_suffix}").decode(
                        "utf-8", errors="replace"
                    ),
                    "output": archive.read(f"{stem}{output_suffix}").decode(
                        "utf-8", errors="replace"
                    ),
                    "testtype": "stdin",
                }
            )
        return TestcaseArchive(
            tests=tests,
            declared_cases=declared,
            unpaired=tuple(sorted(unpaired)),
            ignored_extra_cases=ignored_extra_cases,
        )


def parse_testcase_zip(data: bytes) -> list[dict]:
    """Cases only; see `read_testcase_archive` for completeness evidence."""
    return read_testcase_archive(data).tests


class LiveCodeBenchProAdapter(LiveCodeBenchAdapter):
    info = AdapterInfo(
        name="livecodebench-pro",
        display_name="LiveCodeBench Pro",
        metric="pass@1",
        binary_outcomes=True,
        hf_dataset="QAQAQAQAQ/LiveCodeBench-Pro",
        hf_revision=_PROBLEM_REVISION,
        # The archives decide the tests, so repinning them must invalidate the
        # cache rather than leave stale bytes "ready" under a new methodology.
        extra_sources=(("QAQAQAQAQ/LiveCodeBench-Pro-Testcase", _TESTCASE_REVISION),),
        gated=True,
        needs_execution=True,
        annotations=(
            f"split {_SPLIT} ({_SPLIT_PROBLEMS} problems) — Fugu's 2025 Q2 slice, "
            "text only, no tools",
            "graded by per-line whitespace-normalized comparison against the "
            "shipped .ans files, NOT the official LightCPVerifier: the per-problem "
            "testlib checker.cpp is not compiled, so multi-answer problems can only "
            "lose points — treat the score as a lower bound",
            "solutions are requested in Python; Fugu reports C++ solutions",
            "the pinned official testcase repository has no archives for "
            f"{', '.join(sorted(_KNOWN_MISSING_TESTCASES))}; those source rows "
            "are retained and skipped rather than shrinking the 167-row split",
        ),
        comparable_to_published=False,
        incomparable_reason=(
            "the pinned official testcase repository omits 3 of the 167 source "
            "problems"
        ),
    )

    _TESTCASE_DATASET = "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import download_file, load_hf_rows

        problems = load_hf_rows(
            self.info.hf_dataset,
            split=_SPLIT,
            revision=self.info.hf_revision,
            gated=True,
        )
        # A short problem list means the split moved or the fetch was partial;
        # either way the denominator would silently shrink.
        if len(problems) != _SPLIT_PROBLEMS:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} {_SPLIT} yielded "
                f"{len(problems)} problems, expected {_SPLIT_PROBLEMS}"
            )

        assets = ctx.cache.assets_dir(self.info.name)
        normalized: list[dict] = []
        for row in problems:
            key = row.get("problem_id")
            statement = row.get("problem_statement")
            if not key or not statement:
                raise DatasetUnavailable(
                    f"{self.info.hf_dataset} format drift: expected problem_id + "
                    f"problem_statement, got fields {sorted(row)}"
                )
            key = str(key)
            archive = download_file(
                self._TESTCASE_DATASET,
                f"{key}.zip",
                assets / f"{key}.zip",
                revision=_TESTCASE_REVISION,
            )
            # `download_file` maps every failure to None -- 404, but equally a
            # timeout, a 401, or a rate limit. Excluding the problem would cache
            # a smaller denominator as if it were the whole split, and the
            # resulting rate is not even a lower bound on the full 167.
            if archive is None:
                if key in _KNOWN_MISSING_TESTCASES:
                    normalized.append(
                        {
                            "id": f"lcb-pro-{key}",
                            "question": statement,
                            "starter_code": "",
                            "fn_name": None,
                            "tests": [],
                            "testcase_unavailable": (
                                f"official testcase archive {key}.zip is absent from "
                                f"{self._TESTCASE_DATASET}@{_TESTCASE_REVISION}"
                            ),
                            "time_limit": row.get("time_limit"),
                            "memory_limit": row.get("memory_limit"),
                        }
                    )
                    continue
                raise DatasetUnavailable(
                    f"{self._TESTCASE_DATASET}@{_TESTCASE_REVISION} has no usable "
                    f"archive for problem {key} (missing, or the fetch failed); "
                    "retry rather than score a partial split"
                )
            parsed = read_testcase_archive(archive.read_bytes())
            archive.unlink(missing_ok=True)  # keep only the normalized JSONL
            if not parsed.complete:
                declared = (
                    "no n_cases declared in config.yaml"
                    if parsed.declared_cases is None
                    else f"{parsed.declared_cases} declared in config.yaml"
                )
                raise DatasetUnavailable(
                    f"{self._TESTCASE_DATASET} archive for problem {key} is "
                    f"incomplete: {len(parsed.tests)} usable cases, {declared}"
                    + (f", unpaired {list(parsed.unpaired)}" if parsed.unpaired else "")
                )
            normalized.append(
                {
                    "id": f"lcb-pro-{key}",
                    "question": statement,
                    "starter_code": "",
                    "fn_name": None,
                    "tests": parsed.tests,
                    "declared_cases": parsed.declared_cases,
                    "ignored_extra_cases": list(parsed.ignored_extra_cases),
                    "time_limit": row.get("time_limit"),
                    "memory_limit": row.get("memory_limit"),
                }
            )
        return normalized

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        reason = item.payload.get("testcase_unavailable")
        if reason:
            return SkipItem(reason=reason)
        return super().build_request(item, target, ctx)

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base.pop("release", None)
        base.pop("release_problems", None)
        base["split"] = _SPLIT
        base["split_problems"] = _SPLIT_PROBLEMS
        base["testcases"] = f"{self._TESTCASE_DATASET}@{_TESTCASE_REVISION}"
        base["missing_testcase_archives"] = sorted(_KNOWN_MISSING_TESTCASES)
        base["grading"] = (
            "per-line whitespace-normalized comparison against .ans; the shipped "
            "testlib checker.cpp is not compiled (lower bound)"
        )
        return base
