"""LiveCodeBench Pro slot: same sandboxed pass@1 scorer, harder problem set.

Problems and testcases live in separate community repos whose schema has
drifted before; any fetch/format failure degrades to
`skipped: dataset unavailable` at run time (mirrors the docker-skip pattern).
"""

from __future__ import annotations

from kairyu.bench.adapters.base import AdapterInfo, DownloadContext
from kairyu.bench.adapters.livecodebench import LiveCodeBenchAdapter
from kairyu.bench.types import DatasetUnavailable


class LiveCodeBenchProAdapter(LiveCodeBenchAdapter):
    info = AdapterInfo(
        name="livecodebench-pro",
        display_name="LiveCodeBench Pro",
        metric="pass@1",
        hf_dataset="QAQAQAQAQ/LiveCodeBench-Pro",
        needs_execution=True,
        annotations=(
            "community-hosted problem/testcase mirror of LiveCodeBench Pro; "
            "scored with the local sandbox scorer, not the official OJ",
        ),
    )

    _TESTCASE_DATASET = "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        problems = load_hf_rows(self.info.hf_dataset, split="train")
        testcases = load_hf_rows(self._TESTCASE_DATASET, split="train")

        def problem_key(row: dict) -> str | None:
            for key in ("problem_id", "question_id", "id"):
                if row.get(key) is not None:
                    return str(row[key])
            return None

        cases_by_problem: dict[str, list[dict]] = {}
        for row in testcases:
            key = problem_key(row)
            if key is None or "input" not in row or "output" not in row:
                raise DatasetUnavailable(
                    f"{self._TESTCASE_DATASET} format drift: expected "
                    "problem_id/input/output fields"
                )
            cases_by_problem.setdefault(key, []).append(
                {"input": row["input"], "output": row["output"], "testtype": "stdin"}
            )

        normalized = []
        for row in problems:
            key = problem_key(row)
            statement = row.get("problem_statement") or row.get("question_content")
            if key is None or statement is None:
                raise DatasetUnavailable(
                    f"{self.info.hf_dataset} format drift: expected "
                    "problem_id + problem_statement fields"
                )
            tests = cases_by_problem.get(key)
            if not tests:
                continue  # statement without testcases cannot be graded
            normalized.append(
                {
                    "id": f"lcb-pro-{key}",
                    "question": statement,
                    "starter_code": "",
                    "fn_name": None,
                    "tests": tests,
                }
            )
        if not normalized:
            raise DatasetUnavailable(
                "LiveCodeBench-Pro problems and testcases did not join on problem_id"
            )
        return normalized
