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

from kairyu.bench.adapters.base import AdapterInfo, DownloadContext, RunContext
from kairyu.bench.adapters.livecodebench import LiveCodeBenchAdapter
from kairyu.bench.types import DatasetUnavailable

# Fugu's condition: 2025 Q2, text only, no tools.
_SPLIT = "quater_2025_4_6"
_SPLIT_PROBLEMS = 167
_PROBLEM_REVISION = "adebffce047dddb7768a86bace6aea4f7425e3bc"
_TESTCASE_REVISION = "5257736c0a4e30ba0949d41c56a257c323d9c600"
_DEFAULT_INPUT_SUFFIX = ".in"
_DEFAULT_OUTPUT_SUFFIX = ".ans"


def parse_testcase_zip(data: bytes) -> list[dict]:
    """ZIP bytes -> stdin test rows, ordered by numeric case name.

    `config.yaml` carries the suffixes; both default to the observed
    `.in`/`.ans`. A case is emitted only when both halves are present.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        input_suffix, output_suffix = _DEFAULT_INPUT_SUFFIX, _DEFAULT_OUTPUT_SUFFIX
        if "config.yaml" in names:
            import yaml

            config = yaml.safe_load(archive.read("config.yaml").decode("utf-8")) or {}
            if isinstance(config, dict):
                input_suffix = config.get("input_suffix") or input_suffix
                output_suffix = config.get("output_suffix") or output_suffix

        stems: list[tuple[float, str, str]] = []
        for name in names:
            if not name.endswith(input_suffix) or name.endswith("/"):
                continue
            stem = name[: -len(input_suffix)]
            expected = f"{stem}{output_suffix}"
            if expected not in names:
                continue
            leaf = stem.rsplit("/", maxsplit=1)[-1]
            order = float(leaf) if leaf.isdigit() else float("inf")
            stems.append((order, leaf, stem))

        tests = []
        for _, _, stem in sorted(stems):
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
        return tests


class LiveCodeBenchProAdapter(LiveCodeBenchAdapter):
    info = AdapterInfo(
        name="livecodebench-pro",
        display_name="LiveCodeBench Pro",
        metric="pass@1",
        hf_dataset="QAQAQAQAQ/LiveCodeBench-Pro",
        hf_revision=_PROBLEM_REVISION,
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
        assets = ctx.cache.assets_dir(self.info.name)
        normalized: list[dict] = []
        missing_testcases: list[str] = []
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
            if archive is None:
                missing_testcases.append(key)
                continue
            tests = parse_testcase_zip(archive.read_bytes())
            archive.unlink(missing_ok=True)  # keep only the normalized JSONL
            if not tests:
                missing_testcases.append(key)
                continue
            normalized.append(
                {
                    "id": f"lcb-pro-{key}",
                    "question": statement,
                    "starter_code": "",
                    "fn_name": None,
                    "tests": tests,
                    "time_limit": row.get("time_limit"),
                    "memory_limit": row.get("memory_limit"),
                }
            )
        if not normalized:
            raise DatasetUnavailable(
                f"no {self.info.hf_dataset} {_SPLIT} problem could be joined with a "
                f"{self._TESTCASE_DATASET} archive ({len(problems)} problems seen)"
            )
        if missing_testcases:
            # Visible in the download detail; never a silent denominator change.
            print(
                f"[livecodebench-pro] {len(missing_testcases)} of {len(problems)} "
                f"problems have no usable testcase archive and are excluded: "
                f"{', '.join(missing_testcases[:10])}"
                f"{' ...' if len(missing_testcases) > 10 else ''}"
            )
        return normalized

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base.pop("release", None)
        base.pop("release_problems", None)
        base["split"] = _SPLIT
        base["split_problems"] = _SPLIT_PROBLEMS
        base["testcases"] = f"{self._TESTCASE_DATASET}@{_TESTCASE_REVISION}"
        base["grading"] = (
            "per-line whitespace-normalized comparison against .ans; the shipped "
            "testlib checker.cpp is not compiled (lower bound)"
        )
        return base
