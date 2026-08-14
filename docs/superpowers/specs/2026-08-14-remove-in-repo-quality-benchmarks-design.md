# Remove In-Repository Quality Benchmarks Design

**Date:** 2026-08-14

**Status:** Approved for implementation planning

## Summary

Kairyu no longer owns answer-quality benchmark execution. The independent
[`ytworks/kairyu-bench`](https://github.com/ytworks/kairyu-bench) repository
owns the 12 official Accuracy benchmarks, their harness isolation, result
normalization, reporting, and published-model comparison.

This repository will remove the complete installed quality-benchmark surface:
the `kairyu bench` command, all Accuracy/Core/Quantization/Structured/Long
Context suite code, benchmark datasets and fixtures, quality-only dependencies,
tests, CI, examples, and documentation. The importable `kairyu.bench` namespace
will also disappear completely.

Repository-local performance benchmarks, GPU gates, correctness gates, and
retained evidence remain in scope. The small support library they currently
share with the quality feature moves to the explicitly internal
`kairyu.benchmarking` namespace.

## Goals

1. Make `kairyu bench` unavailable from the installed `kairyu` CLI.
2. Make `import kairyu.bench` fail in a clean source checkout and in the built
   wheel.
3. Remove every quality-suite implementation and its direct test, dependency,
   CI, example, result-summary, and documentation surface from this repository.
4. Preserve the top-level `bench/` performance and formal-gate programs that
   validate the inference framework itself.
5. Preserve shared evidence integrity, result indexing, source ownership,
   profiling, endpoint parsing, batch-invariance, and KV-equivalence behavior
   under `kairyu.benchmarking`.
6. Point users to `ytworks/kairyu-bench` without adding a submodule, runtime
   dependency, wrapper command, or compatibility shim.
7. Preserve frozen historical records exactly where repository policy forbids
   rewriting them.

## Non-goals

- Do not modify the independent `../kairyu-bench` working tree.
- Do not add a `kairyu bench` redirect, downloader, subprocess wrapper, or
  deprecation command.
- Do not remove latency, throughput, goodput, transfer, kernel, distributed,
  fleet, or correctness benchmarks solely because their filenames contain
  `bench` or `accuracy`.
- Do not remove model-output correctness gates such as NVFP4 agreement,
  batch-shape invariance, cache answer equivalence, quant checkpoint parity, or
  exact-token parity.
- Do not rewrite historical Change Log entries, archived progress, or retained
  evidence to erase old `kairyu.bench` strings.
- Do not change serving behavior or the OpenAI-compatible API.

## Ownership Boundary

### External repository

`https://github.com/ytworks/kairyu-bench` owns:

- the 12 official Accuracy rows;
- target model discovery and OpenAI-compatible request transport;
- official or source-pinned benchmark harness execution;
- nested-Docker benchmark isolation;
- normalized raw and aggregate result schemas;
- answer-quality comparison and published reference data;
- user-facing quality benchmark instructions.

Kairyu documents this ownership in one short README migration note. It does not
vendor, import, install, invoke, or validate the external repository.

### This repository

This repository continues to own:

- `bench/serving_bench.py`, `bench/frontier_compare.py`, and other performance
  operators;
- GPU/kernel/distributed/fleet/formal-gate programs under `bench/`;
- retained performance and correctness evidence under `bench/results/`;
- the benchmark entrypoint inventory and results-index validation;
- reusable internal mechanics needed by those programs.

`bench/tiered_auto_bench.py` is removed even though it was a formal G6 program:
its quality half directly embeds a fixed LiveCodeBench slice and scorer. The
closed historical gate remains documented as history; future answer-quality
measurement belongs to the external runner.

## Package Restructure

### Remove the public quality package

Delete `kairyu/bench/` in full after moving retained support. This removes:

- `kairyu.bench.cli` and the `kairyu bench` command implementation;
- all adapters and suite registries;
- all packaged fixtures and vendored IFEval code;
- runner, download/cache, execution sandbox, scoring, judge, aggregation,
  reporting-history, comparison, calibration, config A/B, and quant-sweep
  features;
- packaged benchmark entrypoint metadata;
- the public benchmark schemas in `types.py`.

`kairyu/entrypoints/cli.py` will expose only `serve` and `validate`. There is no
compatibility parser for `bench`, so argparse rejects it as an unknown command.

### Add the internal support package

Create `kairyu/benchmarking/` and move only the behavior still consumed by
repository-local benchmarks or tests:

| Old path | New path | Responsibility |
|---|---|---|
| `kairyu/bench/profiling.py` | `kairyu/benchmarking/profiling.py` | lazy PyTorch profiling and trace publication |
| `kairyu/bench/evidence.py` | `kairyu/benchmarking/evidence.py` | canonical JSON, hashing, JSONL, and evidence replay helpers |
| `kairyu/bench/reporting.py` | `kairyu/benchmarking/reporting.py` | atomic report writes and nearest-rank percentile |
| `kairyu/bench/results_index.py` | `kairyu/benchmarking/results_index.py` | retained `bench/results/` index validation |
| `kairyu/bench/ownership.py` | `kairyu/benchmarking/ownership.py` | repository benchmark entrypoint ownership validation |
| `kairyu/bench/batch_invariance.py` | `kairyu/benchmarking/batch_invariance.py` | pure A12 evidence contract and replay |
| `kairyu/bench/kv_equivalence.py` | `kairyu/benchmarking/kv_equivalence.py` | pure G5 cache-answer evidence contract and replay |
| `kairyu/bench/targets.py` | `kairyu/benchmarking/targets.py` | endpoint normalization, credential-name resolution, and target parsing |

The new `kairyu.benchmarking.__init__` exports nothing. Consumers import the
specific internal module they use.

The old `BenchTarget` type will not move. It carries sampling, judging,
quantization, vision, long-context, and served-config fields belonging to the
deleted quality suites. `kairyu.benchmarking.targets` instead defines one
frozen `EndpointTarget` with only `name`, `base_url`, `model`, and
`api_key_env`, plus `label()`. `bench/serving_bench.py` and
`bench/frontier_compare.py` migrate to that contract.

### Entrypoint manifest

Move `kairyu/bench/entrypoints.toml` to `bench/entrypoints.toml`. The manifest
describes repository-only programs and therefore must not be installed in the
wheel. `kairyu.benchmarking.ownership` and
`scripts/verify_bench_entrypoints.py` load the repository path explicitly.

Remove the `bench` extra label from manifest entries. No retained top-level
program may claim the deleted quality-suite dependency group.

## Quality Feature Deletion

Delete the following quality-owned repository surfaces:

- `bench/configs/accuracy.yaml`
- `bench/configs/core.yaml`
- `bench/configs/quantization.yaml`
- `bench/configs/structured.yaml`
- `deploy/bench/`
- `bench/tiered_auto_bench.py`
- all quality-only modules formerly under `kairyu/bench/`
- all suite-specific tests under `tests/bench/`
- the suite-level `tests/bench/conftest.py`

Retain files such as `bench/g4_ma1_nvfp4_accuracy_bench.py`: they test numerical
correctness of Kairyu implementation choices and are not general answer-quality
suites.

The retained `tests/bench/` directory continues to test top-level performance
and formal-gate programs. The following shared-support tests also remain and
move their imports to `kairyu.benchmarking`:

- evidence helpers;
- entrypoint ownership;
- results-index validation;
- profiling;
- batch invariance;
- KV answer equivalence;
- serving/frontier target parsing and reporting.

Tests for CLI suites, adapters, datasets, scoring, judges, quality reports,
history, config comparison, quant sweeps, sandboxed code execution, and agentic
harnesses are deleted with their production code.

## Example Cleanup

The example benchmark launchers currently combine serving performance with
LiveCodeBench, CharXiv, or Terminal-Bench. Keep their serving performance
paths, but remove quality paths:

- `examples/qwen3.6-27b-1gpu/benchmark.py` retains `serving` only;
- `examples/deepseek-v4-flash-0731-8gpu/benchmark.py` retains `serving` only;
- `examples/qwen3.6-deepseek-v4-8gpu/benchmark.py` retains
  `serving-auto-max` only;
- the corresponding `example.json` files retain only serving benchmark
  configuration;
- example READMEs and `MEASUREMENTS.md` files lose quality invocation and score
  sections while retaining serving evidence;
- `examples/qwen3.6-deepseek-v4-8gpu/terminalbench-result.json` is deleted.

The examples do not shell out to the sibling checkout. Users run the external
tool independently against the example's public `/v1` endpoint.

## Dependencies, Build, and CI

Remove the `bench` and `bench-agentic` optional dependency groups from
`pyproject.toml`. Remove quality-only development dependencies: `datasets`,
`immutabledict`, `langdetect`, `nltk`, `tiktoken`, and `jsonschema`. Retain
`pillow` because the product vision tests import it. Regenerate `uv.lock` from
the resulting project metadata.

Remove the quality-package coverage omission for `kairyu/bench/_vendor/*`.

From `.github/workflows/ci.yml`, remove:

- the tiktoken cache and prefetch steps;
- the `jsonschema` prerequisite assertion;
- the quality execution-image build;
- the `bench-exec-container` job and Docker execution-runner conformance test.

Keep the `tests/bench` test split because the remaining performance/formal-gate
tests are still a large independent group. Keep entrypoint-inventory and
results-index verification. Rewrite wheel verification to assert the new
boundary:

- required `kairyu.benchmarking` helpers are installed;
- `kairyu.bench` is absent and cannot be imported;
- quality adapters, fixtures, manifests, and CLI modules are absent;
- top-level `bench/` scripts and `bench/entrypoints.toml` are not in the wheel;
- profiling remains torch-lazy when disabled.

Update `tests/unit/test_ci_workflow_policy.py` to enforce the reduced workflow
instead of asserting the deleted dependencies and Docker job.

## Documentation Cleanup

Delete the installed-suite guide `docs/benchmarks.md`. Move its still-current
repository performance-inventory and retained-results instructions into
`bench/README.md` or `docs/gpu-runbook.md` before deletion.

Delete design history that exists solely to specify the removed in-repository
quality feature:

- `docs/design/issue-365-config-ab.md`
- `docs/design/issue-367-core-evals.md`
- `docs/design/issue-368-loglikelihood.md`
- `docs/design/issue-369-cross-commit-scoreboards.md`
- `docs/design/issue-372-quantization-sweep.md`
- `docs/design/issue-374-long-context-sweep.md`
- the Accuracy naming, reference-source, and SWE-bench Verified specs and plans
  under `docs/superpowers/`.

Update retained active design documents that describe performance or
correctness mechanics, including the evidence-library, batch-invariance, G6,
GPU-runbook, roadmap, and product design documents, to use
`kairyu.benchmarking` and the external ownership boundary.

Replace the long quality-benchmark sections in `README.md` and `README.ja.md`
with a short link to `ytworks/kairyu-bench`. The README must not reproduce its
installation or command guide.

Clean `.gitignore` of quality-suite cache/history exceptions while preserving
the generic ignore policy for runtime output beneath `bench/results/`.

## Progress Log Policy

This is a significant package and product-boundary change, so it requires a
`PROGRESS.md` update.

`PROGRESS.md` already contains ten Change Log entries. Before adding the new
entry, follow `docs/progress/archiving.md` in a separate commit: move the oldest
entries verbatim to `docs/progress/archive/change-log.md` until only five remain.

Then:

- replace the Current Status claim that this repository provides the five
  quality suites with a short statement that quality benchmarking is owned by
  `ytworks/kairyu-bench`;
- update current example capability text so it no longer claims bundled
  CharXiv or Terminal-Bench launchers;
- add a newest-first English `[amendment]` entry describing the ownership and
  package-boundary change;
- do not rewrite or delete any prior Change Log entry.

Historical archived text and retained evidence may still contain the literal
string `kairyu.bench`. They are inert records, not an importable namespace or
current documentation, and repository policy requires preserving them.

## Evidence Contract Migration

Moving batch-invariance and KV-equivalence contracts changes their
source-bound paths. Update:

- imports in their top-level operators;
- checkout-origin assertions;
- `REQUIRED_SOURCE_PATHS` inventories;
- source-snapshot test fixtures and expected error messages;
- design references naming the package path.

No tracked retained A12 or B7 artifact currently depends on the old source
inventory, so no retained result needs rewriting. The existing schema names and
gate semantics remain unchanged; only future clean-source identity names the
new module paths.

Other retained evidence files may list old `kairyu/bench/*` files inside a
historical source snapshot. Do not mutate those immutable artifacts. The
results index continues to validate their exact retained bytes.

## Failure Behavior

- `kairyu --help` lists only `serve` and `validate`.
- `kairyu bench ...` exits through argparse with an invalid-choice error before
  importing any benchmark support.
- `import kairyu.bench` fails in an isolated installed-wheel environment.
- An invalid performance target continues to fail before requests are sent.
- Explicit profiling continues to fail closed when PyTorch/CUDA support is
  unavailable; disabled profiling remains import-light.
- Evidence and results-index validation remain fail closed on non-canonical,
  missing, stale, unsafe, or source-inconsistent input.

## Verification

The implementation is complete only when all of the following pass:

1. Focused CLI tests prove `bench` is absent from help and rejected as a
   command.
2. Focused wheel tests prove `kairyu.bench` and quality assets are absent while
   `kairyu.benchmarking` support is present.
3. `scripts/verify_bench_entrypoints.py` validates `bench/entrypoints.toml`.
4. `scripts/verify_bench_results_index.py` validates unchanged retained
   evidence coverage.
5. Retained batch-invariance, KV-equivalence, evidence, profiling, endpoint,
   serving, and frontier-comparison tests pass under the new namespace.
6. All remaining `tests/bench` tests pass without the deleted shared conftest.
7. The rest of the CPU suite passes with the existing no-skip policy and
   coverage threshold.
8. `ruff check .` passes.
9. `scripts/check_progress_size.py` passes.
10. A search of active source, tests, scripts, workflows, and current
    documentation finds no `kairyu.bench` import or current usage instruction.
    Frozen progress archives and retained evidence are explicitly excluded from
    this textual check.

## Worktree Safety

At design time the Kairyu worktree contains untracked `tests/examples/` and
`tmp/`. They belong to the user and are outside this change. Implementation
must not delete, stage, or rewrite them. If they add tests that exercise the
example benchmark launchers, implementation should satisfy them where their
expectations match this approved design, without modifying the untracked files.
