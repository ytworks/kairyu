# Accuracy Benchmark Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename Kairyu's vendor-derived benchmark suite key to `accuracy` everywhere without changing benchmark behavior.

**Architecture:** Keep the existing suite registry, adapters, schemas, runner, aggregation, comparison, and example flow intact. Apply one canonical naming map at every interface boundary, rename the four tracked paths containing the old name, and retain `Fugu` only when it is Sakana's real product, a published score column/URL, or immutable progress history.

**Tech Stack:** Python 3.11+, Pydantic, pytest/pytest-asyncio, Ruff, POSIX shell, YAML, Markdown, Git/GitHub CLI.

## Global Constraints

- Canonical suite key: `accuracy`.
- Canonical display name: `Accuracy`.
- Canonical row-order constant: `ACCURACY_ROW_ORDER`.
- Default package result directory: `bench/results/accuracy`.
- Default Qwen example result directory: `results/accuracy`.
- Do not retain a compatibility alias for the legacy suite key.
- Do not change the eleven adapter identifiers or their order.
- Do not change datasets, prompts, sampling, limits, attempts, scoring, aggregation, withholding, calibration, published values, or schema version.
- Preserve `Fugu` and `Fugu Ultra` as Sakana product/model names and preserve their source URLs/assets.
- Do not rewrite `docs/progress/archive/` or past `PROGRESS.md` Change Log entries.

---

### Task 1: Canonical suite identity and defaults

**Files:**
- Modify: `tests/bench/test_bench_aggregate.py`
- Modify: `tests/bench/test_bench_config.py`
- Modify: `tests/bench/test_bench_pins.py`
- Modify: `kairyu/bench/adapters/__init__.py`
- Modify: `kairyu/bench/types.py`
- Modify: `kairyu/bench/cli.py`
- Modify: `kairyu/bench/aggregate.py`
- Modify: `kairyu/bench/calibration.py`
- Modify: `scripts/verify_bench_wheel.py`

**Interfaces:**
- Consumes: existing `SuiteInfo`, `suite_names()`, `suite_info()`, `suite_adapters()`, and `BenchConfig` interfaces.
- Produces: canonical suite key `accuracy`, display name `Accuracy`, row constant `ACCURACY_ROW_ORDER`, and default path `bench/results/accuracy` through the same interfaces.

- [ ] **Step 1: Change the registry behavior test first**

Update `test_only_and_exclude_names_are_validated_within_the_selected_suite` so it calls `suite_adapters("accuracy", exclude=("gsm8k",))` and expects:

```python
match="available: accuracy, core, quantization, structured, long-context"
```

Update the config default test to expect:

```python
assert config.suite == "accuracy"
assert config.results_dir == "bench/results/accuracy"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/bench/test_bench_aggregate.py::test_only_and_exclude_names_are_validated_within_the_selected_suite tests/bench/test_bench_config.py::test_models_shorthand_builds_targets -q
```

Expected: FAIL because `accuracy` is not registered and `BenchConfig` still uses the legacy default.

- [ ] **Step 3: Apply the minimal production rename**

Apply this exact mapping without changing the tuple contents:

```text
legacy row-order constant -> ACCURACY_ROW_ORDER
canonical suite key/name  -> "accuracy"
suite display name        -> "Accuracy"
default result directory  -> bench/results/accuracy
```

Update the registry, Pydantic literal/default/path derivation, CLI defaults/help/fallbacks, aggregation fallback, calibration default, and wheel smoke expectation. Do not replace `Fugu` in sampling-methodology prose because it identifies the source product.

- [ ] **Step 4: Update the remaining tests for the canonical interface**

Replace internal suite values and expected default paths in the three listed test files. Rename test function identifiers that describe Kairyu's suite, while leaving factual comments about published Sakana conditions intact.

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run:

```bash
uv run pytest tests/bench/test_bench_aggregate.py tests/bench/test_bench_config.py tests/bench/test_bench_pins.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the canonical identity change**

```bash
git add kairyu/bench/adapters/__init__.py kairyu/bench/types.py kairyu/bench/cli.py kairyu/bench/aggregate.py kairyu/bench/calibration.py scripts/verify_bench_wheel.py tests/bench/test_bench_aggregate.py tests/bench/test_bench_config.py tests/bench/test_bench_pins.py
git commit -m "refactor(bench): rename default suite to accuracy"
```

### Task 2: Runner, comparison, and test consumers

**Files:**
- Modify: `kairyu/bench/compare.py`
- Modify: `kairyu/bench/runner.py`
- Modify: `tests/bench/test_bench_agentic.py`
- Modify: `tests/bench/test_bench_cli_compare.py`
- Modify: `tests/bench/test_bench_compare.py`
- Modify: `tests/bench/test_bench_config_ab.py`
- Modify: `tests/bench/test_bench_progress.py`
- Modify: `tests/bench/test_bench_run_compare.py`
- Modify: `tests/bench/test_bench_runner.py`

**Interfaces:**
- Consumes: `accuracy` suite identity and `ACCURACY_ROW_ORDER` from Task 1.
- Produces: unchanged run/scoreboard/comparison semantics carrying `suite: "accuracy"`, plus the renamed `Accuracy benchmark scoreboard` heading.

- [ ] **Step 1: Change comparison and full-suite tests first**

Update test scoreboards/configs to `"suite": "accuracy"`, imports to
`ACCURACY_ROW_ORDER`, and the heading assertion to:

```python
assert out.index("# Accuracy benchmark scoreboard") < out.index(
    "# Accuracy vs published"
)
```

Keep `delta_against == "Fugu"`, `published["Fugu"]`, and the Sakana URL assertions unchanged because they verify published provenance.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/bench/test_bench_compare.py tests/bench/test_bench_agentic.py tests/bench/test_bench_runner.py -q
```

Expected: FAIL at stale internal suite values, imports, or headings.

- [ ] **Step 3: Rename production fallbacks and internal prose**

Change only Kairyu-owned suite fallbacks to `accuracy`. Rename test/function/docstring wording to describe the Accuracy suite. Retain `Fugu` in `DELTA_AGAINST`, published comparison headings, source conditions, and annotations.

- [ ] **Step 4: Update all Task 2 test consumers**

Change internal suite keys, imports, expected validation messages, and test identifiers across the listed tests. Do not alter numeric expectations, benchmark IDs, status expectations, or withholding behavior.

- [ ] **Step 5: Run Task 2 tests and verify GREEN**

Run:

```bash
uv run pytest tests/bench/test_bench_agentic.py tests/bench/test_bench_cli_compare.py tests/bench/test_bench_compare.py tests/bench/test_bench_config_ab.py tests/bench/test_bench_progress.py tests/bench/test_bench_run_compare.py tests/bench/test_bench_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit consumer migration**

```bash
git add kairyu/bench/compare.py kairyu/bench/runner.py tests/bench/test_bench_agentic.py tests/bench/test_bench_cli_compare.py tests/bench/test_bench_compare.py tests/bench/test_bench_config_ab.py tests/bench/test_bench_progress.py tests/bench/test_bench_run_compare.py tests/bench/test_bench_runner.py
git commit -m "refactor(bench): migrate accuracy suite consumers"
```

### Task 3: Qwen example, configuration, and tracked paths

**Files:**
- Rename: legacy config filename -> `examples/bench_accuracy.yaml`
- Rename: legacy direct-run script -> `examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh`
- Rename: legacy one-command script -> `examples/qwen3-32b-multi-gpu/run-accuracy-benchmark.sh`
- Rename: legacy static-contract test -> `tests/unit/test_qwen_accuracy_example.py`
- Modify: `.gitignore`
- Modify: `examples/qwen3-32b-multi-gpu/README.md`
- Modify: `examples/deploy_multi_orchestrator.yaml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: CLI suite key `accuracy` and unchanged shell/environment contract.
- Produces: example entry points `accuracy-benchmark.sh` and `run-accuracy-benchmark.sh`, config `bench_accuracy.yaml`, `[accuracy]` log prefix, and result path `results/accuracy`.

- [ ] **Step 1: Rename the example test and set desired filenames first**

Use `apply_patch` move directives for the four tracked paths. In the renamed test set:

```python
ACCURACY = EXAMPLE / "accuracy-benchmark.sh"
RUN_ACCURACY = EXAMPLE / "run-accuracy-benchmark.sh"
```

Rename the associated fixtures and test identifiers. Update the entry-point assertion to require `exec ./accuracy-benchmark.sh`.

- [ ] **Step 2: Run the example contract and verify RED**

Run:

```bash
uv run pytest tests/unit/test_qwen_accuracy_example.py -q
```

Expected: FAIL because the renamed scripts/config still contain old internal suite values, paths, prefixes, or entry-point references.

- [ ] **Step 3: Migrate the example implementation and configuration**

Apply this exact internal mapping:

```text
config filename         -> bench_accuracy.yaml
direct-run script       -> accuracy-benchmark.sh
one-command script      -> run-accuracy-benchmark.sh
suite key               -> accuracy
default results path    -> results/accuracy
shell log prefix        -> [accuracy]
```

Update `.gitignore`, shell usage text, README commands, deployment comment, and the pyproject comment. Retain factual descriptions of published Sakana scores and run conditions.

- [ ] **Step 4: Run the example and config tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_qwen_accuracy_example.py tests/bench/test_bench_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Shell syntax-check both scripts**

Run:

```bash
sh -n examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh
sh -n examples/qwen3-32b-multi-gpu/run-accuracy-benchmark.sh
```

Expected: both exit 0.

- [ ] **Step 6: Commit example migration**

```bash
git add .gitignore examples/bench_accuracy.yaml examples/deploy_multi_orchestrator.yaml examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh examples/qwen3-32b-multi-gpu/run-accuracy-benchmark.sh examples/qwen3-32b-multi-gpu/README.md pyproject.toml tests/unit/test_qwen_accuracy_example.py
git commit -m "docs(bench): rename accuracy benchmark examples"
```

### Task 4: Documentation and live-status terminology

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `bench/README.md`
- Modify: `docs/benchmarks.md`
- Modify: `docs/design/issue-367-core-evals.md`
- Modify: `docs/design/issue-372-quantization-sweep.md`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: renamed CLI, files, and paths from Tasks 1-3.
- Produces: documentation that calls Kairyu's eleven-row suite `Accuracy` while preserving Sakana attribution.

- [ ] **Step 1: Rewrite only Kairyu-owned naming**

Use `Accuracy suite`, `accuracy`, `examples/bench_accuracy.yaml`, and
`bench/results/accuracy` for Kairyu's suite, config, commands, and result paths.
Where the eleven rows are introduced, say they follow or are based on Sakana's
published release table rather than naming the Kairyu suite after that product.

- [ ] **Step 2: Preserve factual third-party references**

Do not change product discussions in `docs/roadmap.md`,
`docs/goals/g6-product-surface.md`, `docs/design/m1-orchestration-and-interface.md`,
or `docs/design/m11-product.md`. Preserve published model columns, source links,
and methodology statements in benchmark docs and code.

- [ ] **Step 3: Update the current status without adding history**

In `PROGRESS.md` change only the current-status suite list to
`Accuracy/Core/...`. Do not modify `## Product` or any Change Log entry; this
routine naming refactor does not require a new entry under the progress rules.

- [ ] **Step 4: Audit live internal names**

Run:

```bash
rg -n -i --hidden --glob '!.git/**' --glob '!docs/progress/archive/**' 'fugu'
git ls-files | rg -i 'fugu'
```

Expected: no tracked filename contains `fugu`; each text match is manually
classified as Sakana product/model provenance, URL/asset path, or factual
methodology attribution. No match may denote Kairyu's suite, default, path,
prefix, config, script, test, or display label.

- [ ] **Step 5: Commit documentation migration**

```bash
git add README.md README.ja.md bench/README.md docs/benchmarks.md docs/design/issue-367-core-evals.md docs/design/issue-372-quantization-sweep.md PROGRESS.md
git commit -m "docs(bench): describe the accuracy suite"
```

### Task 5: Full verification and pull request

**Files:**
- Modify only files required to fix verified naming-regression failures.

**Interfaces:**
- Consumes: all completed rename tasks.
- Produces: a verified branch and GitHub pull request targeting `main`.

- [ ] **Step 1: Verify the full benchmark/unit surface**

Run:

```bash
uv run pytest tests/bench tests/unit/test_qwen_accuracy_example.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 2: Verify the complete repository test suite**

Run:

```bash
uv run pytest
```

Expected: PASS with zero failures.

- [ ] **Step 3: Verify lint**

Run:

```bash
uv run ruff check .
```

Expected: exit 0 with no diagnostics.

- [ ] **Step 4: Review the exact diff and invariants**

Run:

```bash
git diff origin/main...HEAD --check
git diff --stat origin/main...HEAD
git status --short
```

Confirm the eleven adapter identifiers and their order match the pre-change
tuple in design commit `7bd9281^:kairyu/bench/adapters/__init__.py`; confirm
published numeric values are unchanged with:

```bash
git diff origin/main...HEAD -- kairyu/bench/reference.py
```

Expected: no whitespace errors, only planned files changed, clean worktree, and
no diff in `kairyu/bench/reference.py`.

- [ ] **Step 5: Push and create the PR**

```bash
git push -u origin codex/rename-benchmark-suite-to-accuracy
gh pr create --base main --head codex/rename-benchmark-suite-to-accuracy --title "refactor(bench): rename default suite to Accuracy" --body-file /tmp/kairyu-accuracy-pr.md
```

The PR body must summarize the internal naming migration, explicitly state the
functional invariants, list test/lint commands, and explain why Sakana product
names and immutable progress history remain.
