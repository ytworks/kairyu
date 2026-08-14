# In-Repository Quality Benchmark Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Kairyu's installed answer-quality benchmark feature and the complete `kairyu.bench` namespace while preserving repository-local performance, correctness, GPU, and evidence gates under `kairyu.benchmarking`.

**Architecture:** The standalone `ytworks/kairyu-bench` repository is the sole owner of answer-quality execution. Kairyu keeps only a small internal `kairyu.benchmarking` support package for checkout-owned `bench/` programs; the public CLI, suites, adapters, fixtures, quality dependencies, example runners, and feature documentation are deleted without a compatibility shim.

**Tech Stack:** Python 3.11/3.12, argparse, Pydantic/FastAPI application code, pytest, Ruff, Hatchling wheels, uv dependency locking, GitHub Actions, Markdown/TOML/YAML.

## Global Constraints

- Do not modify `../kairyu-bench`; link only to `https://github.com/ytworks/kairyu-bench`.
- Do not touch or stage the user's untracked `tests/examples/` or `tmp/` paths.
- Preserve top-level performance and correctness programs under `bench/`, including NVFP4 accuracy, batch invariance, KV equivalence, serving, frontier comparison, quant parity, GPU, distributed, and fleet gates.
- Remove `bench/tiered_auto_bench.py` because it embeds LiveCodeBench data and scoring.
- Do not leave a `kairyu bench` redirect, alias, deprecation parser, import shim, or empty `kairyu/bench/` directory.
- Never rewrite or delete past `PROGRESS.md` Change Log entries; archive them verbatim according to `.claude/rules/progress-log.md` and `docs/progress/archiving.md`.
- Never rewrite retained evidence solely to remove historical `kairyu.bench` paths.
- Use `apply_patch` for tracked file changes; preserve unrelated worktree changes.
- Every commit must leave its stated test boundary green. Do not claim full completion until Task 6 passes.

---

### Task 1: Create progress-log capacity

**Files:**
- Modify: `PROGRESS.md`
- Modify: `docs/progress/archive/change-log.md`

**Interfaces:**
- Consumes: the append-only Change Log and the archive insertion marker defined by `docs/progress/archiving.md`.
- Produces: five live Change Log entries and room for the migration progress/amendment entries in later tasks.

- [ ] **Step 1: Verify the current budget and entry order**

Run:

```bash
python3 scripts/check_progress_size.py
rg -n '^### ' PROGRESS.md
```

Expected: the size check passes and `PROGRESS.md` contains ten Change Log entries, newest first.

- [ ] **Step 2: Archive the five oldest live entries verbatim**

Move these complete entries from `PROGRESS.md` to immediately below `ARCHIVE-INSERT-POINT` in `docs/progress/archive/change-log.md`, preserving this order and every byte of their content:

```text
2026-08-13 — [progress] Tiered Chat UI publication defaults to visible content
2026-08-13 — [progress] Example vision CharXiv validation closes
2026-08-13 — [verified] Single-Qwen CharXiv vision closes 10/10
2026-08-13 — [progress] Example vision orchestration enters GPU validation
2026-08-13 — [amendment] Kind CI tool setup is shared and verified
```

Do not edit `Product`, `Current Status`, or the five retained newer entries in this task.

- [ ] **Step 3: Verify the archive invariants**

Run:

```bash
python3 scripts/check_progress_size.py
test "$(rg -c '^### ' PROGRESS.md)" -eq 9
```

Expected: the size check passes. The count is nine because it includes four non-Change-Log level-three headings plus five live Change Log entries.

- [ ] **Step 4: Commit the archive-only change**

```bash
git add PROGRESS.md docs/progress/archive/change-log.md
git commit -m "docs(progress): archive old change log entries"
```

---

### Task 2: Cut repository benchmarks over to `kairyu.benchmarking`

**Files:**
- Create: `kairyu/benchmarking/__init__.py`
- Create: `kairyu/benchmarking/profiling.py`
- Create: `kairyu/benchmarking/evidence.py`
- Create: `kairyu/benchmarking/reporting.py`
- Create: `kairyu/benchmarking/results_index.py`
- Create: `kairyu/benchmarking/batch_invariance.py`
- Create: `kairyu/benchmarking/kv_equivalence.py`
- Create: `kairyu/benchmarking/targets.py`
- Create: `tests/unit/test_benchmarking_targets.py`
- Create: `tests/unit/test_benchmarking_profiling.py`
- Delete: `tests/unit/test_bench_profiling.py`
- Modify: `scripts/verify_bench_results_index.py`
- Modify: `bench/__init__.py`
- Modify: `bench/agentic_kv_tier_f4b_bench.py`
- Modify: `bench/batch_invariance_bench.py`
- Modify: `bench/batched_prefill_qwen.py`
- Modify: `bench/dp_scaling_g2_a8_bench.py`
- Modify: `bench/fleet_churn_bench.py`
- Modify: `bench/frontier_compare.py`
- Modify: `bench/future_token_bench.py`
- Modify: `bench/g2_a9_dp_tp_crossover_bench.py`
- Modify: `bench/g4_ma1_nvfp4_accuracy_bench.py`
- Modify: `bench/g4_ma1_qwen3_235b_nvfp4_bench.py`
- Modify: `bench/g4_ma1_qwen3_235b_nvfp4_capture.py`
- Modify: `bench/g4_ma2_qwen3_235b_ep_kv_bench.py`
- Modify: `bench/g4_ma3_sglang_bench.py`
- Modify: `bench/kv_answer_equivalence_bench.py`
- Modify: `bench/serving_bench.py`
- Modify: `tests/bench/test_batch_invariance.py`
- Modify: `tests/bench/test_batch_invariance_bench.py`
- Modify: `tests/bench/test_bench_evidence.py`
- Modify: `tests/bench/test_bench_results_index.py`
- Modify: `tests/bench/test_kv_answer_equivalence_bench.py`
- Modify: `tests/bench/test_kv_equivalence.py`
- Modify: `tests/gpu/test_batched_prefill_gpu.py`
- Modify: `tests/gpu/test_batched_sampler_gpu.py`
- Modify: `tests/gpu/test_decode_input_slots_gpu.py`
- Modify: `tests/gpu/test_ep_attention_dp_status_gpu.py`
- Modify: `tests/gpu/test_flashinfer_tensor_decode.py`
- Modify: `tests/gpu/test_overlap_future_token_gpu.py`
- Modify: `tests/gpu/test_packed_dense_projections_gpu.py`
- Modify: `tests/gpu/test_rms_norm_gpu.py`
- Modify: `tests/gpu/test_sampling_rng_parity_gpu.py`
- Modify: `tests/gpu/test_tensor_decode_gpu.py`
- Modify: `tests/server/test_m11_product.py`
- Modify: `tests/unit/test_frontier_compare.py`
- Modify: `tests/unit/test_serving_bench.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: the existing implementations in `kairyu/bench/{profiling,evidence,reporting,results_index,batch_invariance,kv_equivalence,targets}.py`.
- Produces: import-compatible internal modules under `kairyu.benchmarking`, plus `EndpointTarget(name, base_url, model, api_key_env)` for retained HTTP performance tools.
- Produces: future A12/B7 source inventories bound to `kairyu/benchmarking/*.py` rather than `kairyu/bench/*.py`.

- [ ] **Step 1: Point retained tests at the new namespace first**

Apply these exact import substitutions in the retained test files:

```text
kairyu.bench.profiling       -> kairyu.benchmarking.profiling
kairyu.bench.evidence        -> kairyu.benchmarking.evidence
kairyu.bench.results_index   -> kairyu.benchmarking.results_index
kairyu.bench.batch_invariance -> kairyu.benchmarking.batch_invariance
kairyu.bench.kv_equivalence  -> kairyu.benchmarking.kv_equivalence
```

Move `tests/unit/test_bench_profiling.py` to `tests/unit/test_benchmarking_profiling.py` and update its `sys.modules` and dynamic-import strings to `kairyu.benchmarking.profiling`.

- [ ] **Step 2: Add focused endpoint-target tests**

Create `tests/unit/test_benchmarking_targets.py` with the retained performance contract, independent of deleted quality sampling fields:

```python
import pytest

from kairyu.benchmarking.targets import (
    EndpointTarget,
    normalize_base_url,
    parse_target_spec,
    resolve_api_key_env,
)


def test_endpoint_target_parser_normalizes_url_and_keeps_secret_by_name():
    target = parse_target_spec("candidate=http://localhost:8000=model=API_TOKEN")
    assert target == EndpointTarget(
        name="candidate",
        base_url="http://localhost:8000/v1",
        model="model",
        api_key_env="API_TOKEN",
    )
    assert target.label() == "candidate"


@pytest.mark.parametrize("spec", ["", "name=url", "=url=model", "n=url==KEY"])
def test_endpoint_target_parser_rejects_incomplete_specs(spec):
    with pytest.raises(ValueError, match="--target"):
        parse_target_spec(spec)


def test_endpoint_credentials_fail_closed_without_exposing_the_name():
    with pytest.raises(ValueError, match="API_TOKEN"):
        resolve_api_key_env("API_TOKEN", environ={}, required=True)


def test_endpoint_url_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="userinfo"):
        normalize_base_url("https://user:secret@example.test/v1")
```

- [ ] **Step 3: Run the new imports to prove the namespace does not exist yet**

Run:

```bash
uv run pytest -q \
  tests/unit/test_benchmarking_targets.py \
  tests/unit/test_benchmarking_profiling.py \
  tests/bench/test_bench_evidence.py \
  tests/bench/test_bench_results_index.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'kairyu.benchmarking'`.

- [ ] **Step 4: Create the internal package and copy retained implementations**

Create `kairyu/benchmarking/__init__.py` with exactly:

```python
"""Internal support for Kairyu's repository-local performance and correctness benchmarks."""
```

Copy the existing implementations byte-for-byte into their new module paths, then apply only these internal import changes:

```text
kairyu.bench.reporting      -> kairyu.benchmarking.reporting
kairyu.bench.kv_equivalence -> kairyu.benchmarking.kv_equivalence
```

Keep the old files temporarily so the still-installed quality feature continues to pass until Task 3 deletes the whole old package.

- [ ] **Step 5: Replace the quality-owned target schema with `EndpointTarget`**

In `kairyu/benchmarking/targets.py`, keep the current URL and environment-name validation logic, remove the `TYPE_CHECKING` import of `BenchTarget`, and define this exact public surface:

```python
from dataclasses import dataclass

TARGET_SPEC_FORMAT = "name=base_url=model[=api_key_env]"


@dataclass(frozen=True, slots=True)
class EndpointTarget:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        name = self.name.strip()
        model = self.model.strip()
        if not name or not model:
            raise ValueError("endpoint target name and model must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self,
            "api_key_env",
            validate_api_key_env(self.api_key_env),
        )

    def label(self) -> str:
        return self.name or self.model


def parse_target_spec(spec: str) -> EndpointTarget:
    """Parse ``name=base_url=model[=api_key_env]`` into an endpoint target."""
    if not isinstance(spec, str):
        raise ValueError(f"--target: expected {TARGET_SPEC_FORMAT}")
    parts = spec.split("=")
    if len(parts) not in (3, 4):
        raise ValueError(f"--target: expected {TARGET_SPEC_FORMAT}")

    name, base_url, model = (part.strip() for part in parts[:3])
    api_key_env = parts[3].strip() if len(parts) == 4 else None
    if not name or not base_url or not model:
        raise ValueError("--target: name, base_url, and model must be non-empty")
    try:
        return EndpointTarget(
            name=name,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
        )
    except ValueError as error:
        raise ValueError(f"--target: {error}") from error


def target_api_key(
    target: EndpointTarget,
    *,
    environ: Mapping[str, str] | None = None,
    required: bool | None = None,
) -> str | None:
    """Resolve the configured credential without storing its secret value."""
    if required is None:
        required = target.api_key_env is not None
    return resolve_api_key_env(
        target.api_key_env,
        environ=environ,
        required=required,
    )
```

Keep `normalize_base_url`, `validate_api_key_env`, and `resolve_api_key_env` from the copied module unchanged. Export `EndpointTarget` from `__all__`, remove `BenchTarget`, and do not add any sampling, judging, quantization, vision, or context fields.

- [ ] **Step 6: Cut retained production consumers over**

Apply the same namespace substitutions from Step 1 to every listed `bench/`, `scripts/`, GPU-test, server-test, and unit-test file. In `bench/serving_bench.py` and `bench/frontier_compare.py`, replace `BenchTarget` with `EndpointTarget` in imports, annotations, constructors, and `isinstance` checks.

Update the wrapper source-origin pairs and required source paths exactly:

```text
kairyu.bench.batch_invariance / kairyu/bench/batch_invariance.py
  -> kairyu.benchmarking.batch_invariance / kairyu/benchmarking/batch_invariance.py

kairyu.bench.kv_equivalence / kairyu/bench/kv_equivalence.py
  -> kairyu.benchmarking.kv_equivalence / kairyu/benchmarking/kv_equivalence.py
```

Update the corresponding expected error strings in the four A12/B7 tests. Do not change the A12/B7 schema IDs, thresholds, or replay semantics.

- [ ] **Step 7: Run focused retained-support tests**

Run:

```bash
uv run pytest -q \
  tests/unit/test_benchmarking_targets.py \
  tests/unit/test_benchmarking_profiling.py \
  tests/unit/test_serving_bench.py \
  tests/unit/test_frontier_compare.py \
  tests/server/test_m11_product.py \
  tests/bench/test_bench_evidence.py \
  tests/bench/test_bench_results_index.py \
  tests/bench/test_batch_invariance.py \
  tests/bench/test_batch_invariance_bench.py \
  tests/bench/test_kv_equivalence.py \
  tests/bench/test_kv_answer_equivalence_bench.py
uv run pytest --collect-only -q tests/gpu
uv run python scripts/verify_bench_results_index.py
```

Expected: focused CPU tests pass, all GPU modules collect without the old support imports, and the retained results index validates unchanged evidence.

- [ ] **Step 8: Record the completed internal cutover before committing**

Add this newest-first entry to `PROGRESS.md` without changing `Current Status` yet:

```markdown
### 2026-08-14 — [progress] Repository benchmark support leaves the quality namespace
- What: Performance, profiling, retained-evidence, A12, and B7 consumers now use the internal `kairyu.benchmarking` package; the installed quality CLI remains pending removal.
- Refs: `kairyu/benchmarking/`; `bench/{serving_bench,frontier_compare,batch_invariance_bench,kv_answer_equivalence_bench}.py`; `tests/{bench,gpu,unit}/`
```

Run `python3 scripts/check_progress_size.py` after the insertion.

- [ ] **Step 9: Commit the internal support cutover**

```bash
git add PROGRESS.md kairyu/benchmarking scripts/verify_bench_results_index.py \
  tests/unit/test_benchmarking_targets.py \
  tests/unit/test_benchmarking_profiling.py
git add -u -- bench tests
git commit -m "refactor(bench): isolate internal benchmark support"
```

---

### Task 3: Delete the public quality feature, old namespace, dependencies, and CI

**Files:**
- Create: `kairyu/benchmarking/ownership.py`
- Create: `bench/entrypoints.toml`
- Delete: `kairyu/bench/`
- Delete: `bench/configs/accuracy.yaml`
- Delete: `bench/configs/core.yaml`
- Delete: `bench/configs/quantization.yaml`
- Delete: `bench/configs/structured.yaml`
- Delete: `deploy/bench/`
- Delete: `bench/tiered_auto_bench.py`
- Delete: `tests/bench/conftest.py`
- Delete: `tests/bench/test_tiered_auto_bench.py`
- Modify: `kairyu/entrypoints/cli.py`
- Modify: `bench/__init__.py`
- Modify: `scripts/verify_bench_entrypoints.py`
- Modify: `scripts/verify_bench_wheel.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/bench/test_bench_ownership.py`
- Modify: `tests/bench/test_checkout_benchmark_entrypoints.py`
- Modify: `tests/unit/test_ci_workflow_policy.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: the fully cut-over `kairyu.benchmarking` consumers from Task 2.
- Produces: a `kairyu` CLI with exactly `serve` and `validate`; a checkout-owned 67-entry `bench/entrypoints.toml`; a wheel with `kairyu.benchmarking` and no importable `kairyu.bench`.
- Produces: `load_entrypoints(manifest_path)`, `load_compatibility_imports(manifest_path)`, and `validate_repository(root)` in `kairyu.benchmarking.ownership`.

- [ ] **Step 1: Add the CLI and manifest boundary tests first**

Append to `tests/unit/test_cli.py`:

```python
def test_quality_benchmark_command_is_not_registered():
    parser = cli._build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert set(command_action.choices) == {"serve", "validate"}
    with pytest.raises(SystemExit):
        parser.parse_args(["bench"])
```

Rewrite the retained portions of `tests/bench/test_bench_ownership.py` and `tests/bench/test_checkout_benchmark_entrypoints.py` to load `ROOT / "bench/entrypoints.toml"` and assert:

```python
entries = load_entrypoints(ROOT / "bench/entrypoints.toml")
assert len(entries) == 67
assert "bench/tiered_auto_bench.py" not in {entry.path for entry in entries}
assert all("bench" not in entry.requires for entry in entries)
assert validate_repository(ROOT) == ()
```

Delete ownership tests for packaged fixtures, `entrypoints_payload()`, and `kairyu bench entrypoints`; those surfaces are intentionally removed.

- [ ] **Step 2: Run the boundary tests to verify they fail against the old feature**

Run:

```bash
uv run pytest -q \
  tests/unit/test_cli.py \
  tests/bench/test_bench_ownership.py \
  tests/bench/test_checkout_benchmark_entrypoints.py
```

Expected: failures show the still-registered `bench` command and missing checkout-owned manifest.

- [ ] **Step 3: Move and narrow the entrypoint ownership implementation**

Copy `kairyu/bench/entrypoints.toml` to `bench/entrypoints.toml`, then:

```text
- remove the complete bench/tiered_auto_bench.py entry
- remove compatibility_imports."bench.tiered_auto_bench"
- remove the literal "bench" requirement from the four remaining requirement arrays
- keep all paths and compatibility imports sorted
```

Create `kairyu/benchmarking/ownership.py` from the existing validator, remove `importlib.resources`, installed CLI/fixture ownership metadata, `entrypoints_payload()`, and `render_entrypoints_json()`, and use these exact signatures:

```python
MANIFEST_RELATIVE_PATH = Path("bench/entrypoints.toml")


def load_entrypoints(
    manifest_path: str | Path,
) -> tuple[BenchmarkEntrypoint, ...]:
    payload = _load_manifest_payload(Path(manifest_path))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark entrypoint manifest schema: "
            f"{payload.get('schema_version')!r}"
        )
    raw_entries = payload.get("entrypoints")
    if not isinstance(raw_entries, list):
        raise ValueError("benchmark entrypoint manifest requires [[entrypoints]]")
    entries = tuple(BenchmarkEntrypoint.from_mapping(value) for value in raw_entries)
    paths = [entry.path for entry in entries]
    if paths != sorted(paths):
        raise ValueError("benchmark entrypoint manifest paths must be sorted")
    duplicates = sorted(path for path in set(paths) if paths.count(path) > 1)
    if duplicates:
        raise ValueError(f"duplicate benchmark entrypoints: {', '.join(duplicates)}")
    return entries


def load_compatibility_imports(
    manifest_path: str | Path,
) -> dict[str, tuple[str, ...]]:
    payload = _load_manifest_payload(Path(manifest_path))
    raw_imports = payload.get("compatibility_imports")
    if not isinstance(raw_imports, dict):
        raise ValueError(
            "benchmark entrypoint manifest requires [compatibility_imports]"
        )
    entries = load_entrypoints(manifest_path)
    modules = {entry.module for entry in entries}
    imports: dict[str, tuple[str, ...]] = {}
    for source, raw_targets in raw_imports.items():
        if not isinstance(source, str) or source not in modules:
            raise ValueError(f"unknown compatibility import source: {source!r}")
        targets = _string_tuple(raw_targets, f"compatibility_imports.{source}")
        if not targets:
            raise ValueError(f"{source}: compatibility imports must not be empty")
        if targets != tuple(sorted(targets)):
            raise ValueError(f"{source}: compatibility imports must be sorted")
        unknown = sorted(set(targets) - modules)
        if unknown:
            raise ValueError(
                f"{source}: unknown compatibility import targets: "
                + ", ".join(unknown)
            )
        imports[source] = targets
    if list(imports) != sorted(imports):
        raise ValueError("compatibility import sources must be sorted")
    return imports
```

In the existing `validate_repository()` body, replace its initialization with these exact lines, then retain the existing read-only AST, documentation, main-guard, invocation, and compatibility-allowlist checks:

```python
root = Path(root)
manifest_path = root / MANIFEST_RELATIVE_PATH
entries = load_entrypoints(manifest_path)
declared_compatibility_imports = load_compatibility_imports(manifest_path)
errors: list[str] = []
bench_root = root / REPOSITORY_ENTRYPOINT_ROOT
package_root = root / "kairyu"
by_path = {entry.path: entry for entry in entries}
wrapper_stems = frozenset(Path(entry.path).stem for entry in entries)
```

Delete the old zero-argument assignments to `entries` and `declared_compatibility_imports` from the retained body. Define the loader exactly as:

```python
def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    return tomllib.loads(manifest_path.read_text(encoding="utf-8"))
```

Update `scripts/verify_bench_entrypoints.py` so every `load_entrypoints()` call becomes `load_entrypoints(root / MANIFEST_RELATIVE_PATH)`, including the two summary counts. Import `MANIFEST_RELATIVE_PATH` from `kairyu.benchmarking.ownership`.

- [ ] **Step 4: Remove `kairyu bench` from argparse dispatch**

Delete the top-level `add_bench_parser` import, its call in `_build_parser()`, and the `elif args.command == "bench"` branch from `kairyu/entrypoints/cli.py`. Update its module docstring to:

```python
"""``kairyu`` console entrypoint: serve and validate commands."""
```

Do not catch or redirect the resulting argparse invalid-choice error.

- [ ] **Step 5: Delete the quality production tree and related repository assets**

Delete `kairyu/bench/` in full now that retained consumers use Task 2's package. Delete the four config files, `deploy/bench/`, and `bench/tiered_auto_bench.py`.

After tracked deletions, remove only ignored bytecode left under the exact old package path:

```bash
find kairyu/bench -type f -name '*.pyc' -delete 2>/dev/null || true
find kairyu/bench -depth -type d -empty -delete 2>/dev/null || true
test ! -e kairyu/bench
```

Do not run recursive deletion against `kairyu`, the repository root, `tests`, or any user-owned path.

- [ ] **Step 6: Delete quality-only tests**

Delete exactly these files; retain the evidence, ownership, results-index, performance, A12, B7, GPU, serving, and frontier tests:

```text
tests/bench/conftest.py
tests/bench/test_bench_agentic.py
tests/bench/test_bench_agentic_conditions.py
tests/bench/test_bench_aggregate.py
tests/bench/test_bench_cli_compare.py
tests/bench/test_bench_cli_config_compare.py
tests/bench/test_bench_cli_quant_sweep.py
tests/bench/test_bench_code_adapters.py
tests/bench/test_bench_compare.py
tests/bench/test_bench_config.py
tests/bench/test_bench_config_ab.py
tests/bench/test_bench_config_compare_stats.py
tests/bench/test_bench_core_adapters.py
tests/bench/test_bench_download_hf.py
tests/bench/test_bench_download_resilience.py
tests/bench/test_bench_exec_docker.py
tests/bench/test_bench_exec_runners.py
tests/bench/test_bench_helpers.py
tests/bench/test_bench_history.py
tests/bench/test_bench_judge.py
tests/bench/test_bench_judge_calibration.py
tests/bench/test_bench_lcb_datasets.py
tests/bench/test_bench_loglikelihood.py
tests/bench/test_bench_long_context.py
tests/bench/test_bench_mcq_adapters.py
tests/bench/test_bench_pins.py
tests/bench/test_bench_progress.py
tests/bench/test_bench_quant_sweep.py
tests/bench/test_bench_run_compare.py
tests/bench/test_bench_runner.py
tests/bench/test_bench_sampling.py
tests/bench/test_bench_sampling_sensitivity.py
tests/bench/test_bench_sandbox.py
tests/bench/test_bench_scicode_sequential.py
tests/bench/test_bench_shared_contracts.py
tests/bench/test_bench_store.py
tests/bench/test_bench_streaming.py
tests/bench/test_bench_structured_output.py
tests/bench/test_bench_tau.py
tests/bench/test_tiered_auto_bench.py
```

- [ ] **Step 7: Remove quality dependency and marker surfaces**

In `pyproject.toml`:

```text
- delete optional dependency groups bench and bench-agentic
- delete dev dependencies datasets, immutabledict, langdetect, nltk, tiktoken, jsonschema
- retain pillow
- remove docker_exec from addopts and marker declarations
- remove kairyu/bench/_vendor/* from coverage omit
```

Regenerate the lockfile:

```bash
uv lock
uv sync --frozen --dev
```

Expected: the lock and environment resolve without the deleted direct quality dependencies.

- [ ] **Step 8: Rewrite isolated wheel verification around absence**

In `scripts/verify_bench_wheel.py`, remove fixture/vendor/manifest/suite-command checks. Define required and forbidden members:

```python
REQUIRED_BENCHMARKING_MODULES = {
    "kairyu/benchmarking/__init__.py",
    "kairyu/benchmarking/batch_invariance.py",
    "kairyu/benchmarking/evidence.py",
    "kairyu/benchmarking/kv_equivalence.py",
    "kairyu/benchmarking/ownership.py",
    "kairyu/benchmarking/profiling.py",
    "kairyu/benchmarking/reporting.py",
    "kairyu/benchmarking/results_index.py",
    "kairyu/benchmarking/targets.py",
}
FORBIDDEN_PREFIXES = ("kairyu/bench/", "bench/", "tests/")
```

The isolated runtime check must execute the equivalent of:

```python
import importlib
import sys

sys.modules["torch"] = None
from kairyu.benchmarking.profiling import profile_scope
from kairyu.entrypoints.cli import _build_parser

with profile_scope(False) as native:
    assert native is None
assert set(next(a for a in _build_parser()._actions if a.dest == "command").choices) == {
    "serve",
    "validate",
}
try:
    importlib.import_module("kairyu.bench")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("wheel still exposes kairyu.bench")
```

Keep the `kairyu` console-script metadata check and unsafe-wheel-member check.

- [ ] **Step 9: Remove quality-only CI setup and assertions**

From `.github/workflows/ci.yml`, remove the tokenizer-cache configuration, cache restore, tiktoken prefetch, `jsonschema` prerequisite, and the entire `bench-exec-container` job. Keep the three repository benchmark verifier commands and the two-part CPU test run.

Update `tests/unit/test_ci_workflow_policy.py` to assert:

```python
assert "tiktoken" not in workflow_text
assert "jsonschema" not in dependency_check
assert "bench-exec-container" not in workflow["jobs"]
assert "docker_exec" not in workflow_text
```

Remove its old positive assertions for the deleted dependency and Docker job.

- [ ] **Step 10: Update Current Status and record the ownership amendment**

In `PROGRESS.md`, replace the `Benchmark/eval tooling` bullet with:

```markdown
- Answer-quality benchmarking is owned by the independent `ytworks/kairyu-bench` runner; this repository retains performance, GPU, correctness, and evidence gates only
```

Add this newest-first Change Log entry above Task 2's progress entry:

```markdown
### 2026-08-14 — [amendment] Answer-quality benchmarking moves out of Kairyu
- What: Removed the installed `kairyu bench` suites and complete `kairyu.bench` namespace; the independent `ytworks/kairyu-bench` runner now owns answer-quality execution while repository performance and correctness support lives in `kairyu.benchmarking`.
- Why: Quality benchmarking has moved to a dedicated repository with isolated official harnesses, so keeping a second bundled implementation would create conflicting ownership and dependencies.
- Refs: `kairyu/benchmarking/`; `bench/entrypoints.toml`; `https://github.com/ytworks/kairyu-bench`
```

Run `python3 scripts/check_progress_size.py`.

- [ ] **Step 11: Run the public-removal and retained-benchmark gates**

Run:

```bash
uv run pytest -q \
  tests/unit/test_cli.py \
  tests/unit/test_ci_workflow_policy.py \
  tests/bench/test_bench_ownership.py \
  tests/bench/test_checkout_benchmark_entrypoints.py
uv run python scripts/verify_bench_entrypoints.py
uv run python scripts/verify_bench_results_index.py
uv run python scripts/verify_bench_wheel.py
uv run pytest --fail-on-skip -ra -n 2 --dist loadfile tests/bench --no-cov
```

Expected: all commands pass; the isolated wheel verifier proves `kairyu.bench` is absent; the checkout manifest inventories 67 remaining programs.

- [ ] **Step 12: Commit the public feature removal**

```bash
git add PROGRESS.md bench/entrypoints.toml kairyu/benchmarking/ownership.py
git add -u -- .github/workflows/ci.yml bench deploy kairyu pyproject.toml uv.lock scripts tests
git commit -m "refactor: remove bundled quality benchmarks"
```

---

### Task 4: Remove quality benchmarking from product examples

**Files:**
- Modify: `examples/README.md`
- Modify: `examples/qwen3.6-27b-1gpu/benchmark.py`
- Modify: `examples/qwen3.6-27b-1gpu/example.json`
- Modify: `examples/qwen3.6-27b-1gpu/README.md`
- Modify: `examples/qwen3.6-27b-1gpu/MEASUREMENTS.md`
- Modify: `examples/deepseek-v4-flash-0731-8gpu/benchmark.py`
- Modify: `examples/deepseek-v4-flash-0731-8gpu/example.json`
- Modify: `examples/deepseek-v4-flash-0731-8gpu/README.md`
- Modify: `examples/deepseek-v4-flash-0731-8gpu/MEASUREMENTS.md`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/benchmark.py`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/example.json`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/README.md`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/MEASUREMENTS.md`
- Delete: `examples/qwen3.6-deepseek-v4-8gpu/terminalbench-result.json`
- Modify: `tests/unit/test_frontier_examplectl.py`
- Modify: `tests/unit/test_tiered_frontier_examplectl.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: each example's existing serving measurement path and `bench/serving_bench.py`.
- Produces: performance-only example launchers: `serving` for both single-model examples and `serving-auto-max` for the tiered example.

- [ ] **Step 1: Replace quality assertions with performance-only assertions**

Delete test functions that call or validate `livecodebench`, `charxiv`, `terminalbench`, `terminalbench_pilot`, or `terminalbench-result.json`. Keep all deployment, serving-dataset, serving-validation, and serving-dispatch tests.

Add these assertions to the retained example tests:

```python
assert set(json.loads((QWEN_EXAMPLE / "example.json").read_text())["benchmarks"]) == {
    "serving"
}
assert set(json.loads((DEEPSEEK_EXAMPLE / "example.json").read_text())["benchmarks"]) == {
    "serving"
}
assert set(json.loads((EXAMPLE / "example.json").read_text())["benchmarks"]) == {
    "serving"
}
```

For imported benchmark modules, assert the removed callables are absent and the one serving callable remains:

```python
assert hasattr(benchmark, "serving") or hasattr(benchmark, "serving_auto_max")
assert not hasattr(benchmark, "livecodebench")
assert not hasattr(benchmark, "charxiv")
assert not hasattr(benchmark, "terminalbench")
```

- [ ] **Step 2: Run the focused tests to prove embedded quality paths still exist**

Run:

```bash
uv run pytest -q \
  tests/unit/test_frontier_examplectl.py \
  tests/unit/test_tiered_frontier_examplectl.py
```

Expected: assertions fail because quality configuration and callables are still present.

- [ ] **Step 3: Reduce the two single-model launchers to serving only**

In the Qwen and DeepSeek `benchmark.py` files, delete `_execution_image`, quality runner functions, quality validators, and imports used only by those functions. Set command choices and list output to:

```python
parser.add_argument("benchmark", choices=("serving", "list"))

if args.benchmark == "list":
    print("serving  fixed 8K-input/256-output TTFT and throughput at c=1,8,16,32")
    return
```

After environment setup, call `serving(run_dir)` directly and record only `{"serving": exit_code}`. Remove the redundant `all` alias.

Delete `livecodebench` and `charxiv` objects from each `example.json`, leaving `benchmarks.serving` unchanged.

- [ ] **Step 4: Reduce the tiered launcher to `serving-auto-max` only**

Delete Terminal-Bench dataset preparation, CharXiv execution/validation, Terminal-Bench pilot/full execution/validation, Harbor environment plumbing, and their imports. Set:

```python
BENCHMARKS = ("serving-auto-max",)
parser.add_argument("benchmark", choices=(*BENCHMARKS, "list"))
```

The `list` output contains only the verifier-gated `serving-auto-max` description. Remove `all` because there is one selectable benchmark. Delete `terminalbench` and `charxiv` from `example.json`, leaving `benchmarks.serving` unchanged. Delete `terminalbench-result.json`.

- [ ] **Step 5: Remove quality instructions and results from example documents**

In all listed example READMEs and `MEASUREMENTS.md` files:

```text
- remove LiveCodeBench, CharXiv, Terminal-Bench commands and score sections
- remove quality run IDs, result hashes, dataset setup, judge, Harbor, and execution-image instructions
- retain deployment, serving methodology, throughput, TTFT/TPOT, topology, and serving artifact identity
- point answer-quality users only to https://github.com/ytworks/kairyu-bench where a pointer is useful
```

Update `examples/README.md` to advertise serving performance only.

- [ ] **Step 6: Refresh the mutable Current Status example snapshot**

In `PROGRESS.md` Current Status, replace the example bullet's bundled CharXiv claim with a performance-only description and the fact that external `kairyu-bench` evaluates the public endpoint. Do not edit any existing Change Log entry mentioning historical CharXiv or Terminal-Bench evidence.

- [ ] **Step 7: Run focused example tests**

```bash
uv run pytest -q \
  tests/unit/test_frontier_examplectl.py \
  tests/unit/test_tiered_frontier_examplectl.py
python3 scripts/check_progress_size.py
```

Expected: both example test modules and the progress size check pass.

- [ ] **Step 8: Commit the example cleanup**

```bash
git add -u -- PROGRESS.md examples \
  tests/unit/test_frontier_examplectl.py \
  tests/unit/test_tiered_frontier_examplectl.py
git commit -m "refactor(examples): remove embedded quality benchmarks"
```

---

### Task 5: Remove obsolete quality documentation and publish the external boundary

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `.gitignore`
- Modify: `bench/README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/gpu-runbook.md`
- Modify: `docs/goals/g6-product-surface.md`
- Modify: `docs/design/frontier-native-runtime.md`
- Modify: `docs/design/issue-360-batch-invariance.md`
- Modify: `docs/design/issue-382-evidence-library.md`
- Modify: `docs/design/m11-product.md`
- Modify: `docs/design/m12-model-zoo.md`
- Modify: `docs/design/m16-distributed.md`
- Modify: `docs/design/m17-graphs-drafts.md`
- Delete: `docs/benchmarks.md`
- Delete: `docs/design/issue-365-config-ab.md`
- Delete: `docs/design/issue-367-core-evals.md`
- Delete: `docs/design/issue-368-loglikelihood.md`
- Delete: `docs/design/issue-369-cross-commit-scoreboards.md`
- Delete: `docs/design/issue-372-quantization-sweep.md`
- Delete: `docs/design/issue-374-long-context-sweep.md`
- Delete: `docs/superpowers/specs/2026-08-09-accuracy-benchmark-naming-design.md`
- Delete: `docs/superpowers/specs/2026-08-11-accuracy-reference-sources-design.md`
- Delete: `docs/superpowers/specs/2026-08-12-swe-bench-verified-design.md`
- Delete: `docs/superpowers/plans/2026-08-09-accuracy-benchmark-rename.md`
- Delete: `docs/superpowers/plans/2026-08-11-accuracy-reference-sources.md`
- Delete: `docs/superpowers/plans/2026-08-12-swe-bench-verified.md`

**Interfaces:**
- Consumes: the final code and example ownership boundary from Tasks 3-4.
- Produces: current documentation that sends answer-quality users to `ytworks/kairyu-bench` and documents only this repository's performance/correctness gates.

- [ ] **Step 1: Capture the stale active instructions before editing**

Run:

```bash
rg -n '(kairyu bench|bench/configs/(accuracy|core|quantization|structured))' \
  README.md README.ja.md bench/README.md docs examples/README.md \
  --glob '!docs/progress/archive/**' \
  --glob '!docs/superpowers/**'
rg -n 'kairyu\.bench' \
  README.md README.ja.md bench/README.md docs/gpu-runbook.md docs/design \
  --glob '!docs/progress/archive/**'
```

Expected: both commands report the old installed-suite instructions and namespace.

- [ ] **Step 2: Replace public benchmark chapters with migration pointers**

Replace the long Section 8 quality instructions in both root READMEs with concise equivalent text:

```markdown
## 8. Benchmarks

Answer-quality benchmarking is maintained independently in
[`ytworks/kairyu-bench`](https://github.com/ytworks/kairyu-bench). This
repository retains performance and implementation-correctness tools under
[`bench/`](bench/README.md).
```

Write the Japanese README with the same ownership and links in Japanese. Remove `docs/benchmarks.md` from both documentation indexes.

- [ ] **Step 3: Make `bench/README.md` repository-only**

Change its title to `# Repository performance and correctness benchmarks`. Replace package-owned wording with:

```markdown
`bench/entrypoints.toml` is the checkout-owned inventory. Reusable internal
support lives under `kairyu.benchmarking`; no benchmark CLI or benchmark data is
installed with the Kairyu wheel.
```

Replace `kairyu bench entrypoints --check-repo .` with:

```bash
uv run --frozen python scripts/verify_bench_entrypoints.py
```

Update every retained `kairyu.bench.evidence`, `kairyu.bench.profiling`, `kairyu.bench.reporting`, and `kairyu.bench.results_index` reference to its same-named `kairyu.benchmarking` module. Keep all performance/formal-gate commands and retained-results instructions.

- [ ] **Step 4: Update active roadmap, runbook, and cross-cutting design references**

Apply these ownership rules consistently:

```text
- current answer-quality execution -> external ytworks/kairyu-bench
- repository performance/correctness helpers -> kairyu.benchmarking
- checkout inventory command -> scripts/verify_bench_entrypoints.py
- historical completed LiveCodeBench/CharXiv/Terminal-Bench evidence -> retain as historical wording, not a current in-repo command
- docker_exec marker list in m12 -> remove docker_exec
```

In `docs/gpu-runbook.md`, keep the clean-commit and results-index provenance requirements while replacing package-owned and CLI inventory instructions. In G6/roadmap documents, name the external runner as the current quality gate owner while keeping `bench/frontier_compare.py` for performance comparison.

- [ ] **Step 5: Delete feature-only documentation**

Delete the exact feature guide, six issue design documents, and six old Accuracy/SWE-bench specs/plans listed in this task's Files block. Do not delete the newly approved design or this plan.

- [ ] **Step 6: Clean quality-specific ignore rules**

In `.gitignore`, retain:

```gitignore
bench/results/*
!bench/results/index.json
```

Delete `bench/data/` and the Accuracy/Core/Quantization/Structured/Long Context history exceptions that existed only for the installed suites. Keep any unrelated retained-evidence negations unchanged.

- [ ] **Step 7: Verify the current documentation boundary**

Run:

```bash
rg -n '(uv run.*kairyu bench|kairyu bench (run|download|report|list|compare|entrypoints))' \
  README.md README.ja.md bench/README.md docs examples/README.md \
  --glob '!docs/progress/archive/**' \
  --glob '!docs/superpowers/**' \
  && exit 1 || true
rg -n '(from kairyu\.bench|import kairyu\.bench)' \
  kairyu bench scripts tests .github \
  && exit 1 || true
rg -n 'ytworks/kairyu-bench' README.md README.ja.md
```

Expected: neither forbidden active-use search finds a match, and both root READMEs link to the external repository.

- [ ] **Step 8: Commit the documentation boundary**

```bash
git add -u -- README.md README.ja.md .gitignore bench/README.md docs
git commit -m "docs: point quality benchmarks to standalone runner"
```

---

### Task 6: Run full verification and audit the final tree

**Files:**
- Verify: `kairyu/`
- Verify: `bench/`
- Verify: `tests/`
- Verify: `scripts/`
- Verify: `.github/workflows/ci.yml`
- Verify: `pyproject.toml`
- Verify: `uv.lock`
- Verify: `PROGRESS.md`
- Modify: `docs/superpowers/specs/2026-08-14-remove-in-repo-quality-benchmarks-design.md`

**Interfaces:**
- Consumes: all deliverables from Tasks 1-5.
- Produces: evidence that the old public feature is absent and every retained CPU/repository benchmark contract remains green.

- [ ] **Step 1: Verify repository policy and package boundaries**

Run:

```bash
python3 scripts/check_progress_size.py
uv run --frozen python scripts/verify_bench_entrypoints.py
uv run --frozen python scripts/verify_bench_results_index.py
uv run --frozen python scripts/verify_bench_wheel.py
```

Expected: all four commands exit zero; wheel verification explicitly reports that `kairyu.bench` is absent.

- [ ] **Step 2: Run lint**

```bash
uv run --frozen ruff check .
```

Expected: exit zero with no violations.

- [ ] **Step 3: Run the two-part CPU suite with the coverage gate exactly like CI**

```bash
uv run --frozen coverage erase
uv run --frozen pytest --fail-on-skip -ra \
  -n 2 --dist loadfile tests/bench \
  --cov=kairyu --cov-report= --cov-fail-under=0
uv run --frozen pytest --fail-on-skip -ra \
  tests --ignore=tests/bench \
  --cov=kairyu --cov-append --cov-report=term-missing \
  --cov-fail-under=80
```

Expected: both halves pass without selected skips and combined coverage is at least 80%.

- [ ] **Step 4: Audit namespace, CLI, assets, and user worktree safety**

Run:

```bash
test ! -e kairyu/bench
test ! -e deploy/bench
test ! -e bench/tiered_auto_bench.py
test ! -e examples/qwen3.6-deepseek-v4-8gpu/terminalbench-result.json
uv run --frozen python -m kairyu.entrypoints.cli --help
rg -n '(from kairyu\.bench|import kairyu\.bench)' kairyu bench scripts tests .github \
  && exit 1 || true
rg -n '(kairyu bench (run|download|report|list|compare|entrypoints)|bench/configs/(accuracy|core|quantization|structured))' \
  README.md README.ja.md bench/README.md docs examples \
  --glob '!docs/progress/archive/**' \
  --glob '!docs/superpowers/**' \
  && exit 1 || true
git diff --check
git status --short
```

Expected:

```text
- help lists serve and validate only
- no active import or usage search matches
- no whitespace errors
- tests/examples/ and tmp/ remain untracked and otherwise untouched
```

If a command fails, return to the task that owns that file or contract; do not weaken a check, restore a compatibility shim, edit retained evidence, or modify the user's untracked paths.

- [ ] **Step 5: Mark the approved design as implemented after all gates pass**

Change only the status line in `docs/superpowers/specs/2026-08-14-remove-in-repo-quality-benchmarks-design.md`:

```markdown
**Status:** Implemented
```

- [ ] **Step 6: Commit the completed status and any verification-driven corrections**

Stage the design status plus only the explicit tracked corrections required by Steps 1-4:

```bash
git add docs/superpowers/specs/2026-08-14-remove-in-repo-quality-benchmarks-design.md
git commit -m "docs: mark quality benchmark removal implemented"
```

If verification required corrections, include only their explicit paths in the same `git add` invocation; do not stage the user's untracked paths.
