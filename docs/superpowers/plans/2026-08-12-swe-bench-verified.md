# SWE-bench Verified Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an officially invoked, fail-closed `swe-bench-verified` Accuracy benchmark whose score, history, and sourced frontier comparison behave like the existing benchmark rows.

**Architecture:** Extract SWE-bench prediction validation, schema-v2 report parsing, and the two subprocess stages from the Pro adapter into one shared adapter module. Keep Pro and Verified as thin specifications so their datasets and 1,000/250-step policies cannot drift together accidentally. Register Verified in the normal Accuracy/report/history paths and add one sparse, source-bound Fable 5 reference record while leaving every unverified exact-model value absent.

**Tech Stack:** Python 3.11, asyncio/subprocess, mini-SWE-agent 2.x, swebench 4.x, Pydantic benchmark result types, pytest, Ruff, Markdown/YAML documentation.

## Global Constraints

- The public benchmark key is exactly `swe-bench-verified` and its display name is `SWE-bench Verified`.
- Generation uses `mini-extra swebench --subset verified --split test` with `--config swebench.yaml`; the standard config remains at 250 steps and a $3 cost limit.
- Evaluation uses `python -m swebench.harness.run_evaluation`, dataset `princeton-nlp/SWE-bench_Verified`, split `test`, the exact pre-generation selection as `--instance_ids`, `--max_workers`, `--run_id`, and `--report_dir`.
- `--limit N` must yield an official denominator of exactly the selected IDs, even when generation omits one, never the whole 500-row split or only the emitted predictions.
- Official schema version 2 is parsed fail-closed; resolved, unresolved, empty-patch, error, and incomplete IDs all remain in the denominator.
- Existing `swe-bench-pro` keeps dataset `ScaleAI/SWE-bench_Pro` and `agent.step_limit=1000`.
- The new comparison row publishes only Fable 5 at 95.0; all other exact-model values remain absent.
- A local one-trial mini-SWE-agent run is not comparable to Anthropic's five-trial mean, so all published deltas for the row are withheld.
- Documentation must state the Linux x86 Docker constraint and OpenAI's retirement/contamination caveat.
- Do not modify or stage the pre-existing untracked `tests/examples/` tree.

---

### Task 1: Strict prediction and official report contracts

**Files:**
- Create: `kairyu/bench/adapters/swebench.py`
- Modify: `tests/bench/test_bench_agentic.py`

**Interfaces:**
- Produces: `load_prediction_ids(path: Path) -> tuple[str, ...]`.
- Produces: `parse_swebench_report(report: dict, *, selected_ids: Sequence[str] | None = None) -> tuple[list[ItemResult], int]`.
- Produces: `find_swebench_report(workdir: Path) -> dict`.
- Consumes: official schema-v2 count/list fields and `ItemResult`.

- [ ] **Step 1: Replace the loose report fixture with the complete official schema and write failing category tests**

```python
SWEBENCH_REPORT = {
    "schema_version": 2,
    "total_instances": 5,
    "submitted_instances": 4,
    "completed_instances": 2,
    "resolved_instances": 1,
    "unresolved_instances": 1,
    "empty_patch_instances": 1,
    "error_instances": 1,
    "completed_ids": ["resolved-1", "unresolved-1"],
    "incomplete_ids": ["incomplete-1"],
    "empty_patch_ids": ["empty-1"],
    "submitted_ids": ["resolved-1", "unresolved-1", "empty-1", "error-1"],
    "resolved_ids": ["resolved-1"],
    "unresolved_ids": ["unresolved-1"],
    "error_ids": ["error-1"],
}

def test_parse_swebench_report_preserves_every_official_outcome():
    selected = (*SWEBENCH_REPORT["submitted_ids"], "incomplete-1")
    items, total = parse_swebench_report(SWEBENCH_REPORT, selected_ids=selected)
    by_id = {item.item_id: item for item in items}
    assert total == 5
    assert by_id["resolved-1"].score == 1.0
    assert by_id["unresolved-1"].score == 0.0
    assert by_id["empty-1"].score == 0.0
    assert by_id["error-1"].status == "failed"
    assert by_id["incomplete-1"].status == "failed"
```

- [ ] **Step 2: Add malformed-schema tests before implementation**

Use parametrized mutations for a non-2 schema version, boolean/negative/mismatched counts, duplicate IDs, invalid overlap between terminal categories, `completed_ids - error_ids != resolved_ids | unresolved_ids`, `submitted_ids != completed | empty | error`, and `selected_ids != submitted | incomplete`. SWE-bench 4.1's documented completed/error overlap for an empty or malformed report file must be accepted with error precedence. Each rejected mutation must expect a `ValueError` naming the violated field or set relationship.

- [ ] **Step 3: Add prediction mapping and report-discovery tests**

```python
def test_load_prediction_ids_validates_and_sorts_mini_swe_mapping(tmp_path):
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({
        "b": {"instance_id": "b", "model_name_or_path": "m", "model_patch": "diff"},
        "a": {"instance_id": "a", "model_name_or_path": "m", "model_patch": ""},
    }), encoding="utf-8")
    assert load_prediction_ids(path) == ("a", "b")
```

Also reject non-objects, empty objects, empty/non-string IDs, non-object rows, mismatched `instance_id`, missing/non-string `model_name_or_path`, a missing `model_patch`, and a `model_patch` that is neither a string nor null. Report discovery must accept exactly one root schema-v2 report and reject zero or multiple candidates.

- [ ] **Step 4: Run the focused tests and observe the expected import/assertion failures**

Run: `uv run pytest tests/bench/test_bench_agentic.py -k 'swebench and (parse or prediction or report)' -v`

Expected: FAIL because the shared strict functions and complete validation do not exist yet.

- [ ] **Step 5: Implement minimal strict helpers in the shared module**

Implement count/list readers that reject booleans and enforce exact official relationships:

```python
_COUNT_TO_IDS = {
    "submitted_instances": "submitted_ids",
    "completed_instances": "completed_ids",
    "resolved_instances": "resolved_ids",
    "unresolved_instances": "unresolved_ids",
    "empty_patch_instances": "empty_patch_ids",
    "error_instances": "error_ids",
}

def _required_count(report: dict, name: str) -> int:
    value = report.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value

def _required_id_set(report: dict, name: str) -> set[str]:
    value = report.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate IDs")
    return set(value)

def parse_swebench_report(report, *, selected_ids=None):
    if report.get("schema_version") != 2:
        raise ValueError("SWE-bench report schema_version must be 2")
    counts = {name: _required_count(report, name) for name in (
        "total_instances", *_COUNT_TO_IDS,
    )}
    ids = {
        name: _required_id_set(report, name)
        for name in (*_COUNT_TO_IDS.values(), "incomplete_ids")
    }
    for count_name, ids_name in _COUNT_TO_IDS.items():
        if counts[count_name] != len(ids[ids_name]):
            raise ValueError(f"{count_name} does not match {ids_name}")
    resolved = ids["resolved_ids"] | ids["unresolved_ids"]
    if ids["resolved_ids"] & ids["unresolved_ids"]:
        raise ValueError("resolved_ids and unresolved_ids must be disjoint")
    if resolved & ids["error_ids"] or ids["completed_ids"] & ids["empty_patch_ids"]:
        raise ValueError("submitted outcome IDs have an invalid overlap")
    # v4.1 can mark an existing malformed task report completed and errored.
    if ids["completed_ids"] - ids["error_ids"] != resolved:
        raise ValueError("non-error completed IDs must equal resolved plus unresolved")
    terminal = ids["completed_ids"] | ids["empty_patch_ids"] | ids["error_ids"]
    if ids["submitted_ids"] != terminal:
        raise ValueError("submitted_ids must equal completed, empty-patch, and error IDs")
    selected = set(selected_ids) if selected_ids is not None else (
        ids["submitted_ids"] | ids["incomplete_ids"]
    )
    if ids["submitted_ids"] & ids["incomplete_ids"]:
        raise ValueError("submitted_ids and incomplete_ids must be disjoint")
    if selected != ids["submitted_ids"] | ids["incomplete_ids"]:
        raise ValueError("selected IDs must equal submitted_ids plus incomplete_ids")
    if counts["total_instances"] != len(selected):
        raise ValueError("total_instances does not match selected IDs")
```

The returned item errors are exactly `"SWE-bench harness error"` and `"SWE-bench evaluation incomplete"`; empty patches are completed zero-score items with `details={"outcome": "empty_patch"}`.

- [ ] **Step 6: Run the focused contract tests until green**

Run: `uv run pytest tests/bench/test_bench_agentic.py -k 'swebench and (parse or prediction or report)' -v`

Expected: PASS.

- [ ] **Step 7: Commit the contract layer**

```bash
git add kairyu/bench/adapters/swebench.py tests/bench/test_bench_agentic.py
git commit -m "refactor(bench): validate SWE-bench evidence"
```

---

### Task 2: Shared two-stage adapter and Verified specialization

**Files:**
- Modify: `kairyu/bench/adapters/swebench.py`
- Replace implementation: `kairyu/bench/adapters/swebench_pro.py`
- Create: `kairyu/bench/adapters/swebench_verified.py`
- Modify: `tests/bench/test_bench_agentic.py`
- Modify: `tests/bench/test_bench_agentic_conditions.py`

**Interfaces:**
- Produces: immutable `SweBenchSpec(name, display_name, subset, dataset, step_limit, comparable_to_published, incomparable_reason, annotations)`.
- Produces: `SweBenchAdapter`, with `_generate_command`, `_evaluate_command`, `_preconditions`, `run`, and `_failed` shared by both wrappers.
- Produces: `_evaluate_command(predictions: Path, instance_ids: Sequence[str], run_id: str, report_dir: Path, workers: int) -> list[str]`.
- Produces: `SweBenchProAdapter` and `SweBenchVerifiedAdapter` thin subclasses.
- Consumes: Task 1 helpers and existing `RunContext`, `BenchTarget`, `AdapterInfo`, `summarize_items`, and subprocess conventions.

- [ ] **Step 1: Write failing command-shape tests for both specifications**

```python
@pytest.mark.parametrize(
    ("adapter", "subset", "dataset", "steps"),
    [
        (SweBenchProAdapter(), "ScaleAI/SWE-bench_Pro", "ScaleAI/SWE-bench_Pro", 1000),
        (SweBenchVerifiedAdapter(), "verified", "princeton-nlp/SWE-bench_Verified", 250),
    ],
)
def test_swebench_variants_keep_their_official_conditions(tmp_path, adapter, subset, dataset, steps):
    command = adapter._generate_command(_target(), _ctx(tmp_path), Path("mini-output"))
    configs = [command[i + 1] for i, value in enumerate(command) if value == "--config"]
    assert configs[:2] == ["swebench.yaml", f"agent.step_limit={steps}"]
    assert _flag_value(command, "--subset") == subset
    evaluation = adapter._evaluate_command(
        Path("preds.json"), ("a", "b"), "run", Path("reports"), 3
    )
    assert _flag_value(evaluation, "--dataset_name") == dataset
    assert evaluation[evaluation.index("--instance_ids") + 1:] == ["a", "b"]
```

In the actual assertion, check `--max_workers`, `--run_id`, and `--report_dir` before slicing the IDs, or place `--instance_ids` last so the final equality is exact.

- [ ] **Step 2: Write the failing end-to-end fake-subprocess test**

The fake dataset selection establishes four exact IDs before generation. The fake generation stage writes a valid four-row mini-SWE prediction mapping. The fake evaluation stage asserts those selected IDs and writes a complete schema-v2 report with two resolved, one unresolved, and one harness error. Add a second case where one prediction is omitted but remains incomplete in the four-item denominator. Assert score `0.5`, total `4`, partial status, both stored commands, report category counts, persistent artifact paths, redacted captured logs, and no credential value in serialized methodology.

- [ ] **Step 3: Add fail-closed tests**

Cover generation timeout/non-zero exit, missing/malformed predictions, evaluation timeout/non-zero exit, zero/multiple report candidates, and invalid schema. Assert every result has `status == "failed"`, `metrics["score"] is None`, and a stage-specific reason.

- [ ] **Step 4: Run the focused flow tests and observe failure**

Run: `uv run pytest tests/bench/test_bench_agentic.py tests/bench/test_bench_agentic_conditions.py -k swebench -v`

Expected: FAIL because Verified and the shared adapter do not exist.

- [ ] **Step 5: Implement the specification and shared runner**

```python
@dataclass(frozen=True)
class SweBenchSpec:
    name: str
    display_name: str
    subset: str
    dataset: str
    step_limit: int
    comparable_to_published: bool = True
    incomparable_reason: str = ""
    annotations: tuple[str, ...] = ()

class SweBenchAdapter:
    spec: SweBenchSpec
    info: AdapterInfo

    def _evaluate_command(self, predictions, instance_ids, run_id, report_dir, workers):
        return [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.spec.dataset,
            "--split", "test",
            "--predictions_path", str(predictions),
            "--max_workers", str(workers),
            "--run_id", run_id,
            "--report_dir", str(report_dir),
            "--instance_ids", *instance_ids,
        ]
```

In `run`, load and persist the official dataset selection before generation, validate predictions as its subset between stages, derive a run id containing `ctx.run_id` and the adapter name, pass the full selection to evaluation, parse exactly one report, and call `summarize_items` with the specification's comparability fields. Persist the mini-SWE output, selection manifest, redacted stage logs, evaluator logs, `official_report`, selected/predicted/missing IDs, both shell-joined commands, dataset, subset, split, scaffold, step limit, and concurrency below the run artifact root; never persist environment values. Reuse failed-run artifacts normally, but isolate explicit `--rerun` attempts and evaluator-taint recovery.

- [ ] **Step 6: Replace Pro with a thin specification while retaining compatibility exports**

```python
PRO_SPEC = SweBenchSpec(
    name="swe-bench-pro",
    display_name="SWE-Bench Pro",
    subset="ScaleAI/SWE-bench_Pro",
    dataset="ScaleAI/SWE-bench_Pro",
    step_limit=1000,
    annotations=PRO_ANNOTATIONS,
)

class SweBenchProAdapter(SweBenchAdapter):
    def __init__(self):
        super().__init__(PRO_SPEC)
```

Re-export `parse_swebench_report` from this module so existing imports remain valid.

- [ ] **Step 7: Add the Verified thin specification**

Its annotations state the standard 250-step/$3 config, one-trial limitation, forwarded sampling fields, Linux x86 Docker image constraint, and OpenAI retirement URL. Set `comparable_to_published=False` with the exact reason that the local result is one mini-SWE-agent trial while Anthropic's Fable 5 score is a five-trial mean.

- [ ] **Step 8: Run both adapter test files until green**

Run: `uv run pytest tests/bench/test_bench_agentic.py tests/bench/test_bench_agentic_conditions.py -v`

Expected: PASS.

- [ ] **Step 9: Commit the shared runner and adapter**

```bash
git add kairyu/bench/adapters/swebench.py kairyu/bench/adapters/swebench_pro.py kairyu/bench/adapters/swebench_verified.py tests/bench/test_bench_agentic.py tests/bench/test_bench_agentic_conditions.py
git commit -m "feat(bench): run SWE-bench Verified officially"
```

---

### Task 3: Accuracy registry, binary scoring, and history policy

**Files:**
- Modify: `kairyu/bench/adapters/__init__.py`
- Modify: `kairyu/bench/history.py`
- Modify: `tests/bench/test_bench_agentic.py`
- Modify: `tests/bench/test_bench_runner.py`
- Modify: `tests/bench/test_bench_aggregate.py`
- Modify: `tests/bench/test_bench_history.py`

**Interfaces:**
- Consumes: `SweBenchVerifiedAdapter` from Task 2.
- Produces: a 12-row Accuracy order with Verified directly after Pro.
- Produces: unresolved-runtime history withholding for both SWE-bench rows.

- [ ] **Step 1: Write failing registry/inventory tests**

Assert:

```python
assert ACCURACY_ROW_ORDER[:3] == (
    "swe-bench-pro", "swe-bench-verified", "terminal-bench"
)
assert all_adapters()["swe-bench-verified"].info.binary_outcomes is True
assert all_adapters()["swe-bench-verified"].info.agentic is True
assert all_adapters()["swe-bench-verified"].info.comparable_to_published is False
```

Update the full-suite smoke test to twelve rows and verify both SWE-bench rows skip cleanly without Docker.

- [ ] **Step 2: Write failing history-policy tests**

Add `swe-bench-verified` to the unresolved-runtime expected set in runner/history tests. Parameterize dataset identity expectations so Verified uses `princeton-nlp/SWE-bench_Verified`, no immutable revision, and `history_provenance.complete == False`.

- [ ] **Step 3: Run focused registry/history tests and observe failure**

Run: `uv run pytest tests/bench/test_bench_agentic.py tests/bench/test_bench_runner.py tests/bench/test_bench_aggregate.py tests/bench/test_bench_history.py -k 'accuracy or binary or unresolved or swe_bench or swebench' -v`

Expected: FAIL because the registry and policy sets omit Verified.

- [ ] **Step 4: Register the adapter and update cross-run policy**

Import and instantiate `SweBenchVerifiedAdapter`, insert its name after Pro in `ACCURACY_ROW_ORDER`, and add it to `_UNRESOLVED_RUNTIME_ADAPTERS`. Keep it out of `_NON_DATASET_ADAPTERS` because it declares its Hugging Face dataset even though the harness fetch remains mutable.

- [ ] **Step 5: Run the focused tests until green**

Run the command from Step 3.

Expected: PASS.

- [ ] **Step 6: Commit registry and history integration**

```bash
git add kairyu/bench/adapters/__init__.py kairyu/bench/history.py tests/bench/test_bench_agentic.py tests/bench/test_bench_runner.py tests/bench/test_bench_aggregate.py tests/bench/test_bench_history.py
git commit -m "feat(bench): register Verified in Accuracy history"
```

---

### Task 4: Sparse sourced comparison row

**Files:**
- Modify: `kairyu/bench/reference.py`
- Modify: `tests/bench/test_bench_compare.py`

**Interfaces:**
- Produces: `PUBLISHED_SCORES["swe-bench-verified"] == {"Fable 5": 95.0}`.
- Produces: one selected `FRONTIER_SCORE_RECORDS` entry bound to the Anthropic system card.
- Consumes: existing eight-model `COMPARISON_MODELS`, comparison rendering, and adapter-level non-comparability.

- [ ] **Step 1: Write failing sparse-catalog tests**

```python
def test_verified_reference_keeps_only_the_exact_fable_score():
    assert PUBLISHED_SCORES["swe-bench-verified"] == {"Fable 5": 95.0}
    records = FRONTIER_SCORE_RECORDS["swe-bench-verified"]
    assert [(row["model"], row["score"]) for row in records] == [("Fable 5", 95.0)]
    assert records[0]["condition"] == (
        "SWE-bench Verified; standard configuration; mean of five trials; "
        "thinking blocks included"
    )
```

Update the full selected-matrix expectation with the sparse row. Replace the blanket `"Fugu" in scores` assertion with percentage validation plus an assertion that legacy Fugu-table rows still contain Fugu/Fugu Ultra and the new Verified row is explicitly sparse.

- [ ] **Step 2: Write the failing rendered-report test**

Build a completed local Verified cell with `comparable=False` and its one-vs-five-trials reason. Assert the main row contains Fable `95.0` with a source marker; the other seven published cells are `—`; both legacy delta and the Fable gap are withheld; Markdown contains `n/c`, the five-trial condition, and OpenAI's retirement URL from the run annotation/footnote.

- [ ] **Step 3: Run comparison tests and observe failure**

Run: `uv run pytest tests/bench/test_bench_compare.py -k 'published or catalog or verified or source' -v`

Expected: FAIL because the constructor assumes every row has Fugu/Fugu Ultra and there is no Verified source record.

- [ ] **Step 4: Add source metadata and make record construction sparse-safe**

Add sources:

```python
"anthropic-fable5-system-card": {
    "title": "Claude Fable 5 & Claude Mythos 5 System Card",
    "url": "https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf",
    "publisher": "Anthropic",
    "published_on": "2026-06-09",
    "retrieved_on": "2026-08-12",
    "tier": "primary",
},
"openai-swebench-verified-retirement": {
    "title": "Why we no longer evaluate SWE-bench Verified",
    "url": "https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/",
    "publisher": "OpenAI",
    "published_on": "2026-02-23",
    "retrieved_on": "2026-08-12",
    "tier": "primary",
},
```

Construct legacy Fugu/Fugu Ultra records only when those keys exist, then append `_FRONTIER_ROWS`. Add the exact Fable row and no other models. Keep the retirement source in catalog metadata and the adapter annotation so it is auditable without attaching it as the numeric score's source.

- [ ] **Step 5: Run all comparison tests until green**

Run: `uv run pytest tests/bench/test_bench_compare.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the comparison extension**

```bash
git add kairyu/bench/reference.py tests/bench/test_bench_compare.py
git commit -m "feat(bench): source Verified comparison score"
```

---

### Task 5: Operator documentation and launch surface

**Files:**
- Modify: `README.md`
- Modify: `docs/benchmarks.md`
- Modify: `bench/configs/accuracy.yaml`
- Test: `tests/bench/test_bench_agentic.py`

**Interfaces:**
- Produces: the supported command `kairyu bench run --config bench/configs/accuracy.yaml --only swe-bench-verified`.
- Documents: local official evaluation, exact denominator, standard 250-step/$3 method, Docker architecture constraint, score artifacts, comparison caveat, and retirement warning.

- [ ] **Step 1: Add a failing documentation contract test**

Read `README.md`, `docs/benchmarks.md`, and the Accuracy YAML in a focused test. Assert the docs say `12-benchmark` / `12 Accuracy slots`, list `SWE-bench Verified`, contain the exact launch command, dataset ID, `--subset verified`, `agent.step_limit=250`, the x86 Docker warning, comparison report names, and the OpenAI retirement URL. Assert the config's chat-only exclusion example includes `swe-bench-verified`.

- [ ] **Step 2: Run the documentation contract test and observe failure**

Run: `uv run pytest tests/bench/test_bench_agentic.py -k documentation -v`

Expected: FAIL because docs still describe eleven rows and only Pro.

- [ ] **Step 3: Update README and the full benchmark guide**

Add Verified after Pro in the slot table. Document:

```bash
uv run kairyu bench run \
  --config bench/configs/accuracy.yaml \
  --only swe-bench-verified
```

Explain that Kairyu fixes the official test-split selection first, mini-SWE-agent generates with `verified`/`test` and standard `swebench.yaml` (250 steps, $3 task cap), then swebench 4.x evaluates all selected IDs in Docker. State that Linux x86 images may not run on other architectures, omitted predictions and empty/error/incomplete tasks stay in the denominator, a full run has 500 tasks, and `--limit` evaluates only its exact selected IDs. Document persistent raw artifacts and explicit-rerun isolation. List `scoreboard.{json,md}` and `comparison.{json,md}` and explain the Fable 95.0 orientation-only five-trial value.

- [ ] **Step 4: Update the Accuracy config comment**

Include `swe-bench-verified` among agentic rows excluded for chat-only sampling policies and state both SWE-bench adapters require `attempts=1`.

- [ ] **Step 5: Run the documentation test and benchmark guide checks until green**

Run: `uv run pytest tests/bench/test_bench_agentic.py -k documentation -v`

Expected: PASS.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/benchmarks.md bench/configs/accuracy.yaml tests/bench/test_bench_agentic.py
git commit -m "docs(bench): document Verified execution"
```

---

### Task 6: Final verification, progress record, and pull request

**Files:**
- Modify: `PROGRESS.md`
- Verify: all files changed by Tasks 1–5

**Interfaces:**
- Consumes: the completed implementation and all acceptance tests.
- Produces: a green branch and a pull request closing issue #472.

- [ ] **Step 1: Exercise CLI discovery and the no-Docker launch path**

Run:

```bash
uv run kairyu bench list --suite accuracy
uv run kairyu bench run --config bench/configs/accuracy.yaml --only swe-bench-verified --limit 1 --no-download --no-progress
```

Expected: list shows Verified directly after Pro; the run selects only Verified and either starts the official harness or records an explicit unavailable Docker/dependency precondition without a fabricated score.

- [ ] **Step 2: Run focused benchmark tests**

Run: `uv run pytest tests/bench/test_bench_agentic.py tests/bench/test_bench_agentic_conditions.py tests/bench/test_bench_compare.py tests/bench/test_bench_runner.py tests/bench/test_bench_aggregate.py tests/bench/test_bench_history.py -v`

Expected: PASS.

- [ ] **Step 3: Run repository verification**

Run:

```bash
uv run pytest
uv run ruff check .
uv run kairyu bench entrypoints --check-repo .
uv run python scripts/verify_bench_entrypoints.py
uv run python scripts/verify_bench_wheel.py
```

Expected: every command exits 0. A full 500-instance Docker evaluation is intentionally not run as part of portable verification.

- [ ] **Step 4: Update progress memory before the final implementation commit**

Change the Current Status benchmark-tooling sentence to include official SWE-bench Verified execution and sourced comparison. Remove the open item saying implementation is pending. Add a newest-first English `[progress]` entry referencing issue #472, the adapter/shared module, tests, docs, and the design specification. Run `python3 scripts/check_progress_size.py` and archive first if the enforced budget is exceeded.

- [ ] **Step 5: Review the complete diff and commit the progress record**

Run `git diff --check`, `git status --short`, and `git diff main...HEAD --stat`. Confirm `tests/examples/` is untracked and unstaged.

```bash
git add PROGRESS.md
git commit -m "docs(progress): record Verified benchmark support"
```

- [ ] **Step 6: Push and create the pull request**

```bash
git push -u origin codex/add-swe-bench-verified
gh pr create --base main --head codex/add-swe-bench-verified \
  --title "feat(bench): add SWE-bench Verified" \
  --body $'## Summary\n- add official mini-SWE-agent plus SWE-bench Verified execution\n- preserve schema-v2 outcomes and exact selected denominators\n- extend sourced Accuracy comparison and documentation\n\n## Verification\n- uv run pytest\n- uv run ruff check .\n- benchmark entrypoint and wheel checks\n\nA full 500-task Docker run is not part of portable CI.\n\nCloses #472'
```

The PR body summarizes official generation/evaluation, strict denominator and outcome handling, shared Pro regression protection, sourced comparison/report behavior, documentation, exact verification commands, and the fact that portable CI does not execute all 500 Docker tasks. End with `Closes #472`.
