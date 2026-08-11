# Accuracy Reference Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare every completed Accuracy-suite run with the supplied eight-model score matrix and render an auditable source for every selected or alternate reference score.

**Architecture:** Keep immutable source and score metadata in `kairyu.bench.reference`; keep comparison construction and Markdown presentation in `kairyu.bench.compare`. Benchmark runs remain offline and reproducible: reports embed committed source identifiers and metadata, while the renderer assigns deterministic compact source markers.

**Tech Stack:** Python 3.12, pytest, Pydantic-backed benchmark artifacts, Markdown reports, Ruff, uv.

## Global Constraints

- Reference columns are exactly `Fugu`, `Fugu Ultra`, `Fable 5`, `GPT-5.6 Sol`, `DeepSeek-V4-Flash-0731`, `Qwen3.8 MAX`, `GLM-5.2`, and `Kimi K3`, in that order.
- DeepSeek-V4-Flash-0731 has a selected score only for Terminal-Bench 2.1; older-snapshot values must stay absent.
- Missing and inapplicable scores remain missing and render as `—`, never zero.
- Every selected and alternate score is source-bound and carries source class, condition, and optional notes.
- Reports perform no network requests and do not change adapter, dataset, scoring, sampling, or run-comparison behavior.
- The untracked `tests/examples/` tree belongs to the user and must not be modified or committed.

---

### Task 1: Replace the six-model reference catalog with the final eight-model matrix

**Files:**
- Modify: `tests/bench/test_bench_compare.py:46-105`
- Modify: `kairyu/bench/reference.py:1-435`

**Interfaces:**
- Consumes: `PUBLISHED_SCORES`, the existing eleven adapter keys, and the supplied final matrix.
- Produces: `COMPARISON_MODELS: tuple[str, ...]`, `REFERENCE_SOURCES: dict[str, dict[str, object]]`, `FRONTIER_SCORE_RECORDS: dict[str, tuple[dict[str, object], ...]]`, `FRONTIER_SCORE_VARIANTS`, and unchanged `comparison_published()` / `comparison_records()` call shapes.

- [ ] **Step 1: Add failing catalog contract tests**

Add literal expectations to `tests/bench/test_bench_compare.py`:

```python
def test_frontier_catalog_uses_the_supplied_eight_model_order():
    assert COMPARISON_MODELS == (
        "Fugu",
        "Fugu Ultra",
        "Fable 5",
        "GPT-5.6 Sol",
        "DeepSeek-V4-Flash-0731",
        "Qwen3.8 MAX",
        "GLM-5.2",
        "Kimi K3",
    )


def test_frontier_catalog_matches_selected_matrix_values_and_absences():
    selected = {
        benchmark: {record["model"]: record["score"] for record in records}
        for benchmark, records in FRONTIER_SCORE_RECORDS.items()
    }
    assert selected["swe-bench-pro"] == {
        "Fugu": 59.0,
        "Fugu Ultra": 73.7,
        "Fable 5": 80.3,
        "GPT-5.6 Sol": 64.6,
        "Qwen3.8 MAX": 67.7,
        "GLM-5.2": 62.1,
    }
    assert selected["terminal-bench"] == {
        "Fugu": 80.2,
        "Fugu Ultra": 82.1,
        "Fable 5": 88.0,
        "GPT-5.6 Sol": 88.8,
        "DeepSeek-V4-Flash-0731": 82.7,
        "Qwen3.8 MAX": 86.6,
        "GLM-5.2": 81.0,
        "Kimi K3": 88.3,
    }
    assert selected["livecodebench"] == {"Fugu": 92.9, "Fugu Ultra": 93.2}
    deepseek_rows = {
        benchmark
        for benchmark, scores in selected.items()
        if "DeepSeek-V4-Flash-0731" in scores
    }
    assert deepseek_rows == {"terminal-bench"}
```

Extend `test_frontier_catalog_records_are_source_bound_and_unambiguous` to assert `record["source_class"] in {"provider", "third_party"}` and every source has a non-empty HTTPS URL, publication date, retrieval date, publisher, and tier.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
uv run pytest tests/bench/test_bench_compare.py::test_frontier_catalog_uses_the_supplied_eight_model_order tests/bench/test_bench_compare.py::test_frontier_catalog_matches_selected_matrix_values_and_absences tests/bench/test_bench_compare.py::test_frontier_catalog_records_are_source_bound_and_unambiguous -q
```

Expected: failures show the old six-column order, old selected values, missing `source_class`, and missing source records.

- [ ] **Step 3: Implement the final committed catalog**

Update `_record` to require the source classification:

```python
def _record(
    model: str,
    score: float,
    source: str,
    *,
    condition: str,
    source_class: str,
    metric: str = "score_percent",
    comparable: bool = True,
    notes: str | None = None,
) -> dict[str, object]:
    return {
        "model": model,
        "score": score,
        "metric": metric,
        "condition": condition,
        "source": source,
        "source_class": source_class,
        "comparable": comparable,
        "notes": notes,
    }
```

Replace `COMPARISON_MODELS` with the exact eight-name tuple from Step 1. Add the supplied official and third-party URLs to `REFERENCE_SOURCES`, including Fugu, Anthropic/Vals, OpenAI, DeepSeek, Qwen, GLM, Kimi, and the Artificial Analysis evaluation pages. Replace `_FRONTIER_ROWS` and `_VARIANT_ROWS` with records matching the final matrix; generate both Fugu columns from `PUBLISHED_SCORES`; do not fill an absent selected cell from `PUBLISHED_SCORES` or a variant.

- [ ] **Step 4: Run the focused catalog tests and confirm GREEN**

Run the exact command from Step 2.

Expected: `3 passed`.

- [ ] **Step 5: Run all comparison tests**

Run:

```bash
uv run pytest tests/bench/test_bench_compare.py -q
```

Expected: existing assertions that intentionally describe old values fail only where their expected final-matrix values must be updated; update those literal expectations, then rerun to a clean pass.

- [ ] **Step 6: Commit the catalog**

```bash
git add kairyu/bench/reference.py tests/bench/test_bench_compare.py
git commit -m "feat(bench): update frontier accuracy references"
```

---

### Task 2: Render cell-level source markers and reference details

**Files:**
- Modify: `tests/bench/test_bench_compare.py:176-233`
- Modify: `kairyu/bench/compare.py:51-313`

**Interfaces:**
- Consumes: each row's `published_records`, `comparison["reference"]["sources"]`, and `COMPARISON_MODELS`.
- Produces: `_source_markers(reference: dict) -> dict[str, str]`, `_published_cell(row: dict, model: str, markers: dict[str, str]) -> str`, and Markdown sections `Published reference details` and `Published-source catalog`.

- [ ] **Step 1: Add failing renderer behavior tests**

Add these tests with literal, source-linked expectations:

```python
def test_report_marks_each_selected_score_with_its_source():
    comparison = build_comparison(
        _scoreboard(**{"terminal-bench": {"status": "completed", "score": 0.8}})
    )
    markdown = render_comparison_markdown(comparison)
    main_row = next(
        line for line in markdown.splitlines() if line.startswith("| terminal-bench |")
    )
    assert "80.2 [S1]" in main_row
    assert "82.7 [S4]" in main_row
    assert "## Published reference details" in markdown
    assert "Terminal-Bench 2.1; max effort; temperature=1; top_p=.95" in markdown


def test_report_source_catalog_links_markers_and_dates():
    markdown = render_comparison_markdown(
        build_comparison(
            _scoreboard(**{"scicode": {"status": "completed", "score": 0.5}})
        )
    )
    assert "[S1]: https://sakana.ai/fugu-release/" in markdown
    assert "Published 2026-07-23; retrieved 2026-07-25" in markdown
    assert "third_party" in markdown
    assert "AA snapshot reproduced by provider" in markdown


def test_json_records_keep_source_traceability():
    comparison = build_comparison(
        _scoreboard(**{"hle": {"status": "completed", "score": 0.4}})
    )
    records = comparison["rows"][0]["published_records"]
    qwen = next(record for record in records if record["model"] == "Qwen3.8 MAX")
    assert qwen["source"] in comparison["reference"]["sources"]
    assert qwen["source_class"] == "provider"
    assert qwen["condition"]
```

Use the actual deterministic marker number produced by the final `REFERENCE_SOURCES` order; keep the source order stable so the literal markers are reviewable.

- [ ] **Step 2: Run renderer tests and confirm RED**

Run:

```bash
uv run pytest tests/bench/test_bench_compare.py::test_report_marks_each_selected_score_with_its_source tests/bench/test_bench_compare.py::test_report_source_catalog_links_markers_and_dates tests/bench/test_bench_compare.py::test_json_records_keep_source_traceability -q
```

Expected: Markdown assertions fail because selected score cells have no markers and no reference-details section; JSON traceability passes only after Task 1 provides `source_class`.

- [ ] **Step 3: Implement deterministic markers and selected score cells**

Add:

```python
def _source_markers(reference: dict) -> dict[str, str]:
    return {
        source_id: f"S{index}"
        for index, source_id in enumerate(reference["sources"], start=1)
    }


def _published_cell(row: dict, model: str, markers: dict[str, str]) -> str:
    selected = next(
        (
            record
            for record in row.get("published_records", [])
            if record["model"] == model and not record.get("variant")
        ),
        None,
    )
    if selected is None:
        return "—"
    return f"{float(selected['score']):.1f} [{markers[str(selected['source'])]}]"
```

Compute `markers` once in `render_comparison_markdown` and replace bare reference score formatting with `_published_cell`.

- [ ] **Step 4: Render per-record details and linked source definitions**

After the gap table, render one detail row for every selected and variant record with columns `Benchmark`, `Model`, `Score`, `Kind`, `Source`, `Class`, `Condition`, and `Notes`. Use `selected` / `alternate` for `Kind`, the deterministic marker for `Source`, and escape pipe/newline characters with the existing `_markdown_table_text` helper. Render the catalog with marker-prefixed entries containing title, publisher, source tier, publication date, retrieval date, and add Markdown link definitions such as `[S1]: https://sakana.ai/fugu-release/`.

- [ ] **Step 5: Run renderer tests and confirm GREEN**

Run the exact command from Step 2, followed by:

```bash
uv run pytest tests/bench/test_bench_compare.py -q
```

Expected: all comparison tests pass and old delta/comparability behavior remains green.

- [ ] **Step 6: Commit report provenance**

```bash
git add kairyu/bench/compare.py tests/bench/test_bench_compare.py
git commit -m "feat(bench): cite published scores in reports"
```

---

### Task 3: Update durable status and verify the complete change

**Files:**
- Modify: `PROGRESS.md:69-83,98-108`
- Verify: `docs/superpowers/specs/2026-08-11-accuracy-reference-sources-design.md`
- Verify: `docs/superpowers/plans/2026-08-11-accuracy-reference-sources.md`

**Interfaces:**
- Consumes: the passing implementation and the progress-log rules in `.claude/rules/progress-log.md`.
- Produces: a current-status statement describing the eight-model sourced comparison and a final implementation Change Log entry.

- [ ] **Step 1: Update the progress snapshot and implementation entry**

Change `six-model sourced Accuracy comparison` to `eight-model sourced Accuracy comparison with cell-level provenance`. Add a newest-first `[progress]` entry stating that generated Markdown/JSON comparisons bind every selected and alternate reference score to a committed source, condition, and source class. Keep `PROGRESS.md` within 200 lines and at most ten Change Log entries.

- [ ] **Step 2: Run complete verification**

Run each command and inspect its exit code and full summary:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/check_progress_size.py
git diff --check
```

Expected: all tests pass, Ruff reports no errors, progress size validation exits 0, and `git diff --check` prints nothing.

- [ ] **Step 3: Review the final diff and scope**

Run:

```bash
git status --short
git diff --stat main...HEAD
git diff --check main...HEAD
git log --oneline main..HEAD
```

Confirm only `PROGRESS.md`, the approved spec and plan, `kairyu/bench/reference.py`, `kairyu/bench/compare.py`, and `tests/bench/test_bench_compare.py` are included. Confirm `tests/examples/` remains untracked and absent from commits.

- [ ] **Step 4: Commit final status**

```bash
git add PROGRESS.md docs/superpowers/plans/2026-08-11-accuracy-reference-sources.md
git commit -m "docs(progress): record sourced accuracy comparison"
```

- [ ] **Step 5: Push and open the pull request**

```bash
git push -u origin codex/accuracy-reference-sources
gh pr create --base main --head codex/accuracy-reference-sources --title "Add sourced eight-model accuracy comparisons" --body-file /tmp/kairyu-accuracy-reference-pr.md
```

The PR body must summarize the final matrix, cell-level source markers/details, deliberate missing-value policy, and list the focused and full verification commands with their observed results.
