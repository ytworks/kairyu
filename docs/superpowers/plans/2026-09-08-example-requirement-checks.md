# Example Requirement Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the ensemble example by extracting verifiable user requirements alongside initial image analysis and checking every requirement before final output.

**Architecture:** Add one dependency-free requirements role to the existing primary ensemble. Pass its checklist to planning, candidate answers, synthesis, and the inline audit; use existing bounded refinement. This is general answer verification, not program execution.

**Tech Stack:** Existing example YAML, model prompts, example configuration and scripted-backend tests.

**Spec:** DTO-D16 in `docs/design/example-dual-track-orchestration.md`; user instructions in this task, including the confirmed Qwen 3.8 correction after updating main.

## Global Constraints

- Do not modify `kairyu/` or introduce a verification runtime/service.
- Keep requirements grounded in the original request and applicable to all task types.
- Requirements and existing image analysis use Qwen 3.8 27B with the existing example's medium-thinking mapping (`reasoning_effort: high`, DTO-D14).
- Preserve the direct profiles and existing head/publication behavior; do not claim machine-enforced correctness.
- Do not overwrite unrelated untracked work.

## Updated baseline

The initial plan inspected an outdated local main. On 2026-09-08, remote main
097affb1 already had Qwen 3.8, conditional `image_description`, per-role effort,
five routes, and the dual-track ensemble. Preserve that implementation instead of restoring the retired
Qwen 3.6 seven-role DAG. The user confirmed Qwen 3.8 after this discovery.
The actual pre-change primary has 11 roles and max_steps 19; this change adds
one role and one budget step.

Local main's three unrelated commits were retained on
`codex/preserve-local-main-20260908`; main was updated and pulled before
implementation on `codex/ensemble-requirement-checks`.

## Task 1: Checklist and per-item audit

**Files:** `examples/qwen3.8-deepseek-v4-8gpu/auto-max.yaml`;
`tests/unit/test_tiered_frontier_examplectl.py`;
`tests/unit/test_tiered_requirement_dag.py`.

- [x] Add independent `requirements` on tier1 beside `image_description`, with fixed medium effort, a 4096-token cap, and seed offset 10.
- [x] Extract stable IDs, minimum/optional priority, requirements, observable acceptance criteria, and source instructions; flag ambiguity and reject invented task obligations.
- [x] Pass the checklist as untrusted derived data to policies, all answerers, critique, synthesis/headless synthesis, and audit. The original request stays authoritative.
- [x] Audit every ID with status, evidence, and actionable corrections after the existing PASS/FAIL first line. Recover omissions from the request, identify unsupported items, and FAIL valid minimum requirements that are unmet or unverifiable.
- [x] Preserve existing refinement and publication behavior. Cover writing, research, image and tool-call tasks without requiring program execution.

## Task 2: Configuration consistency and validation

**Files:** Example `example.json`, `verification.py`, `README.md`;
`docs/design/example-dual-track-orchestration.md`; `PROGRESS.md`.

- [x] Add the role to the manifest used by readiness; require its trace in primary serving validation. Set normal text calls to 11 and maximum budget to 20, retaining two refinements.
- [x] Correct diagrams to the actual four scheduler waves and document per-item audit, call accounting, medium mapping, and publication limitations.
- [x] Exercise actual YAML through Conductor with scripted engines: simultaneous image/checklist roots; text/image and headed/headless cases; immutable checklist through FAIL→repair→PASS and exhaustion; final-answer separation.
- [x] Render the existing Qwen template with each root's effort to confirm the medium preamble and open thinking span.
- [x] Obtain independent read-only review; no actionable functional findings.
- [x] Record DTO-D16 and the pending GPU gates/digest re-pin without claiming new GPU evidence.

## Validation and practical limits

Focused tests: `UV_CACHE_DIR=/private/tmp/kairyu-uv-cache uv run --no-sync pytest tests/unit/test_tiered_frontier_examplectl.py tests/unit/test_tiered_requirement_dag.py -q --no-cov`.
Final validation: 201 tests passed across the focused tests and existing
Conductor/head/DSL suites; ruff passed for all changed Python files. The
progress-size check and git diff whitespace check also passed.

Model inference on real GPUs is not exercised by scripted tests. The already
committed head, exhausted FAIL, inconclusive-verdict fallback, and streaming
multi-choice audit bypass keep their existing semantics; direct routes have
no checklist. This is a simple model-based minimum-quality check, not a proof
that every published response satisfies every requirement. A live quality and
latency rerun plus served-config digest re-pin remain pending.
