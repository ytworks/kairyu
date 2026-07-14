# Issue 88 Per-Choice Tool Satisfaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject required or named tool-choice responses whenever any returned choice lacks a matching tool call.

**Architecture:** Keep the existing response construction, per-choice named filtering, controlled 502, and buffered tool-stream path. Tighten only the shared satisfaction predicate from existential to universal, so non-streaming, buffered streaming, HTTP, and batch callers inherit one deterministic rejection rule without regeneration.

**Tech Stack:** Python predicates, FastAPI/HTTPX, SSE parsing, usage ledger, pytest/pytest-asyncio, Ruff.

## Global Constraints

- For `required` and named tool choices, every returned choice must contain at least one retained tool call.
- Preserve per-choice declared/named function filtering in response construction.
- Reject mixed results with the existing HTTP 502 and `tool_choice_not_satisfied` code before emitting buffered SSE chunks.
- Do not regenerate, retry, or synthesize output.
- Record the consumed generation exactly once even when the response is rejected.
- Leave `auto` and `none` semantics unchanged.
- Base this stacked PR on `codex/issue-86-batch-request-boundary`; its diff must contain only Issue #88.

---

### Task 1: Pin mixed and all-valid multi-choice behavior

**Files:**
- Modify: `tests/server/test_openai_api.py`

- [x] **Step 1: Add a fixed multi-choice backend**

Return two indexed `CompletionOutput` values with configurable text and reported usage so tests can model mixed and fully compliant generations.

- [x] **Step 2: Add failing mixed-choice tests**

For both non-streaming and buffered streaming requests, assert `n=2` plus `[tool call, plain text]` under `required` returns the existing controlled 502. Repeat for named choice with `[named call, non-matching call]` to prove filtering is evaluated per choice.

- [x] **Step 3: Add positive all-choice tests**

Assert both choices calling tools under `required`, and both choices retaining the selected function under named choice, return 200. For buffered streaming, parse SSE and assert both choice indexes, per-choice tool calls, finish reasons, and per-call indexes.

- [x] **Step 4: Verify RED**

Run: `uv run pytest tests/server/test_openai_api.py -k 'multi_choice_tool' -q`

Expected: mixed cases incorrectly return 200 because the predicate uses `any`.

### Task 2: Pin usage accounting on rejection

**Files:**
- Modify: `tests/server/test_m11_product.py`

- [x] **Step 1: Add a mixed-choice ledger regression**

Return one valid tool choice and one plain choice with explicit backend usage. Assert the 502 still records exactly one ledger request with the reported prompt/completion counts.

- [x] **Step 2: Verify RED/contract state**

Run: `uv run pytest tests/server/test_m11_product.py -k 'mixed_tool_choice' -q`

Expected before the predicate fix: the route returns 200, proving the rejected-response contract is not exercised.

### Task 3: Require every choice and record the decision

**Files:**
- Modify: `kairyu/entrypoints/server/chat_service.py`
- Modify: `docs/design/m1-orchestration-and-interface.md`
- Modify: `PROGRESS.md`

- [x] **Step 1: Tighten the shared predicate**

Replace `any(choice.message.tool_calls for choice in choices)` with `all(...)` for required/named modes. Keep the empty-choice behavior explicit and fail closed.

- [x] **Step 2: Verify focused GREEN**

Run: `uv run pytest tests/server/test_openai_api.py tests/server/test_m11_product.py tests/server/test_chat_parity.py tests/server/test_batches.py -q`

Expected: PASS.

- [x] **Step 3: Update design and progress**

Record per-choice satisfaction, deliberate rejection rather than regeneration, buffered-stream parity, and exact-once usage.

### Task 4: Verify and publish the stacked PR

**Files:**
- No additional source files.

- [x] **Step 1: Run complete verification**

Run: `uv run pytest`

Run: `uv run ruff check .`

- [x] **Step 2: Audit the parent-relative diff and commit**

Run: `git diff --check` and compare against `codex/issue-86-batch-request-boundary`.

Commit: `fix: require tool calls in every choice`

- [ ] **Step 3: Push and create the stacked PR**

Push `codex/issue-88-tool-choice-all-choices`, create a ready PR against `codex/issue-86-batch-request-boundary`, link parent PR #94, list verification, and include `Fixes #88`.
