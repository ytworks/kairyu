# Issue #90 Usage Ledger Resilience Implementation Plan

> **For Codex:** Execute this plan test-first and keep the pull request stacked on
> Issue #86 because both changes own the `create_app` lifecycle boundary.

**Goal:** Ensure every app-created usage ledger is flushed and closed after all
inner lifespan cleanup, while preserving valid usage totals when the JSONL file
contains a truncated tail or malformed complete records.

**Architecture:** `create_app` owns the ledger and wraps the optional caller
lifespan with an outer async context manager. Its `finally` closes the ledger
after batch workers and backend resources finish or fail. `UsageLedger.totals`
validates records independently, skips corruption with severity-specific logs,
and exposes the most recent malformed-record count for inspection.

**Tech stack:** Python 3.12, FastAPI lifespan contexts, JSONL, pytest, Ruff.

---

### Task 1: Pin corruption recovery behavior

**Files:**
- Modify: `tests/server/test_m11_product.py`
- Modify: `kairyu/entrypoints/server/tenancy.py`

1. Add failing tests for a truncated final record, malformed complete records,
   whitespace, partial `/admin/usage` results, and the malformed-record count.
2. Run the focused tests and confirm the current fail-fast parser is red.
3. Parse each non-whitespace line independently; validate tenant and token types.
4. Warn once for a malformed non-newline tail, error for every complete malformed
   record, preserve valid totals, and publish the latest scan count.
5. Make `close()` flush then close while retaining reopen-on-next-write behavior.

### Task 2: Make lifecycle ownership unconditional

**Files:**
- Modify: `tests/server/test_m11_product.py`
- Modify: `tests/server/test_serve_builder.py`
- Modify: `kairyu/entrypoints/server/app.py`
- Modify: `kairyu/deploy/builder.py`

1. Add failing direct-app tests for normal shutdown and a caller lifespan whose
   shutdown raises after yielding.
2. Add a failing deployment-builder test where backend shutdown raises an
   `ExceptionGroup`, while the opened ledger must still be closed.
3. Wrap the provided lifespan (or an empty one) in `create_app`; close the
   app-state ledger in the wrapper's outer `finally`.
4. Document in the builder that worker/backend cleanup is inner cleanup and
   ledger ownership remains with `create_app`.

### Task 3: Record the amended contract and verify

**Files:**
- Modify: `docs/design/m11-product.md`
- Modify: `PROGRESS.md`

1. Amend M11 D3/A7 with lifecycle ownership and malformed JSONL policy.
2. Update the Current Status snapshot and prepend an English Change Log entry.
3. Run focused tests, the full non-GPU suite, and `uv run ruff check .`.
4. Commit, push, open a ready stacked PR against
   `codex/issue-86-batch-request-boundary`, and audit its metadata/checks.
