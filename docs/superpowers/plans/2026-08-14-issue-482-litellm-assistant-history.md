# Issue 482 LiteLLM Assistant-History Implementation Plan

**Goal:** Make LiteLLM assistant messages reusable as history without exposing
provider metadata or widening Kairyu's typed API contract.

**Architecture:** Extend the existing shared message-extra predicate in
`chat_service.py` to make role- and value-sensitive decisions. Both preflight
validation and the retained wire shape call the same predicate.

## Task 1: Prove the compatibility gap

**Files:**
- Modify: `tests/server/test_chat_template_policy.py`
- Modify: `tests/server/test_openai_api.py`
- Modify: `tests/server/test_orchestration_usage_trace.py`

Add parameterized prompt-boundary coverage for absent, null, empty-object, and
LiteLLM refusal metadata. Add an HTTP regression using the complete dumped
assistant shape and negative tests for wrong roles, wrong value kinds, non-null
legacy calls, and unrelated extras. Change the L2 round-trip fixture to the
real dictionary-valued metadata. Run the focused tests and observe failures
only for the newly accepted cases.

## Task 2: Implement the narrow boundary exception

**File:** `kairyu/entrypoints/server/chat_service.py`

Change the predicate to receive `ChatMessage`, accept the two explicitly
approved assistant-only shapes, preserve null provider compatibility, and use
it from both validation and wire filtering. Run focused tests and ruff.

## Task 3: Record the superseding contract

**Files:**
- Modify: `docs/design/m11-product.md`
- Modify: `PROGRESS.md`

Append a new M11 amendment rather than rewriting the #480 history, update the
current snapshot, and prepend a concise #482 change-log entry. Run the progress
size check and documentation-sensitive tests.

## Task 4: Verify and publish

Run the focused suite, the full portable suite, ruff, the progress-size check,
`git diff --check`, and a self-review of the final diff. Commit, push
`codex/issue-482-litellm-history`, and open a PR that closes #482 while noting
the remaining live SWE-bench gate.
