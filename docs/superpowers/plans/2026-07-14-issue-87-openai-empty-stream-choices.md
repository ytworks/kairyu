# Issue 87 OpenAI Empty Stream Choices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every observed OpenAI-compatible streamed choice, including valid choices whose final text is empty.

**Architecture:** Treat upstream choice observation, text deltas, and finish reasons as separate state. Initialize text state as soon as an index appears, build partial/final outputs from the union of text and finish indexes with empty defaults, and reject only streams that never contained a choice.

**Tech Stack:** Python async iterators, HTTPX mock SSE streams, `GenerationResult`/`CompletionOutput`, pytest/pytest-asyncio, Ruff.

## Global Constraints

- An observed choice index always survives into the final result even with no text delta.
- Empty text is a valid completed output and retains its finish reason.
- Mixed `n > 1` streams preserve indexes and ordering for both empty and non-empty choices.
- Content-bearing partial yield behavior and delta-derived token placeholders remain unchanged.
- Usage-only or `[DONE]`-only streams still fail because no choice was observed.
- Base this independent PR on `main`; its diff must contain only Issue #87.

---

### Task 1: Pin empty and missing-choice stream behavior

**Files:**
- Modify: `tests/unit/test_openai_backend.py`

- [x] **Step 1: Add a reusable SSE fixture helper**

Construct mock transports from explicit SSE chunks so empty, mixed, and usage-only streams remain readable and deterministic.

- [x] **Step 2: Add an empty single-choice regression**

Send role/empty-content/finish chunks for index 0 and assert one final completion with empty text, empty token IDs, and the upstream `stop` reason.

- [x] **Step 3: Add a mixed `n > 1` regression**

Stream content for index 0 and only role/finish state for index 1; assert the final result contains both ordered indexes and preserves index 1 as empty.

- [x] **Step 4: Preserve the no-choice guard**

Send only a usage chunk plus `[DONE]` and assert a `RuntimeError` whose message states that no choices were streamed.

- [x] **Step 5: Verify RED**

Run: `uv run pytest tests/unit/test_openai_backend.py -k 'empty_text_for_valid or preserves_empty_choice or no_choices_observed' -q`

Expected: valid empty choices raise or disappear, while the no-choice case already raises with the old message.

### Task 2: Track observed choices independently of content

**Files:**
- Modify: `kairyu/engine/openai_backend.py`

- [x] **Step 1: Initialize state on index observation**

Call `texts.setdefault(index, "")` before content accumulation for every upstream choice object.

- [x] **Step 2: Build from the complete observed index set**

Iterate `texts.keys() | finish.keys()` and default missing text/delta state so finish-only choices produce valid empty `CompletionOutput` values.

- [x] **Step 3: Narrow the final guard**

Reject only when neither text nor finish state observed a choice, and update the error message from “no content” to “no choices”.

- [x] **Step 4: Verify focused GREEN**

Run: `uv run pytest tests/unit/test_openai_backend.py -q`

Expected: PASS.

### Task 3: Record the compatibility decision

**Files:**
- Modify: `docs/design/m1-orchestration-and-interface.md`
- Modify: `PROGRESS.md`

- [x] **Step 1: Amend the backend contract**

Record that streamed choice presence is index-based, empty text is valid, and zero-choice streams remain failures.

- [x] **Step 2: Update progress memory**

Update Current Status and prepend the required English amendment entry before committing.

### Task 4: Verify and publish the independent PR

**Files:**
- No additional source files.

- [x] **Step 1: Run complete verification**

Run: `uv run pytest`

Run: `uv run ruff check .`

- [x] **Step 2: Audit the main-relative diff and commit**

Run: `git diff --check` and compare against `main`.

Commit: `fix: preserve empty streamed choices`

- [ ] **Step 3: Push and create the ready PR**

Push `codex/issue-87-openai-empty-stream-choices`, create a ready PR against `main`, list verification, and include `Fixes #87`.
