# Issue 86 Batch Request Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make batch chat lines pass through the same validation, normalization, dispatch, and backend-error boundary as interactive chat requests.

**Architecture:** Introduce a frozen typed batch envelope and a transport-neutral chat service. The service owns validation through backend preflight plus buffered execution and response construction; HTTP keeps its streaming/orchestrator transport branches, while batch becomes a thin adapter. Shared error helpers preserve controlled client errors and expose only backend exception class names.

**Tech Stack:** Pydantic v2 models/validators, async Python services, FastAPI/HTTPX, pytest/pytest-asyncio, Ruff.

## Global Constraints

- Reject non-object lines, missing/blank/duplicate `custom_id`, non-`POST` methods, and URLs unequal to the batch job endpoint before engine dispatch.
- Preserve existing HTTP status codes, error payloads, streaming behavior, orchestrator behavior, session affinity, usage accounting, and chat-template rendering.
- Apply the same regular-engine validation order to HTTP and batch: tools, stream/logprob relationships, response format, image rejection, model lookup, `supports_n`, sampling parameters, and backend `validate_request`.
- Convert validation failures into controlled per-line errors; do not meter them.
- Sanitize dispatch failures to exception-class-only payloads in both transports while retaining full server-side logging.
- Keep the batch worker's fixed consumer count and bounded input queue.
- Follow `.claude/rules/progress-log.md` before committing the m7 D7 boundary amendment.

---

### Task 1: Pin the envelope and shared error payload contracts

**Files:**
- Create: `tests/unit/test_batch_envelope.py`
- Create: `kairyu/batch/envelope.py`
- Modify: `kairyu/entrypoints/server/errors.py`

- [x] **Step 1: Write failing unit tests**

Cover a valid frozen envelope and rejections for a non-object, missing/blank `custom_id`, non-POST method, endpoint mismatch, and non-object body. Assert `sanitize_backend_error` returns only the exception class form and never the raw message.

- [x] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/test_batch_envelope.py -q`

Expected: collection fails because the envelope and sanitizer do not exist.

- [x] **Step 3: Implement the typed envelope and errors**

Use a frozen Pydantic model with endpoint validation context. Grow the shared error module with payload builders plus `model_not_found`, `upstream_error`, and `sanitize_backend_error`; retain the current HTTP response shapes.

- [x] **Step 4: Verify GREEN**

Run: `uv run pytest tests/unit/test_batch_envelope.py -q`

Expected: PASS.

### Task 2: Pin batch envelope, deduplication, and leak prevention

**Files:**
- Modify: `tests/server/test_batches.py`

- [x] **Step 1: Add failing envelope integration tests**

Use a counting backend and submit DELETE, wrong URL, missing/blank `custom_id`, and duplicate-ID lines. Assert every invalid line creates a controlled error record, duplicates are rejected exactly once, and no invalid line dispatches.

- [x] **Step 2: Add a failing backend leak regression**

Make the backend raise with an internal URL and secret in its message. Assert the batch error record contains the same class-only payload as the HTTP helper and contains neither secret nor topology.

- [x] **Step 3: Verify RED**

Run: `uv run pytest tests/server/test_batches.py -k 'envelope or duplicate_custom_id or backend_error_is_sanitized' -q`

Expected: FAIL because the worker ignores method/URL/ID invariants and serializes `str(error)`.

### Task 3: Pin HTTP/batch validation parity

**Files:**
- Create: `tests/server/test_chat_parity.py`

- [x] **Step 1: Add paired transport helpers**

Drive one request through `POST /v1/chat/completions` and the same body through `BatchWorker.process`, recording status/error payloads and backend dispatch counts.

- [x] **Step 2: Add parametrized validation parity cases**

Cover invalid/named tool choice before generation, unsatisfied required tool choice, invalid response format, image content, `supports_n=False`, unknown model, invalid sampling parameters, backend `validate_request`, and backend exception sanitization.

- [x] **Step 3: Verify RED**

Run: `uv run pytest tests/server/test_chat_parity.py -q`

Expected: batch cases diverge from HTTP or leak raw backend details.

### Task 4: Extract and adopt the shared chat service

**Files:**
- Create: `kairyu/entrypoints/server/chat_service.py`
- Modify: `kairyu/entrypoints/server/app.py`
- Modify: `kairyu/batch/worker.py`
- Modify: `tests/server/test_batches.py`

- [x] **Step 1: Move transport-neutral chat construction helpers**

Move sampling conversion, tool normalization/parsing, response-format checks, prompt rendering, completion response construction, and usage shaping into `chat_service.py`; import/re-export private helpers needed by the streaming code without changing behavior.

- [x] **Step 2: Add typed validation and execution results**

Implement `ChatRequestError`, `validate_chat_request`, and `execute_chat`. Validation returns the selected engine, rendered prompt, `GenerationRequest`, and normalized tool choice; execution returns both the response and generation result and enforces tool-choice satisfaction.

- [x] **Step 3: Refactor the HTTP regular-engine path**

Map `ChatRequestError` through shared HTTP errors, retain the auto-model branch in `app.py`, retain direct engine streaming for tool-free streams, and preserve usage/session-affinity behavior.

- [x] **Step 4: Refactor the batch adapter**

Validate `BatchLineEnvelope` against `job.endpoint`, atomically track seen IDs before awaits, execute the shared service, meter only successes, map typed request failures to controlled errors, and sanitize all other dispatch exceptions.

- [x] **Step 5: Verify focused GREEN**

Run: `uv run pytest tests/unit/test_batch_envelope.py tests/server/test_chat_parity.py tests/server/test_batches.py tests/server/test_openai_api.py -q`

Expected: PASS.

### Task 5: Record, verify, and publish

**Files:**
- Modify: `docs/design/m7-productionization.md`
- Modify: `PROGRESS.md`

- [x] **Step 1: Amend m7 D7 and cross-session progress**

Record typed envelopes, per-job ID uniqueness, the shared request service, pre-dispatch 4xx behavior, and exception-class-only backend failures.

- [x] **Step 2: Run complete verification**

Run: `uv run pytest`

Expected: all non-GPU/non-Hub tests pass.

Run: `uv run ruff check .`

Expected: `All checks passed!`.

- [x] **Step 3: Review and commit**

Run: `git diff --check && git status --short && git diff --stat`

Commit: `fix: unify batch and chat request validation`

- [ ] **Step 4: Push and create the ready PR**

Push `codex/issue-86-batch-request-boundary`, create a ready PR against `main`, list focused/full verification, and include `Fixes #86`.
