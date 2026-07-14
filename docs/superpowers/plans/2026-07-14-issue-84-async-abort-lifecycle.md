# Issue 84 Async Abort Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AsyncLLMEngine.abort()` a no-op for inactive IDs and promptly close exactly one active backend stream without poisoning later ID reuse.

**Architecture:** Replace the historical abort-marker set with an in-flight registry mapping request IDs to request-local `asyncio.Event` signals. Each `generate()` call races its pending backend `anext()` against that signal and owns all iterator/task cleanup in one `finally` block.

**Tech Stack:** Python 3.11+ async generators, `asyncio`, pytest/pytest-asyncio, Ruff.

## Global Constraints

- Preserve the public vLLM-compatible `generate()` and `abort()` signatures.
- Unknown or inactive aborts must retain no state.
- The same request ID may be reused after normal completion or abort.
- Every opened backend async iterator is closed exactly once on completion, abort, exception, cancellation, or consumer close.
- Reject only concurrently active duplicate request IDs with a clear `ValueError`.
- Follow `.claude/rules/progress-log.md` before committing this lifecycle change.

---

### Task 1: Pin inactive-abort compatibility

**Files:**
- Modify: `tests/compat/test_async_engine_compat.py`

**Interfaces:**
- Consumes: `AsyncLLMEngine.abort(request_id: str)` and `AsyncLLMEngine.generate(...)`.
- Produces: a regression contract that abort-before-submit does not suppress future work.

- [x] **Step 1: Extend the compatibility test with an abort-before-submit reproduction**

```python
async def test_abort_is_accepted():
    engine = _engine()
    await engine.abort("future")

    outputs = [
        output
        async for output in engine.generate(
            "future prompt", SamplingParams(), "future"
        )
    ]

    assert outputs
    assert outputs[-1].finished is True
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/compat/test_async_engine_compat.py::test_abort_is_accepted -q`

Expected: FAIL because the stale `_aborted` marker suppresses every output from the first request reusing `future`.

### Task 2: Pin active-request lifecycle and isolation

**Files:**
- Create: `tests/unit/test_async_engine_abort.py`

**Interfaces:**
- Consumes: the same public async engine API.
- Produces: fake async-generator backends exposing `started`, `release`, and close counters.

- [x] **Step 1: Add focused failing lifecycle tests**

```python
class BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = 0

    async def stream(self, request):
        try:
            self.started.set()
            await asyncio.Event().wait()
            if False:
                yield GenerationResult(request.request_id, request.prompt, ())
        finally:
            self.closed += 1
```

Cover these independent behaviors:

1. aborting a blocked active request makes `anext(engine.generate(...))` finish promptly with `StopAsyncIteration` and increments `closed` once;
2. 1,000 inactive aborts leave no tracked entries, and normal completion empties the registry;
3. two request IDs are isolated, aborting one while the other completes;
4. a finished request ID can be reused;
5. a concurrently active duplicate ID raises `ValueError` without deregistering the original;
6. closing the consumer generator after one output deregisters the request and closes the backend generator once;
7. a synchronous failure while constructing the backend iterator deregisters the request ID.

- [x] **Step 2: Run the new file and verify RED**

Run: `uv run pytest tests/unit/test_async_engine_abort.py -q`

Expected: FAIL because inactive aborts accumulate, blocked streams cannot be interrupted, `_active` does not exist, and duplicate active IDs are not rejected.

### Task 3: Implement event-driven active lifecycle

**Files:**
- Modify: `kairyu/entrypoints/async_engine.py`

**Interfaces:**
- Produces: `self._active: dict[str, asyncio.Event]` and event-driven abort signaling.

- [x] **Step 1: Replace abort markers with the active registry**

```python
self._active: dict[str, asyncio.Event] = {}
```

- [x] **Step 2: Register a request and reject only an active collision**

```python
if request_id in self._active:
    raise ValueError(f"request ID {request_id!r} is already active")
abort_event = asyncio.Event()
self._active[request_id] = abort_event
```

- [x] **Step 3: Race backend progress against abort**

For each iteration, create one `anext(stream)` task and one `abort_event.wait()` task, await `asyncio.wait(..., return_when=asyncio.FIRST_COMPLETED)`, let abort win ties, and otherwise convert the backend partial to the existing `RequestOutput` unchanged.

- [x] **Step 4: Centralize cleanup**

Create the backend iterator inside the protected `try`. In a single `finally`, pop only this request ID, cancel/await any pending wait tasks, and call the backend iterator's `aclose()` when present. Repeated `aclose()` on an already-closed async generator is harmless, while the generator's own `finally` observes exactly one close.

- [x] **Step 5: Make inactive abort a pure no-op**

```python
event = self._active.get(request_id)
if event is not None:
    event.set()
```

- [x] **Step 6: Run focused tests and verify GREEN**

Run: `uv run pytest tests/compat/test_async_engine_compat.py tests/unit/test_async_engine_abort.py -q`

Expected: PASS.

### Task 4: Record and verify the lifecycle correction

**Files:**
- Modify: `PROGRESS.md`

**Interfaces:**
- Produces: a newest-first Change Log entry referencing Issue #84 and the affected files.

- [x] **Step 1: Update cross-session progress**

Add a 2026-07-14 `[amendment]` entry explaining explicit active-request ownership, prompt cancellation, inactive no-op semantics, and exactly-once backend iterator cleanup.

- [x] **Step 2: Run the complete verification suite**

Run: `uv run pytest`

Expected: all non-GPU/non-Hub tests pass.

Run: `uv run ruff check .`

Expected: `All checks passed!`.

- [x] **Step 3: Review the diff and commit**

Run: `git diff --check && git status --short && git diff --stat`

Expected: only the plan, async engine, focused tests, and `PROGRESS.md` changed; no whitespace errors.

Commit: `fix: make async abort request-scoped`

### Task 5: Publish the issue PR

**Files:**
- No source changes.

**Interfaces:**
- Produces: branch `codex/issue-84-async-abort-lifecycle` and a GitHub PR closing Issue #84.

- [ ] **Step 1: Push the branch**

Run: `git push -u origin codex/issue-84-async-abort-lifecycle`

- [ ] **Step 2: Create a ready PR against `main`**

The PR body must summarize lifecycle ownership, list the focused and full verification commands, and include `Fixes #84`.
