# Issue 85 Streaming Batch Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream multipart batch uploads into a store-owned transaction so gateway memory is bounded by one configured chunk per request rather than the full upload limit.

**Architecture:** Add an async streaming-save seam to `BatchStore` that owns the temporary file, byte accounting, atomic metadata publication, and cleanup. The HTTP route supplies fixed-size `UploadFile.read()` chunks and translates only the typed size-limit exception into the existing 413 response.

**Tech Stack:** Python async iterators, FastAPI/Starlette `UploadFile`, filesystem transactions, pytest/pytest-asyncio, `httpx.ASGITransport`, `tracemalloc`, Ruff.

## Global Constraints

- Preserve the existing 512 MiB production limit and 413 payload.
- Enforce the limit incrementally before writing an over-limit chunk.
- Never retain more than one upload chunk in the route or store.
- Publish content and metadata only after the entire stream succeeds.
- Remove every partial artifact after over-limit, iterator failure, cancellation, close failure, or metadata failure.
- Keep `save_file(bytes, ...)` for internal callers while extending the full store protocol to eleven methods.
- Preserve tenant ownership, file metadata, download, and batch-worker behavior.
- Follow `.claude/rules/progress-log.md` before committing this storage-boundary change.

---

### Task 1: Pin streaming store transactions

**Files:**
- Create: `tests/unit/test_batch_store_streaming.py`

**Interfaces:**
- Consumes: `BatchStore.save_file_streaming(chunks, filename, purpose, owner, max_bytes)`.
- Produces: success, size-boundary, failure, cancellation, and empty-file contracts.

- [x] **Step 1: Write failing store tests**

Use async generators yielding controlled byte chunks and assert:

1. successful chunks publish one `.bin` and one `.json`, preserve exact bytes/owner/metadata, and leave no `.tmp`;
2. exactly `max_bytes` succeeds, while the next byte raises `FileTooLargeError` and leaves `files/` empty;
3. an iterator that raises after one chunk leaves no content or metadata;
4. cancellation while awaiting the next chunk removes the already-written temporary file;
5. an empty stream publishes a valid zero-byte file transaction.

- [x] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/test_batch_store_streaming.py -q`

Expected: collection fails because `FileTooLargeError` and `save_file_streaming` do not exist.

### Task 2: Implement the store-owned streaming transaction

**Files:**
- Modify: `kairyu/batch/store.py`

**Interfaces:**
- Produces: `FileTooLargeError` and the eleventh `BatchStoreProtocol` method.

- [x] **Step 1: Add the typed size-limit exception and protocol method**

```python
class FileTooLargeError(Exception):
    pass

async def save_file_streaming(
    self,
    chunks: AsyncIterator[bytes],
    filename: str,
    purpose: str,
    owner: str = "default",
    max_bytes: int | None = None,
) -> FileObject: ...
```

- [x] **Step 2: Stream into one exclusive temporary file**

Generate the same file ID format as `save_file`, open `<id>.bin.tmp` with `xb`, and for every chunk calculate the prospective total before writing. Raise `FileTooLargeError` when `prospective > max_bytes`; allow equality and zero bytes.

- [x] **Step 3: Reuse `_commit_file` and clean every failure**

After the async iterator ends, close the handle and call `_commit_file`. Catch `BaseException`, unlink the temporary path, and re-raise so `CancelledError` receives the same cleanup guarantee as ordinary failures. `_commit_file` remains the single content-rename/metadata-last publication owner.

- [x] **Step 4: Verify GREEN for store tests**

Run: `uv run pytest tests/unit/test_batch_store_streaming.py tests/unit/test_batch_store_tenancy.py -q`

Expected: PASS.

### Task 3: Stream the HTTP upload boundary

**Files:**
- Modify: `kairyu/entrypoints/server/batch_routes.py`
- Modify: `tests/server/test_batches.py`

**Interfaces:**
- Consumes: `BatchStore.save_file_streaming` and `FileTooLargeError`.
- Produces: fixed-size `_CHUNK_BYTES` reads and the unchanged 413 API contract.

- [x] **Step 1: Strengthen the oversized-upload regression**

After the 413 response, assert `(tmp_path / "files")` contains no `.tmp`, `.bin`, or `.json` artifacts.

- [x] **Step 2: Add a concurrent memory/read-size regression**

Monkeypatch `_MAX_UPLOAD_BYTES` to 256 KiB, `_CHUNK_BYTES` to 8 KiB, and Starlette's multipart spool threshold below the payload. Send eight concurrent near-limit uploads through `httpx.ASGITransport`, record every `UploadFile.read(size)` request, and use `tracemalloc` to assert a small fixed peak budget. Assert all responses are 200, no requested read exceeds `_CHUNK_BYTES`, and no `.tmp` remains.

- [x] **Step 3: Verify route tests RED**

Run: `uv run pytest tests/server/test_batches.py -k 'oversized_upload or concurrent_upload' -q`

Expected: FAIL because the old route requests `_MAX_UPLOAD_BYTES + 1` in one read and cannot call the new store seam.

- [x] **Step 4: Replace the bulk read with a fixed-size async generator**

```python
_CHUNK_BYTES = 1024 * 1024

async def chunks():
    while chunk := await file.read(_CHUNK_BYTES):
        yield chunk
```

Await `store.save_file_streaming(..., max_bytes=_MAX_UPLOAD_BYTES)` and translate `FileTooLargeError` to the existing 413 JSON body. Read both module constants at request time so tests and operators retain the current monkeypatch/config seam.

- [x] **Step 5: Verify route and end-to-end batch behavior GREEN**

Run: `uv run pytest tests/server/test_batches.py -q`

Expected: PASS, including upload/download and batch lifecycle coverage.

### Task 4: Record and verify the bounded upload boundary

**Files:**
- Modify: `PROGRESS.md`

**Interfaces:**
- Produces: updated Current Status and newest-first Change Log entry for Issue #85.

- [x] **Step 1: Update cross-session progress**

Record the eleventh streaming store method, fixed-size route reads, incremental limit, metadata-last publication, and cleanup on cancellation/failure.

- [x] **Step 2: Run complete verification**

Run: `uv run pytest`

Expected: all non-GPU/non-Hub tests pass.

Run: `uv run ruff check .`

Expected: `All checks passed!`.

- [x] **Step 3: Review and commit**

Run: `git diff --check && git status --short && git diff --stat`

Expected: only the plan, store, route, focused tests, and progress log changed.

Commit: `fix: stream batch uploads into storage`

### Task 5: Publish the issue PR

**Files:**
- No source changes.

**Interfaces:**
- Produces: branch `codex/issue-85-stream-batch-upload` and a ready PR closing Issue #85.

- [ ] **Step 1: Push the branch**

Run: `git push -u origin codex/issue-85-stream-batch-upload`

- [ ] **Step 2: Create the PR against `main`**

Summarize incremental upload enforcement and transactional cleanup, list focused/full verification, and include `Fixes #85`.
