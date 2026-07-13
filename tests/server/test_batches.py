"""Batch API lifecycle, caps, cancel, restart recovery (goal G3 gate C4)."""

import asyncio
import json

import httpx

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.backend import GenerationResult
from kairyu.engine.mock import MockBackend
from kairyu.outputs import CompletionOutput


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _batch_line(custom_id: str, content: str, model: str = "m") -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": content}],
            },
        }
    )


def _spec_yaml(tmp_path, max_concurrency: int = 2) -> str:
    return f"""
engines:
  m: {{ backend: mock }}
batch:
  data_dir: {tmp_path / "batch-data"}
  max_concurrency: {max_concurrency}
"""


async def _wait_status(client, batch_id: str, wanted: str, attempts: int = 100) -> dict:
    for _ in range(attempts):
        job = (await client.get(f"/v1/batches/{batch_id}")).json()
        if job["status"] == wanted:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"batch never reached {wanted}: {job}")


async def test_batch_lifecycle_end_to_end(tmp_path):
    app = build_app_from_spec(load_deployment_spec(_spec_yaml(tmp_path)))
    # httpx's ASGITransport never runs the lifespan; drive it directly so the
    # worker task is live while requests flow.
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            content = "\n".join(
                [
                    _batch_line("a", "hello"),
                    _batch_line("b", "world"),
                    _batch_line("c", "!", model="missing"),
                ]
            ).encode()
            upload = await client.post(
                "/v1/files",
                files={"file": ("input.jsonl", content, "application/jsonl")},
                data={"purpose": "batch"},
            )
            assert upload.status_code == 200
            file_id = upload.json()["id"]

            created = await client.post(
                "/v1/batches",
                json={"input_file_id": file_id, "endpoint": "/v1/chat/completions"},
            )
            assert created.status_code == 200
            batch_id = created.json()["id"]

            job = await _wait_status(client, batch_id, "completed")
            assert job["request_counts"] == {"total": 3, "completed": 2, "failed": 1}

            output = await client.get(f"/v1/files/{job['output_file_id']}/content")
            lines = [json.loads(line) for line in output.text.splitlines()]
            assert {line["custom_id"] for line in lines} == {"a", "b"}
            assert all(line["response"]["status_code"] == 200 for line in lines)
            assert all(
                line["response"]["body"]["choices"][0]["message"]["content"]
                for line in lines
            )

            errors = await client.get(f"/v1/files/{job['error_file_id']}/content")
            error_lines = [json.loads(line) for line in errors.text.splitlines()]
            assert error_lines[0]["custom_id"] == "c"
            assert "not found" in error_lines[0]["error"]["message"]

            listing = (await client.get("/v1/batches")).json()
            assert listing["data"][0]["id"] == batch_id


async def test_worker_cap_holds_while_interactive_traffic_flows(tmp_path):
    """The batch worker never exceeds its own cap (strictly below the server's)."""

    class CountingBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.batch_active = 0
            self.max_batch_active = 0

        async def generate(self, request):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            is_batch = request.request_id.startswith("batch-")
            if is_batch:
                self.batch_active += 1
                self.max_batch_active = max(
                    self.max_batch_active, self.batch_active
                )
            try:
                await asyncio.sleep(0.02)
                return await super().generate(request)
            finally:
                if is_batch:
                    self.batch_active -= 1
                self.active -= 1

    backend = CountingBackend()
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": backend}, max_concurrency=2)
    lines = "\n".join(_batch_line(f"r{i}", f"prompt {i}") for i in range(8))
    file = store.save_file(lines.encode(), "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")

    interactive = asyncio.create_task(backend.generate(_interactive_request()))
    await worker.process(job.id)
    await interactive  # interactive request completed alongside the batch

    finished = store.get_batch(job.id)
    assert finished.status == "completed"
    assert finished.request_counts.completed == 8
    assert backend.max_batch_active == 2
    assert backend.max_active <= 3  # 2 batch slots + the 1 interactive request


async def test_worker_task_count_is_constant_for_large_batches(tmp_path, monkeypatch):
    max_concurrency = 4
    line_count = 300
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": MockBackend()}, max_concurrency=max_concurrency)
    content = "\n".join(
        _batch_line(f"r{index}", f"prompt {index}") for index in range(line_count)
    )
    file = store.save_file(content.encode(), "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")
    release = asyncio.Event()
    all_consumers_started = asyncio.Event()
    consumer_tasks = set()
    active = 0
    max_active = 0
    peak_task_count = 0
    baseline_task_count = len(asyncio.all_tasks())

    async def recording_run_line(line):
        nonlocal active, max_active, peak_task_count
        task = asyncio.current_task()
        assert task is not None
        consumer_tasks.add(task)
        active += 1
        max_active = max(max_active, active)
        peak_task_count = max(peak_task_count, len(asyncio.all_tasks()))
        if active == max_concurrency:
            all_consumers_started.set()
        try:
            await release.wait()
            return {"custom_id": line["custom_id"], "ok": True}, None
        finally:
            active -= 1

    monkeypatch.setattr(worker, "_run_line", recording_run_line)
    process_task = asyncio.create_task(worker.process(job.id))
    try:
        await asyncio.wait_for(all_consumers_started.wait(), timeout=1)
    finally:
        release.set()
    await process_task

    finished = store.get_batch(job.id)
    assert finished.request_counts.completed == line_count
    assert max_active == max_concurrency
    assert len(consumer_tasks) == max_concurrency
    assert peak_task_count <= baseline_task_count + max_concurrency + 3


async def test_worker_streams_input_without_bulk_read(tmp_path, monkeypatch):
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": MockBackend()}, max_concurrency=2)
    content = (
        "\n\n"
        + _batch_line("a", "first")
        + "\n   \n"
        + _batch_line("b", "second")
        + "\n"
    ).encode()
    file = store.save_file(content, "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")

    def fail_bulk_read(*args, **kwargs):
        raise AssertionError("batch worker must stream input lines")

    monkeypatch.setattr(store, "read_file_content", fail_bulk_read)

    await worker.process(job.id)

    finished = store.get_batch(job.id)
    assert finished.status == "completed"
    assert finished.request_counts.total == 2
    assert finished.request_counts.completed == 2


async def test_streaming_parse_error_fails_after_admitted_lines(tmp_path):
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": MockBackend()}, max_concurrency=1)
    content = (
        _batch_line("accepted", "first")
        + "\nnot-json\n"
        + _batch_line("never-admitted", "last")
    ).encode()
    file = store.save_file(content, "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")

    await worker.process(job.id)

    failed = store.get_batch(job.id)
    assert failed.status == "failed"
    assert failed.request_counts.total == 1
    assert failed.output_file_id is None
    assert failed.error_file_id is None
    assert list((tmp_path / "files").glob("*.tmp")) == []


def _interactive_request():
    from kairyu.engine.backend import GenerationRequest
    from kairyu.sampling_params import SamplingParams

    return GenerationRequest(
        request_id="interactive", prompt="hi", sampling_params=SamplingParams()
    )


async def test_cancel_stops_remaining_lines(tmp_path):
    class SlowBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def generate(self, request):
            self.calls += 1
            await asyncio.sleep(0.05)
            return await super().generate(request)

    backend = SlowBackend()
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": backend}, max_concurrency=1)
    lines = "\n".join(_batch_line(f"r{i}", f"p{i}") for i in range(10))
    file = store.save_file(lines.encode(), "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")

    task = asyncio.create_task(worker.process(job.id))
    await asyncio.sleep(0.08)  # let a line or two start
    cancelled = store.get_batch(job.id)
    cancelled.status = "cancelled"
    store.update_batch(cancelled)
    await task

    cancelled = store.get_batch(job.id)
    assert cancelled.status == "cancelled"
    assert cancelled.output_file_id is None
    assert cancelled.error_file_id is None
    assert backend.calls < 10  # remaining lines were skipped
    assert list((tmp_path / "files").glob("*.tmp")) == []


async def test_restart_marks_inflight_jobs_failed(tmp_path):
    store = BatchStore(tmp_path)
    file = store.save_file(b"", "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")
    job.status = "in_progress"
    store.update_batch(job)

    recovered = BatchStore(tmp_path)  # same data dir, fresh process
    assert recovered.recover_orphans() == (job.id,)
    failed = recovered.get_batch(job.id)
    assert failed.status == "failed"
    assert "restarted" in failed.errors["message"]


async def test_invalid_input_file_fails_job(tmp_path):
    store = BatchStore(tmp_path)
    file = store.save_file(b"not json\n", "bad.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")
    worker = BatchWorker(store, {"m": MockBackend()})
    await worker.process(job.id)
    failed = store.get_batch(job.id)
    assert failed.status == "failed"
    assert "invalid input file" in failed.errors["message"]


async def test_unsupported_endpoint_and_missing_file(tmp_path):
    app = build_app_from_spec(load_deployment_spec(_spec_yaml(tmp_path)))
    async with _client(app) as client:
        missing = await client.post(
            "/v1/batches",
            json={"input_file_id": "file-nope", "endpoint": "/v1/chat/completions"},
        )
        assert missing.status_code == 404

        upload = await client.post(
            "/v1/files",
            files={"file": ("i.jsonl", b"", "application/jsonl")},
            data={"purpose": "batch"},
        )
        bad_endpoint = await client.post(
            "/v1/batches",
            json={"input_file_id": upload.json()["id"], "endpoint": "/v1/embeddings"},
        )
        assert bad_endpoint.status_code == 400
        assert (await client.get("/v1/batches/batch_nope")).status_code == 404
        assert (await client.get("/v1/files/file-nope")).status_code == 404


async def test_files_and_batches_are_isolated_across_tenants(tmp_path, monkeypatch):
    # C3: a tenant must never read, list, or cancel another tenant's files/batches.
    from kairyu.entrypoints.server.app import create_app
    from kairyu.entrypoints.server.batch_routes import add_batch_routes
    from kairyu.entrypoints.server.settings import ServerSettings
    from kairyu.entrypoints.server.tenancy import TenantConfig

    monkeypatch.setenv("KAIRYU_TEST_KEYS", "key-a,key-b")
    settings = ServerSettings(api_keys_env="KAIRYU_TEST_KEYS")
    tenants = TenantConfig(key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"})
    app = create_app({"m": MockBackend()}, settings=settings, tenant_config=tenants)
    store = BatchStore(tmp_path)
    add_batch_routes(app, store, BatchWorker(store, {"m": MockBackend()}))
    a = {"Authorization": "Bearer key-a"}
    b = {"Authorization": "Bearer key-b"}

    async with _client(app) as client:
        up = await client.post(
            "/v1/files", headers=a,
            files={"file": ("in.jsonl", _batch_line("c1", "hi"))},
            data={"purpose": "batch"},
        )
        file_id = up.json()["id"]
        created = await client.post(
            "/v1/batches", headers=a,
            json={"input_file_id": file_id, "endpoint": "/v1/chat/completions"},
        )
        batch_id = created.json()["id"]

        # tenant A sees its own objects
        assert (await client.get(f"/v1/files/{file_id}", headers=a)).status_code == 200
        assert (await client.get(f"/v1/batches/{batch_id}", headers=a)).status_code == 200

        # tenant B cannot read, download, list, or cancel them
        assert (await client.get(f"/v1/files/{file_id}", headers=b)).status_code == 404
        assert (await client.get(f"/v1/files/{file_id}/content", headers=b)).status_code == 404
        assert (await client.get(f"/v1/batches/{batch_id}", headers=b)).status_code == 404
        assert (await client.post(f"/v1/batches/{batch_id}/cancel", headers=b)).status_code == 404
        listing = (await client.get("/v1/batches", headers=b)).json()
        assert listing["data"] == []  # B's list never shows A's batch


async def test_oversized_upload_is_rejected(tmp_path, monkeypatch):
    # S7: an upload above the size cap returns 413 instead of buffering
    # unboundedly and risking an OOM.
    import kairyu.entrypoints.server.batch_routes as batch_routes
    from kairyu.entrypoints.server.app import create_app

    monkeypatch.setattr(batch_routes, "_MAX_UPLOAD_BYTES", 16)
    app = create_app({"m": MockBackend()})
    store = BatchStore(tmp_path)
    batch_routes.add_batch_routes(app, store, BatchWorker(store, {"m": MockBackend()}))
    async with _client(app) as client:
        resp = await client.post(
            "/v1/files",
            files={"file": ("big.jsonl", b"x" * 64)},
            data={"purpose": "batch"},
        )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "file_too_large"


async def test_non_object_input_line_does_not_wedge_the_job(tmp_path):
    # S1: a JSON line that is not an object (e.g. `5`) must become a per-line
    # error, not raise out of gather and leave the job stuck in_progress.
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": MockBackend()})
    content = ("5\n" + _batch_line("ok", "hi") + "\n").encode()
    file = store.save_file(content, "i.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")
    await worker.process(job.id)  # must not raise
    done = store.get_batch(job.id)
    assert done.status == "completed"
    assert done.request_counts.failed == 1  # the bad line recorded as an error
    assert done.request_counts.completed == 1  # the good line still ran


async def test_empty_result_still_completes(tmp_path):
    """A GenerationResult with no completions must not crash the worker."""

    class EmptyBackend(MockBackend):
        async def generate(self, request):
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0, text="", token_ids=(), cumulative_logprob=0.0,
                        finish_reason="stop",
                    ),
                ),
            )

    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": EmptyBackend()})
    file = store.save_file(_batch_line("a", "x").encode(), "i.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")
    await worker.process(job.id)
    assert store.get_batch(job.id).status == "completed"
