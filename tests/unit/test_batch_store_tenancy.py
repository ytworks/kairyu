"""C3: batch/file store tenant isolation — one tenant can never see another's."""

import json
from pathlib import Path

import pytest

from kairyu.batch.store import BatchStore


class _CloseFailingHandle:
    """Delegate a real handle, then report its flush-on-close failure."""

    def __init__(self, handle):
        self._handle = handle

    def close(self):
        self._handle.close()
        raise OSError("simulated flush-on-close failure")


def _input_file(store: BatchStore, owner: str):
    return store.save_file(
        b'{"custom_id": "a"}\n', filename="in.jsonl", purpose="batch", owner=owner
    )


def test_get_file_is_scoped_to_owner(tmp_path):
    store = BatchStore(tmp_path)
    a_file = store.save_file(b"secret", filename="a.txt", purpose="batch", owner="tenant-a")
    # the owner reads it; another tenant gets a not-found (existence never leaks)
    assert store.get_file(a_file.id, owner="tenant-a").id == a_file.id
    with pytest.raises(KeyError):
        store.get_file(a_file.id, owner="tenant-b")
    with pytest.raises(KeyError):
        store.read_file_content(a_file.id, owner="tenant-b")
    # the internal/worker path (owner=None) still sees everything
    assert store.read_file_content(a_file.id) == b"secret"


def test_list_batches_only_returns_callers_jobs(tmp_path):
    store = BatchStore(tmp_path)
    a_in = _input_file(store, "tenant-a")
    b_in = _input_file(store, "tenant-b")
    a_job = store.create_batch(a_in.id, "/v1/chat/completions", owner="tenant-a")
    b_job = store.create_batch(b_in.id, "/v1/chat/completions", owner="tenant-b")
    a_ids = {job.id for job in store.list_batches(owner="tenant-a")}
    assert a_ids == {a_job.id}
    assert b_job.id not in a_ids
    # unscoped list (worker recovery) sees both
    assert {job.id for job in store.list_batches()} == {a_job.id, b_job.id}


def test_get_batch_cross_tenant_is_not_found(tmp_path):
    store = BatchStore(tmp_path)
    a_in = _input_file(store, "tenant-a")
    a_job = store.create_batch(a_in.id, "/v1/chat/completions", owner="tenant-a")
    assert store.get_batch(a_job.id, owner="tenant-a").id == a_job.id
    with pytest.raises(KeyError):
        store.get_batch(a_job.id, owner="tenant-b")


def test_cannot_batch_over_another_tenants_file(tmp_path):
    store = BatchStore(tmp_path)
    a_in = _input_file(store, "tenant-a")
    # tenant B references tenant A's file id -> reads as missing, no batch created
    with pytest.raises(KeyError):
        store.create_batch(a_in.id, "/v1/chat/completions", owner="tenant-b")


def test_default_owner_preserves_single_tenant_behavior(tmp_path):
    store = BatchStore(tmp_path)
    f = store.save_file(b"x", filename="a", purpose="batch")  # default owner
    job = store.create_batch(f.id, "/v1/chat/completions")  # default owner
    # the default tenant (keyless mode) sees its own objects normally
    assert store.get_file(f.id, owner="default").id == f.id
    assert store.get_batch(job.id, owner="default").id == job.id
    assert [j.id for j in store.list_batches(owner="default")] == [job.id]


def test_iter_file_lines_streams_binary_lines_without_bulk_read(tmp_path, monkeypatch):
    store = BatchStore(tmp_path)
    file = store.save_file(
        b"first\nsecond\nlast",
        filename="input.jsonl",
        purpose="batch",
        owner="tenant-a",
    )

    def fail_bulk_read(*args, **kwargs):
        raise AssertionError("streaming iteration must not bulk-read file content")

    monkeypatch.setattr(Path, "read_bytes", fail_bulk_read)
    monkeypatch.setattr(store, "read_file_content", fail_bulk_read)

    lines = store.iter_file_lines(file.id, owner="tenant-a")

    assert iter(lines) is lines
    assert list(lines) == [b"first\n", b"second\n", b"last"]


def test_iter_file_lines_checks_owner_before_opening_content(tmp_path, monkeypatch):
    store = BatchStore(tmp_path)
    file = store.save_file(
        b"secret\n", filename="input.jsonl", purpose="batch", owner="tenant-a"
    )
    opened_content = []
    real_open = Path.open

    def tracking_open(path, *args, **kwargs):
        if path.name == f"{file.id}.bin":
            opened_content.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    with pytest.raises(KeyError):
        store.iter_file_lines(file.id, owner="tenant-b")
    with pytest.raises(KeyError):
        store.iter_file_lines("file-missing", owner="tenant-a")

    assert opened_content == []


def test_iter_file_lines_closes_handle_when_iteration_stops_early(
    tmp_path, monkeypatch
):
    store = BatchStore(tmp_path)
    file = store.save_file(
        b"first\nsecond\n", filename="input.jsonl", purpose="batch"
    )
    content_handles = []
    real_open = Path.open

    def tracking_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if path.name == f"{file.id}.bin":
            content_handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    lines = store.iter_file_lines(file.id)

    assert next(lines) == b"first\n"
    assert len(content_handles) == 1
    assert not content_handles[0].closed

    lines.close()

    assert content_handles[0].closed


def test_jsonl_writer_is_lazy_and_appends_each_line_immediately(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="output.jsonl", purpose="batch_output", owner="tenant-a"
    )
    files_dir = tmp_path / "files"

    assert writer.state == "new"
    assert writer.has_content is False
    assert list(files_dir.iterdir()) == []

    writer.append({"custom_id": "a", "ok": True})

    expected = json.dumps({"custom_id": "a", "ok": True}).encode("utf-8") + b"\n"
    temporary_files = list(files_dir.glob("*.tmp"))
    assert writer.state == "open"
    assert writer.has_content is True
    assert list(files_dir.glob("*.bin")) == []
    assert list(files_dir.glob("*.json")) == []
    assert len(temporary_files) == 1
    assert temporary_files[0].read_bytes() == expected

    writer.append({"custom_id": "b", "ok": False})
    expected += json.dumps({"custom_id": "b", "ok": False}).encode("utf-8") + b"\n"
    assert temporary_files[0].read_bytes() == expected

    writer.abort()


def test_jsonl_writer_commit_publishes_exact_owner_scoped_file(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="output.jsonl", purpose="batch_output", owner="tenant-a"
    )
    rows = [{"custom_id": "a", "value": "雪"}, {"custom_id": "b", "value": 2}]
    expected = b""
    for row in rows:
        writer.append(row)
        expected += json.dumps(row).encode("utf-8") + b"\n"

    file = writer.commit()

    assert writer.state == "committed"
    assert file.filename == "output.jsonl"
    assert file.purpose == "batch_output"
    assert file.owner == "tenant-a"
    assert file.bytes == len(expected)
    assert store.get_file(file.id, owner="tenant-a") == file
    assert store.read_file_content(file.id, owner="tenant-a") == expected
    with pytest.raises(KeyError):
        store.get_file(file.id, owner="tenant-b")
    assert list((tmp_path / "files").glob("*.tmp")) == []


def test_jsonl_writer_rollback_hides_an_already_committed_file(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="output.jsonl", purpose="batch_output", owner="tenant-a"
    )
    writer.append({"custom_id": "a", "ok": True})
    file = writer.commit()

    writer.rollback()
    writer.rollback()

    assert writer.state == "aborted"
    with pytest.raises(KeyError):
        store.get_file(file.id, owner="tenant-a")
    assert list((tmp_path / "files").iterdir()) == []


def test_jsonl_writer_abort_removes_temporary_data_and_metadata(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="errors.jsonl", purpose="batch_output", owner="tenant-a"
    )
    writer.append({"error": "bad"})

    writer.abort()

    assert writer.state == "aborted"
    assert list((tmp_path / "files").iterdir()) == []


def test_jsonl_writer_abort_cleans_up_when_close_flush_fails(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="errors.jsonl", purpose="batch_output", owner="tenant-a"
    )
    writer.append({"error": "partial"})
    handle = writer._handle
    writer._handle = _CloseFailingHandle(handle)

    with pytest.raises(OSError, match="flush-on-close"):
        writer.abort()

    assert handle.closed
    assert writer._handle is None
    assert writer.state == "aborted"
    assert list((tmp_path / "files").iterdir()) == []


def test_jsonl_writer_commit_close_failure_aborts_without_visibility(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="output.jsonl", purpose="batch_output", owner="tenant-a"
    )
    writer.append({"custom_id": "partial"})
    handle = writer._handle
    writer._handle = _CloseFailingHandle(handle)

    with pytest.raises(OSError, match="flush-on-close"):
        writer.commit()

    assert handle.closed
    assert writer._handle is None
    assert writer.state == "aborted"
    assert list((tmp_path / "files").iterdir()) == []
    with pytest.raises(KeyError):
        store.get_file(writer._file_id, owner="tenant-a")


def test_jsonl_writer_publish_failure_aborts_without_visibility(tmp_path, monkeypatch):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="output.jsonl", purpose="batch_output", owner="tenant-a"
    )
    writer.append({"custom_id": "partial"})
    real_replace = Path.replace

    def fail_content_publish(path, target):
        if path.name.endswith(".bin.tmp"):
            raise OSError("simulated content publish failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_content_publish)

    with pytest.raises(OSError, match="content publish"):
        writer.commit()

    assert writer._handle is None
    assert writer.state == "aborted"
    assert list((tmp_path / "files").iterdir()) == []
    with pytest.raises(KeyError):
        store.get_file(writer._file_id, owner="tenant-a")


def test_jsonl_writer_metadata_failure_aborts_without_visibility(tmp_path, monkeypatch):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="output.jsonl", purpose="batch_output", owner="tenant-a"
    )
    writer.append({"custom_id": "partial"})

    def fail_metadata_publish(path, payload):
        path.with_suffix(".tmp").write_text(json.dumps(payload), encoding="utf-8")
        raise OSError("simulated metadata publish failure")

    monkeypatch.setattr(store, "_write_json", fail_metadata_publish)

    with pytest.raises(OSError, match="metadata publish"):
        writer.commit()

    assert writer._handle is None
    assert writer.state == "aborted"
    assert list((tmp_path / "files").iterdir()) == []
    with pytest.raises(KeyError):
        store.get_file(writer._file_id, owner="tenant-a")


def test_jsonl_writer_context_exception_aborts_transaction(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer(
        filename="errors.jsonl", purpose="batch_output", owner="tenant-a"
    )

    with pytest.raises(RuntimeError, match="stop"):
        with writer:
            writer.append({"error": "partial"})
            raise RuntimeError("stop")

    assert writer.state == "aborted"
    assert list((tmp_path / "files").iterdir()) == []


def test_jsonl_writer_rejects_empty_commit_and_closed_state_operations(tmp_path):
    store = BatchStore(tmp_path)
    writer = store.create_jsonl_writer("output.jsonl", "batch_output")

    with pytest.raises(RuntimeError, match="empty"):
        writer.commit()

    writer.append({"ok": True})
    writer.commit()
    with pytest.raises(RuntimeError, match="committed"):
        writer.append({"late": True})
    with pytest.raises(RuntimeError, match="committed"):
        writer.commit()

    aborted = store.create_jsonl_writer("errors.jsonl", "batch_output")
    aborted.append({"error": "bad"})
    aborted.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        aborted.append({"late": True})
    with pytest.raises(RuntimeError, match="aborted"):
        aborted.commit()


def test_save_and_read_file_content_remain_byte_compatible(tmp_path):
    store = BatchStore(tmp_path)
    content = b"\x00binary\xff\n"

    file = store.save_file(
        content,
        filename="upload.bin",
        purpose="batch",
        owner="tenant-a",
    )

    assert file.bytes == len(content)
    assert file.owner == "tenant-a"
    assert store.read_file_content(file.id, owner="tenant-a") == content
