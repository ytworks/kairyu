"""C3: batch/file store tenant isolation — one tenant can never see another's."""

import pytest

from kairyu.batch.store import BatchStore


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
