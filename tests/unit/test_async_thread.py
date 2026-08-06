import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from kairyu.async_thread import (
    run_prompt_work,
    run_request_body_work,
    run_serialized_prompt_work,
)


async def test_repeated_fast_prompt_work_releases_each_lane():
    loop_thread = threading.get_ident()

    worker_threads = [await run_prompt_work(threading.get_ident) for _ in range(3)]

    assert all(thread_id != loop_thread for thread_id in worker_threads)


async def test_prompt_lane_does_not_occupy_default_executor(monkeypatch):
    import kairyu.async_thread as async_thread

    monkeypatch.setattr(async_thread, "_MAX_PROMPT_WORKERS", 1)
    loop = asyncio.get_running_loop()
    async_thread._limiters.pop(loop, None)
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-default"))
    prompt_started = threading.Event()
    release_prompt = threading.Event()

    def blocked_prompt():
        prompt_started.set()
        assert release_prompt.wait(2)

    prompt_task = asyncio.create_task(run_prompt_work(blocked_prompt))
    try:
        while not prompt_started.is_set():
            await asyncio.sleep(0)
        default_thread = await asyncio.wait_for(
            asyncio.to_thread(threading.get_ident),
            timeout=0.5,
        )

        assert default_thread != threading.get_ident()
        assert not prompt_task.done()
    finally:
        release_prompt.set()
        await asyncio.gather(prompt_task, return_exceptions=True)


async def test_request_body_backlog_cannot_convoy_prompt_work(monkeypatch):
    import kairyu.async_thread as async_thread

    monkeypatch.setattr(async_thread, "_MAX_REQUEST_BODY_WORKERS", 1)
    monkeypatch.setattr(async_thread, "_MAX_PROMPT_WORKERS", 1)
    loop = asyncio.get_running_loop()
    async_thread._request_body_limiters.pop(loop, None)
    async_thread._limiters.pop(loop, None)
    body_started = threading.Event()
    release_body = threading.Event()
    prompt_started = threading.Event()

    def blocked_body():
        body_started.set()
        assert release_body.wait(2)

    body_task = asyncio.create_task(run_request_body_work(blocked_body))
    try:
        assert await asyncio.to_thread(body_started.wait, 2)
        prompt_task = asyncio.create_task(run_prompt_work(prompt_started.set))
        assert await asyncio.to_thread(prompt_started.wait, 0.5)
        assert not body_task.done()
    finally:
        release_body.set()
        await asyncio.gather(body_task, return_exceptions=True)
    await prompt_task


async def test_prompt_work_releases_event_loop_and_propagates_context():
    sentinel = contextvars.ContextVar("sentinel", default="missing")
    sentinel.set("request-context")
    worker_started = threading.Event()
    release_worker = threading.Event()
    loop_thread = threading.get_ident()

    def blocking_work():
        worker_started.set()
        assert release_worker.wait(2)
        return threading.get_ident(), sentinel.get()

    work = asyncio.create_task(run_prompt_work(blocking_work))
    assert await asyncio.to_thread(worker_started.wait, 2)
    heartbeat = False

    async def beat():
        nonlocal heartbeat
        heartbeat = True

    await beat()
    assert heartbeat
    release_worker.set()
    worker_thread, context_value = await work

    assert worker_thread != loop_thread
    assert context_value == "request-context"


async def test_cancelled_prompt_work_keeps_lane_until_thread_finishes(
    monkeypatch,
):
    import kairyu.async_thread as async_thread

    monkeypatch.setattr(async_thread, "_MAX_PROMPT_WORKERS", 1)
    # Isolate this test from a limiter already created for pytest's loop.
    async_thread._limiters.pop(asyncio.get_running_loop(), None)
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first():
        first_started.set()
        assert release_first.wait(2)

    def second():
        second_started.set()

    first_task = asyncio.create_task(run_prompt_work(first))
    assert await asyncio.to_thread(first_started.wait, 2)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    second_task = asyncio.create_task(run_prompt_work(second))
    await asyncio.sleep(0)
    assert not second_started.is_set()

    release_first.set()
    await second_task
    assert second_started.is_set()


async def test_cancelled_serialized_work_keeps_gate_until_thread_finishes():
    gate = asyncio.Lock()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first():
        first_started.set()
        assert release_first.wait(2)

    def second():
        second_started.set()

    first_task = asyncio.create_task(run_serialized_prompt_work(gate, None, first))
    assert await asyncio.to_thread(first_started.wait, 2)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    second_task = asyncio.create_task(run_serialized_prompt_work(gate, None, second))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    await second_task
    assert second_started.is_set()
