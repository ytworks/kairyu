"""Bounded, cancellation-safe submission of prompt preparation work.

Prompt rendering and tokenizer preflight are CPU work that must not serialize
the serving event loop.  They also must not fill asyncio's shared default
executor: the native engine uses that executor for progress and shutdown work.
This module keeps small, per-event-loop admission lanes in front of dedicated
process-owned executors and propagates ContextVars explicitly.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_ResultT = TypeVar("_ResultT")

# Prompt work owns its executor so native engine progress, shutdown, and other
# ``asyncio.to_thread`` users cannot deadlock behind request CPU work.
_MAX_PROMPT_WORKERS = 4
_prompt_executor = ThreadPoolExecutor(
    max_workers=_MAX_PROMPT_WORKERS,
    thread_name_prefix="kairyu-prompt",
)
# JSON decoding and Pydantic graph construction retain the GIL for material
# portions of large bodies.  One ingress worker avoids multiplying loop stalls
# and body memory while the independent prompt lane still forms a pipeline.
_MAX_REQUEST_BODY_WORKERS = 1
_request_body_executor = ThreadPoolExecutor(
    max_workers=_MAX_REQUEST_BODY_WORKERS,
    thread_name_prefix="kairyu-request-body",
)
_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.ReferenceType[asyncio.Semaphore]
] = weakref.WeakKeyDictionary()
_request_body_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.ReferenceType[asyncio.Semaphore]
] = weakref.WeakKeyDictionary()
_limiters_lock = threading.Lock()


def _limiter_for_running_loop(
    limiters: weakref.WeakKeyDictionary[
        asyncio.AbstractEventLoop,
        weakref.ReferenceType[asyncio.Semaphore],
    ],
    workers: int,
) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    with _limiters_lock:
        limiter_ref = limiters.get(loop)
        limiter = limiter_ref() if limiter_ref is not None else None
        if limiter is None:
            limiter = asyncio.Semaphore(workers)
            # A Semaphore binds back to its loop once callers queue.  Keep the
            # value weak as well as the key so repeated asyncio.run()/embedded
            # loops do not form a global loop -> semaphore -> loop leak.
            limiters[loop] = weakref.ref(limiter)
    return limiter


def _release_abandoned_work(
    limiter: asyncio.Semaphore,
    task: asyncio.Future[object],
) -> None:
    """Release a cancelled caller's lane and retrieve a late worker failure."""

    limiter.release()
    if not task.cancelled():
        task.exception()


def _release_serial_gate(
    gate: asyncio.Lock,
    task: asyncio.Task[object],
) -> None:
    """Transfer gate release to an abandoned but still-running worker task."""

    gate.release()
    if not task.cancelled():
        task.exception()


async def _run_bounded_work(
    executor: ThreadPoolExecutor,
    limiter: asyncio.Semaphore,
    function: Callable[..., _ResultT],
    /,
    *args: object,
    **kwargs: object,
) -> _ResultT:
    await limiter.acquire()
    task: asyncio.Future[_ResultT] | None = None
    release_when_done = False
    try:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        call = functools.partial(function, *args, **kwargs)
        task = asyncio.ensure_future(
            loop.run_in_executor(
                executor,
                context.run,
                call,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            release_when_done = True
            task.add_done_callback(lambda done: _release_abandoned_work(limiter, done))
            raise
    finally:
        if not release_when_done:
            limiter.release()


async def run_prompt_work(
    function: Callable[..., _ResultT],
    /,
    *args: object,
    **kwargs: object,
) -> _ResultT:
    """Run pure prompt CPU work without blocking or flooding the event loop.

    Cancellation before lane acquisition submits no work.  Once submitted,
    cancellation returns promptly but the lane remains occupied until the
    underlying thread actually finishes; this prevents abandoned work from
    exceeding the concurrency bound.  Callers must publish caches or other
    state only after this coroutine returns successfully.
    """

    limiter = _limiter_for_running_loop(_limiters, _MAX_PROMPT_WORKERS)
    return await _run_bounded_work(
        _prompt_executor,
        limiter,
        function,
        *args,
        **kwargs,
    )


async def run_request_body_work(
    function: Callable[..., _ResultT],
    /,
    *args: object,
    **kwargs: object,
) -> _ResultT:
    """Run request decoding separately from TTFT-critical prompt preparation."""

    limiter = _limiter_for_running_loop(
        _request_body_limiters,
        _MAX_REQUEST_BODY_WORKERS,
    )
    return await _run_bounded_work(
        _request_body_executor,
        limiter,
        function,
        *args,
        **kwargs,
    )


async def run_serialized_prompt_work(
    gate: asyncio.Lock,
    still_requires_serialization: Callable[[], bool] | None,
    function: Callable[..., _ResultT],
    /,
    *args: object,
    **kwargs: object,
) -> _ResultT:
    """Run prompt work while an event-loop gate outlives caller cancellation.

    Cancellation while waiting for ``gate`` submits no work. Once acquired, an
    owned task is shielded so a cancelled caller returns promptly while the
    gate remains held until the actual worker settles. ``still_requires_`` lets
    a lazy tokenizer become explicitly concurrency-safe while queued.
    """

    await gate.acquire()
    task: asyncio.Task[_ResultT] | None = None
    gate_owned = True
    release_when_done = False
    try:
        if still_requires_serialization is not None and not still_requires_serialization():
            gate.release()
            gate_owned = False
            return await run_prompt_work(function, *args, **kwargs)
        task = asyncio.create_task(run_prompt_work(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            release_when_done = True
            task.add_done_callback(lambda done: _release_serial_gate(gate, done))
            raise
    finally:
        if gate_owned and not release_when_done:
            gate.release()
