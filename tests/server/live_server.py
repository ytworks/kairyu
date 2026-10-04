"""Live uvicorn server and official OpenAI SDK clients for the server tests.

openai-python 3.x talks HTTPX2; an in-process Starlette ``TestClient`` or
``httpx.ASGITransport`` reaches the SDK only through a temporary legacy-httpx
escape hatch. The SDK tests therefore drive the app the way a deployment does:
a real uvicorn server on an ephemeral loopback port, launched with the CLI's
uvicorn options (``ws="none"``, loop/http selection, logging), with the app
lifespan running, and reached by the SDK's own default transport.

The schema gate (``tests/contracts``) records live servers through
``uvicorn.Config``. Every helper here stops its server before it returns, so a
test's exchanges are recorded before the gate validates them at teardown.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import openai
import uvicorn
from starlette.types import ASGIApp

from kairyu.entrypoints.cli import uvicorn_options

LOOPBACK_HOST = "127.0.0.1"
SERVER_TIMEOUT_S = 10.0
SDK_TIMEOUT_S = 30.0
_STARTUP_POLL_S = 0.01
# The SDK requires a key even for an unauthenticated local server.
_LOCAL_API_KEY = "sk-local"


@contextmanager
def serve(config: uvicorn.Config) -> Iterator[int]:
    """Run ``config`` in a background thread; yield the bound port.

    Startup waits for the lifespan and the listening socket. On exit the server
    shuts down gracefully (connections, then lifespan) and the thread joins.
    """

    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="live-uvicorn", daemon=True)
    thread.start()
    deadline = time.monotonic() + SERVER_TIMEOUT_S
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("live uvicorn server exited during startup")
        if time.monotonic() > deadline:
            server.should_exit = True
            raise TimeoutError(f"live uvicorn server did not start in {SERVER_TIMEOUT_S}s")
        time.sleep(_STARTUP_POLL_S)
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(SERVER_TIMEOUT_S)
    if thread.is_alive():
        raise RuntimeError(f"live uvicorn server did not stop in {SERVER_TIMEOUT_S}s")


@contextmanager
def live_server(app: ASGIApp) -> Iterator[str]:
    """Serve ``app`` as ``kairyu serve`` would, on an ephemeral port; yield its base URL."""

    config = uvicorn.Config(app, **{**uvicorn_options(), "host": LOOPBACK_HOST, "port": 0})
    with serve(config) as port:
        yield f"http://{LOOPBACK_HOST}:{port}"


def _sdk_options(base_url: str, *, strict: bool = True) -> dict[str, Any]:
    return {
        "base_url": f"{base_url}/v1",
        "api_key": _LOCAL_API_KEY,
        # Every response body and stream event must match the SDK's own types.
        "_strict_response_validation": strict,
        # A retry would hide a failed exchange and dispatch the request twice.
        "max_retries": 0,
        "timeout": SDK_TIMEOUT_S,
    }


@contextmanager
def openai_client(app: ASGIApp, *, strict: bool = True) -> Iterator[openai.OpenAI]:
    """The official sync SDK with its default transport, against ``app`` on a live server.

    ``strict=False`` only where the SDK's own design rules strict validation
    out: it validates a default (base64) embedding as ``list[float]`` before
    its post-parser decodes it.
    """

    with live_server(app) as base_url:
        with openai.OpenAI(**_sdk_options(base_url, strict=strict)) as client:
            yield client


@asynccontextmanager
async def async_openai_client(app: ASGIApp) -> AsyncIterator[openai.AsyncOpenAI]:
    """The official async SDK with its default transport, against ``app`` on a live server."""

    with live_server(app) as base_url:
        async with openai.AsyncOpenAI(**_sdk_options(base_url)) as client:
            yield client
