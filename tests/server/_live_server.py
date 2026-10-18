"""Serve an ASGI app on a real localhost socket, for tests of the HTTP stack itself."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn


def serve(app) -> tuple[str, uvicorn.Server]:
    """Run ``app`` under uvicorn in a daemon thread; return its URL and server."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}", server
