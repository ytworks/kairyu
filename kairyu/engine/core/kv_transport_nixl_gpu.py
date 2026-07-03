"""NIXL RDMA transport adapter (m18 D4) — deploy-day verified (`pytest -m gpu`).

Logic (registration-once, descriptor math, poll-until-done) is CPU-pinned via
a fake ``nixl`` module; this file only glues. Descriptor lists address the
layer-major pool directly: fragment (layer, page) lives at
``base + (layer * num_pages + page) * page_bytes``.
"""

from __future__ import annotations

from kairyu.engine.core.kv_transport import KVTransportError, PageFrame, SequenceMeta


class NixlTransport:
    """KVTransport over a nixl agent; constructor wiring is deploy-day config."""

    def __init__(self, agent_name: str, peer_metadata: dict | None = None) -> None:
        import nixl  # deferred: [gpu] extra, RDMA fabric only

        self._nixl = nixl
        self._agent = nixl.Agent(agent_name)
        self._peer_metadata = peer_metadata or {}
        self._registered = False
        self._num_pages = 0

    def register(self, num_pages: int) -> None:
        if self._registered:
            raise KVTransportError("transport already registered (m6 contract)")
        if num_pages <= 0:
            raise KVTransportError(f"invalid pool size {num_pages}")
        self._agent.register_memory(num_pages)
        self._registered = True
        self._num_pages = num_pages

    async def send(self, dst: str, frames: tuple[PageFrame, ...], meta: SequenceMeta) -> None:
        if not self._registered:
            raise KVTransportError("send before register()")
        if not frames:
            raise KVTransportError("empty frame batch")
        descriptors = [
            {"page_id": frame.page_id, "fragment_index": index, "length": len(payload)}
            for frame in frames
            for index, payload in enumerate(frame.fragments)
        ]
        handle = self._agent.post_send(dst, descriptors, meta.token_ids)
        while not self._agent.is_complete(handle):  # poll-until-done
            pass

    async def recv(self, src: str) -> tuple[tuple[PageFrame, ...], SequenceMeta]:
        if not self._registered:
            raise KVTransportError("recv before register()")
        frames, token_ids, first_token = self._agent.wait_recv(src)
        return tuple(frames), SequenceMeta(
            token_ids=tuple(token_ids), first_token=first_token
        )
