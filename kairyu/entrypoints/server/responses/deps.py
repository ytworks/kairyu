"""What the ``/v1/responses`` route needs, bundled once in the route factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import Response

from kairyu.engine.backend import EngineBackend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.responses.compaction import CompactionCodec
from kairyu.entrypoints.server.responses.store import ResponseStore

# The late-bound Chat Completions handler that serves AUTO (orchestrated) models.
ChatDispatch = Callable[[ChatCompletionRequest, Request], Awaitable[Response]]


@dataclass(frozen=True)
class ResponsesDeps:
    """Route dependencies: engines, rendering policy, AUTO dispatch, and state."""

    engines: Mapping[str, EngineBackend]
    store: ResponseStore
    compaction_codec: CompactionCodec
    chat_templates: Mapping[str, ChatTemplate] | None = None
    legacy_chat_models: AbstractSet[str] | None = None
    orchestrated_models: AbstractSet[str] | None = None
    chat_dispatch: ChatDispatch | None = None
