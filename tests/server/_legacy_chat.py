"""Explicit legacy-chat constructors for server tests unrelated to templates."""

from __future__ import annotations

from kairyu.batch.worker import BatchWorker
from kairyu.entrypoints.server.app import AUTO_MODEL, create_app


def create_legacy_app(engines, orchestrator=None, **kwargs):
    """Build a strict app while opting every untemplated test model into legacy chat."""

    chat_templates = kwargs.get("chat_templates") or {}
    legacy_models = set(engines) - set(chat_templates)
    if orchestrator is not None and AUTO_MODEL not in chat_templates:
        legacy_models.add(AUTO_MODEL)
    legacy_models.update(
        name
        for name in (kwargs.get("orchestrators") or {})
        if name not in chat_templates
    )
    if "legacy_chat_models" in kwargs:
        raise TypeError("create_legacy_app owns legacy_chat_models")
    return create_app(
        engines,
        orchestrator,
        legacy_chat_models=frozenset(legacy_models),
        **kwargs,
    )


class LegacyBatchWorker(BatchWorker):
    """BatchWorker test seam with an explicit per-engine legacy chat policy."""

    def __init__(self, store, engines, *args, **kwargs) -> None:
        chat_templates = kwargs.get("chat_templates") or {}
        if "legacy_chat_models" in kwargs:
            raise TypeError("LegacyBatchWorker owns legacy_chat_models")
        super().__init__(
            store,
            engines,
            *args,
            legacy_chat_models=frozenset(set(engines) - set(chat_templates)),
            **kwargs,
        )
