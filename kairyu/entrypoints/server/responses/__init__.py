"""OpenAI Responses API adapter (``/v1/responses``), registered by ``routes``.

Module map (m20 D2): ``request`` (accepted surface), ``canonical`` (input
items), ``tools`` and ``to_chat`` (Chat Completions translation), ``compaction``
(remote compaction) over ``sealing`` (``kst2`` sealed items and the key ring),
``output``/``envelope``/``events``/``errors``
(wire shapes), ``framework_errors`` (scoped 422/404/405 envelopes), ``store``
(``previous_response_id`` state), ``deps`` (route dependencies), and the
temporary ``paths_legacy_{live,buffered,relay}`` stream paths that WP-17a and
WP-18 replace with the single emitter pipeline.
"""
