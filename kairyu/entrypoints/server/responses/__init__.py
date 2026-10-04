"""OpenAI Responses API adapter (``/v1/responses``), registered by ``routes``.

Module map (m20 D2): ``request`` (accepted surface), ``canonical`` (input
items), ``tools`` and ``to_chat`` (Chat Completions translation), ``compaction``
(sealed remote compaction), ``output``/``envelope``/``events``/``errors``
(wire shapes), ``store`` (``previous_response_id`` state), ``deps`` (route
dependencies), and the temporary ``paths_legacy_{live,buffered,relay}`` stream
paths that WP-17a and WP-18 replace with the single emitter pipeline.
"""
