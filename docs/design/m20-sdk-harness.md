# M20 companion: SDK v3.24 harness (WP-13)

*2026-10-05, WP-13* (m20 D1, G-conformance-4, M-ST-8). The dev group pins
`openai[realtime]>=3.24,<4` and `openai-agents==0.23.*`. openai-python 3.x
defaults to HTTPX2, so the 11 SDK tests drive a live uvicorn server instead of
an in-process transport; they were ported, not added (collected count
unchanged). The server lives in `tests/server/live_server.py`. It starts with
`cli.uvicorn_options()` on port 0 with the lifespan running, and stops before
teardown so the schema gate records it. The tests use the SDK's own
transport, `_strict_response_validation=True` and `max_retries=0`.

v3.24 requires `usage.input_tokens_details.cache_write_tokens`. Kairyu emits
`0` (D11). This is the G-usage-2 field, pulled forward from WP-08b, and both of
its divergences are deleted.

The base64 embedding round trip stays lenient: the SDK validates base64 as
`list[float]` before decoding it. `responses.stream(response_id=…)` resume
waits for WP-33/38b.
