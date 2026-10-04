# M20 companion: SDK v3.24 harness (WP-13)

*2026-10-05, WP-13* (m20 D1, G-conformance-4, M-ST-8). The dev group pins `openai[realtime]>=3.24,<4`
and `openai-agents==0.23.*`. openai-agents needs urllib3>=2.8, so the shared lock moves urllib3 from
2.7.0 to 2.8.0, and the images that install requests ship it too (the embeddings variants and Dockerfile.cuda).
The 11 SDK tests were ported, with the count unchanged, to a live uvicorn server in
`tests/server/live_server.py` that uses `cli.uvicorn_options()`. They run with strict validation, no
retries and no proxy variables. v3.24 requires `usage.input_tokens_details.cache_write_tokens`; Kairyu
emits `0` (D11). This G-usage-2 field was pulled forward from WP-08b, and both of its divergences are
deleted. The base64 embedding test stays lenient because the SDK validates base64 as `list[float]`
before decoding it. The `responses.stream(response_id=…)` resume test waits for WP-33/38b.
