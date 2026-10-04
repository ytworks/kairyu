"""Shared test engines for gate and conformance suites (M20 WP-02b).

- ``scenario_script``: the immutable turn/scenario model both engines replay.
- ``scenario_backend``: ``ScenarioBackend``, an in-process ``EngineBackend``.
- ``fake_vllm_upstream``: ``FakeVLLMUpstream``, a vLLM OpenAI-compatible server
  for ``OpenAICompatBackend`` (httpx ``MockTransport`` or ASGI/uvicorn).

Importing these modules registers nothing; launchers opt in explicitly.
"""
