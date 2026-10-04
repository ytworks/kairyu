"""Codex conformance gate tooling (docs/design/m20-responses-compat.md, D1).

- ``record_proxy``: a recording reverse proxy between a real Codex binary and a
  Kairyu server; writes normalized request captures and promotes them into
  ``tests/fixtures/codex/rust-v<ver>/`` fixtures, re-applying a derived
  fixture's edits (``derivation``).
- ``extension_inventory``: walks the codex-rs serde request types at a pinned
  tag and writes ``extensions.json`` (spec field vs Codex extension).
- ``run_matrix`` (WP-05): real Codex binaries (``codex_cli``) against a Kairyu
  ``launcher`` serving ``scenarios`` on ScenarioBackend, through the proxy's
  ``exchange_log``; ``--live`` targets a deployment. ``drift`` compares the
  OpenAPI and Codex pins with upstream (``.github/workflows/codex-gate.yml``).

The fixtures are replayed by ``tests/server/test_codex_fixture_replay.py``.
"""
