"""Open Responses compliance gate (docs/design/m20-open-responses-ci.md, D1).

- ``launcher``: Kairyu on ScenarioBackend, scripted for the suite's scenarios
  (``tests/contracts/openresponses/kairyu-mock.yaml``).
- ``verdict``: the pinned suite and its expected failures
  (``tests/contracts/openresponses/expected-failures.toml``), and the
  PASS/XFAIL/FAIL/XPASS/MISSING verdict of one ``--json`` report.
- ``run``: fetches and installs the pinned suite, runs it with bun against the
  launcher, and exits 1 unless every scenario is PASS or XFAIL
  (``.github/workflows/open-responses.yml``).
- ``drift``: the weekly run of upstream ``main`` against the same list; it
  reports drift as an issue and gates nothing.
"""
