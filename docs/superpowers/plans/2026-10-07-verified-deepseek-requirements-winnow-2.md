# Verified route: DeepSeek max requirements, two Winnow replicas (2026-10-07)

Approved by the owner on 2026-10-07. Design record: VCO-D19 amendment of
2026-10-07 in `docs/design/example-verified-checklist-orchestration.md`.

## Context

DeepSWE r1 of the three-wave verified route was stopped at 24 of 113 tasks
(8 passed). Qwen `requirements` was the wave-1 bottleneck (median 107 s
against 36 s for the five DeepSeek drafts). Owner decisions:

1. `requirements` is written by DeepSeek at max effort instead of Qwen.
2. Qwen leaves the GPUs; GPU 6 hosts a second Winnow-12B replica.
3. The two Winnow replicas split by use: GPU 6 the route judge
   (`profile_judge`), GPU 7 the judgments.
4. The example is renamed without `qwen3.8`.
5. The running DeepSWE is stopped.

## Changes

- Framework: withdraw `{conversation_without_reasoning}` (added in this PR
  for Qwen only; DeepSeek needs the replayed reasoning). Code, test and the
  m1 D8 text get a withdrawal note. No other `kairyu/` change: the split uses
  two `systemone:` entries and one `systemone_ref` worker per use.
- Example `examples/deepseek-v4.1-winnow-8gpu/` (renamed):
  - `verified.yaml` / `verified-always.yaml`: workers `winnow_route`,
    `winnow_judge`; `requirements` on DeepSeek, `reasoning_effort: max`,
    T=1.0/top_p=0.95, `max_tokens` 131072, `{conversation}`.
  - `kairyu.yaml`: no Qwen pool; `winnow-12b` pool with two replicas;
    `winnow-route-systemone` and `winnow-judge-systemone`.
  - `compose.yaml`: no `qwen`; `winnow-route` (GPU 6), `winnow-judge` (GPU 7).
  - `example.json`, `control.py`, `verify.sh`, `verification.py`, README.
  - `qwen3.8-chat.jinja` removed.
- Docs: VCO-D19 amendment, PROGRESS entry, examples index.

## Verification

- CPU: ruff; the example test, `test_conductor.py`, `test_frontier_examplectl.py`.
- GPU (after redeploying with `run.sh`): every gate in `verification.GATES`
  (l1, routing, think-route, effort, verified-route, fallback, serving,
  serving-routed, browser), recorded in the example's `MEASUREMENTS.md`.
- No DeepSWE re-run until the owner asks.
