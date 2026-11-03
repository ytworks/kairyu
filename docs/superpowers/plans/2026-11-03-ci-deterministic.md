# CI verdicts depend only on the code

Status: **Plan approved 2026-11-03 (owner), including the verification plan.**

## Principle (owner)

A CI test passes or fails only because of the code under test. Runner speed or
load, wall-clock timing, network or registry behaviour, the runner's Docker
configuration and execution order must not change a verdict. Trigger: PR #667's
CI failed in `cpu-microbench` (`scheduler.priority_speedup` 0.83 on a shared
runner, 1.59 locally on main and the branch) and in F1c `gateway` (`kind load
docker-image` of a multi-platform postgres image failed under the runner's
containerd image store); both were wrongly set aside as "not caused by this PR".

## Rules

1. **No time in verdicts.** Elapsed times, speed ratios, p99 budgets and "within
   N seconds" checks leave pass/fail and are recorded as diagnostics (as F2b
   already does). The property they protected is checked with deterministic
   quantities: operation counts, event order, or an injected virtual clock.
2. **Synchronize with events, not sleeps.** A test waits for the other side's
   signal, never for a number of seconds before asserting.
3. **Timeouts only detect hangs.** They are far above the expected time and a
   breach is reported as a hang, never as slowness.
4. **External inputs do not depend on the runner.** Images are pinned to a
   single-platform manifest digest, loaded into kind from an archive, and
   compared by the config digest inside that archive.

## Changes

| Area | Today | Change |
|---|---|---|
| `cpu-microbench` (`scripts/cpu_microbench_gate.py`) | six wall-clock speedup floors, router p99 < 10 ms, 45 s per benchmark, retries on timing noise | ratios and p99 become diagnostics; verdicts use operation counts (heap operations, nodes visited, allocations) and their growth across sizes; 45 s becomes a generous hang bound; the retry machinery goes |
| nightly CPU perf (`scripts/cpu_perf_series.py`, `nightly-cpu-perf.yml`) | fails on a 15 % drop against the last seven medians; fails when history cannot be fetched | the trend is reported, never failing; verdicts are the operation counts above; missing history is a warm-up |
| pytest wall-clock assertions | `test_moa.py`, `test_conductor.py`, `test_audit_io.py`, `test_pipeline.py`, `test_unified_engine_loop.py`, `test_router_learning.py`, `tests/dist/test_distributed.py` | overlap by event order (every call started before the first finished); "admission does not wait" by completing admission while the write is held; time-derived values from a fake clock; latency budgets become diagnostics |
| pytest sleep-then-assert | to be listed | every case found is rewritten to rule 2; the list goes in the PR |
| compose smoke (`scripts/compose_smoke.sh`) | trace in logs within 10 s; replica ejected within 20 s, restored within 60 s | flush, then check once; ejection and restoration by state transitions; seconds only bound hangs |
| kind F1a / F1b / F1c (`scripts/kind_*_gate.sh`, `verification/fleet/resilience/*`, `deploy/kind/*`) | 1 s membership withdrawal, 60 s recovery, rollout time, 99 % pacing, 10 s / 5 s request timeouts, 1 s probes with one failure | causal checks (no placement on an old UID after the deletion is applied; replacement completed); pacing and durations become diagnostics; timed-out and errored requests are told apart; test-manifest probe thresholds widened |
| kind images | loaded from the runner's Docker and compared with Docker's `.Id` | rule 4 |
| F1c gateway and async smoke | kill inside a 3 s lease; `/v1/models` within 5 s; a deadline 1.5 s ahead must expire | fencing tokens and order only; `/v1/models` answers before the lock is released; the deadline is already past |
| F2b (`verification/fleet/correctness/kv_event_f2b_bench.py`) | freshness judged by wall clock (a 200 ms loop stall fails) | a virtual clock drives `KvEventIndex` and the tick scheduler |
| F2a | untracked files fail the clean-source check | only the gate's inputs are checked |
| networked setup in CI | dependency sync, tokenizer, image pulls inside test steps | setup separated from tests, cached and retried; it is not part of a verdict |

## Structural changes (approved)

1. A separate PR from `main`, merged before PR #667, which is then rebased.
2. `kairyu/` gains a clock injection point for F2b (`KvEventIndex` and related); product behaviour is unchanged.
3. Performance regressions no longer fail CI; speed is recorded and reported nightly. CI fails only on regressions expressible as operation counts.
4. CLAUDE.md's test policy gains the principle.

## Verification

- Every changed test and gate gives the same verdict normally, with the process
  pinned to one CPU while other processes keep every core busy, and over ten
  repetitions.
- Every replaced speed check still fails on a deliberate regression (for
  example indexed removal turned back into a linear scan).
- kind F1a / F1b / F1c run on this host, whose Docker uses the containerd image
  store that broke CI.
- Every CI job green, and green again on a re-run.
