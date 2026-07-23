# Qwen3-32B benchmark shell progress

## Goal

Make `examples/qwen3-32b-multi-gpu/run-benchmark.sh` visibly active while it
waits for Kairyu to become ready and while the serving benchmark runs. A user
must be able to distinguish startup, readiness waiting, benchmark execution,
and report generation without changing Kairyu or the shared benchmark harness.

## Scope

Changes are limited to:

- shell scripts in `examples/qwen3-32b-multi-gpu/`;
- that example's README;
- dedicated tests for those scripts.

The implementation must not modify `kairyu/`, `bench/serving_bench.py`, the
Compose topology, or the model configuration.

## User-visible behavior

`run-benchmark.sh` prints named stages for starting the Compose service,
waiting for readiness, and handing off to the benchmark. During readiness
waiting it prints an update every poll interval with elapsed seconds. The
existing 15-minute timeout and diagnostic Compose logs remain unchanged.

`benchmark.sh` prints the benchmark configuration before launching the
benchmark container. While that container is running, it prints an update every
poll interval in one of these forms:

```text
[benchmark] completed 48/128 (elapsed 35s)
[benchmark] running (elapsed 35s)
```

The first form is used when the local Kairyu `/metrics` endpoint exposes a
request counter for `qwen3-32b`. Progress is the increase from the counter value
captured immediately before the benchmark starts. It is bounded by the
configured request total so unrelated requests cannot produce a percentage
above 100%. If metrics are absent, temporarily unavailable, or unparsable, the
script falls back to the elapsed-time form without failing the benchmark.

The report-generation stage is announced after the benchmark succeeds. Existing
benchmark output and the final report path remain visible.

## Process and failure behavior

The benchmark container remains the source of truth for success or failure.
The progress monitor is auxiliary: it must be stopped when the benchmark exits,
and its failures must not replace the benchmark's exit status. Interrupting the
script must not leave the progress monitor running.

The implementation remains POSIX `sh` compatible and uses tools already listed
or implied by the example requirements: Docker Compose, Docker, `curl`, and
standard shell text utilities.

## Tests

Dedicated shell-script tests run with fake `docker`, `curl`, and timing commands
so they do not require a GPU, model download, or live service. They verify:

- readiness waiting emits elapsed progress before success;
- readiness timeout keeps the existing failure diagnostics;
- benchmark configuration is printed;
- metric deltas produce bounded `completed N/total` progress;
- missing metrics fall back to elapsed-only progress;
- benchmark failures remain failures and do not proceed to report generation;
- script syntax remains valid under `/bin/sh -n`.
