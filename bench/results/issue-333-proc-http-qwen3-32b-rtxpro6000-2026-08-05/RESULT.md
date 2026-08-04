# Issue #333 v2 result

This is the sole fresh v2 ABBA rerun precommitted by the issue #333 evidence
amendment. It is retained regardless of its performance direction.

- Evidence schema: `kairyu.issue-333.proc-http-diagnostic.v2`
- Source commit: `65ba4779c118d534f1e34a1a4dcb1b579cbcfe73`
- Trace SHA-256: `e96abab24db5090c11b1f59e3e870b8fe86a31e031ad37189e287bac97f28525`
- Raw SHA-256: `21ba4789cf34fcf4beab5ab5862952574693f80c8b608c2cede002da05088925`
- Manifest SHA-256: `bd321c32654c0602d6e799167d16a7b375b8edf5dea5bd95faa899acde24286c`
- Completed-record SHA-256: `e05a5dc17293a49bc267d0bae326f0f816c6c0829e0e74258dd3c787839a109e`
- Integrity: all 15 binding checks pass independent raw replay
- Traffic: four fresh TP4 servers in `kairyu`, `kairyu-proc`,
  `kairyu-proc`, `kairyu` order; 512/512 synchronized measurement requests
  succeeded once with strict SSE, exact usage, and 128 completion tokens
- Cleanup: all four containers exited gracefully with exit code zero, were
  removed without force, and restored the selected GPUs to their run-start
  idle baseline

The paired process/in-process ratios were:

| Repeat | Goodput | TTFT p50 | TTFT p99 |
| --- | ---: | ---: | ---: |
| 0 | 1.0872651589542448 | 0.9187326369845356 | 0.9189755344057482 |
| 1 | 1.0851378253734132 | 0.9378290330934350 | 0.9219867334510442 |
| Median | 1.0862014921638290 | 0.9282808350389853 | 0.9204811339283963 |

The predeclared report-only material line was paired-median
`kairyu-proc/kairyu` TTFT p99 <= 0.90. The observed 0.9204811339283963 does not
meet it, so the diagnostic classification is `no_material_reduction` and the
dominant process/GIL-contention hypothesis is `not_supported`. This is not a
formal G2 A6 PASS/FAIL, and the treatment is the complete backend swap including
process isolation, ZMQ, msgpack/delta transport, and lifecycle overhead rather
than pure GIL isolation.

The six non-binding measurement output-hash agreement counts in order-pair
order were 00–01: 27/128, 00–02: 21/128, 00–03: 29/128, 01–02: 54/128,
01–03: 34/128, and 02–03: 32/128. Binding integrity instead comes from the
strict request contract, unique response IDs, and all-four parity for every
serialized ShareGPT warm-up output.

Replay with `bench/issue_333_proc_http_bench.py verify --assert-integrity`
from the source commit above against the `artifact/` directory.
