# Issue #318 pipeline-depth diagnostic result

Issue #318 closes with a measured negative result. Kairyu retains the existing
two-step unresolved horizon whenever admission or prefill work remains. The
experimental scheduler and metric changes were reverted; this bundle is the
only repository change associated with the result.

The issue correctly observed that sustained waiting work keeps the effective
pipeline depth at two. It did not establish that this was the production
ShareGPT bottleneck. The historical 35.98% result compared depth one with depth
five, so it measured the benefit of host/device overlap without isolating any
benefit beyond depth two.

## GPU result

All retained runs report Qwen3-32B TP4 on the RTX PRO 6000 Blackwell host. The
four schema-v2 runs bind UUIDs for GPUs 0–3; the first eight runs do not retain
GPU telemetry. Every measurement completed 128/128 requests and 16,384 output
tokens. Positive latency deltas below mean the candidate was slower.

| Comparison | TPS delta | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial depth-5 candidate vs primary main, means | -4.010% | +5.380% | +4.478% | +2.128% | +3.412% |
| Initial depth-5 candidate vs primary main, medians | -3.351% | +4.021% | +3.716% | +0.995% | +2.404% |
| Refined candidate vs refined main, aggregate | -0.096% | +0.266% | +0.164% | +0.944% | +1.115% |
| Intended depth-2 candidate vs adjacent drift main | -0.257% | -1.220% | +0.609% | -1.993% | +0.971% |

The initial candidate was uniformly slower. Its three-run median was 514.20
output tokens/s versus 532.03 for the primary three-run main group. The refined
two-pair experiment was an aggregate tie, not a demonstrated improvement: its
first pair favored main by 2.00% TPS while its second favored the candidate by
1.87%. Main itself ranged from 524.20 to 543.04 tokens/s in those two runs.

The intended depth-2 diagnostic measured 525.00 tokens/s beside a 526.35 main
control. That schema-v1 artifact binds the setting only by its label and
campaign config, and it is a single non-randomized adjacent comparison. It
shows that the initial regression was not reproduced in that run, but cannot
attribute the cause.

## Mechanism diagnostic

The initial CPU fake-runner replay was reported to process the same 16,430 rows
and 77,150 device tokens while deep scheduling increased submissions from
1,042 to 1,078 and underfilled plans from 81 to 213. Its raw output and exact
candidate diff were not retained, so those counts are not independently
reproducible. TP execution uses one serial control and executor path; extra
small plans do not add device concurrency and can add launch/collective
overhead.

A refined selector restored exact replay geometry (1,042 submissions and 81
underfilled plans) while structurally reaching deeper pending depth. Its GPU
result returned only to parity. Subsequent correctness review also found that
maintaining a deeper immutable tail required increasingly complex fairness,
P-D, late-arrival, and metric-lifecycle policy. That complexity had no measured
production benefit.

## Disposition

- Keep the existing depth-two admission/prefill horizon.
- Do not ship the experimental strict-decode tail or its event counter.
- Treat prefill/launch cost, serving-layer overhead, and active-sequence sizing
  as separate A6 hypotheses; this experiment does not identify one of them as
  the causal fix.
- If effective-depth observability is pursued later, scope it independently to
  a direct achieved-depth signal with explicit reset semantics.

## Evidence strength and limits

The twelve JSON files were independently replayed from their 128 measurement
rows. Stored TPS, nearest-rank TTFT/TPOT, usage totals, and per-sequence output
hashes all recompute exactly. The request-identity digest is common across all
runs, and backend evidence consistently reports Qwen3-32B TP4, FlashInfer, and
BF16 KV.

This is diagnostic evidence, not a formal A6 gate:

- The first eight schema-v1 files do not bind config, trace, script, image,
  container, or GPU telemetry. Dirty candidate source diffs are identified by
  hash but their bodies/resulting trees were not retained.
- The initial CPU replay counts are reported diagnostics; its raw output and
  exact candidate diff were not retained for independent reproduction.
- The four refined schema-v2 files bind a common depth-five config, trace,
  script, image, and GPU UUIDs, but contain only two pairs. The measured refined
  candidate predates the final abandoned experimental diff.
- The trace requests are identical, but the trace bundle's formal A6 metadata
  declares depth one. These runs deliberately override it for issue #318 and
  must not be presented as a formal A6 result.
- Initial run order was not randomized; no artifact contains pending-depth or
  graph-transaction histograms; per-run p99 is noisy at 128 requests; and this
  closed burst is not a long-duration open-loop or SLO-goodput gate.

Detailed group statistics, source identities, hashes, and limitations are in
`summary.json`. The files under `scripts/` and `configs/` retain the diagnostic
methodology; they are evidence provenance, not supported current-checkout
entrypoints.
