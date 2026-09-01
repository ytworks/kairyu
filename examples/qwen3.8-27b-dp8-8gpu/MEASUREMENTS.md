# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The locked run is the second matrix
against an already-running stack (warm replicas: every replica had served the
warm-up row and one full matrix before it), with row-unique prompt prefixes so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the
matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Model tree SHA-256:
  `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- vLLM: `v0.23.0`, image ID
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Kairyu verification commit: `1d80d0d1d8d11ea8d5d2fa3a160f400350057528`
  (PR #584 branch; the served config is byte-identical to the committed files)
- Served-config SHA-256: `3b2fd507c45c2de8bf9fe596024bfe5faa9629316686030ccda09b0014217f0b`
- L1: 8 x TP1 replicas, each with `max_num_batched_tokens=32768`,
  `max_num_seqs=32`, FP8 KV, FP16 Gated-DeltaNet state, piecewise CUDA Graphs,
  MTP off — the single-GPU example's measured envelope.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`.

## Warm serving result

Each operating point completed 64/64 requests with approximately 8K input
tokens and exactly 256 output tokens per request (success rate 1.0 in every
row). Percentiles use the harness's nearest-rank method. "Placements" is the
per-replica request count from the pool's placement log for that row; the
gate requires every replica to receive traffic and no replica to exceed 2x
the even share (8 of 64) at concurrency >= 8.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements (replica 0..7) | Placement gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 941.13 | 959.75 | 6,514.52 | 21.856 | 0.15 | 39.30 | 64, 0, 0, 0, 0, 0, 0, 0 | reported only |
| 8 | 938.84 | 985.59 | 6,515.96 | 21.881 | 1.23 | 313.72 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |
| 16 | 1,805.64 | 2,328.04 | 7,766.62 | 23.369 | 2.02 | 518.34 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |
| 32 | 3,855.16 | 9,322.62 | 9,999.02 | 24.073 | 2.52 | 644.02 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |
| 64 | 14,044.98 | 20,635.47 | 20,233.00 | 26.286 | 2.39 | 611.21 | 8, 8, 8, 8, 8, 8, 8, 8 | pass |

Run ID: `20260901T133331Z` (the first, cold-shape matrix `20260901T123716Z` in the
same results directory produced the same placement split and output
throughput within 1%; only its c64 TTFT p50 differed, 20,592.92 ms).

## Reading the result

- **Placement is exactly even.** At every concurrency >= 8 each of the eight
  replicas received exactly 8 of the 64 requests; c1 sends all 64 requests to
  replica 0 because least-outstanding ties resolve to the lowest replica id
  (reported only, by design).
- **c1 and c8 are indistinguishable** (TTFT ~940 ms, TPOT ~21.9 ms): eight
  concurrent requests land one-per-replica, so each replica sees the c1 load.
  Aggregate output throughput scales 8.0x (39.30 -> 313.72 tok/s).
- **From c16 the matrix is prefill-bound within each replica.** Every replica
  holds 2/4/8 concurrent 8K prompts at c16/c32/c64; chunked prefill of
  16K/32K/64K prompt tokens per replica dominates TTFT (p50 1.8 s / 3.9 s /
  14.0 s) while TPOT stays 23-26 ms. Output throughput saturates around
  611-644 tok/s at c32-c64 because each replica runs only 4-8 sequences; the
  single-GPU example reached 867.58 tok/s at c32 with 32 sequences on one
  card. This matrix therefore characterizes the latency regime (<= 8
  sequences per replica). Reaching the replicas' full batching regime needs
  concurrency near 8 x 32 = 256, which is not part of the committed matrix.

## Raw evidence identity

Artifacts are below
`/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-dp8-8gpu/verification-results/20260901T133331Z/`
on the measurement host (`run.json`, per-row `*-serving.json`,
`placement.json`, and logs).

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `5040354780943149054cae7000815ef310833b956f6b954d2fc90f5c2295f84b` |
| Serving concurrency 8 | `8e9e79a9c1f2663970a22b90c515501b5fd8750bcd8e8cdbba26cdba42cc69be` |
| Serving concurrency 16 | `2486b341b468bdeb39300fe94638a22b07c1340221f4dff334c6d91b9d8e51aa` |
| Serving concurrency 32 | `2ae0d0f7749841788ca2c57fe2085c26f242a4c2ec5a15f125aecccaa69ef651` |
| Serving concurrency 64 | `230dab727e4d9b4325f3c6c5e2097cb523b36a5bc0a034d518e7acd5d2025bff` |
