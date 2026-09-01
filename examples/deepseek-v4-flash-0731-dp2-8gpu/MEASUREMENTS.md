# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The locked run is the second matrix
against an already-running stack (warm replicas: both had served the warm-up
row and one full matrix before it), with row-unique prompt prefixes so neither
vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256:
  `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM revision: `aa0d51302747ea80f282e26949708b3253409fe2`, image ID
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Kairyu verification commit: `1d80d0d1d8d11ea8d5d2fa3a160f400350057528`
  (PR #584 branch; the served config is byte-identical to the committed files)
- Served-config SHA-256: `b6ccf1d629b42383f329011398bccc89949d89029e9a42c320182a4aa84564c0`
- L1: 2 x TP4+EP4 replicas (GPU 0-3, GPU 4-7), each with DSpark-5,
  `max_num_batched_tokens=16384`, `max_num_seqs=32`, FP8 KV, 256-token blocks,
  prefix caching, full/piecewise CUDA Graphs; SM100-only MegaMoE and FP4
  indexer cache disabled on SM120. Each replica reported a 2,947,608-token
  KV cache (2.81x the 1M context).
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`.
- Cold start: both replicas initialized in parallel from empty compilation
  caches in about 11 minutes (`run.sh` start to Kairyu ready); the caches are
  persisted below the example's NVMe directory for later starts.

## Warm serving result

Each operating point completed 64/64 requests with approximately 8K input
tokens and exactly 256 output tokens per request (success rate 1.0 in every
row). Percentiles use the harness's nearest-rank method. "Placements" is the
per-replica request count from the pool's placement log for that row; the
gate requires both replicas to receive traffic and neither to exceed 2x the
even share (32 of 64) at concurrency >= 8.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements (replica 0, 1) | Placement gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 773.21 | 781.35 | 2,015.99 | 4.853 | 0.50 | 127.30 | 64, 0 | reported only |
| 8 | 802.44 | 3,210.70 | 5,187.67 | 15.851 | 1.48 | 378.04 | 32, 32 | pass |
| 16 | 812.71 | 6,074.07 | 9,147.94 | 29.932 | 1.57 | 401.99 | 32, 32 | pass |
| 32 | 2,415.77 | 11,913.45 | 16,291.66 | 47.319 | 1.84 | 471.33 | 32, 32 | pass |
| 64 | 12,428.06 | 23,549.00 | 28,951.44 | 64.912 | 1.84 | 471.37 | 32, 32 | pass |

Run ID: `20260901T140112Z` (the first, cold-shape matrix `20260901T135600Z` in the
same results directory produced the same numbers within a few percent and a
33/31 split at c16 — also inside the gate).

## Reading the result

- **Placement is even.** At every concurrency >= 8 the two replicas received
  32 and 32 of the 64 requests; c1 sends all 64 requests to replica 0 because
  least-outstanding ties resolve to the lowest replica id (reported only, by
  design).
- **c1 matches the single-replica TP4+EP4 row measured by the tiered example**
  through Kairyu L3 (TTFT p50 773 vs 779 ms, 127.3 vs 130.8 output tok/s,
  `../qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md`, "Tier2 … selection"): the
  pool adds no measurable overhead on the single-request path.
- **Two replicas roughly double saturated throughput**: 471 output tok/s at
  c32/c64 versus 241.62 tok/s for one TP4+EP4 replica at c32 (1.95x). TTFT
  at c32/c64 (16/32 concurrent 8K prompts per replica) is prefill-bound
  within each replica, as in the single-replica rows.
- The TP8+EP8 single-replica example reached 972.93 tok/s at c32 on a
  shared-prefix dataset with `max_num_seqs=64`; that figure is not comparable
  to this unique-prefix matrix.

## Raw evidence identity

Artifacts are below
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-dp2-8gpu/verification-results/20260901T140112Z/`
on the measurement host (`run.json`, per-row `*-serving.json`,
`placement.json`, and logs).

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `312ccb728f7b713003769d2446bad950557f621e265fe69c5a5c290bcaa65e93` |
| Serving concurrency 8 | `e58c7a7c63630d915f74732524aef5f029bb60878d4cadb65d6437fbed8ca64a` |
| Serving concurrency 16 | `954cd5efbe018b974a578f180317949e69f975078bda63f47c70baf176b2768d` |
| Serving concurrency 32 | `93737ea5a7f8b6545fb3d7148125f4e2c3485dfce24eda45e132acf9675fc1c6` |
| Serving concurrency 64 | `00d502d57a4599aec172366b9059609966cd4a614223554c8769501d046ed341` |
