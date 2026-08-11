# Measurements

Runtime validation is in progress. All performance values in this document are
measured at the public Kairyu L3 OpenAI-compatible endpoint, never at a vLLM L1
endpoint. Terminal-Bench 2.1 evidence and the final served-configuration hash
will be added only after the L1/L2 selection gates close.

## Tier1 topology selection

The comparison uses the same Qwen3.6-27B FP8 checkpoint and vLLM settings on
GPUs 0-3. Every row has an explicit topology-sized warm-up followed by 32
requests with approximately 8,192 prompt tokens and exactly 256 generated
tokens. Each concurrency row has a different namespace in its prompt prefix,
so a later row cannot become a full-prefix-cache benchmark. Kairyu trace-v2 was
requested and validated for every request; every trace selected the Tier1
direct-generation route.

| Tier1 topology | Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | TPOT mean (ms/token) | per-request generation TPS p50 | aggregate output TPS | success / valid trace |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP1 x 4 replicas | 1 | 2,620.97 / 2,630.50 | 8,631.38 / 8,642.58 | 23.571 | 42.59 | 29.66 | 32/32 / 32/32 |
| TP1 x 4 replicas | 8 | 5,177.94 / 5,304.99 | 11,622.34 / 11,688.60 | 25.277 | 39.66 | 175.90 | 32/32 / 32/32 |
| TP1 x 4 replicas | 16 | 10,169.72 / 10,489.80 | 16,971.69 / 17,130.61 | 30.098 | 37.47 | 240.41 | 32/32 / 32/32 |
| TP1 x 4 replicas | 32 | 16,373.42 / 20,904.68 | 27,545.04 / 27,728.81 | 44.838 | 22.84 | 295.19 | 32/32 / 32/32 |
| TP2 x 2 replicas | 1 | 2,078.98 / 2,092.62 | 5,868.14 / 5,879.39 | 14.858 | 67.57 | 43.62 | 32/32 / 32/32 |
| TP2 x 2 replicas | 8 | 6,338.02 / 8,524.49 | 12,423.15 / 12,828.95 | 24.848 | 39.88 | 163.48 | 32/32 / 32/32 |
| TP2 x 2 replicas | 16 | 11,802.63 / 16,197.90 | 20,753.74 / 20,852.76 | 36.580 | 28.49 | 196.87 | 32/32 / 32/32 |
| TP2 x 2 replicas | 32 | 19,487.37 / 32,171.86 | 37,069.33 / 37,186.92 | 67.419 | 14.56 | 220.17 | 32/32 / 32/32 |
| TP4 x 1 replica | 1 | 1,644.54 / 1,662.67 | 4,387.91 / 4,406.00 | 10.758 | 93.33 | 58.33 | 32/32 / 32/32 |
| TP4 x 1 replica | 8 | 9,773.81 / 13,226.58 | 15,897.35 / 16,627.64 | 27.789 | 37.66 | 127.40 | 32/32 / 32/32 |
| TP4 x 1 replica | 16 | 15,120.87 / 24,992.45 | 28,644.03 / 28,898.91 | 52.447 | 18.73 | 142.39 | 32/32 / 32/32 |
| TP4 x 1 replica | 32 | 27,005.18 / 49,913.22 | 54,199.46 / 54,421.42 | 102.705 | 9.41 | 150.48 | 32/32 / 32/32 |

Run IDs:

- `l3-auto-tp1x4-baseline-unique-20260811`
- `l3-auto-tp2x2-unique-20260811`
- `l3-auto-tp4x1-unique-20260811`

TP4 is the best c1 topology, improving median TTFT by 37.3% and aggregate
output TPS by 96.7% over TP1. It loses decisively once requests overlap: at c8
TP1 has 47.0% lower median TTFT and 38.1% higher output TPS; at c32 it has
39.4% lower median TTFT and 96.2% higher output TPS. TP2 is the same compromise
at a smaller scale: it improves c1 but loses TTFT and throughput to TP1 at every
tested concurrency from c8 onward.

The selected Tier1 topology is therefore **TP1 x 4 replicas**. This example is
quality-first and its principal `auto-max` path fans out 2-4 Qwen proposals.
Four independent replicas let those proposals execute concurrently, while TP2
and TP4 necessarily queue part of the fan-out. The selection is based on the L3
matrix and the intended orchestration workload, not on the fastest isolated L1
request.

The PCIe server required vLLM's supported `--disable-custom-all-reduce` option
for TP2/TP4. Without it, the TP2 candidate failed during CUDA-graph memory
profiling in the custom all-reduce kernel; with the option, both NCCL candidates
started and completed the matrix. TP4's first cache build reported 201.41 s for
engine initialization, including 109.54 s compilation. These candidate-only
transport settings are not present in the selected TP1 deployment.

## Tier2 speculation, batch-budget, and CUDA Graph selection

The Tier2 comparison keeps DeepSeek TP4+EP4, FP8 KV, max sequences 32, prefix
caching, and full/piecewise CUDA Graphs fixed. It changes only the named
candidate variable. The dataset and L3 measurement protocol are the same as
the Tier1 matrix. Direct-model requests enter through Kairyu L3; they do not
call the DeepSeek vLLM endpoint directly.

| Tier2 candidate | Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | TPOT mean (ms/token) | per-request generation TPS p50 | aggregate output TPS | success |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSpark-5, 16K batch | 1 | 779.22 / 791.75 | 1,926.21 / 2,240.48 | 4.617 | 223.00 | 130.79 | 32/32 |
| DSpark-5, 16K batch | 8 | 833.16 / 6,641.27 | 9,700.31 / 15,773.82 | 32.336 | 29.39 | 187.17 | 32/32 |
| DSpark-5, 16K batch | 16 | 2,710.53 / 11,899.12 | 17,749.63 / 29,026.48 | 49.887 | 19.82 | 221.88 | 32/32 |
| DSpark-5, 16K batch | 32 | 12,399.45 / 23,535.78 | 30,373.40 / 33,890.59 | 66.807 | 14.65 | 241.62 | 32/32 |
| DSpark-5, 32K batch | 1 | 771.59 / 786.90 | 2,007.40 / 2,251.42 | 4.789 | 207.32 | 128.39 | 32/32 |
| DSpark-5, 32K batch | 8 | 809.94 / 6,040.70 | 9,718.16 / 15,700.83 | 29.201 | 31.96 | 211.44 | 32/32 |
| DSpark-5, 32K batch | 16 | 3,306.91 / 11,602.01 | 16,664.29 / 28,367.76 | 46.221 | 23.27 | 234.90 | 32/32 |
| DSpark-5, 32K batch | 32 | 12,219.31 / 23,008.60 | 31,538.33 / 35,025.73 | 70.727 | 13.40 | 233.78 | 32/32 |
| No speculation, 16K batch | 1 | 774.25 / 1,186.28 | 3,406.64 / 3,817.17 | 10.326 | 97.21 | 74.88 | 32/32 |
| No speculation, 16K batch | 8 | 3,965.50 / 6,191.15 | 9,524.89 / 10,508.00 | 21.526 | 44.22 | 213.53 | 32/32 |
| No speculation, 16K batch | 16 | 6,549.86 / 11,780.05 | 16,120.20 / 21,357.75 | 38.401 | 26.73 | 252.94 | 32/32 |
| No speculation, 16K batch | 32 | 12,414.10 / 23,381.04 | 29,184.28 / 29,296.87 | 63.230 | 15.27 | 279.41 | 32/32 |
| DSpark-5, 16K, Graph NONE | 1 | 906.50 / 910.99 | 15,710.47 / 18,312.96 | 59.105 | 17.13 | 16.06 | 32/32 |
| DSpark-5, 16K, Graph NONE | 8 | 1,099.37 / 6,302.06 | 26,115.42 / 35,678.58 | 96.659 | 10.65 | 71.66 | 32/32 |
| DSpark-5, 16K, Graph NONE | 16 | 1,834.91 / 12,025.04 | 31,821.59 / 47,515.43 | 110.630 | 9.19 | 108.23 | 32/32 |
| DSpark-5, 16K, Graph NONE | 32 | 12,495.09 / 23,732.88 | 40,037.46 / 56,320.56 | 112.944 | 8.59 | 145.41 | 32/32 |

Run IDs:

- `l3-deepseek-tp4ep4-dspark5-b16k-unique-20260811`
- `l3-deepseek-tp4ep4-dspark5-b32k-unique-20260811`
- `l3-deepseek-tp4ep4-nospec-b16k-nvmecache-unique-20260811`
- `l3-deepseek-tp4ep4-dspark5-b16k-cudagraph-none-unique-20260811`

DSpark-3 is not a supported candidate in the pinned vLLM revision. Startup
fails closed because DeepSeek's DSpark block size is five and values below five
can produce incorrect output. It was rejected before serving any request.

The current winner is **DSpark-5 with a 16K batch-token budget**. Against no
speculation it gives nearly identical c1 median TTFT but 43.5% lower median E2E
latency and 74.7% higher aggregate output TPS. No-spec improves aggregate TPS
under c8-c32 saturation, but it delays the first token at c8/c16 and is much
slower for the single DeepSeek synthesis request on the principal auto-max
path. The 32K budget improves c8/c16 throughput but loses c1 E2E/generation
speed and c32 throughput/TPOT, so 16K is the better latency-first balance.

`FULL_AND_PIECEWISE` is also retained. Disabling CUDA Graph shortened a
persistent-cache engine initialization from 73.05 s to 29.55 s, but that
startup-only saving is not worth the serving regression. At c1, Graph NONE
increased median E2E latency from 1.93 s to 15.71 s and reduced aggregate
output throughput from 130.79 to 16.06 tok/s. It remained slower in E2E and
throughput at c8, c16, and c32; even its lower c16 median TTFT was paired with
79.3% higher median E2E latency. The selected Tier2 configuration is therefore
TP4+EP4, DSpark-5, 16K batch tokens, max 32 sequences, FP8 KV, prefix caching,
and full/piecewise CUDA Graphs.

Before the cache fix, a DSpark-5/32K cold engine initialization took 560.68 s,
including 375.98 s of mHC warm-up. The no-spec/16K initial build in the
corrected layout took 519.00 s, including 404.55 s of mHC warm-up. After
restoring the selected DSpark-5/16K configuration, the first restart against
that persistent cache reduced mHC warm-up to 11.98 s and total engine
initialization to 73.05 s. The generated caches exist outside the containers at
`/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/compile-cache/`:
94 MiB for DeepSeek and approximately 239-240 MiB for each Qwen replica after
the reuse check. The selected DeepSeek process also reported 45.74 GiB of KV
cache, or 2,947,608 tokens (2.81x the configured 1M-token context), with FP8 KV.
