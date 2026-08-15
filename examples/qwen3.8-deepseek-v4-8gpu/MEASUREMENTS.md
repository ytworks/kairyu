# Measurements

## Current deployment validation

On 2026-08-14 UTC (2026-08-15 JST), the complete eight-GPU deployment reached
healthy status with four Qwen3.8 TP1 replicas on official vLLM v0.23.0 and one
DeepSeek TP4+EP4 replica on the measured `aa0d513027` DSpark build. The Qwen
replicas all reported the ported 32K/32-sequence/FP8-KV/piecewise-graph/no-MTP
configuration.

The launcher validated the unchanged seven-role verifier-gated DAG and exposed
exactly `kairyu-auto-max` plus `embed-small`. A 384-dimensional embedding
request completed, followed by an end-to-end product chat that returned a
non-empty final answer and 3,750 characters of model-attributed
`reasoning_content`. The stack was stopped after validation to release all
eight GPUs. This is a deployment/correctness smoke, not a new layered-path
performance matrix.

> **Historical evidence:** these measurements predate both the current
> verifier-gated `kairyu-auto-max` role DAG and the Qwen3.8 replacement. They
> must not be attributed to the current deployment; a fresh layered-path run
> is required.

Runtime validation was complete for the measured policy. All performance values
in this document were measured at the Kairyu L3 OpenAI-compatible endpoint used
by ChatUI, never at a vLLM L1 endpoint.

## Selected deployment

- L1: four Qwen3.6-27B-FP8 TP1 replicas on GPUs 0-3, plus one
  DeepSeek-V4-Flash-0731 TP4+EP4 replica on GPUs 4-7.
- Tier2: DSpark-5, 16,384 batch tokens, 32 sequences, FP8 KV, prefix caching,
  chunked prefill, and `FULL_AND_PIECEWISE` CUDA Graphs.
- L2 `kairyu-auto-max`: fixed quality route, three parallel Qwen proposals,
  private-thinking DeepSeek synthesis, `internal_max_tokens=2048`, four total
  steps, and no recursive refinement. `kairyu-auto` remains the static-rule
  low-latency mode and uses direct routes for ordinary requests.
- L3: Kairyu OpenAI-compatible API on loopback port 8003. Only unauthenticated
  Open WebUI is externally exposed, and it defaults to `kairyu-auto-max`.

## L3 auto-max performance selection

These rows measure complete Qwen proposal fan-out, DeepSeek synthesis, L2, and
L3 streaming on unique approximately 8K-token prompts. `public TPS` counts only
the assistant answer visible to the user; `internal TPS` is the cumulative
proposal-plus-synthesis output reported by orchestration. Every selected row
has a non-empty public answer and a valid trace with the exact proposal count.

| L2 candidate | Concurrency | semantic TTFT p50/p99 (ms) | E2E p50/p99 (ms) | TPOT mean (ms/public token) | req/s | public TPS | internal TPS | success / valid trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoA-3, ordinary DeepSeek synthesis | 1 | 14,762.34 / 19,008.44 | 15,965.11 / 19,695.69 | 4.828 | 0.06 | 14.56 | 85.86 | 32/32 / 32/32 |
| MoA-3, ordinary DeepSeek synthesis | 8 | 32,257.80 / 37,529.51 | 39,267.84 / 43,239.06 | 35.025 | 0.20 | 44.26 | 276.16 | 32/32 / 32/32 |
| MoA-3, ordinary DeepSeek synthesis | 16 | 53,530.21 / 68,127.32 | 69,949.14 / 85,593.48 | 87.895 | 0.23 | 73.68 | 336.97 | 32/32 / 32/32 |
| MoA-3, ordinary DeepSeek synthesis | 32 | 100,263.47 / 126,709.95 | 130,182.47 / 133,476.95 | 172.232 | 0.24 | 48.98 | 324.82 | 32/32 / 32/32 |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **1** | **15,559.95 / 19,801.79** | **16,899.93 / 20,997.39** | **4.384** | **0.06** | **17.05** | **96.61** | **32/32 / 32/32** |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **8** | **38,325.65 / 48,881.13** | **41,234.34 / 50,111.05** | **13.707** | **0.19** | **47.81** | **302.65** | **32/32 / 32/32** |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **16** | **68,184.60 / 85,067.02** | **72,678.04 / 85,897.75** | **15.862** | **0.21** | **50.62** | **357.03** | **32/32 / 32/32** |
| **MoA-3, private-thinking DeepSeek, 2048 internal cap** | **32** | **129,700.49 / 150,492.27** | **138,592.39 / 153,445.40** | **29.284** | **0.21** | **54.32** | **351.29** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **1** | **14,968.53 / 18,565.29** | **16,285.04 / 19,684.75** | **4.897** | **0.06** | **14.88** | **65.73** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **8** | **26,741.72 / 33,050.82** | **34,450.35 / 37,173.33** | **36.660** | **0.23** | **54.50** | **235.91** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **16** | **42,287.00 / 55,686.19** | **58,878.42 / 64,166.79** | **83.318** | **0.26** | **65.04** | **279.93** | **32/32 / 32/32** |
| **MoA-2, ordinary DeepSeek synthesis** | **32** | **79,506.41 / 104,720.95** | **108,613.41 / 114,351.48** | **169.822** | **0.28** | **56.15** | **285.42** | **32/32 / 32/32** |

Run IDs:

- `l3-auto-max-chat-moa3-public-v1-20260812`
- `l3-auto-max-chat-moa2-public-v1-20260812`
- `l3-auto-max-thinking2048-public-v1-20260812`

Against the selected warm `kairyu-auto` Tier1-direct path, `auto-max` pays the
intended quality cost. At c1/c8/c16/c32 its median semantic TTFT is
5.94x/7.40x/6.70x/7.92x higher and median E2E is
1.96x/3.55x/4.28x/5.03x higher. The exact corresponding `auto` rows are the
TP1 x 4 rows in the Tier1 table below. The comparison is a user-visible
latency envelope, not a same-output-length TPS A/B: `auto` emits exactly 256
tokens while thinking `auto-max` ends naturally after its public answer and
also performs three private proposals plus synthesis.

MoA-2 retains MoA-3's c1 envelope while reducing cumulative internal output
by 21.5%. At c8 it improves median TTFT by 17.1%, median E2E by 12.3%, and
public aggregate TPS by 23.1%. At c16 it improves TTFT/E2E by 21.0%/15.8%; at
c32 by 20.7%/16.6%, while request throughput rises from 0.24 to 0.28 req/s.
The lower c16 aggregate public TPS reflects shorter answers (247 versus 327
tokens/request), not slower delivery: per-request visible generation rises
from 10.25 to 11.37 tok/s.

A separate 1024-versus-512 private-proposal cap A/B was stopped after c1/c8.
The 512 cap produced 32/32 valid responses but changed c1 TTFT from 14,762.34
to 14,821.93 ms and c8 from 32,257.80 to 32,645.11 ms, while internal output
did not decrease. Qwen proposals usually ended naturally below 512 tokens, so
the performance candidate restores the quality-preserving 1024-token default.
Run ID: `l3-auto-max-chat-moa3-private512-public-v1-20260812`.

The hidden-thinking MoA-3 candidate was rejected before c16: c1 completed
32/32 with TTFT 15,375.10 ms, but c8 returned one response containing private
reasoning usage and no public answer (31/32 usable). L2 now converts that case
to an explicit failure instead of a successful empty response, but a quality
path must be reliable before scoring. Run ID:
`l3-auto-max-moa3-public-v2-20260812`.

The 2048-token private-thinking MoA-3 configuration passed the full L3 matrix: all 128 requests returned
non-empty public answers with 128/128 valid traces, exactly three proposals,
and zero request errors. At c1 its median TTFT/E2E are only 1.2%/0.8% above the
old 1024 candidate. At c8 it trades 18.8% higher median TTFT than ordinary MoA-3
for a 5.0% E2E increase and 8.0% higher public TPS, while eliminating the old
hidden-thinking empty-answer failure. At c16 the TTFT increase is 27.4%, but
median E2E increases only 3.9%. This establishes its serving envelope; product evaluation is owned by `evals/`.

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

## Cold/warm separation and runtime identity

All request-latency tables exclude an explicit warm-up and use a fresh
row-specific prefix namespace; they are warm serving measurements and cannot be
misread as cold-start latency or a full-prefix replay. Cold startup was measured
separately. Before the corrected persistent-cache layout, DeepSeek engine
initialization took 560.68 s; restarting the selected DSpark-5/16K configuration
with the warmed NVMe cache took 73.05 s, an 87.0% reduction. The first TP4 Qwen
candidate cache build took 201.41 s, including 109.54 s compilation; those
candidate-only artifacts were not used as warm request samples.

The measurements use eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs.
Qwen revision is `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`; DeepSeek revision
is `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`; vLLM source revision is
`aa0d51302747ea80f282e26949708b3253409fe2`; and the vLLM image ID is
`sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`.
The externally reachable no-auth ChatUI validated after the run is
`http://61.206.39.14:3000`; Kairyu L3 remains loopback-only.
