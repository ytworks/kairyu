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
