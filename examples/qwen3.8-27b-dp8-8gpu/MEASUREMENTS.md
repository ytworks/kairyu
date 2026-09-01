# Measured performance

**Status: pending.** No serving run of this exact configuration has completed
yet. The table below is filled only from `./verify.sh serving` artifacts of the
committed configuration; nothing here is estimated or copied from another
topology.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, compute
  capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Model tree SHA-256:
  `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- vLLM: `v0.23.0`, image ID
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- L1: 8 x TP1 replicas, each with `max_num_batched_tokens=32768`,
  `max_num_seqs=32`, FP8 KV, FP16 Gated-DeltaNet state, piecewise CUDA Graphs,
  MTP off — the single-GPU example's measured envelope.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`.
- Kairyu verification commit / served-config SHA-256: recorded in each run's
  `run.json`.

## Comparison baseline

The single-GPU example measured 867.58 output tok/s at concurrency 32 on the
same L1 envelope (`../qwen3.8-27b-1gpu/MEASUREMENTS.md`). The tiered example's
Tier1 matrix through Kairyu L3 measured TP1 x 4 replicas at 295.19 aggregate
output tok/s (c32) with its role-shaped 8K/256 protocol
(`../qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md`, "Tier1 topology selection").
Neither is a prediction for eight replicas; they bound what this example
should exceed.

## Warm serving result

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements per replica (0..7) | Gate |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | pending | pending | pending | pending | pending | pending | reported only |
| 8 | pending | pending | pending | pending | pending | pending | pending |
| 16 | pending | pending | pending | pending | pending | pending | pending |
| 32 | pending | pending | pending | pending | pending | pending | pending |
| 64 | pending | pending | pending | pending | pending | pending | pending |

Run ID: pending.
