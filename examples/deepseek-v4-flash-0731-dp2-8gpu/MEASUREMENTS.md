# Measured performance

**Status: pending.** No serving run of this exact configuration has completed
yet. The table below is filled only from `./verify.sh serving` artifacts of the
committed configuration; nothing here is estimated or copied from another
topology.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, compute
  capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256:
  `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM revision: `aa0d51302747ea80f282e26949708b3253409fe2`, image ID
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- L1: 2 x TP4+EP4 replicas (GPU 0-3, GPU 4-7), each with DSpark-5,
  `max_num_batched_tokens=16384`, `max_num_seqs=32`, FP8 KV, 256-token blocks,
  prefix caching, full/piecewise CUDA Graphs; SM100-only MegaMoE and FP4
  indexer cache disabled on SM120.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`.
- Kairyu verification commit / served-config SHA-256: recorded in each run's
  `run.json`.

## Comparison baselines

Same host, same checkpoint and vLLM build, different topology:

- TP8+EP8, one replica, `max_num_seqs=64`
  (`../deepseek-v4-flash-0731-8gpu/MEASUREMENTS.md`): 168.01 output tok/s at
  c1, 972.93 at c32 (shared-prefix dataset).
- TP4+EP4, one replica, DSpark-5 / 16K, through Kairyu L3
  (`../qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md`, "Tier2 … selection"):
  130.79 aggregate output tok/s at c1, 241.62 at c32 (unique-prefix dataset).

Neither is a prediction for two replicas; the second bounds what this example
should roughly double at saturation.

## Warm serving result

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements per replica (0, 1) | Gate |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | pending | pending | pending | pending | pending | pending | reported only |
| 8 | pending | pending | pending | pending | pending | pending | pending |
| 16 | pending | pending | pending | pending | pending | pending | pending |
| 32 | pending | pending | pending | pending | pending | pending | pending |
| 64 | pending | pending | pending | pending | pending | pending | pending |

Run ID: pending.
