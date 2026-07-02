# Goal G2: Multi-GPU Serving — Intra-Node and Inter-Node

Status: Goal defined (drives M5 and M6). Design docs `docs/design/m5-*.md` and
`docs/design/m6-*.md` must exist and pass design review (APPROVE-WITH-AMENDMENTS or
better) before implementation of each milestone begins — same flow as M1–M4.
Depends on: M2 GPU phase Gates 1–3 (`docs/gpu-runbook.md` §1–2: single-GPU 8B runner
correct and benchmarked).
Date: 2026-07-02

## 1. Goal

Kairyu serves Llama-3.3-70B-class dense FP8 models across multiple GPUs (M5, one node)
and multiple nodes (M6, two nodes) with best-in-class latency, matching or beating
vLLM V1 at identical parallel configurations, via four strategies:

- **TP** — tensor parallelism inside one node (M5)
- **DP** — data-parallel replicas load-balanced by the existing L2 Router (M5, M6)
- **P-D** — prefill/decode disaggregation with page-granular KV handoff (M5 intra-node,
  M6 inter-node)
- **PP** — pipeline parallelism across nodes (M6)

All numbers must come from `bench/` reproduction scripts; no estimated or extrapolated
results are ever reported (goal acceptance criteria, carried from G1).

## 2. Hardware and model contract

- **M5**: one 8×H100-80GB node, NVLink/NVSwitch.
- **M6**: two such nodes, ≥400 Gb/s InfiniBand or RoCE. Record the actual fabric, link
  rate, and a raw fabric microbenchmark in `bench/results/env-<date>.json` (extends
  gpu-runbook §0).
- **Models**: Llama-3.3-70B FP8 W8A8 (primary); Llama-3.1-8B (correctness stepping
  stone — the only model that fits TP=1).
- **Memory arithmetic (fixes the baseline)**: 70B FP8 weights ≈ 70 GB. A single 80 GB
  H100 cannot hold them with usable KV headroom, so **TP=1 is not a valid 70B config;
  TP=2 is the minimal viable config and the base for all scaling-efficiency ratios.**
- **KV size**: 70B GQA (80 layers × 8 KV heads × 128 head_dim × K+V × FP8) =
  **160 KB/token**. All KV-transfer budgets below derive from this number.

## 3. Definitions and measurement regimes

- Metrics: TTFT (first streamed token incl. tokenize + queueing, m2 §5 convention),
  TPOT, output tokens/s, goodput at a stated SLO, KV cache hit rate. Report p50/p99
  over ≥3 runs, fixed seeds, warmup excluded.
- **Two regimes** — every target below names its regime:
  - *latency-bound*: closed loop, concurrency 8.
  - *saturation*: open-loop arrival sweep to peak goodput.
- **Scaling efficiency** at TP=N := (metric improvement over TP=2) ÷ (N/2), per regime.
- **vLLM baseline rule**: vLLM V1, pinned version, same box, same model/dtype/TP degree,
  same `max_num_batched_tokens`, same trace, prefix caching on. CUDA-graph parity status
  disclosed (m2 §5 carry-over).

## 4. M5 — Intra-node (8×H100)

### Stage 5.1 — TP runner

| Gate | Target | Regime |
|---|---|---|
| A1 (correctness anchor) | 8B TP=2: greedy token parity vs 8B TP=1 (HF-verified per gpu-runbook Gate 1) on the 64 fixed prompts, overlap ON and OFF | — |
| A2 (correctness, 70B) | 70B TP=4 and TP=8 vs TP=2: greedy output-match rate ≥99% on 64 prompts + logprob tolerance (m2 §2.5 style; reduction order shifts argmax ties — match-rate precedent from m3) | — |
| A3 (TTFT scaling) | TTFT p50 at TP=8 ≤ ⅓ × TP=2 on 4k-token prompts (≥75% efficiency; prefill is compute-bound and parallelizes near-linearly over NVLink) | latency-bound |
| A4 (TPOT scaling) | TPOT p50 at TP=8 ≤ ½ × TP=2 (≥50% efficiency; decode is bandwidth-bound and all-reduce latency does not shrink with N — linear TPOT scaling is not a defensible promise) | latency-bound |
| A5 (throughput scaling) | Output tokens/s at TP=8 ≥ 2.8 × TP=2 (≥70% efficiency) | saturation |
| A6 (vLLM comparison) | vs vLLM TP=4 and TP=8, ShareGPT @128 conc: goodput ≥ 0.95× vLLM AND TTFT p99 ≤ vLLM. On the 50%-shared-prefix multi-turn trace: TTFT p50 ≥20% better than vLLM (radix-KV structural edge — the G1 claim, preserved where it is defensible) | saturation |
| A7 (KV invariance) | KV hit rate >80% @50% shared prefix preserved at TP=4/8 (`bench/multiturn_prefix.py`; guards that KV sharding does not break radix reuse) | — |

### Stage 5.2 — DP replicas + routing (blocked on 5.1)

| Gate | Target | Regime |
|---|---|---|
| A8 (DP scaling) | DP=2 × TP=4 (same 8 GPUs) vs 1 × TP=4: goodput ≥1.9× (replicas are independent — near-linear is fair to demand); L2 Router added latency p99 <10 ms (m4 budget); session-affinity routing keeps multi-turn KV hit rate ≥90% of the single-replica value (naive round-robin destroys prefix locality — affinity is part of the acceptance contract) | saturation |
| A9 (DP vs TP crossover) | Report DP=2×TP=4 vs TP=8 goodput and TPOT across the arrival sweep. No threshold — the crossover concurrency must appear in the results file | saturation |

### Stage 5.3 — P-D disaggregation, intra-node (blocked on 5.1)

| Gate | Target | Regime |
|---|---|---|
| A10 (P-D value) | Mixed workload (long prefills + latency-SLO decodes): TPOT p99 ≤ 0.8× the best colocated chunked-prefill config at equal goodput; goodput ≥ 0.9× colocated (caps the capacity cost); NVLink page-granular KV handoff adds ≤5 ms p99 to TTFT for ≤4k-token prompts (4k tok × 160 KB ≈ 640 MB ≈ 1.4 ms raw at NVLink rates; 5 ms allows paging scatter) | saturation |

## 5. M6 — Inter-node (2 nodes; prereq: all M5 gates green)

Stage order rationale: DP first (reuses the L2 Router, near-zero engine change,
validates the 2-node harness) → KV-transfer primitive (the riskiest new plane, gated
standalone before any end-to-end claim) → P-D → PP last (delivers capability for
bigger-than-node models rather than a latency win for a model that fits one node).

### Stage 6.1 — 2-node DP

| Gate | Target | Regime |
|---|---|---|
| B1 | 2-node DP (each node at its best M5 config) vs 1 node: goodput ≥1.85×; router p99 <10 ms including the network hop | saturation |

### Stage 6.2 — KV transfer plane

| Gate | Target | Regime |
|---|---|---|
| B2 | Page-granular inter-node KV transfer, standalone microbench (`bench/kv_transfer_bench.py`): sustained effective ≥20 GB/s on 400 Gb/s IB (≥40% of line rate) for batches ≥64 contiguous pages — i.e. **≤8 µs/token amortized** at 160 KB/token | — |

### Stage 6.3 — P-D inter-node (blocked on 6.2)

| Gate | Target | Regime |
|---|---|---|
| B3 | Prefill node → decode node: TTFT p50 inflation ≤20% vs intra-node colocated at matched load (2k-token prompt = 320 MB ≈ 16 ms raw at 20 GB/s vs hundreds-of-ms 70B TTFT — achievable only if transfer is overlapped page/layer-wise with prefill, which is what this target forces); the A10 TPOT p99 improvement (≥20% vs colocated) must survive the network | saturation |

### Stage 6.4 — PP across nodes

| Gate | Target | Regime |
|---|---|---|
| B4 | PP=2 across nodes (TP=4 or 8 per stage): TPOT p50 inflation ≤10% vs single-node equal-TP (per-token inter-node hop is a ~0.5 MB hidden-state transfer — tens of µs vs 20–50 ms TPOT); saturation throughput ≥1.6× one node (≥80% efficiency, accounting for stage imbalance and bubbles under continuous batching); pipeline bubble fraction measured and reported | both |
| B5 (vLLM comparison) | Where vLLM supports the equivalent config (multi-node TP/PP via Ray): parity — goodput ≥0.95×, TTFT p99 ≤ vLLM. Where it does not (P-D over this fabric), the stated baseline is single-node colocated Kairyu | saturation |

## 6. Non-goals

- MoE / expert parallelism (dense 70B only).
- Sequence/context parallelism; >2 nodes; heterogeneous GPU mixes; A100 tuning (H100 only).
- Fault tolerance beyond removing unhealthy replicas from routing (no live migration,
  no elastic autoscaling).
- Custom collectives — NCCL (or equivalent) is assumed, not built.
- Training/fine-tuning parallelism.
- CPU/disk KV offload tiers — the transfer plane is GPU↔GPU only in this goal.
- Multi-model serving per replica; the M4 learned router's cost/quality objective (this
  goal uses the router only as a load/affinity balancer).

## 7. Seams (informative, non-binding)

The goal expects the following existing seams to be sufficient. These are blast-radius
constraints, not designs — the m5/m6 design docs decide the how. If a design doc must
break one, that is an amendment to this goal, flagged in review.

- **TP/PP live inside `ModelRunner` implementations**
  (`kairyu/engine/core/engine_core.py`): the scheduler, radix KV, and step loop keep
  their public contracts unchanged (m2 §2.6 stated intent).
- **P-D admission policy is already implemented** (`kairyu/engine/core/scheduler.py`:
  `pd_separation`, `decode_token_budget`, `decode_watermark_pages`). This goal covers
  only the missing halves: replica/stream topology and KV handoff.
- **Contiguous KV pages** (`kairyu/engine/core/pages.py`, `radix_kv.py`) are the unit
  of all KV transfer, intra- and inter-node. Page granularity is a goal-level
  requirement (B2 is defined against it); the transport is not.
- **DP routing goes through the existing L2 Router**
  (`kairyu/orchestration/router.py`), inheriting its <10 ms budget and JSONL decision
  log — replica choice must appear in the decision log like any routing decision.
- **vLLM-compat surface**: `tensor_parallel_size` (`kairyu/entrypoints/llm.py`,
  `async_engine.py`) stops being a no-op; the API shape does not change.

### Amendments (2026-07-02, flagged by the m5/m6 design review per this section's rule)

- **DP routing seam reworded**: the `Router` protocol returns tiers, not replicas, so DP
  placement lives in a sibling L2 component (`ReplicaPool`), inheriting the router's
  <10 ms budget and JSONL decision log (via a new `record_replica` entry kind). The
  intent of the seam — DP is an orchestration-layer concern, engine untouched — is
  unchanged (m5 D4).
- **Step-loop contract extension**: the `ModelRunner` protocol gains an async
  submit/handle form (already reserved by m2 §5 item 3 for CUDA graphs); PP=2's B4
  gates are unreachable under the synchronous contract (m6 D5). Scheduler and RadixKV
  contracts remain unchanged except the additive `resume_with_kv` entry point (m5 D5).

## 8. Evidence and reporting rules

G1 rules carried forward verbatim, plus:

- Every number from a committed `bench/` script; results in
  `bench/results/<date>-<gpu-topology>.json` with the full config beside it (parallel
  degrees, fabric, NCCL version, measured — not nominal — link rate from a raw fabric
  microbench run in the same session).
- **Scaling-efficiency claims must include the TP=2 base measurement in the same
  results file** (no cross-session bases).
- Required bench additions (named here, designed in m5/m6): topology arguments for
  `bench/serving_bench.py` (TP/PP/DP sweep), `bench/kv_transfer_bench.py` (B2), a P-D
  mixed-workload trace. `bench/router_latency.py` and `bench/multiturn_prefix.py` are
  reused as-is for A7/A8.
- ≥3 runs, fixed seeds, warmup excluded, open-loop sweep for saturation claims, goodput
  SLO stated per result, pinned vLLM/NCCL/driver versions, CUDA-graph parity disclosed
  (m2 §5 controls).

## 9. Human sign-off checklist (blocking)

- [ ] `docs/design/m5-*.md` written and design-reviewed (amendments applied)
- [ ] `docs/design/m6-*.md` written and design-reviewed (amendments applied)
- [ ] All M5 gates (A1–A10) green with results files pushed
- [ ] All M6 gates (B1–B5) green with results files pushed
