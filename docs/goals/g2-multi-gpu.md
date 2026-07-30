# Goal G2: Multi-GPU Serving — Intra-Node and Inter-Node

Status: Goal defined (drives M5 and M6). Design docs `docs/design/m5-*.md` and
`docs/design/m6-*.md` must exist and pass design review (APPROVE-WITH-AMENDMENTS or
better) before implementation of each milestone begins — same flow as M1–M4.
**Amended 2026-07-03** (see §7): the hardware contract widens from "8×H100 only" to
capability profiles covering **all NVIDIA GPUs from A100 (SM80) onward**. The
original NVLink arithmetic and all gates stand as written on NVLink-HBM profiles
(A100/H100/H200/B200 nodes); a PCIe-GDDR profile (RTX PRO 6000 fleet) gets its own
arithmetic and replaces A3–A5 with a placement-crossover gate, per
`docs/roadmap.md` §2. The §6 MoE non-goal is lifted into goal G4.
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
- **Models**: `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` at immutable
  revision `f50dbad2c84590ca17dc51e207c34321b65ff14b` is the primary dense
  anchor: compressed-tensors FP8 E4M3 per-output-channel weights, dynamic
  per-token FP8 activations, and BF16 model/KV dtype. Llama-3.1-8B is the
  correctness stepping stone and the only model that fits TP=1 on the original
  80 GB profile. For A6 on the PCIe-GDDR/SM120 profile, the executable closure
  model is `Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`
  with BF16 weights and KV (2026-07-30 amendment below). This profile result is
  not a performance extrapolation to the 70B anchor.
- **Memory arithmetic (fixes the baseline)**: 70B FP8 weights ≈ 70 GB. A single 80 GB
  H100 cannot hold them with usable KV headroom, so **TP=1 is not a valid 70B config;
  TP=2 is the minimal viable config and the base for all scaling-efficiency ratios.**
- **KV size**: 70B GQA (80 layers × 8 KV heads × 128 head_dim × K+V × FP8) =
  **160 KB/token**. All KV-transfer budgets below derive from this number.

## 3. Definitions and measurement regimes

- Metrics: TTFT (first streamed token incl. tokenize + queueing, m2 §5 convention),
  TPOT, output tokens/s, goodput at a stated SLO, KV cache hit rate. For
  latency/throughput distributions, report p50/p99 over ≥3 runs with fixed seeds
  and warmup excluded. Deterministic accounting gates run one complete fixed trace
  per declared topology/path unless their gate explicitly requires repetition.
- **Two regimes** — every target below names its regime:
  - *latency-bound*: closed loop, concurrency 8.
  - *saturation*: open-loop arrival sweep to peak goodput.
- **Scaling efficiency** at TP=N := (metric improvement over TP=2) ÷ (N/2), per regime.
- **vLLM baseline rule**: vLLM V1, pinned version, same box, same model/dtype/TP degree,
  same `max_num_batched_tokens`, same trace, prefix caching on. CUDA-graph parity status
  disclosed (m2 §5 carry-over).
- **A6 aggregation**: four paired rounds use server-start order
  Kairyu/vLLM, vLLM/Kairyu, vLLM/Kairyu, Kairyu/vLLM. Each arm therefore has
  four runs, and all raw request samples are pooled without trimming or outlier
  removal. Percentiles use nearest rank (`ceil(p*N)-1`). Exact per-repeat
  Kairyu/vLLM ratios plus their median and MAD disclose order effects and
  variability, but are non-binding diagnostics; only the issue-specified pooled
  thresholds determine the verdict. The binding ShareGPT
  point is one synchronized 128-request/concurrency-128 burst; a predeclared
  `{1,2,4,8,16}` request/s open-loop sweep is retained as saturation
  diagnostics and cannot replace or select the binding point after measurement.
  A request contributes to goodput only after an exact successful completion and
  TTFT at or below the predeclared 10 s Qwen serving SLO. The 10 s value is
  fixed before A6 measurement and reuses the retained Qwen serving gate's SLO;
  it is not fitted to either arm's observed latency.

## 4. M5 — Intra-node (8×H100)

### Stage 5.1 — TP runner

| Gate | Target | Regime |
|---|---|---|
| A1 (correctness anchor) | Llama-3.1-8B TP=1/2 on the same 64 fixed prompts: retain full greedy continuations with overlap ON/OFF, require each overlap pair to be exact, and require TP1 and TP2 to pass the amended teacher-forced agreement and logprob criteria. `bench/gate_a1.py` assembles and enforces the self-contained gate | — |
| A2 (correctness, 70B) | 70B TP=4 and TP=8 vs TP=2, teacher-forced next-token agreement on 64 prompts: (a) **zero substantive disagreements** — every disagreement inside the reference's own top-k and within the measured tie gap; (b) agreement rate **at or above the reference's self-agreement rate**, both measured by `bench/parity_hf.py`. See the 2026-07-25 amendment below | — |
| A3 (TTFT scaling) | TTFT p50 at TP=8 ≤ ⅓ × TP=2 on 4k-token prompts (≥75% efficiency; prefill is compute-bound and parallelizes near-linearly over NVLink) | latency-bound |
| A4 (TPOT scaling) | TPOT p50 at TP=8 ≤ ½ × TP=2 (≥50% efficiency; decode is bandwidth-bound and all-reduce latency does not shrink with N — linear TPOT scaling is not a defensible promise) | latency-bound |
| A5 (throughput scaling) | Output tokens/s at TP=8 ≥ 2.8 × TP=2 (≥70% efficiency) | saturation |
| A6 (vLLM comparison) | `bench/g2_a6_vllm_bench.py` independently verifies the complete TP=4/8 matched matrix against pinned stock vLLM. ShareGPT @128 conc: goodput ≥ 0.95× vLLM AND TTFT p99 ≤ vLLM. On the 50%-shared-prefix multi-turn trace: TTFT p50 ≥20% better than vLLM (radix-KV structural edge — the G1 claim, preserved where it is defensible) | saturation |
| A7 (KV invariance) | On the fixed 50%-shared-prefix trace, the real native engine's prompt-token KV hit rate (`sum(cached_tokens) / sum(prompt_tokens)`, recomputed from engine-originated response usage) is strictly >80% independently at TP=4 and TP=8, both against the replica directly and through a single-replica gateway. `bench/tp_kv_hit_g2_a7_bench.py` verifies the committed raw trace, config, and physical topology; routing counters are diagnostic and never cache-hit truth | — |

**A7 closure (2026-07-29):** the retained Qwen3-32B artifact records
TP4 direct/gateway at 87.6725%/87.3531% and TP8 direct/gateway at
87.6725%/87.3531%, with 512/512 successful requests in every cell. Independent
raw replay passes all eight binding checks:
`bench/results/g2-a7-kv-hit-qwen3-32b-rtxpro6000-2026-07-29/`.

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

### Amendment (2026-07-02, flagged by the M7 productionization design)

- **">2 nodes" scope clarified**: §6's node cap bounds the TP/PP **coherence domain**
  (collectives, KV transfer plane) that this goal validates — it does not cap the
  number of independent DP replica endpoints behind `ReplicaPool`, which share no
  collective state. Serving fleets of N replica endpoints are a G3/M7 concern
  (`docs/goals/g3-production-deployment.md` §5); each endpoint internally uses a
  layout validated under this goal.

### Amendments (2026-07-03, hardware contract widened to capability profiles — `docs/roadmap.md`)

Production spans **all NVIDIA GPUs from A100 (SM80) onward**, in two fleet shapes:
NVLink-HBM nodes (A100/H100/H200/B200 — this goal's original assumption) and a
PCIe-only RTX PRO 6000 Blackwell fleet (96 GB GDDR7, no NVLink). The original
entries above stay as the record and **remain binding on NVLink-HBM profiles**; the
following define how the goal applies to the PCIe-GDDR profile and to non-H100
NVLink parts:

- **§2 memory arithmetic is per-profile**: on 80 GB H100 the original "TP=2 is the
  minimal viable 70B config" stands. On 96 GB SM120 parts, 70B FP8 fits one GPU
  with usable KV headroom (NVFP4 ≈ 37 GB) — there, **TP=1/DP is the scaling base**,
  and TP over the PCIe root complex is an anti-pattern (per-layer all-reduce latency
  exceeds decode compute); TP is limited to PCIe-switch pairs and prefill-heavy use.
- **A3–A5 (TP-scaling gates) are NVLink-profile gates**: their premises (near-linear
  prefill over NVLink, TP=2 base) do not hold on PCIe. On the PCIe-GDDR profile they
  are replaced by the **placement-crossover report** (roadmap E3) — DP×1 vs PP=2 vs
  TP=2 goodput/TTFT/TPOT across the arrival sweep per model class, generalizing A9's
  "no threshold, publish the crossover" discipline. A1/A2 (correctness anchors),
  A6 (vLLM comparison), A7 (KV invariance), A8 (DP+affinity), A9, A10 (P-D value)
  apply on **every** profile.
- **B2's fabric assumption restated for portability**: "≥20 GB/s on 400 Gb/s IB"
  becomes "≥70% of the **measured** NIC line rate" (G5 F3a); the ≤8 µs/token
  amortized budget stands everywhere. A10's "NVLink page-granular handoff ≤5 ms"
  becomes the same budget over the profile's fast path (NVLink, PCIe-P2P, or NIC),
  measured.
- **§6 MoE / expert-parallelism non-goal lifted** into goal G4
  (`docs/goals/g4-moe-engine.md`): two of the four production model classes are MoE.
- **§6 autoscaling/elasticity non-goal lifted** into goal G5
  (`docs/goals/g5-fleet-scale.md`).
- **§6 "A100 tuning (H100 only)" non-goal lifted**: A100 (SM80) is a supported
  compatibility profile — no FP8/FP4 tensor cores, so its quant path is W4A16
  (AWQ/GPTQ/Marlin) + INT8 W8A8; correctness gates apply as written.
- **§2 model contract widened**: Llama-3.3-70B stays the dense anchor; NVFP4 joins
  FP8 as a planning-default weight format where the SM supports it (KV stays BF16 on
  SM120 pending the FP8-KV correctness bake, G4 E-KV).

### Amendment (2026-07-25, A1/A2 restated after the first real-hardware run)

**A2's "output-match rate ≥99%" is not achievable by any implementation, including
the reference's.** Measured on Qwen3-32B, bf16, 8× RTX PRO 6000 Blackwell
(`bench/results/gate1-hf-parity-tp{1,8}-2026-07-26.json`, whose provenance carries
the full SHA-256 of every weight file — the same digests Hugging Face publishes as
the LFS oids of `Qwen/Qwen3-32B@main`, so §8's "reviewable next to the config that
produced it" holds against an upstream revision and not a path on one machine):

| | agreement |
|---|---|
| HF transformers vs **itself** — `generate()` against a teacher-forced forward over the same sequence | **251/256 = 0.9805** |
| kairyu TP=1 vs HF | 253/256 = 0.9883 |
| kairyu TP=8 vs HF | 251/256 = 0.9805 |

Two code paths through one set of bf16 weights do not always pick the same token.
A gate asking an engine to match a reference *more closely than the reference
matches itself* measures the reference's instability, not the engine.

The same applies to the logprob half. The first tolerance tried here was 0.1
nats; bf16 quantizes the gaps to multiples of ~0.125 at these magnitudes, so 0.1
could only ever classify 0.0 as a tie and everything else as a fault — and one
observed gap was **negative**, HF's forward scoring kairyu's pick above HF's own.

A1/A2 are therefore restated against measured quantities:

1. **zero substantive disagreements** — every disagreement lands inside the
   reference's top-k AND within the tie gap, where the tie gap is *measured* as
   the largest gap the reference produces disagreeing with itself (never below a
   bf16 resolution floor);
2. **agreement at or above the reference's self-agreement rate**.

Both are computed and reported by `bench/parity_hf.py`, which records the noise
floor next to every result — so the comparison travels with the number. A fixed
percentage may return once a reference is available at a precision that supports
it (an fp32 forward, which needs more memory than one card holds for a 32B).

**Free-running greedy sequence equality is not a correctness gate.** A1 keeps its
full-continuation, overlap-ON-and-OFF definition; what changes is that its
verdict is the teacher-forced agreement above rather than sequence equality.
At the time of this amendment, `bench/parity_hf.py` measured that agreement but
did not yet satisfy A1: it ran single-token requests through `EngineCore` only,
so it covered neither full continuations nor Llama-3.1-8B. The overlap path was
already unblocked by `PagedModelRunner`'s in-flight token buffer.

**A1 closure (2026-07-26).** The self-contained
`bench/results/g2-a1-llama31-8b-rtxpro6000-2026-07-26.json` now retains all
Llama-3.1-8B TP1/2 overlap OFF/ON continuations over the same 64 BOS-free prompt
token sequences and passes `bench/gate_a1.py`. Overlap ON reproduces OFF 64/64
at both TP degrees. Against HF's 1010/1024 self-agreement, TP1 and TP2 each
agree on 1014/1024 positions, with zero substantive disagreements, zero missing
logprob samples, and agreeing-position max absolute deltas 0.10440 and 0.10331
under the 0.25 bound. The result embeds the HF reference, exact checkpoint
digests, CUDA 13.0/NCCL 2.29.7 config, raw continuations, and clean commit
provenance. The device-side half of m2 §2.2 stays open as a performance
invariant, not as a gate. The same engine measured 0.786 free-running and 0.988
teacher-forced: once one token differs, every later token is compared against a
prefix the other side never produced, so a single moved near-tie is
indistinguishable from a broken shard. `bench/parity_tp.py` still reports
free-running match rates for orientation; only the teacher-forced numbers gate.

**A2 closure (2026-07-27).** The dense anchor is
`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic@f50dbad2c84590ca17dc51e207c34321b65ff14b`
(compressed-tensors per-channel FP8 weights, dynamic per-token FP8
activations, BF16 model/KV dtype). On the fixed 64×16 BOS-free prefixes, HF
agrees with its own greedy reference on 1005/1024 positions (0.981445) and
sets the measured tie gap at 0.5 nats. Kairyu achieves TP2 1006/1024, TP4
1005/1024, and TP8 1006/1024; every TP degree has zero substantive
disagreements and no missing raw positions/logprobs. Direct TP4/8-vs-TP2 each
agree on 1004/1024 with zero substantive differences. The self-contained
`bench/results/g2-a2-llama33-70b-fp8-rtxpro6000-2026-07-27.json` embeds all
four source envelopes, 15 full safetensors SHA-256s, CUDA 13.0/NCCL 2.29.7
and physical PCIe topology, and passes all ten `bench/gate_a2.py` checks.
The result retains agreeing-position maximum logprob deltas
0.56900/0.54147/0.25563 as diagnostics; they are not a third A2 criterion
beyond the two binding amended criteria above.

### Amendment (2026-07-30, A6 PCIe-GDDR closure profile and measurement contract)

A6 remains binding on every hardware profile, but its performance result must
name the model actually measured. The RTX PRO 6000 Blackwell closure profile
uses the already correctness-reviewed
`Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137` checkpoint with BF16
weights/KV. The Llama-3.3-70B FP8 dense anchor remains unchanged; no Qwen result
is reported as 70B evidence.

The comparison sends identical UTF-8 text through `/v1/completions`, so TTFT
continues to include server tokenization and queueing. The pinned tokenizer's
precomputed token IDs, lengths, and hashes travel with every prompt and must
agree with response usage. ShareGPT is pinned to
`anon8231489123/ShareGPT_Vicuna_unfiltered@745745adf6cd15b84e4f1c4a5a051fb4304f9342`,
file `ShareGPT_V3_unfiltered_cleaned_split.json` (SHA-256
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`).
Selection follows the upstream vLLM shape: seed-0 shuffle, at least two turns,
first user turn, 4–1024 prompt tokens, first 128 qualifying rows. The
shared-prefix arm reuses A7's fixed 64-session × 8-turn geometry and identical
round-trippable text in both engines.

Both arms use greedy seed 0, `ignore_eos=true`, and exact lengths (128 ShareGPT
output tokens, one shared-prefix output token). They share TP degree,
`max_num_batched_tokens=1024`, `max_num_seqs=16`, 8,192-token context,
16-token cache blocks, 131,072 KV-token capacity, chunked prefill, and prefix
caching. Qwen's usable local BF16 KV footprint is 8 GiB/GPU at TP4 and
4 GiB/GPU at TP8. Each arm allocates one additional reserved block:
1 MiB/rank at TP4 and 512 KiB/rank at TP8. Stock vLLM is
`vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404`, build
`ffd46bfab2128bb84146050e98b51a617c6575ab`, whose installed distribution is
`vllm==0.26.0+cu129`; alternate common-path settings are diagnostic only. Decode
CUDA Graphs are enabled on both arms, while the artifact must separately
disclose Kairyu's decode-only capture envelope and vLLM's actual compile/capture
mode and sizes rather than claiming identical full-graph strategies.

The formal closure uses `bench/run_g2_a6_formal.py`, not an uncommitted
operator. Before any server starts it regenerates the complete trace bundle
from the pinned dataset and tokenizer and requires byte-equivalent descriptors,
then full-hashes all 17 live weight shards, the safetensors index, model
metadata, tokenizer files, and Hugging Face revision metadata. It repeats the
checkpoint attestation after all 52 fresh-server cells and rejects any change.
The environment artifact supplies its own measurement-session ID and
nanosecond timestamp; the runner never invents either field.

Every cell retains four serialized workload warmups plus 31 graph-size request
warmups: unique 64-token prompts released synchronously at B=1,2,4,8,16, each
producing exactly two tokens. This proves the synchronized request geometry and
records each arm's configured/captured graph status; it does not by itself
claim that a particular request used a graph dispatch. These rows are verified
but excluded from performance aggregates. Both arms allocate 8,193 pages.
Kairyu reserves one graph scratch page and no null page; stock vLLM reserves
its mandatory `BlockPool` null page and no graph scratch page. Both therefore
expose exactly 8,192 usable KV blocks. Kairyu
pins `pipeline_depth=1`; stock vLLM explicitly enables async scheduling,
multiprocessing TP, FlashAttention, and `VLLM_COMPILE` mode 3 with
`FULL_AND_PIECEWISE` graphs. Both disable custom all-reduce and access/request
logging, and both pin uvloop 0.22.1 plus httptools 0.8.0. Each launch uses the
immutable image ID resolved at preflight, then post-start evidence binds the
actual container ID and image ID, complete argv/environment, working directory,
read-only mounts, GPU/IPC/memlock/port host config, imported engine source
paths/hashes, and container-visible GPU UUIDs/PCI addresses. Kairyu additionally
retains the exact mounted YAML bytes, SHA-256, and parsed object; vLLM retains
the raw V1 config and FlashAttention startup messages from which its markers
are recomputed, including the SM120-resolved
`Using FlashAttention version 2` kernel-version message. Package versions and
Kairyu `/backends` are also retained. The
only permitted clean-tree exception is an untracked path exactly at or below
`.claude/worktrees/`; every other porcelain status record fails closed.

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
  mixed-workload trace. `bench/multiturn_prefix.py` remains the deterministic CPU
  workload-geometry source and KV-manager sanity check. Formal A7 real-engine
  evidence is collected and independently replayed by
  `bench/tp_kv_hit_g2_a7_bench.py`; `bench/router_latency.py` remains reusable for
  A8. Formal A6 evidence is collected as one fresh-server shard per scenario by
  `bench/g2_a6_vllm_bench.py`; its assembler requires the exact 32-run binding
  matrix and its verifier replays raw nanosecond timings, failures, prompt
  identity, server generations, matched cache/batching configuration, and
  provenance before accepting any ratio.
- Latency/throughput/goodput claims use ≥3 runs, fixed seeds, warmup excluded,
  open-loop sweeps for saturation claims, a stated goodput SLO, pinned
  vLLM/NCCL/driver versions, and disclosed CUDA-graph parity (m2 §5 controls).
  Deterministic accounting gates such as A7 use one complete fixed trace per
  declared cell unless the gate says otherwise.

## 9. Human sign-off checklist (blocking)

- [ ] `docs/design/m5-*.md` written and design-reviewed (amendments applied)
- [ ] `docs/design/m6-*.md` written and design-reviewed (amendments applied)
- [ ] All M5 gates (A1–A10) green with results files pushed
- [ ] All M6 gates (B1–B5) green with results files pushed
