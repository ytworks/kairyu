# Goal G4: MoE Engine — Fused Experts, EP, MTP, NVFP4 (Roadmap Track E4–E5)

Status: G4.1 M-A1 retains its formal FAIL; M-A2 production integration and
formal evidence are complete; M-A3 is scope-closed by an explicit product-owner
deviation while its unchanged formal performance gate remains FAIL
(2026-08-03).
Lifts the G2 §6 "MoE / expert parallelism" non-goal (amendment recorded in
PROGRESS.md). The reviewed mid-MoE design is `docs/design/g4-mid-moe.md`;
M-A2 passed its independent cache gate; M-A3 did not pass its formal gate.
Depends on: Roadmap Track E1–E3 (`docs/roadmap.md` §4): real single-GPU engine,
scheduler multi-token commit (E2), NcclCommunicator (E3). Frontier-class gates
additionally depend on the E3 hardware decision record (PCIe-switch chassis,
RDMA NICs).
Date: 2026-07-03

## 1. Goal

Kairyu serves MoE models on both hardware profiles (`docs/roadmap.md` §2) at two
tiers:

- **Mid MoE (100–300B; Qwen3-235B-A22B class)** on 2–4 GPUs in one node —
  fused-expert kernels, EP inside the profile's fast domain (NVLink node or
  PCIe-switch domain), MTP speculative decode.
- **Frontier MoE (500B+; DeepSeek-V3/R1, Kimi K2 class)** on 4–8 GPUs per node and
  across nodes — wide EP over RDMA NICs, MLA attention (FlashMLA on SM90/100; an
  FA2-style fallback on SM120), inter-node P-D.

All numbers from committed `bench/` scripts (G2 §8 evidence rules carry forward).

## 2. Hardware and model contract

- **Profiles**: NVLink-HBM (H100/H200/B200 class) and PCIe-GDDR (RTX PRO 6000,
  96 GB GDDR7, no NVLink); see `docs/roadmap.md` §2 for the measured-truth
  requirements (P2P matrix, NIC line rate, kernel-tier smoke tests) that must exist
  in `bench/results/env-<date>.json` before any gate here runs. Gates below run per
  profile; the SM120-specific items are marked.
- **Models**: Qwen3-235B-A22B (mid tier primary); DeepSeek-R1 and/or Kimi K2 NVFP4
  (frontier tier; NVIDIA-published NVFP4 checkpoints are the reference weights).
- **Quantization**: NVFP4 weights are the planning default where the SM supports it
  (235B ≈ 120 GB → 2 GPUs minimum, 4 for KV headroom; DeepSeek ≈ 340 GB → 4–6;
  Kimi ≈ 550 GB → 8); FP8 on SM90. On SM120, KV cache is BF16 until the FP8-KV
  correctness bake (E-KV below) passes.
- **Attention**: DeepSeek/Kimi use MLA. FlashMLA covers SM90/SM100; it does not
  exist for SM120 — an FA2-style MLA fallback kernel is a deliverable of this goal
  (PCIe profile), not an assumption.

## 3. Acceptance gates

### Stage G4.1 — Mid MoE, single node

| Gate | Target | Regime |
|---|---|---|
| M-A1 (correctness anchor) | Qwen3-235B NVFP4 on 2 and 4 GPUs: per-GPU-count TensorRT-LLM reference and Kairyu score the same autoregressive fresh-prefill reference rollout at 16 positions × 64 fixed prompts; require ≥1,014/1,024 token agreement, no substantive disagreement, and the fixed 0.125 near-tie / 0.25 reciprocal selected-logprob bounds. Retain ordinary 16-token retained-KV continuations as diagnostics only; M-A1 does not turn them into a cross-stack decode-parity claim. | — |
| M-A2 (EP does not break KV) | Qwen3-235B NVFP4 EP4, BF16 KV: serialize the fixed A7-lineage 64-session × 8-turn trace (512-token shared prefix, 128 appended tokens/turn) through one persistent production radix/scheduler/engine path. Require 512/512 terminal usages, logical `sum(engine cached_tokens) / sum(engine prompt_tokens) > 80%`, exact raw radix store events, and identical first-prefill allocation/page receipts on all four ranks. Rank rows are witnesses and are never multiplied into the rate. Retain source/checkpoint/container/GPU/topology/kernel evidence and pass manifest verification plus raw-only replay. Timing, OS jitter, and cross-engine output equality are non-binding. | — |
| M-A3 (baseline comparison) | tok/s/GPU and TTFT p99 ≥ SGLang, same box, same checkpoint, same config — SGLang is the credible MoE-on-SM120 baseline; disclose its known SM120 limitations in the results file | saturation |
| M-A4 (MTP value) | MTP acceptance ≥2 tokens/step measured; decode throughput ≥1.5× MTP-off at equal quality (spec ≡ non-spec greedy invariant pinned by test, E2 lineage) | latency-bound |
| E-KV (FP8 KV bake) | `verification/l1/correctness/fp8_kv_g4_ekv_bench.py` runs the pinned Qwen3-32B checkpoint on one SM120 with exact 8K/16K/32K prompts, 2,048-token native ragged-prefill chunks, and 16 greedy decode tokens in BF16-KV and explicit unit-scale E4M3-KV arms. PASS requires exact output token IDs/stopping, finite common-prefix selected logprobs with max absolute delta ≤0.25, complete finite/in-range SATFINITE write audits with bit-exact stored E4M3 bytes and the declared quantization-error bound, and fixed cross-cache samples with NRMSE ≤0.05 and cosine ≥0.99. BF16 remains the default; the operator alone may construct the candidate arm, and any runtime or quality failure is retained as FAIL and keeps public FP8 KV disabled. | — |

M-A3 implementation state (2026-08-03): Kairyu now has a bounded Qwen3-235B
NVFP4 TP1/attention-DP4/EP4 serving path with request-owned attention, KV, and
sampling; grouped direct-NCCL packed-MoE exchange; one compatible packed QKV
projection call; coordinated CUDA-graph decode; and production `/backends`
witnesses. A same-checkpoint implementation diagnostic selected pipeline
depth 5 over depth 1; depth 5 is fixed before the formal comparison and that
diagnostic timing is not gate evidence. The comparison pins SGLang v0.5.16 and
its immutable image/source, uses the same four physical GPUs and checkpoint,
and matches four request owners, EP4, BF16 KV, FCFS, aggregate 65,536-token
cache capacity, and the fixed 128-request ShareGPT workload. Kairyu's internal
TP1 replicated-attention projection layout and SGLang's recorded
TP4/DP4/EP4 layout remain explicitly visible rather than being relabelled as
identical. SGLang additionally fixes `--log-level-http warning`,
`--cuda-graph-max-bs-decode 32`, and disabled prefill CUDA graph.

The binding operator uses exactly ten fresh sequential generations: one fixed
preflight per arm followed by four formal pairs in K/S, S/K, S/K, K/S order.
It gates the exact median of the four ratios: Kairyu completion tok/s/GPU over
SGLang must be at least 1, while Kairyu/SGLang TTFT p99 must be at most 1.
Failures and retries are retained; there is no outlier removal, rounding before
the gate, or exclusion. Every shard now completes and closes its warmup
connection pool before creating a distinct measurement pool with zero prior
requests; assembly, verification, and raw replay reject any lifecycle or
request-order violation. The complete checkpoint is hashed before and after
the matrix, and every generation binds the start capture, a read-only model
volume with no read-write consumer, and live runtime evidence. All operator
commands execute the detached clean `SOURCE_ROOT`; provenance receives
`--checkpoint-start`, while assembly requires both checkpoint boundaries.

The formal matrix at clean commit
`55f3a8ca4513e158182d4b9b4a818c24f5ae7b34` completed all ten generations and
every non-performance binding check, but retained a FAIL: exact median
completion tok/s/GPU was 0.783818× SGLang and TTFT p99 was 1.352633× SGLang.
All four throughput pairs were below one, so the result is not reclassified as
jitter. The optimized candidate keeps configured depth 5, removes replay-side
host drains, and writes ordinary non-aliasing same-device int64 sampled tokens
into persistent decode slots with one vectorized batched D2D copy. A corrected
fresh-server/fresh-pool diagnostic measured Kairyu 536.690626 versus SGLang
551.731445 completion
tok/s/GPU (0.972739×), with a TTFT-p99 ratio of 0.868731. The previous
571.542867-versus-449.965–481.865 comparison is withdrawn because its client
lifecycles were incompatible. A full-server CUTLASS override was also rejected
because its 530.616804 tok/s/GPU was 1.13% below the retained `auto` result.
The product owner accepts the remaining 2.73% diagnostic throughput gap as an
explicit closure deviation so later work can proceed. The formal 1.0/1.0
thresholds and retained FAIL remain unchanged; this is not a formal PASS. The
exact procedure and provenance contract are in `docs/gpu-runbook.md` §9.13.

Recorded 2026-07-31: **FAIL** on the pinned Qwen3-32B revision and one RTX PRO
6000 Blackwell. All K/V write audits passed across 7,522,091,008 values with
zero non-finite inputs, range violations, stored-byte mismatches, or declared
quantization-error violations. The unit-scale candidate nevertheless diverged
from BF16 output at 8K and 32K, exceeded the 0.25 common-prefix selected-logprob
bound at all three lengths (0.3099/0.4518/0.2656), and reached cache NRMSE
0.1047. Public `fp8_e4m3` startup therefore remains fail-closed and BF16 remains
the serving path. Raw evidence is retained under
`bench/results/g4-ekv-fp8-kv-qwen3-32b-sm120-fail-2026-07-31/`; calibrated
per-layer K/V scales require a separate bake before enablement.

### Stage G4.2 — Frontier MoE, multi-node (prereq: G4.1 green + F3 transport gates)

| Gate | Target | Regime |
|---|---|---|
| M-B1 (MLA per profile) | SM90/100: FlashMLA integrated and parity-checked. SM120: MLA fallback kernel — parity vs reference implementation on fixed prompts; kernel microbench published (the highest-risk item on the PCIe profile — the spike starts during G4.1) | — |
| M-B2 (wide EP) | DeepSeek-R1 NVFP4 served with EP dispatch/combine over RDMA NICs (DeepEP/UCCL-EP class); dispatch p99 latency and NIC utilization reported; tok/s/GPU ≥ SGLang same box | saturation |
| M-B3 (fleet integration) | A multi-node MoE group registers as ONE ReplicaPool endpoint and passes the G3 C2 kill/recover drill (kill = whole group) | — |
| M-B4 (frontier latency) | End-to-end TTFT/TPOT for the flagship model measured and published vs Claude/GPT APIs (feeds G6's scoreboard; no threshold here — the scoreboard gate lives in G6) | both |

## 4. Non-goals

- MoE training/fine-tuning parallelism.
- Expert offloading to CPU/NVMe (KV tiering is G5 F4; expert tiering is a future goal).
- Heterogeneous GPU mixes; >8-node coherence domains.
- Custom collectives beyond adopting DeepEP/UCCL-EP-class libraries — we integrate,
  not rebuild.
- Multimodal MoE.

## 5. Seams (informative, non-binding)

- EP lives inside the `ModelRunner` (G2 §7 seam philosophy): scheduler, radix KV,
  and step loop keep their contracts. M-A2 proves rank-invariant logical
  accounting for the current replicated-attention/KV EP path; M-A3 must retain
  that contract when attention-DP makes each attention replica own its
  requests' KV.
- MTP rides the E2 multi-token commit extension of `Scheduler.update()` — an MTP head
  is a different draft source behind `spec_decode.py`'s verify, not a new scheduler.
- Expert-sharded loading extends the E1 safetensors loader + `quant_config.py`
  detection (NVFP4/modelopt parsing added in E1).
- Fleet registration reuses `ReplicaPool`/`DeploymentSpec`; `kairyu.launch` and the
  ClusterSpec 8-node cap come from G5 F3, not this goal.

## 6. Human sign-off checklist (blocking)

- [ ] MoE design doc written and design-reviewed (amendments applied)
- [ ] G4.1 gates green with results files pushed
- [ ] G4.2 gates green with results files pushed
