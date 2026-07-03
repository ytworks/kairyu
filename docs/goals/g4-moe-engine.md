# Goal G4: MoE Engine — Fused Experts, EP, MTP, NVFP4 (Roadmap Track E4–E5)

Status: Goal defined (2026-07-03). Lifts the G2 §6 "MoE / expert parallelism"
non-goal (amendment recorded in PROGRESS.md). A design doc (`docs/design/m8-*.md`
or successor numbering) must pass design review before implementation begins —
same flow as M1–M7.
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
| M-A1 (correctness anchor) | Qwen3-235B NVFP4 on 2 and 4 GPUs: greedy output-match ≥99% on the 64 fixed prompts vs a reference serving stack of the same checkpoint, + logprob tolerance (m2 §2.5 style) | — |
| M-A2 (EP does not break KV) | Radix hit >80% @50% shared prefix with EP on (A7 lineage; attention-DP must keep per-replica KV accounting rank-invariant) | — |
| M-A3 (baseline comparison) | tok/s/GPU and TTFT p99 ≥ SGLang, same box, same checkpoint, same config — SGLang is the credible MoE-on-SM120 baseline; disclose its known SM120 limitations in the results file | saturation |
| M-A4 (MTP value) | MTP acceptance ≥2 tokens/step measured; decode throughput ≥1.5× MTP-off at equal quality (spec ≡ non-spec greedy invariant pinned by test, E2 lineage) | latency-bound |
| E-KV (FP8 KV bake) | FP8 KV cache enabled only after a dedicated correctness run (long-context bit-exactness vs BF16 KV) passes on SM120; result recorded either way | — |

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

- EP lives inside the `ModelRunner` (G2 §7 seam philosophy): scheduler, radix KV, and
  step loop keep their contracts; attention-DP means each rank owns its requests' KV.
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
