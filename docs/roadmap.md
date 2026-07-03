# Kairyu Roadmap: Fugu-Class Orchestrated Serving on a Dual-Profile GPU Fleet

Status: **Accepted** (2026-07-03). Master roadmap; extends the hardware contract of
`docs/goals/g2-multi-gpu.md` to two fleet profiles (amended, see G2 §7) and lifts
the scope caps of G3.
Drives goals G4 (`docs/goals/g4-moe-engine.md`), G5 (`docs/goals/g5-fleet-scale.md`),
G6 (`docs/goals/g6-product-surface.md`).
Date: 2026-07-03

## 1. Product goal

Serve a Fugu-class orchestration product (multi-model auto-routing + agent
ensemble/synthesis behind one OpenAI-compatible API and a chat UI) from an on-prem DC
of **thousands of GPUs across two hardware profiles — NVLink nodes (8×H100 class) and
PCIe nodes (RTX PRO 6000 Blackwell)** — with TTFT/TPOT/goodput that beat the
Claude/GPT frontier APIs as measured by a committed benchmark harness (G6 gate P-C1).
Both profiles are first-class: parallelism strategy is selected per profile from
measured topology, not assumed. L1 remains Kairyu's own engine (kernel libraries like
FlashInfer are used; scheduler, radix KV, spec decode, orchestration stay ours).

Target model classes (all four):

| Class | Examples | Role |
|---|---|---|
| Small dense ~14B | Llama/Qwen 8–14B | Thousands of single-GPU replicas; latency tier |
| Mid dense ~70B | Llama-3.3-70B | Quality tier, single-GPU (NVFP4) or PP=2 |
| Mid MoE 100–300B | Qwen3-235B-A22B, GLM-4.5 | High-quality tier, 2–4 GPUs |
| Frontier MoE 500B+ | DeepSeek-V3/R1, Kimi K2 | Flagship tier, multi-GPU/multi-node EP |

## 2. Hardware capability profiles (A100 and later)

Kairyu supports **every NVIDIA datacenter GPU from A100 (SM80) onward**. Nothing in
the engine or control plane hard-codes a GPU: at bring-up each node is probed into a
**hardware profile** — interconnect (NVLink domain vs measured PCIe P2P), memory
(capacity, measured bandwidth), compute formats (BF16/INT8/FP8/NVFP4), and kernel
tier (which attention/GEMM backends exist for the SM) — recorded in
`bench/results/env-<date>.json`. Parallelism and quantization strategy are functions
of the profile, never assumptions.

| Profile | Arch (SM) | Memory | Interconnect | Formats | Kernel tier | Default parallelism |
|---|---|---|---|---|---|---|
| **NVLink-HBM** (peak-perf) | H100/H200 (SM90), B200 (SM100) | 80–192 GB HBM, 3.3–8 TB/s | NVLink ≥900 GB/s | BF16, FP8 (+NVFP4 on SM100) | Full: FA2/FA3, FlashMLA, DeepGEMM, CUTLASS | **TP-first** in-node — G2's original §2 arithmetic and gates A1–A10 apply as written |
| **PCIe-GDDR** (fleet-scale) | RTX PRO 6000 Blackwell (SM120) | 96 GB GDDR7, ~1.6 TB/s | PCIe Gen5 (~24 GB/s P2P via root complex; switch topologies better); GPUDirect P2P/RDMA; MIG 4×24 GB (vBIOS ≥98.02.55 — audit) | BF16, FP8, native NVFP4 | FA2-path only (FlashMLA/FA3/DeepGEMM are SM90/100-only); ~99 KB smem/SM; Triton-first FP8; several NVFP4 grouped-GEMM/MoE paths immature; FP8-KV silent-corruption reports → **BF16 KV default** until a correctness bake passes | **DP-first, PP for capacity, EP over RDMA NICs**; TP only within a PCIe-switch pair, mainly prefill |
| **Ampere-compat** | A100 (SM80) | 40/80 GB HBM2e, ~2 TB/s | NVLink 600 GB/s | BF16, INT8 (**no FP8/FP4 tensor cores**) | FA2; Marlin-class W4A16 (AWQ/GPTQ), INT8 W8A8 | TP-first (NVLink rules); quant paths differ |

Consequences that drive the designs below:

1. **Strategy is per-profile.** On NVLink-HBM the original G2 plan (TP=2/4/8 for
   70B FP8) stands unchanged. On PCIe-GDDR the memory arithmetic flips — 70B FP8
   (~70 GB weights) fits ONE 96 GB GPU with usable KV headroom, NVFP4 is ~37 GB —
   and TP over the root complex is an anti-pattern (per-layer all-reduces at ~50 µs
   PCIe latency add ms-scale TPOT; community 8×TP measurements land at ~1/3 of
   8×H100). E3's placement-crossover report is produced **per profile**.
2. **Quantization matrix**: BF16 everywhere; W4A16 (AWQ/GPTQ/Marlin) + INT8 W8A8 on
   SM80; FP8 W8A8 on SM90+; NVFP4 on SM100/SM120. `quant_config.py` detection must
   cover all of these; the loader picks the best format the profile supports.
3. **On PCIe nodes, all inter-GPU KV movement rides the NIC** (P-D, tiering,
   migration) — an RDMA-capable transport (NIXL/UCX class) is mandatory there, and
   still needed on NVLink profiles for inter-node paths (G2 §5 unchanged).
4. **Kernel tiers are verified, not trusted**: every kernel path is smoke-tested on
   the actual SM before any gate runs on it (the SM120 gotcha list above is the
   motivating example; SM80's missing FP8 is the other).

### Deployment matrix — PCIe-GDDR planning defaults (E3's crossover report finalizes per profile)

On NVLink-HBM profiles the G2 original configs apply (70B FP8 TP=2/4/8; MoE with the
full kernel tier). The PCIe-GDDR profile needs its own defaults:

| Class | Quant | GPUs | Parallelism | Spec decode |
|---|---|---|---|---|
| ~14B dense | FP8 W8A8 (NVFP4 later) | 1 | Pure DP | EAGLE-3 |
| ~70B dense | NVFP4 primary, FP8 quality tier | 1 (NVFP4) / 2 PP (FP8) | TP=1 DP-first; PP=2 when KV-bound | EAGLE-3 |
| 100–300B MoE | NVFP4 weights, BF16 KV | 2–4 | PP across pairs; attention-DP + EP inside a switch domain | MTP (native heads) |
| 500B+ MoE | NVFP4, BF16 KV | 4–8/node, multi-node | Attention-DP + EP over NIC RDMA; PP across nodes; P-D pools | MTP (mandatory) |

**Hardware procurement dependency (blocking E4/E5 on the PCIe profile, decided during
E3)**: chassis with PCIe Gen5 switches (pairs/quads behind a switch, not all through
the root complex) and per-node RDMA NICs sized for EP dispatch (≥400 Gb/s class).
E1's measured P2P matrix is the evidence basis for this order.

## 3. Where Kairyu stands (2026-07-03)

The control plane is real and CPU-proven (scheduler, radix KV, overlap contract,
PDCoordinator, KVTransport protocol, ReplicaPool, learned router, M7 serving/deploy
layer — 328 tests). The compute is placeholder: toy attention, word-hash tokenizer,
greedy-only sampling, FakeCommunicator, TCP-loopback KV transport, no quant kernels,
no CUDA graphs, no MoE. **Every protocol seam carries over; the GPU halves and the
fleet/product layers are the work.**

Competitive facts grounding the plan (sources in §7):

- **Sakana Fugu** (GA 2026-06) is a "multi-agent system as a model": one
  OpenAI-compatible endpoint (`fugu`, `fugu-ultra`) whose model is a learned
  orchestrator over frontier LLMs. Chat Completions + Responses API; usage exposes
  `orchestration_input/output_tokens`; cached-input discount is on the price sheet;
  no consumer chat UI. Independent first-touch latency: ~7–8 s light tasks, 11–269 s
  Ultra. **Fugu does not win on latency** — Kairyu's wedge is Fugu-class orchestration
  at direct-call latency, plus orchestration transparency (trace disclosure).
- **Fleet control planes converged** (Dynamo, llm-d, SGLang gateway, Mooncake, AIBrix):
  prefix-cache-aware routing is the single largest serving lever; P/D disaggregation
  pays on goodput under mixed workloads; SLO-based early rejection beats queueing;
  KV tiering (HBM→DRAM→NVMe) multiplies cache. Kairyu owns its L1 radix KV, so it can
  emit precise KV events to the router natively — a vertical integration the
  vLLM-wrapper control planes have to standardize across engines. Multi-model
  cost/latency routing with a learning loop (M4) is white space none of them cover.

## 4. Roadmap — three parallel tracks

Dependency spine: **E1 + P-A first** (they unblock every honest number), then
E2 + F1 + P-B, then E3 / F2 / P-C, then E4 / F3, then E5 / F4 / F5.
Track F phases 0–2 and all of Track P's CPU work proceed without GPUs, preserving the
repo's CPU-first discipline.

### Track E — L1 engine (goal: G2-as-amended, then G4)

| Phase | Deliverables | Exit gates | Builds on |
|---|---|---|---|
| **E1 Single-GPU real engine** (start now on available RTX 6000 Pro; identical steps on any A100+/H100 box) | HF tokenizer + incremental detokenizer (replace `KairyuBackend._tokenize`); safetensors loader for Llama/Qwen dense; FlashInfer FA2-path `ModelRunner` (works SM80–SM120; verify per SM first, page_size=16); **hardware-profile probe** (measured bandwidth, P2P matrix, format support → env json); full sampling (SamplingParams→EngineRequest plumbing, temperature/top-k/top-p/penalties/logprobs on GPU, xgrammar mask wired into the sampler); per-arch quant paths — FP8 W8A8 via Triton (SM90+), W4A16 AWQ/GPTQ + INT8 for SM80, NVFP4/modelopt detection added to `quant_config.py` | Greedy parity vs HF transformers (64 prompts, overlap ON/OFF); full CPU suite stays green; 14B goodput ≥1.0× pinned vLLM same box + TTFT p99 ≤ vLLM (best common quant format for the box); radix hit >80% @50% shared prefix on the real engine; profile probe recorded in `bench/results/env-<date>.json` | `engine_core.py` ModelRunner seam, `radix_kv.py`, `scheduler.py`, `bench/serving_bench.py` |
| **E2 Decode-speed stack** | Decode CUDA graphs (batch buckets, on the async submit/handle runner form from `pipeline.py`); **scheduler multi-token commit** (the one real CPU-half change — `Scheduler.update()` accepts per-request accept/reject token lists); wire `spec_decode.py` n-gram, then EAGLE-3 (SpecForge-trained heads); 70B NVFP4 W4A16 on one GPU; ZMQ/msgpack API-server/engine process split | Spec output ≡ non-spec greedy; 14B decode ≥2.2× (EAGLE-3 + graphs, conc 8); **70B NVFP4 TP=1 beats vLLM's best any-TP config same box in tok/s/GPU**; NVFP4-vs-FP8 quality delta ≤1% on an eval suite | `overlap.py`, `pipeline.py` async contract, `spec_decode.py` |
| **E3 Multi-GPU, profile-aware** | G2 amendment PR (§7 of G2, this doc §5); `NcclCommunicator` behind `comm.py` (TPModelRunner → multi-process SPMD, rank-agreement becomes debug mode) — serves TP-first NVLink profiles AND switch-pair TP on PCIe; PP=2/4 on `PipelinedEngineCore`; DP fleets via existing `ReplicaPool` (unchanged — the primary PCIe scaling axis); intra-node P-D device-copy behind the `KVHandoff` seam; **hardware decision record** (PCIe-switch chassis + NICs) | On NVLink profile: G2 A1–A10 as written. On PCIe profile: placement-crossover report (DP×1 vs PP=2 vs TP=2 per class, arrival sweep — replaces A3–A5 there); PP=2 70B-FP8 TPOT inflation ≤10% + throughput ≥1.7×; A8/A10 as written | `comm.py`, `tp_runner.py`, `pipeline.py`, `pd.py`, `replica.py` |
| **E4 MoE engine** (goal G4) | MoE model defs (Qwen3-235B class) + expert-sharded loader; Triton fused-MoE + grouped GEMM (FP8→NVFP4); EP v1 = NCCL all-to-all inside the profile's fast domain (NVLink node or PCIe-switch domain), attention-DP hybrid; MTP spec decode (reuses E2 multi-token machinery); **MLA-on-SM120 kernel spike started early** (gates E5 on the PCIe profile; SM90/100 use FlashMLA) | G4 gates (see goal doc): correctness anchor, ≥ SGLang same box, MTP acceptance ≥2 tok/step | E2 scheduler extension, E3 NCCL |
| **E5 Frontier MoE** | DeepEP/UCCL-EP over RDMA NICs; GPUDirect-RDMA `KVTransport` impl (fragment protocol carries over; JSON header → msgpack); inter-node P-D via `PDCoordinator` (copy-before-commit semantics verbatim) with layer-group streaming; MLA + NVFP4 full path; EPLB-style expert balancing v1 | G2 B2/B3 as amended (≥70% of measured NIC line rate, ≤8 µs/token; TTFT inflation ≤20%); DeepSeek-R1 NVFP4 ≥ SGLang same box; end-to-end TTFT/TPOT vs Claude/GPT measured (feeds G6 P-C1) | `kv_transport.py`, `pd.py`, E4 |

### Track F — Fleet control plane (goal: G5)

| Phase | Deliverables | Exit gates (G5) | Builds on |
|---|---|---|---|
| **F0 Goal + amendments** | `docs/goals/g5-fleet-scale.md`; amendment entries in PROGRESS.md (§5 below); SLO vocabulary fixed (goodput@SLO, prefix-hit-rate, placement p99, scale-up latency) | Docs merged | — |
| **F1 Elastic control plane** (CPU-mock) | Dynamic `ReplicaPool` membership (add/remove/drain; decision/health state keyed by replica id); replica registry + reconciler (`discovery: static | k8s-endpoints | register`, static stays default); Helm/k8s manifests + kind CI smoke next to compose-smoke; gateway horizontal scale-out (consistent-hash session partitioning at the LB; `BatchStore` behind a protocol); OTel tracing (amends m7 D8); `/drain` endpoint | 200 mock replicas, 10%/min churn ×10 min → zero 5xx, placement p99 <10 ms; rolling restart of 100 mock replicas with zero failures; 3 gateways behind an LB pass the C1 affinity assertion; one request = one full trace | `replica.py`, `spec.py`, `prober.py`, compose smoke pattern |
| **F2 KV-aware routing** (CPU-first) | Approximate prefix-trie scorer in the gateway (α·prefix-overlap − β·load, power-of-two fallback, session affinity as tiebreak); **RadixKV KV events** (BlockStored/BlockRemoved over ZMQ, vLLM-schema-compatible) → precise block-hash→replica index; learned placement (feed decisions+TTFT into `learning/dataset.py`, bandit tunes α/β); Conductor/MoA shared-prefix co-location hints | Prefix-hit ≥2× vs session hashing on a shared-prefix trace with no goodput loss on uniform traffic; TTFT p95 −30% on multi-turn+RAG trace (4–8 GPU testbed); index staleness <500 ms under churn with graceful fallback | `_select()` in `replica.py`, `radix_kv.py`, JsonlRouterLog, M4 learning pipeline |
| **F3 NIC KV transfer + P/D pools** | m6 D3 bake-off run for real **with NIXL added** (vs NCCL-p2p-staging vs UCX/RDMA); rack-locality-aware prefill↔decode pool pairing; P:D ratio planner v0 from SLO telemetry; ClusterSpec cap 2→8 nodes + `kairyu.launch` (k8s pod-group rendezvous, torchrun-style); a multi-node group registers as ONE ReplicaPool endpoint | Transfer ≥70% of measured NIC line rate at ≥64-page batches; cross-node P-D TTFT inflation ≤20% (B3 carried); mixed-workload goodput@SLO ≥ +25% vs colocated on the 70B tier; 4-node MoE group passes the C2 kill/recover drill as one endpoint | `kv_transport.py`, `pd.py`, `cluster.py`, `bench/kv_transfer_bench.py` |
| **F4 KV tiering** | DRAM offload tier in the engine (page swap on eviction, restore on radix hit), then NVMe (NIXL GDS target); global-pool decision doc gated on F2's measured cross-replica duplicate-prefix mass (buy Mooncake/LMCache vs build) | Restore-from-DRAM beats recompute above a measured prefix-length crossover (published); prefix-hit-rate gain on an agentic trace with TPOT p99 unregressed | `radix_kv.py` eviction, `pages.py`, F3 transport |
| **F5 Tenancy, SLO admission, autoscaling** | Tenant model (key→tenant, per-tenant token quotas/limits, usage-metering export); priority classes (interactive/batch) honored by replica scheduler admission; SLO-based early rejection (predict TTFT from queue/index state; shed or defer-to-batch); per-model replica-count autoscaler on goodput/queue/KV-utilization; weight pre-staging for ≤2 min scale-up; fleet-mix bandit (which model gets marginal GPUs) | 2× overload: interactive TTFT p99 SLO holds while batch absorbs slack; tenant isolation test (10× quota abuser cannot degrade another tenant's p99); 0→50 14B replicas in ≤5 min; metering reconciles with request logs to <0.1% | `middleware.py`, `batch/worker.py`, F1 dynamic membership, M4 bandit |

### Track P — Product surface (goal: G6)

| Phase | Deliverables | Exit gates (G6) | Builds on |
|---|---|---|---|
| **P-A Truthful API core** (CPU, start now) | Tokenizer-backed `Usage` + `prompt_tokens_details.cached_tokens` from radix KV + `stream_options.include_usage`; **HF Jinja chat templates** (`apply_chat_template`, per-model override in DeploymentSpec, tool schemas in-template) replacing the 14-line concatenator; `logprobs`/`top_logprobs`, `/v1/completions`, verified `n>1` streaming; `response_format: json_schema` wired to `structured.py` (not `extra_args` passthrough); bench fixes (auth headers, token-granularity TPOT, results to `bench/results/`) | Usage matches HF tokenizer count exactly; cached_tokens >0 on a repeated prefix; Llama/Qwen template output byte-matches HF reference; OpenAI SDK round-trips logprobs; 100% schema-valid structured outputs on a 50-schema suite | `app.py`, `protocol.py`, `chat_template.py`, `outputs.py`, `serving_bench.py` |
| **P-B Fugu-class product** | Chat-native streaming orchestrator (`Orchestrator.run_chat(messages, tools, stream=True)` — route fast, stream the synthesizer stage token-by-token, keep-alive status events); `usage.orchestration_input/output_tokens` + opt-in trace disclosure (the anti-Fugu transparency feature); Open WebUI as a compose service (custom UI only for the trace viewer); tiered `kairyu-auto` / `kairyu-auto-max`; tenancy v1 (key→tenant map, in-gateway token-bucket limits, append-only usage ledger, `/admin/usage`) | `kairyu-auto` TTFT ≤1.5× the underlying engine on the direct-route path; every auto request logs internal token spend; fresh user chats with `kairyu-auto` in one `docker compose up`; two keys get isolated 429s; ledger reconciles to <0.1% | `orchestrator.py`, `conductor.py`, `moa.py`, `budget.py`, `settings.py`, `middleware.py`, `batch/store.py` pattern |
| **P-C Competitive proof + completeness** | `bench/frontier_compare.py` (multi-target: Kairyu, Anthropic, OpenAI, DeepSeek; identical prompts; TTFT/TPOT/goodput/$-per-Mtok + small quality eval; nightly scoreboard with documented methodology); `/v1/responses` (Responses API subset — vLLM gap, Fugu parity); `/v1/embeddings` (+rerank) as a new backend kind; vision content-parts | One command → dated scoreboard vs ≥3 frontier APIs, runs unattended nightly; OpenAI SDK `responses.create` + a Codex-class agent work unmodified; Open WebUI RAG end-to-end on Kairyu alone | P-A bench, `batch/store.py` state pattern, `registry.py` |

## 5. Design decisions amended by this roadmap

Recorded as amendment entries in PROGRESS.md (2026-07-03); originals untouched per
progress-log rules.

| Decision | Verdict | Amendment |
|---|---|---|
| G2 §2 hardware/memory contract (8×H100 only, TP=2 base) | Too narrow — production spans A100+ incl. a PCIe-only fleet | Extended to capability profiles (§2): original arithmetic and A3–A5 stand **on the NVLink-HBM profile**; the PCIe-GDDR profile uses TP=1 as the 70B base and the E3 placement-crossover gate instead of A3–A5; SM80 gets W4A16/INT8 quant paths (G2 §7) |
| G2 §6 "MoE / expert parallelism" non-goal | Two of four target classes are MoE | Lifted into goal G4 |
| G2 §6 / G3 §4 no autoscaling/elasticity/dynamic registration | Untenable at thousands of GPUs | Lifted into goal G5 (F1/F5) |
| m6 D1 no-Ray, static ClusterSpec | No-Ray **stands** (k8s owns placement); staticness falls | `kairyu.launch` + registry (F1/F3); ClusterSpec stays static per coherence domain |
| ClusterSpec ≤2 nodes | Frontier MoE needs 4–8-node EP/PP domains | Cap raised to 8 (F3); "coherence domain ≠ fleet size" principle preserved |
| m7 D2 no Kubernetes | Its own revisit triggers (fleet >5–8 nodes, weekly-rolling toil) are all hit | k8s adopted as the **machine layer** only; model-aware control (routing, P:D, admission, autoscaling decisions) stays in Kairyu. llm-d not adopted wholesale — it would replace the differentiating L2 |
| m5 D4 / m7 D6 session-hash affinity, per-replica cache only | Blind to cross-request shared prefixes | Two-step: prefix-aware placement (caches stay per-replica, D6 letter intact) → tiering; global pool gated on F2 telemetry (D6's own revisit mechanism) |
| m7 D8 no OTel | A tracing consumer now exists (fleet-scale ops) | OTel spans in F1 |

## 6. Evidence rules

G1/G2 §8 rules carry forward verbatim: every number from a committed `bench/` script,
results + config in `bench/results/`, ≥3 runs, fixed seeds, open-loop sweeps for
saturation claims, pinned baseline versions. New: frontier-API comparisons (P-C1) must
publish prompt sets, sampling params, region/time-of-day, and provider-side caching
state alongside the numbers.

## 7. References

Hardware/SM120: [RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) ·
[TechPowerUp specs](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell-server.c4274) ·
[AWS G7e (P2P/GPUDirect RDMA)](https://aws.amazon.com/blogs/aws/announcing-amazon-ec2-g7e-instances-accelerated-by-nvidia-rtx-pro-6000-blackwell-server-edition-gpus/) ·
[rtx6kpro field guide](https://github.com/voipmonitor/rtx6kpro) ·
[CloudRift RTX6000-vs-DC-GPU bench](https://www.cloudrift.ai/blog/benchmarking-rtx6000-vs-datacenter-gpus) ·
[4×Blackwell PCIe P2P limits](https://oxmiq.ai/blog/4blackwell) ·
SM120 issues: [SGLang #24633 (MLA)](https://github.com/sgl-project/sglang/issues/24633),
[SGLang #21132 (EP FP8)](https://github.com/sgl-project/sglang/issues/21132),
[FlashInfer #2577 (NVFP4)](https://github.com/flashinfer-ai/flashinfer/issues/2577),
[vLLM #32109 (FP8 MoE)](https://github.com/vllm-project/vllm/issues/32109),
[SGLang #19603 (MTP on RTX PRO 6000)](https://github.com/sgl-project/sglang/issues/19603).
Quant/spec-decode: [NVFP4 intro](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) ·
[Kimi-K2 NVFP4 checkpoint](https://huggingface.co/nvidia/Kimi-K2.6-NVFP4) ·
[EAGLE-3](https://arxiv.org/pdf/2503.01840) · [SpecForge](https://arxiv.org/html/2603.18567v1) ·
[DeepEP](https://github.com/deepseek-ai/DeepEP) · [UCCL-EP](https://uccl-project.github.io/posts/uccl-ep/).
Control planes: [NVIDIA Dynamo 1.0](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) ·
[Dynamo KV-aware routing](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing) ·
[llm-d](https://github.com/llm-d/llm-d) ·
[SGLang Model Gateway](https://docs.sglang.io/advanced_features/sgl_model_gateway.html) ·
[Mooncake (FAST'25)](https://arxiv.org/abs/2407.00079) ·
[LMCache × Mooncake](https://blog.lmcache.ai/en/2025/05/08/lmcache-x-mooncake-unite-to-pioneer-kvcache-centric-llm-serving-system/) ·
[AIBrix](https://vllm.ai/blog/aibrix-release) ·
[vLLM KV events](https://docs.vllm.ai/en/stable/api/vllm/config/kv_events/).
Fugu/product: [sakana.ai/fugu](https://sakana.ai/fugu/) ·
[Fugu GA release](https://sakana.ai/fugu-release/) ·
[console.sakana.ai/pricing](https://console.sakana.ai/pricing) ·
[DevelopersIO first-touch](https://dev.classmethod.jp/en/articles/sakana-fugu-ga-first-touch/) ·
[Open WebUI vs LibreChat](https://docs.openwebui.com/alternatives/librechat/) ·
[vLLM Responses API gap #14721](https://github.com/vllm-project/vllm/issues/14721).
