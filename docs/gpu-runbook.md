# GPU-Day Runbook (H100/A100)

> **Hardware profiles (2026-07-03)**: production spans all NVIDIA GPUs from A100
> onward, in two shapes — NVLink-HBM nodes (A100/H100/H200/B200; this runbook's
> original assumption) and a PCIe-only RTX PRO 6000 Blackwell fleet (SM120, some
> units available now). This runbook applies as written on NVLink-HBM boxes. On the
> PCIe/SM120 fleet, the single-GPU sections (§0–§5, §9) apply with the caveats in
> `docs/roadmap.md` §2 (FA2-path kernels only, ~99 KB smem, Triton-first FP8,
> BF16 KV default); the multi-GPU sections (§6–§7) keep their correctness gates but
> the TP-scaling targets are replaced per G2 §7's 2026-07-03 amendments
> (DP-first/PP/EP strategy, placement-crossover report). On A100 (SM80), FP8 steps
> substitute W4A16/INT8. See `docs/roadmap.md` §4 Track E for the phase plan.

Purpose: everything CPU-verifiable is done (M8–M19: the full engine/model/quant/
distributed/transport stack, ~647 tests on main). This is the ordered,
command-level plan for the first GPU session. Est. scope: 2–4 focused days.

> Note (2026-07-04): §1/§3 below predate M8/M12/M13 — the tokenizer seam,
> `PagedModelRunner`, the `AttentionBackend`/`FlashInferBackend` seam, and quant
> loading are already delivered. The GPU-day work is enabling/tuning the real
> FlashInfer path behind the existing seam and running the perf gates, not
> building the seam from scratch. The batched-execution / CUDA-graph seam changes
> tracked in the repo-review Phase 6 land before the perf gates here.

## 0. Environment (30 min)

```bash
# Ubuntu 22.04+, CUDA 12.4+, driver >= 550
git clone https://github.com/ytworks/kairyu.git && cd kairyu
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev                    # the CPU suite must pass here first
uv run pytest                          # gate 0: green before any GPU work
uv sync --extra gpu --extra engine     # flashinfer/triton/nixl + torch/xgrammar
uv sync --extra bench                  # baselines (vllm/sglang installed alongside)
```

Record: GPU model, driver, CUDA, flashinfer/vllm/sglang versions → `bench/results/env-<date>.json`.

## 1. FlashInfer ModelRunner (day 1)

Enable the FlashInfer paged-attention path behind the delivered `AttentionBackend`
seam (`kairyu/engine/core/attention/`; `FlashInferBackend` is CPU-pinned against a
fake and mirrored in `tests/gpu/`). The real runner is `PagedModelRunner`
(`kairyu/engine/core/model_runner.py`, m12); the paging-math reference remains
`kairyu/engine/core/torch_runner.py`.

- Llama-3.1-8B weights (BF16 first) via safetensors; `detect_quantization()`
  (`quant_config.py`) selects the load path.
- Tokenizer: the HF tokenizer seam is delivered (m8, `kairyu/engine/tokenizer.py`);
  point `model_path=`/`tokenizer=` at the checkpoint dir.
- Gate 1 (correctness): greedy token parity vs HF transformers BF16 on 64 fixed prompts
  — run with overlap ON and OFF (`EngineCore` vs `OverlapEngineCore`).
- Gate 2: `uv run pytest` — the whole CPU suite still green with the GPU runner importable.

## 2. FP8 + first benches (day 2)

- FP8 W8A8 checkpoint (llm-compressor) through the compressed-tensors path.
- Gate 3: FP8 logprob tolerance vs BF16 (see m2-engine.md §2.5).
- First measurements (same script already smoke-tested vs mock):

```bash
uv run python examples/serve.py &                 # swap engines= to KairyuBackend(gpu)
uv run python bench/serving_bench.py --model kairyu --num-requests 256 --concurrency 128
uv run python bench/multiturn_prefix.py           # CPU workload/KV-manager diagnostic
# Real-engine TP4/8 direct+gateway A7 procedure: §6.
# baselines on the SAME box, same trace:
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching &
uv run python bench/serving_bench.py --base-url http://localhost:8000 --model <vllm>
python -m sglang.launch_server --model-path ... &
uv run python bench/serving_bench.py --base-url http://localhost:30000 --model <sglang>
```

Controls checklist (m2-engine.md §5) for performance claims: ≥3 runs, fixed
seeds, warmup excluded, open-loop arrival sweep, goodput SLO stated, CUDA-graph
handicap disclosed. Deterministic accounting gates such as A7 use one complete
fixed trace per declared cell. Results → `bench/results/<date>-<gpu>.json`; no
number leaves this file without a config next to it.

## 3. Remaining pre-GPU-deferred interfaces (with the runner, day 3)

From m2-engine.md §5 item 3 (shapes depend on FlashInfer metadata, hence deferred):
typed `StepInput`, per-step streaming out of `OverlapEngineCore` (KairyuBackend already
consumes a queue — wire it), incremental detokenizer, ZMQ/msgpack process split.
Then M3 items per m3 doc: EAGLE draft, CUDA graph capture (decode buckets), spec-decode
scheduler protocol §2.1.

## 4. Acceptance targets (goal)

| Criterion | Where measured |
|---|---|
| TTFT ≥20% better vs vLLM @128 conc (or p99 win at equal tput) | `serving_bench.py`, step 2 |
| KV hit >80% @50% shared prefix | `tp_kv_hit_g2_a7_bench.py` on the real native engine at TP4/8, direct and through the gateway; the existing 88.1% KV-manager result is diagnostic only |
| Router −40% cost @97% quality | needs serving traffic + judge; pipeline ready (`m4-router-learning.md`) |
| vLLM API compat pytest | already green; re-run with vLLM installed to un-skip cross-checks |

## 5. Human sign-off checklist (blocking, before implementation continues)

- [ ] `docs/design/m2-engine.md` (agent-reviewed, amendments applied)
- [ ] `docs/design/m3-spec-decode-and-graphs.md` (same)
- [ ] `docs/design/m4-router-learning.md` (same)
- [ ] Push to origin (nothing pushed yet)

## 6. M5 — multi-GPU day(s), 8×H100 (prereq: Gates 1–3 green; goal G2 gates A1–A10)

CPU half already merged: Communicator/FakeCommunicator, typed StepInput, TP plumbing
(`tensor_parallel_size` no longer a no-op), ReplicaPool + affinity + `record_replica`,
PDCoordinator + `resume_with_kv` + LocalKVHandoff — all tested. This section is the
GPU-only remainder (design m5 §4.2).

- 6.1 `NcclCommunicator`; dedicated non-rank driver process wired over shm/zmq;
  per-step driver budget ≤1 ms measured.
  *(Amended 2026-07-03: NVLS/symmetric-memory is NVSwitch-only — required on
  NVLink-HBM boxes as before, inapplicable on the PCIe fleet. On PCIe add instead:
  NCCL topology tuning for PCIe/EPYC hosts, a measured GPU↔GPU P2P matrix recorded
  in `bench/results/env-<date>.json`, and a MIG/vBIOS audit of the inventory. TP is
  demoted on the PCIe profile per G2 §7 2026-07-03 — run correctness gates A1/A2,
  then the placement-crossover sweep instead of A3–A5 there.)*
- 6.2 Sharded FP8 70B load (per-rank safetensors); FlashInfer paged attention under
  head-sharded KV; pool sized min-over-ranks (m5 D1).
- 6.3 Decode CUDA-graph capture per TP topology (A4 prerequisite, m5 §3).
- Gate A1: Llama-3.1-8B, 64 fixed prompts, 16-token full continuations:
  `bench/parity_tp.py --tp 1,2 --num-prompts 64 --max-new-tokens 16
  --model-path <checkpoint>` records TP1/2 with overlap ON/OFF. Run
  `bench/parity_hf.py` against one shared reference at `--tp 1` and `--tp 2`,
  then pass all four files to `bench/gate_a1.py`. The assembler retains the raw
  continuations/reference and fails unless both amended teacher-forced verdicts,
  overlap transparency, checkpoint/prompt identity, and clean-code provenance pass.
- Gate A2: use
  `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic@f50dbad2c84590ca17dc51e207c34321b65ff14b`
  with compressed-tensors FP8 W8A8 and BF16 model/KV dtype. First create one
  shared 64×16 HF reference with `bench/parity_hf.py --reference-only
  --reference-device-map auto --reference-batch-size 8`; then run the same
  command without `--reference-only` at `--tp 2`, `--tp 4`, and `--tp 8`.
  Always pass the pinned source through `--checkpoint-repo` and
  `--checkpoint-revision`. Assemble the four raw files with
  `bench/gate_a2.py --reference ... --teacher-tp2 ... --teacher-tp4 ...
  --teacher-tp8 ... --out ...`. The assembler requires 64×16 raw positions
  at every TP degree, all 15 complete safetensors SHA-256s, the measured
  reference noise floor, CUDA/NCCL/topology provenance, and one clean commit;
  it recomputes both amended HF-relative criteria and TP4/8-vs-TP2 instead of
  trusting reported verdicts. Closure evidence:
  `bench/results/g2-a2-llama33-70b-fp8-rtxpro6000-2026-07-27.json` (HF
  1005/1024; TP2/4/8 1006/1005/1006; zero substantive disagreements at every
  degree; all ten assembler checks pass).
- Gates A3–A5: `bench/serving_bench.py --sweep-tp 2,4,8` (TP=2 base in same file;
  conc-64 report-only point).
- Gate A6: vs pinned vLLM TP=4/8 (ShareGPT@128 + shared-prefix trace).
- Gate A7: run `bench/tp_kv_hit_g2_a7_bench.py` against Qwen3-32B at TP4
  and TP8, once through each replica's direct endpoint and once through its
  single-replica gateway. Assemble
  `bench/results/g2-a7-kv-hit-qwen3-32b-<gpu>-<date>/`; the verifier must
  recompute each strict >80% verdict from raw engine prompt-token usage and
  validate the fixed trace, configs, `/backends`, and physical topology.
- Gates A8–A9: `--dp-replicas 2` + `multiturn_prefix.py --replicas 2`; DP-vs-TP sweep.
- Gate A10: `bench/pd_mixed.py` (stream-copy KVHandoff on side stream; ≤5 ms p99).

## 7. M6 — 2-node day(s), IB/RoCE ≥400Gb/s (prereq: all M5 gates; goal G2 gates B1–B5)

CPU half already merged: ClusterSpec, KVTransport protocol + LocalFabric +
TCP-loopback, `bench/kv_transfer_bench.py` (CPU-runnable), `openai_backend` remote-
replica fixes (real SSE, pooled client, optional auth, token counts).

- 7.1 Record fabric truth: raw microbench via `kv_transfer_bench.py` →
  `bench/results/env-<date>.json` (measured, not nominal, link rate).
- Gate B1 (first — validates harness): 2-node DP via ReplicaPool over remote
  `openai` backends; goodput ≥1.85×; router p99 <10 ms incl. hop; cross-node affinity
  hit rate reported.
- Gate B2: transport bake-off (NCCL p2p + staging ring vs UCX/RDMA SGL) on the REAL
  sharded fragment layout; sustained ≥20 GB/s, ≤8 µs/token.
- Gate B3: inter-node P-D — execute-hooked chunk sends, layer-group streaming for the
  final chunk, P-D prefill chunk budget ≤1024; TTFT inflation ≤20%.
- Gate B4: PP=2 via `PipelinedModelRunner` (async submit/handle, two in-flight steps,
  full decode batches per stage); TPOT inflation ≤10%, throughput ≥1.6×, bubble
  fraction reported. Serial-commit correctness pass first (m3 §2.1 precedent).
- Gate B5: vLLM comparison for PP=2 and 2-node DP only (m6 §3 pins the set).

## 8. Human sign-off checklist for G2 (blocking)

- [ ] `docs/design/m5-intra-node-parallelism.md` (agent-reviewed, amendments applied)
- [ ] `docs/design/m6-inter-node-parallelism.md` (same)
- [ ] All M5 gates green, results pushed
- [ ] All M6 gates green, results pushed

## 9. M7 — production bring-up on real GPUs (prereq: §1–2 gates; §6/§7 for multi-GPU layouts)

The M7 CPU half (goal G3 gates C1–C7) is proven against mock replicas by
`scripts/compose_smoke.sh`. This section swaps in real engines — the topology,
image, and drill are unchanged.

- 9.1 Replica node: edit `deploy/compose/replica.yaml` — `backend: kairyu`
  (or `vllm`) with the model + `tensor_parallel_size` for this node; add the
  GPU device stanza to the compose service (`docs/deployment.md` §3).
- 9.2 Re-run the smoke drill against the real fleet:
  `scripts/compose_smoke.sh` end to end, including the kill/recover step
  (drains one GPU replica — schedule accordingly).
- 9.3 Repeat A7's formal gateway arm with per-session `user` ids using
  `bench/tp_kv_hit_g2_a7_bench.py`. Record the
  `kairyu_pool_decisions_total{reason="session_affinity"}` count beside the
  engine-derived prompt-token hit rate, but never use routing counters as
  cache-hit truth (G2 A7/A8 through the M7 path).
- 9.4 Rolling-update drill on real weights (`docs/deployment.md` §5), one
  replica at a time, gateway `/metrics` watched throughout — gate C7 on
  hardware.
- 9.5 F5a priority overload: use the Qwen3-32B example's one-replica gateway,
  tenant class mapping, and Batch API, then run
  `uv run python bench/priority_overload_gpu_bench.py --assert-gate
  --model-revision <40-hex-revision> --replica-image-digest <sha256:...>
  --gateway-image-digest <sha256:...>`.
  The harness performs an untimed warmup, calibrates steady-state capacity,
  offers interactive at 0.5x plus batch at 1.5x, checks the fixed 2 s
  interactive TTFT p99 SLO, proves batch completion plus residual backlog, and
  reconciles bounded gateway/replica class counters with native scheduler
  enqueue/admit/complete counters and queue gauges. The formal gate also
  requires a clean source tree and records the source/config/benchmark hashes,
  model revision, image digests, `/backends` topology, and GPU inventory. Save
  the raw JSON under `bench/results/`.
- 9.6 G5 F2c KV-aware TTFT: provision the external
  `kairyu-qwen3-32b_qwen3-32b` volume with the pinned Qwen3-32B revision, and
  make an image built from the clean expected source available locally by its
  immutable `repository@sha256:...` digest. Its
  `org.opencontainers.image.revision` label must equal that source commit.
  The Compose profile assigns TP2 replicas A0/A1/B0/B1 to host GPU pairs
  0–1/2–3/4–5/6–7 and exposes them on ports 8100–8103:

  ```bash
  export KAIRYU_F2C_IMAGE='repository@sha256:<64-lowercase-hex>'
  docker image inspect \
    --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
    "$KAIRYU_F2C_IMAGE"
  examples/qwen3-32b-multi-gpu/f2c-stack.sh config
  examples/qwen3-32b-multi-gpu/f2c-stack.sh up -d --wait
  curl -fsS http://127.0.0.1:8100/readyz
  curl -fsS http://127.0.0.1:8101/readyz
  curl -fsS http://127.0.0.1:8102/readyz
  curl -fsS http://127.0.0.1:8103/readyz
  ```

  Run only the changed-scope smoke needed to establish the harness. Smoke
  binds evidence integrity, routing, usage, prompt identity, valid output
  digests, and provenance; its performance metrics are diagnostic only. The
  cross-arm output-match count, total, and rate are also diagnostic in both
  profiles. Each trace family predeclares a canonical assistant continuation;
  after both turn-1 requests succeed, both arms use that frozen common
  transcript for turn 2 rather than either observed output. If that smoke
  generates tokens, recreate all four containers before the formal run so its
  per-round roots begin with empty independent caches. Run the one binding
  profile from a clean pinned commit; the harness records `nvidia-smi`
  inventory/topology and defaults to the four endpoints above:

  ```bash
  uv run python bench/kv_aware_ttft_f2c_bench.py \
    --profile formal \
    --output-dir bench/results/f2c-kv-aware-ttft-qwen3-32b-<date> \
    --model qwen3-32b \
    --model-revision <40-hex-model-revision> \
    --trace-namespace issue181-run1-<date> \
    --model-digest weights-rollup=<64-hex-sha256> \
    --expected-commit <40-hex-source-commit> \
    --assert-gate

  uv run python bench/kv_aware_ttft_f2c_bench.py \
    --verify-artifact bench/results/f2c-kv-aware-ttft-qwen3-32b-<date> \
    --assert-gate

  examples/qwen3-32b-multi-gpu/f2c-stack.sh down
  ```

  A trace namespace is single-use for a live cache set. Choose a new
  lower-case namespace for a technical retry, or recreate all four containers
  first; never replay a measured namespace against its warmed caches.
  Retain both generated JSON files. The offline command rehashes the raw JSONL
  and recomputes the trace, production routing decisions, exact engine
  `cached_tokens` rates, nearest-rank p95, crossover order statistics,
  goodput, frozen turn-2 transcript identity, diagnostic output-match rate,
  configuration, topology, and source/model provenance.

  Closure evidence (2026-07-29): the exact-source formal artifact at
  `bench/results/f2c-kv-aware-ttft-qwen3-32b-2026-07-29/` passed the online run
  and offline verification with all checks green, 512 binding requests, and
  zero failures. Control-to-candidate pooled TTFT p95 was 527.957623 ms →
  134.357747 ms; candidate/control ratios were 0.2544858548 pooled,
  0.2550841404 at the seventh ordered round, and 0.2530080045 by geometric
  mean. Cache rate was 0.4994645560 → 0.9843917326 with every round
  noninferior. Goodput ratios were 0.9999979014 pooled, 0.9998437390 at the
  second ordered round, and 0.9999978783 by geometric mean. Diagnostic output
  agreement was 239/256 (0.93359375), with maximum paired receipt skew
  5.182959 ms and schedule lateness 7.470463 ms.

  The retained artifact pins source
  `80b039b5d429c656871a480c2740740951b29b97`, image
  `kairyu-f2c@sha256:d2c01580964f461a3d3d2a02ced5303e69c681696d4a38179162084e1624121f`,
  raw SHA-256
  `4cfcdeba2b7473aa6c2b28409dbf21de23d775d9b08e971beed6bdab875abe64`,
  and trace SHA-256
  `51d188671432bf791c02d66d91e6a7d785eb2bd01f64e29a41a62e74f9957dad`.
