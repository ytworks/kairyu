# GPU-Day Runbook (H100/A100)

Purpose: everything CPU-verifiable is done (177 tests, main). This is the ordered,
command-level plan for the first GPU session. Est. scope: 2–4 focused days.

## 0. Environment (30 min)

```bash
# Ubuntu 22.04+, CUDA 12.4+, driver >= 550
git clone https://github.com/ytworks/kairyu.git && cd kairyu
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev                          # 177 CPU tests must pass here first
uv run pytest                          # gate 0: green before any GPU work
uv add flashinfer-python --group gpu   # pin exact version in pyproject
uv add vllm sglang --group bench       # baselines, pin versions
```

Record: GPU model, driver, CUDA, flashinfer/vllm/sglang versions → `bench/results/env-<date>.json`.

## 1. FlashInfer ModelRunner (day 1)

Replace `TorchPagedRunner`'s naive gather+matmul with FlashInfer paged attention behind
the same `ModelRunner` protocol (`kairyu/engine/core/engine_core.py`). Reference
implementation of the paging math: `kairyu/engine/core/torch_runner.py` (tested).

- Llama-3.1-8B weights (BF16 first) via safetensors; `detect_quantization()`
  (`quant_config.py`) selects the load path.
- Tokenizer: HF tokenizers; replace `KairyuBackend._tokenize` placeholder.
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
uv run python bench/multiturn_prefix.py           # now with real engine KV
# baselines on the SAME box, same trace:
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching &
uv run python bench/serving_bench.py --base-url http://localhost:8000 --model <vllm>
python -m sglang.launch_server --model-path ... &
uv run python bench/serving_bench.py --base-url http://localhost:30000 --model <sglang>
```

Controls checklist (m2-engine.md §5): ≥3 runs, fixed seeds, warmup excluded, open-loop
arrival sweep, goodput SLO stated, CUDA-graph handicap disclosed. Results →
`bench/results/<date>-<gpu>.json`; no number leaves this file without a config next to it.

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
| KV hit >80% @50% shared prefix | `multiturn_prefix.py` on real engine (88.1% already shown at KV-manager level) |
| Router −40% cost @97% quality | needs serving traffic + judge; pipeline ready (`m4-router-learning.md`) |
| vLLM API compat pytest | already green (177); re-run with vLLM installed to un-skip cross-checks |

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

- 6.1 `NcclCommunicator` (+NVLS/symmetric-memory — required, not optional, m5 D2);
  dedicated non-rank driver process wired over shm/zmq; per-step driver budget ≤1 ms
  measured.
- 6.2 Sharded FP8 70B load (per-rank safetensors); FlashInfer paged attention under
  head-sharded KV; pool sized min-over-ranks (m5 D1).
- 6.3 Decode CUDA-graph capture per TP topology (A4 prerequisite, m5 §3).
- Gate A1: `bench/parity_tp.py` 8B TP=2 vs TP=1, overlap ON/OFF.
- Gate A2: 70B TP=4/8 vs TP=2 match-rate ≥99% + logprob tolerance.
- Gates A3–A5: `bench/serving_bench.py --sweep-tp 2,4,8` (TP=2 base in same file;
  conc-64 report-only point).
- Gate A6: vs pinned vLLM TP=4/8 (ShareGPT@128 + shared-prefix trace).
- Gate A7: `bench/multiturn_prefix.py --tensor-parallel N` and `--pd`.
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
- 9.3 Measure the cache story under the gateway: `bench/multiturn_prefix.py`
  through the gateway with per-session `user` ids; report
  `kairyu_pool_decisions_total{reason="session_affinity"}` share and the
  engine-level radix hit rate side by side (G2 A7/A8 through the M7 path).
- 9.4 Rolling-update drill on real weights (`docs/deployment.md` §5), one
  replica at a time, gateway `/metrics` watched throughout — gate C7 on
  hardware.
- 9.5 Batch under load: submit a `/v1/batches` job while an interactive
  trace runs; verify interactive p99 is unaffected with the batch cap at its
  configured value (gate C4 on hardware).
