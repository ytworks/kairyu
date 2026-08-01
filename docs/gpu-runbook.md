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

> **Benchmark command provenance:** Every top-level `bench/*.py` command in this
> runbook is a source-checkout-only compatibility entrypoint recorded in
> `kairyu/bench/entrypoints.toml`; it is not installed in the wheel. Run formal
> measurements from the exact clean commit whose wrapper and package-owned
> `kairyu.bench` helpers are recorded by the artifact. Before a GPU session, run
> `uv run --frozen kairyu bench entrypoints --check-repo .`. Result artifacts
> remain under the checkout-only `bench/results/` path so G2/G4/G5/G6 provenance
> and historical replay commands do not change.

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

### 3.1 Quantized EAGLE-3 draft checkpoint gate (#234)

Run the fixed public Qwen3-32B EAGLE-3 checkpoint against the local Qwen3-32B
target on one supported GPU. The operator converts eligible draft projections
offline into the validated dynamic compressed-tensors FP8 dialect, reloads each
generated checkpoint through the serving loader, and gives dense and quantized
arms identical target embeddings, auxiliary hidden states, KV contents, and
verification geometry:

```bash
docker run --rm --gpus device=0 --entrypoint python \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace \
  -e FLASHINFER_WORKSPACE_BASE=/host-tmp/kairyu-issue234-flashinfer \
  -e XDG_CACHE_HOME=/host-tmp/kairyu-issue234-cache \
  -e TRITON_CACHE_DIR=/host-tmp/kairyu-issue234-triton \
  -v "$PWD:/workspace:ro" \
  -v kairyu-qwen3-32b_qwen3-32b:/model:ro \
  -v /tmp/kairyu-issue234-eagle:/draft:ro \
  -v /tmp:/host-tmp \
  -w /workspace \
  kairyu-depth-ab:20260731-issue156 \
  /workspace/bench/draft_quant_qwen.py \
  --model-path /model/qwen3-32b \
  --draft-path /draft \
  --source-commit "$(git rev-parse HEAD)" \
  --output /host-tmp/issue-234-draft-quant-qwen3-32b-<gpu>-<date>.json \
  --assert-gate
```

The public draft weights must match SHA-256
`65fd3a6ad0f78f82e44e948e61096c914159912c31948bbfd90a73af5c973562`.
Do not add `CUDA_VISIBLE_DEVICES` inside this host: its managed NVIDIA runtime
already isolates Docker GPU 0, while the host venv's explicit override prevents
NVML initialization. The retained invocation uses FlashInfer from the pinned
image rather than the host venv's Torch fallback. The artifact full-hashes all
17 target weight shards and retains per-cycle draft/target timing, exact greedy
acceptance, target-correction validity, static and forward-peak CUDA memory,
source identity, generated packed-checkpoint hashes, and environment
provenance. Five prompts × three repeats balance all arm-order positions after
warming every draft and multi-token verification shape.
Quantization is opt-in: the selected arm must reduce memory while retaining at
least 95% of dense acceptance and standalone-cycle goodput, and dense absolute
acceptance must be at least 20%. Accepted prefixes must equal the independently
generated teacher trace. A target correction may differ across sequential and
multi-token verification shapes only when reciprocal selected-token
log-probabilities differ by at most 0.25 nat; the artifact retains exact-match
counts, both token IDs, all four cross-distribution log-probabilities, and the
individual/maximum deltas. The 0.25-nat bound is fixed, not operator-adjustable.
The cycle begins with context tensor
construction and ends after exact target correction; scheduler/HTTP serving is
outside this issue's construction/checkpoint scope. A measured slowdown is
reported and does not become an automatic serving default. Gate failures still
write the complete artifact before returning nonzero.

The retained source-`d8dbdba` run passes all gates at
`bench/results/issue-234-draft-quant-qwen3-32b-rtxpro6000-2026-07-31.json`
(SHA-256 `850191a039edd6e3ff5ae4bf974eadeef3227b3700b1747d281c595daad63c59`).
The selected arm is `fp8_dense_fc`: 33.33% acceptance (identical to dense),
44.94% less module memory, and 98.74% retained standalone-cycle goodput. Its
draft median is 1.2171× dense, so this is retained as an opt-in memory tradeoff
and not enabled as a performance default.

## 4. Acceptance targets (goal)

| Criterion | Where measured |
|---|---|
| A6 ShareGPT goodput ≥0.95× vLLM **and** TTFT p99 ≤ vLLM; shared-prefix TTFT p50 ≤0.80× vLLM | `g2_a6_vllm_bench.py`, §6 |
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
- 6.3a Decode page-table cache (#229): after CPU staleness/shape-churn tests,
  run the exact legacy/cache switch through the production Qwen3-32B TP8 graph
  path from a clean commit:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --frozen python \
    bench/decode_page_table_cache_qwen.py \
    --model-path /models/qwen3-32b \
    --output bench/results/issue-229-page-table-cache-qwen3-32b-tp8-<date>.json \
    --assert-gate
  uv run --frozen python bench/decode_page_table_cache_qwen.py \
    --verify bench/results/issue-229-page-table-cache-qwen3-32b-tp8-<date>.json \
    --model-path /models/qwen3-32b --assert-gate
  ```

  Every rank must report the same 31 decode builds and graph dispatches, zero
  fallback, exact output parity, bounded storage, fewer page-table device
  allocations/H2D elements/graph D2D elements, and zero live cache or graph
  owners after release. The retained OFF path must report its former outer +
  per-row device allocations and full rectangular graph copies. Wall/CUDA
  time, TPOT, throughput, medians, and MAD are integrity-protected diagnostics
  only; OS jitter never changes this verdict.
- TP sampling-ownership gate (#225, blocking before TP performance claims):
  rank 0 alone owns RNG/penalty/grammar/logprob state; every rank must adopt
  rank 0's fixed-layout device token before another forward. Run:

  ```bash
  uv run --frozen pytest --fail-on-skip tests/unit/test_tp_sampling_authority.py tests/unit/test_tp_worker.py -q
  uv run --frozen pytest --fail-on-skip tests/dist/test_distributed.py -k 'tp2_rank0_sampling_matrix or tp_structured_sampling' -v --no-cov
  CUDA_VISIBLE_DEVICES=0,1 uv run --frozen python scripts/test_prerequisites.py \
    --min-gpus 2 --require-nccl --require-module transformers
  CUDA_VISIBLE_DEVICES=0,1 uv run --frozen pytest --fail-on-skip -m gpu tests/gpu/test_tp_sampling_owner_nccl.py -v --no-cov
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --frozen python bench/tp_sampling_owner_bench.py \
    --world-size 8 --assert-gate \
    --output bench/results/issue-225-tp-sampling-comm-<gpu>-<date>.json
  uv run --frozen python bench/tp_sampling_owner_bench.py \
    --verify bench/results/issue-225-tp-sampling-comm-<gpu>-<date>.json --assert-gate
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --frozen python bench/tp_sampling_owner_qwen.py \
    --model-path /models/qwen3-32b --assert-gate \
    --output bench/results/issue-225-tp-sampling-qwen3-32b-<gpu>-<date>.json
  uv run --frozen python bench/tp_sampling_owner_qwen.py \
    --verify bench/results/issue-225-tp-sampling-qwen3-32b-<gpu>-<date>.json \
    --assert-gate
  ```

  The NCCL test injects a divergent follower-local future token, requires
  canonical adoption before the next decode, checks TP1/TP2 seeded parity, and
  relaunches after teardown. The microbenchmark compares the selected broadcast
  with a valid all-rank sampling+equality protocol at B=1/8/16 in rotating order.
  Across these gates, binding fields are exact logical bytes/collective counts,
  exact TP2 production packet/adopted-token equality, all raw trials, clean
  source/script provenance, and worst-rank CUDA-event broadcast steady-state
  p95 <1 ms over at least 256 aligned samples per batch. Complete
  p99/max/tail counts, relative latency,
  end-to-end throughput, and wall-clock timing remain diagnostic: a late host
  launch delays every tiny collective, so OS jitter must not decide protocol
  correctness.
  The Qwen3-32B gate then runs the production TP1 and TP8 runners once each over
  the same fixed greedy/seeded/filter/penalty/logprob request matrix. It must
  retain complete finite raw records and prove the actual topology: TP1 and TP8
  world sizes, complete rank sets, rank 0 as the sole owner with a sampler, and
  every TP8 follower passive without one. Free-running TP1/TP8 token equality is
  reported but is diagnostic only.

  Recorded 2026-07-30 on 8× NVIDIA RTX PRO 6000 Blackwell Server Edition:
  `issue-225-tp-sampling-comm-rtxpro6000-2026-07-30.json` passes all binding and
  replay checks with rank-0 broadcast p95 0.078400/0.070816/0.075648 ms at
  B=1/8/16 (256 samples per cell). The Qwen3-32B TP1/TP8 artifact retains 43
  tokens per degree, proves the complete owner/passive topology, and passes the
  0.25 compatibility bound with aligned maximum 0.148189 and first-divergence
  direct cross-selected maximum 0.101015. Free-running equality is 41/43 and is
  diagnostic, as specified.

  The binding distribution comparison uses only positions with an aligned
  common input prefix. Before the first token divergence, the common selected
  token's TP1/TP8 logprob delta must be at most 0.25. At the first divergence,
  the harness retains each full raw log-softmax in memory long enough to extract
  both selected tokens under both runs; both reciprocal logprob deltas must be
  at most 0.25. Public top-64 membership remains diagnostic because penalties
  are applied after that raw distribution. Positions after divergence remain
  raw evidence but are not compared as if they had the same prefix. The fixed
  texts are bound to the checkpoint's exact prompt-token IDs, and all
  harness/protocol sources must be clean and identical at measurement start and
  end. Exact adoption and next-decode use are binding in the real TP2 injection
  test; TP8 separately binds NCCL buffer overwrite and complete owner/follower
  topology. Both evidence files must independently replay against their
  recorded source commit before closure.
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
- Gate A6: prepare the pinned trace bundle with
  `.venv/bin/python bench/g2_a6_vllm_bench.py prepare-traces
  --tokenizer /tmp/kairyu-a7-qwen3-32b-tokenizer.json
  --dataset /tmp/ShareGPT_V3_unfiltered_cleaned_split.json
  --output /tmp/a6-traces/g2-a6-traces.json`, then run
  `.venv/bin/python bench/run_g2_a6_formal.py --dry-run` and
  `.venv/bin/python bench/run_g2_a6_formal.py` from a clean committed tree
  against
  Qwen3-32B and pinned stock vLLM 0.26.0 at TP4/8. Collect all four paired
  K/V, V/K, V/K, K/V fresh-server rounds for both the 128-request c128
  ShareGPT burst and the serialized 64-session × 8-turn shared-prefix trace.
  Sampling, batching, usable cache capacity (8,193 allocated on each arm;
  Kairyu scratch reservation 1, vLLM `BlockPool` null reservation 1; 8,192
  usable on each), model/tokenizer/dataset identities,
  full live-checkpoint hashes, GPU UUID/PCI assignments, HTTP runtime packages,
  launch argv, and resolved runtime backends must match the G2 2026-07-30
  amendment. Launch every cell by its preflight-resolved immutable image ID and
  retain the post-start container/image IDs, mounts, host resource config,
  imported engine hashes, exact Kairyu YAML bytes/parsed object, and raw vLLM
  startup messages. Each fresh server must retain the unique-prompt synchronized
  B=1/2/4/8/16 graph-size request warmup before formal requests. This validates
  request geometry and discloses configured/captured graph status; it is not
  evidence that each request actually dispatched through a graph. Collect
  the predeclared 1/2/4/8/16 rps open-loop shards separately as report-only
  saturation evidence. Assemble raw JSONL plus manifest and run the independent
  verifier with `--assert-gate`; no retry, failed row, missing usage, post-hoc
  SLO/rate selection, or trimmed sample can pass.
  The four exact per-repeat Kairyu/vLLM ratios and median/MAD are non-binding
  order/variability diagnostics. The verdict remains the three pooled A6
  thresholds. After a passing verification, publish the complete artifact
  before closing the issue:

  ```bash
  cp -a /tmp/g2-a6-formal/artifact \
    bench/results/g2-a6-vllm-qwen3-32b-<gpu>-<date>
  .venv/bin/python bench/g2_a6_vllm_bench.py verify \
    --artifact bench/results/g2-a6-vllm-qwen3-32b-<gpu>-<date> \
    --assert-gate
  sha256sum /tmp/g2-a6-formal/artifact/g2-a6-vllm-manifest.json \
    bench/results/g2-a6-vllm-qwen3-32b-<gpu>-<date>/g2-a6-vllm-manifest.json
  git add bench/results/g2-a6-vllm-qwen3-32b-<gpu>-<date>
  git diff --cached --check
  ```

  The two manifest hashes must be identical. Commit the verified raw JSONL and
  manifest together; do not publish only a summary.
- Gate A7: run `bench/tp_kv_hit_g2_a7_bench.py` against Qwen3-32B at TP4
  and TP8, once through each replica's direct endpoint and once through its
  single-replica gateway. Assemble
  `bench/results/g2-a7-kv-hit-qwen3-32b-<gpu>-<date>/`; the verifier must
  recompute each strict >80% verdict from raw engine prompt-token usage and
  validate the fixed trace, configs, `/backends`, and physical topology.
  Closure evidence:
  `bench/results/g2-a7-kv-hit-qwen3-32b-rtxpro6000-2026-07-29/`
  (TP4 direct/gateway 87.6725%/87.3531%; TP8 direct/gateway
  87.6725%/87.3531%; 512/512 successful requests per cell; all eight binding
  checks pass independent raw replay).
- Gate A8 / issue #158: use the pinned Qwen3-32B checkpoint and the dedicated
  two-replica TP4 stack. The caller owns the evidence directory; the wrapper
  never removes it. Resolve and record the immutable image ID, then launch the
  two TP4 replicas on GPU partitions 0–3 and 4–7 plus the L2 gateway:

  ```bash
  export KAIRYU_A8_IMAGE='<image-reference>'
  export KAIRYU_A8_IMAGE_ID='sha256:<64-lowercase-hex>'
  export KAIRYU_A8_RUN_DIR='bench/results/g2-a8-dp-qwen3-32b-<gpu>-<date>'
  examples/qwen3-32b-multi-gpu/a8-stack.sh "$KAIRYU_A8_RUN_DIR" \
    up -d --force-recreate --wait
  curl -fsS http://127.0.0.1:8200/readyz
  curl -fsS http://127.0.0.1:8201/readyz
  curl -fsS http://127.0.0.1:8202/readyz
  ```

  Run the formal operator from that clean committed tree. The operator verifies
  the immutable A6 trace bundle, retained A7 checkpoint evidence, image ID,
  eight-GPU topology, container layout, Kairyu backend metadata, and the live
  model volume's 17 weight shards/index/tokenizer/config before sending formal
  traffic:

  ```bash
  export KAIRYU_A8_TRACE_BUNDLE='/tmp/a6-traces/g2-a6-traces.json'
  export KAIRYU_A8_EXPECTED_COMMIT='<40-lowercase-hex-commit>'
  uv run --frozen python bench/dp_scaling_g2_a8_bench.py measure \
    --single-url http://127.0.0.1:8200/v1 \
    --dp-url http://127.0.0.1:8202/v1 \
    --trace-bundle "$KAIRYU_A8_TRACE_BUNDLE" \
    --gateway-placement-log "$KAIRYU_A8_RUN_DIR/placements.jsonl" \
    --output-dir "$KAIRYU_A8_RUN_DIR" \
    --image-id "$KAIRYU_A8_IMAGE_ID" \
    --expected-commit "$KAIRYU_A8_EXPECTED_COMMIT" \
    --assert-gate
  uv run --frozen python bench/dp_scaling_g2_a8_bench.py verify \
    --artifact "$KAIRYU_A8_RUN_DIR" --assert-gate
  uv run --frozen python bench/dp_scaling_g2_a8_bench.py replay \
    --artifact "$KAIRYU_A8_RUN_DIR" --assert-gate
  ```

  The operator's formal config predeclares the complete open-loop arrival-rate
  grid, the TTFT SLO, and at least three paired seed-0 repeats. It alternates
  the single/DP arm order and excludes warmup. The single baseline addresses
  replica 0 directly; the DP arm addresses the gateway. Do not use A6's
  synchronized c128 burst as an A8 binding substitute. The operator computes
  each repeat's peak SLO-goodput from the same predeclared grid and binds the
  median paired DP/single ratio to ≥1.9.

  The gateway writes
  `$KAIRYU_A8_RUN_DIR/placements.jsonl`. Every formal DP response must correlate
  to exactly one placement row by request ID. Recompute nearest-rank p99 over
  the recorded outer-ingress-to-selection intervals and require <10 ms. Run the
  fixed 64-session × 8-turn trace (512-token shared prefix, 128 new prompt
  tokens per turn, one output token) against replica 0 and through the
  two-replica gateway, retaining stable `X-Session-ID` values and all response
  usage. Recompute KV hit rates only from engine-originated
  `prompt_tokens_details.cached_tokens / prompt_tokens`; require the
  gateway/single ratio to be ≥0.90. Routing counters and placement reasons are
  supporting evidence only.

  The `verify` and independent `replay` commands must retain and hash the config,
  every raw performance sample, every correlated placement row, every per-turn
  cache-usage record, physical topology, pinned checkpoint identity, immutable
  image ID, and clean source identity. Missing/failed requests, incomplete
  placement correlation, non-finite values, a changed predeclared grid, or any
  failed threshold must fail closed. Offline replay and CPU tests cannot report
  an A8 threshold PASS. Any closure despite a failed threshold must be an
  explicit accepted deviation recorded in the goal and progress log; it must
  not rewrite the artifact or operator verdict.

  Retained result (2026-07-31):
  `bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31/` records all 2,992
  retry-free successful requests and 1,496 correlated placement rows. The
  three paired peak-goodput ratios are 1.9988×, 1.7342×, and 1.7993×, for a
  1.7993× median against the original 1.9× threshold. Router p99 is 3.723 ms
  and affinity-cache retention is 99.53%. Verify and raw-only replay both
  reproduce `passed: false` solely for goodput. The product owner accepted the
  measured median for issue #158 closure as a documented deviation; the
  original 1.9× verdict remains unchanged. SHA-256:
  raw `b637489302a9b818a0c34790c4059946994ff6070f76ac9e1bc7d128bbbd803f`,
  manifest
  `da439f153c04d05178ddf96c489aca7fb1cc270ba982736c1bc98197730e3946`,
  placements
  `76ca1a5f709ccf8492238bdcc4193776f94fdb7a88a40a7e970a517db30270c3`.

  Stop only the scoped A8 project after evidence is safely retained:

  ```bash
  examples/qwen3-32b-multi-gpu/a8-stack.sh "$KAIRYU_A8_RUN_DIR" down
  ```

- Gate A9: retain the separate DP=2×TP4 vs TP8 goodput/TPOT crossover across
  the arrival sweep; A9 remains report-only and is not implied by an A8 pass.
  The formal operator uses the post-SSE-fix A8 comparator retained at
  `bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/` only after
  replaying its raw/manifest/placement SHA-256s. All 2,992 requests succeeded
  without retry, all 1,496 placements correlated, and raw-only replay retained
  only the accepted 1.9× deviation (measured median 1.7711×). A future fully
  passing A8 rerun is also valid; every other false check is rejected. A9 then
  measures the missing TP8 arm through a one-replica gateway. The TP8 engine
  and gateway use the exact A8 image plus a read-only source tree that
  `a9-tp8-stack.sh` materializes directly from `git archive` of A8 commit
  `86d4922`, matching A8's runtime mount rather than the image's older baked
  package or the current checkout. Preflight verifies the fixed archive digest,
  all 199 runtime files, and the delegated A8 helper blob. Runtime-source and
  full-checkpoint attestations both repeat after traffic.
  This keeps both arms on the same runtime while allowing the current clean
  commit to own the A9 operator and evidence schema.

  ```bash
  export KAIRYU_A9_IMAGE='sha256:2c73b577b5213264493c6aeba24c1f6318214d20bf6df7780158fd1ef2c70a50'
  export KAIRYU_A9_IMAGE_ID="$KAIRYU_A9_IMAGE"
  export KAIRYU_A9_RUN_DIR='bench/results/g2-a9-dp-tp-qwen3-32b-rtxpro6000-<date>'
  examples/qwen3-32b-multi-gpu/a9-tp8-stack.sh "$KAIRYU_A9_RUN_DIR" \
    up -d --force-recreate --wait
  curl -fsS http://127.0.0.1:8300/readyz
  curl -fsS http://127.0.0.1:8301/readyz

  export KAIRYU_A9_EXPECTED_COMMIT='<40-lowercase-hex-commit>'
  uv run --frozen python bench/g2_a9_dp_tp_crossover_bench.py measure \
    --engine-url http://127.0.0.1:8300/v1 \
    --gateway-url http://127.0.0.1:8301/v1 \
    --trace-bundle /tmp/a6-traces/g2-a6-traces.json \
    --gateway-placement-log "$KAIRYU_A9_RUN_DIR/placements.jsonl" \
    --output-dir "$KAIRYU_A9_RUN_DIR" \
    --image-id "$KAIRYU_A9_IMAGE_ID" \
    --expected-commit "$KAIRYU_A9_EXPECTED_COMMIT" \
    --assert-gate
  uv run --frozen python bench/g2_a9_dp_tp_crossover_bench.py verify \
    --artifact "$KAIRYU_A9_RUN_DIR" --assert-gate
  uv run --frozen python bench/g2_a9_dp_tp_crossover_bench.py replay \
    --artifact "$KAIRYU_A9_RUN_DIR" --assert-gate
  ```

  The fixed TP8 work is 24 excluded warmups plus 960 measured requests:
  three seed-0 repeats of 64 requests at each predeclared
  `{4,8,16,32,64}` request/s rate, with 32 exact output tokens and no retry.
  TP8 reuses the retained DP arm's one-token namespace, making the full request
  bytes and token IDs identical. Per-engine cache/scheduler/graph settings are
  matched: each engine has 8,192 usable KV pages, 16 sequence slots, 1,024
  batched tokens, graph batch 16, and pipeline depth 5. Consequently the two
  independent DP replicas have twice TP8's aggregate configured capacity; the
  artifact records this explicitly and observed in-flight work at every rate.
  A9 is therefore a production topology comparison, not an equal-aggregate-
  capacity kernel microbenchmark.
  Goodput and TTFT use the A8 definitions. TPOT is explicitly versioned as
  `stream-terminal-token-v1`:
  `(stream terminal - first content) / (completion_tokens - 1)`. This matches
  `bench/serving_bench.py` and includes the final usage/finish/`[DONE]` tail;
  it must not be relabeled as `frontier_compare.py`'s last-content TPOT.
  Report every rate and repeat, median/MAD, observed maximum in-flight count,
  all measured ordering transitions with both arms' observed concurrency at
  each bracket, an explicit `no_order_transition_in_measured_range` when none
  exists, and either the first DP-noninferior arrival-rate bracket or
  `none_in_measured_range`. There is no performance threshold and no
  interpolation.

  Retained result (2026-07-31):
  `bench/results/g2-a9-dp-tp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`
  passed all 14 report-integrity checks. The TP8 arm recorded 984/984 exact,
  retry-free successful requests and 984/984 correlated placements; both
  `verify --assert-gate` and raw-only `replay --assert-gate` pass.

  | Offered req/s | DP goodput | TP8 goodput | DP/TP8 | DP/TP8 TTFT p50 (ms) | DP/TP8 TPOT p50 (ms/token) |
  |---:|---:|---:|---:|---:|---:|
  | 4 | 3.884 | 3.902 | 0.995× | 181.0 / 174.9 | 20.3 / 21.1 |
  | 8 | 7.383 | 7.313 | 1.010× | 190.2 / 339.9 | 25.4 / 50.5 |
  | 16 | 12.948 | 8.994 | 1.440× | 350.8 / 1,258.3 | 49.5 / 42.4 |
  | 32 | 16.042 | 11.707 | 1.370× | 507.7 / 1,304.1 | 42.3 / 30.0 |
  | 64 | 19.612 | 12.440 | 1.577× | 617.6 / 1,627.8 | 33.4 / 29.2 |

  The measured goodput ordering changes between 4 and 8 offered req/s; DP is
  first noninferior at 8 req/s, with median observed in-flight counts 9 for DP
  and 17 for TP8. No interpolated crossover is claimed. TP8's lower TPOT at
  16–64 req/s does not offset DP's higher goodput and lower TTFT under load;
  these are reported as distinct metrics. SHA-256: TP8 raw
  `6c6d11deeb5735ecf37de48663e553a88a90e606196bb432111fe70772d4a664`,
  manifest
  `799813e087bd9aca9d62c45a995e7eed6471534f18fdaa09fca0666412786bdc`,
  placements
  `0276059cd9822ce9e046fce088102b372a324689d620375ce23e13829f4ec020`.

  Stop only this scoped project after retaining and replaying the artifact:

  ```bash
  examples/qwen3-32b-multi-gpu/a9-tp8-stack.sh "$KAIRYU_A9_RUN_DIR" down
  ```

  The PCIe per-class DP×1/PP=2/TP=2 placement report is not fabricated by
  relabeling `pipeline_depth`: production stage-sharded PP serving does not
  yet exist. It remains a separate dependency on real PP wiring rather than
  weakening or blocking this A9 DP-versus-TP report.
- Gate A10 / issue #223: cross-device P-D handoff uses a dedicated source-copy
  stream on the prefill GPU and a destination dependency/completion stream on
  the decode GPU. Select two distinct GPUs for which CUDA peer access is
  available; the deferred production constructor fails rather than falling
  back when the pair cannot perform P2P. A stream wait is not completion:
  `PDCoordinator` may publish destination KV and release the source lease only
  after the retained physical event returns true from `event.query()`.
  `pd_defer_handoff=false` is the serialized control and synchronizes both
  transfer streams before publication. If completion or cleanup ownership is
  lost, the engine must retain/poison the allocation and fail closed rather
  than publish or reuse partial KV.

  Run the formal Qwen3-32B production-path comparison from a clean committed
  tree:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 uv run python bench/pd_overlap_qwen.py \
    --model-path <qwen3-32b-checkpoint> \
    --prefill-device 0 --decode-device 1 \
    --output bench/results/issue-223-pd-overlap-qwen3-32b-<gpu>-<date>.json \
    --assert-gate
  uv run python bench/pd_overlap_qwen.py \
    --verify bench/results/issue-223-pd-overlap-qwen3-32b-<gpu>-<date>.json \
    --assert-gate
  ```

  The artifact binds the full Qwen3-32B checkpoint and tokenizer provenance,
  clean source hashes, the real distinct role devices, production coordinator,
  pool and stream-provider topology, complete raw outputs, and exact
  blocking/deferred token and text parity. Wall time, TTFT, completion latency,
  throughput, and ratios between the two fixed-order conditions are
  **diagnostic only**: OS scheduling, runtime launch, clock, thermal, and
  shared-host jitter must not turn a short performance sample into a correctness
  gate. The historical `bench/pd_mixed.py` ≤5 ms p99 target is therefore not a
  closure criterion for #223.

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

- 9.7 Batched speculative target verification (#215): run from a clean commit
  containing the implementation and formal runner. The gate uses the exact
  reviewed Qwen3-32B checkpoint on all eight GPUs, exercises eager and CUDA
  Graph runners with warmed OFF/ON ABBA/BAAB order, and independently counts
  every rank's requests, target positions, model calls, FlashInfer plan/run
  calls, graph dispatches, and eager fallbacks. End-to-end and fixed-geometry
  target parity are binding. Wall time, TPOT, throughput, CUDA time,
  median/MAD, and cross-kernel BF16 KV distance are diagnostic only.

  The same run releases the TP launchers and loads the full checkpoint once on
  GPU 0 to compare flattened tensor decode with native ragged prefill over the
  identical 8-request/32-target-row page, position, and write geometry. It
  requires identical selected tokens and proves that both paths overwrite all
  poisoned target KV slots with finite values; it records the faster strategy
  without turning host or device timing into a correctness gate.

  ```bash
  uv run python bench/batched_spec_verify_qwen.py \
    --model-path /path/to/qwen3-32b \
    --tp 8 \
    --output \
      bench/results/issue-215-batched-spec-verify-qwen3-32b-tp8-<date>.json \
    --assert-gate

  uv run python bench/batched_spec_verify_qwen.py \
    --verify \
      bench/results/issue-215-batched-spec-verify-qwen3-32b-tp8-<date>.json \
    --model-path /path/to/qwen3-32b \
    --assert-gate
  ```

  A skipped CUDA/FlashInfer/NCCL cell, missing rank, wrong target width, graph
  fallback, source/checkpoint drift, unwritten/non-finite target KV slot, or
  stored-verdict mismatch is a failure. It is never recorded as a pass merely
  because the local environment cannot execute it.

  Closure evidence (2026-07-30): the clean-source artifact at
  `bench/results/issue-215-batched-spec-verify-qwen3-32b-tp8-2026-07-30.json`
  passes all 16 live checks and all 21 independent replay checks against
  implementation commit
  `5dc7dd1591b37b8685fa7c6df6a94c8b8481574d`. The fixed 32-position cell
  reports 32 sequential model calls versus one grouped call on every rank and
  zero graph fallback. Diagnostic median throughput was 12.95 → 260.16 token/s
  in eager and 12.40 → 387.48 token/s with CUDA Graph. The full-model strategy
  comparison selected flattened decode at 59.004 ms versus 75.561 ms native
  ragged CUDA time (0.7809x), with 32/32 selected tokens equal and all poisoned
  target KV slots overwritten with finite values. Artifact SHA-256:
  `58ec81de2a7a1e89dbf7ced1d6f223039037c80be34cf743c1f4939aa10e66c9`.

- 9.8 Global KV pool decision replay (#189, G5 F4c): do not rerun F2a or
  F2c. Their retained raw artifacts already contain the distinct evidence
  needed for this decision. Replay the F2a seed/placement sequence to
  reconstruct logical replica residency, then independently replay F2c's
  paired real-engine token usage:

  ```bash
  uv run --frozen python bench/global_kv_pool_decision.py \
    --verify-artifact \
    bench/results/f4c-global-kv-pool-decision-2026-07-31.json \
    --assert-gate
  ```

  The replay must report session-HRW's 997 redundant family copies (3,988
  logical prefix chunk-copies and 513,809 family-copy/request-steps), zero
  redundant copies under prefix-aware placement, and 319,696 matched avoided
  recomputed model tokens in F2c with no pairwise cache regression. It must
  retain F2c's 0.999998 goodput ratio and classify the remaining 1.5608%
  uncached prompt-token fraction only as a gross upper bound: the trace cannot
  distinguish novel suffixes from compatible remote-resident misses.

  F4c is a decision gate, not a new GPU performance run. The reviewed m7 D6
  amendment keeps per-replica RadixKV plus F2 routing, completes F4a/F4b
  first, and chooses Mooncake Store behind a separate bounded global-KV
  object-store adapter only if the predeclared future exact-event telemetry
  trigger fires across three consecutive 10,000-request windows. This does
  not expand the existing `KVTransport` seam. The artifact claims logical
  copies and token recomputation only; it must never be reported as measured
  physical KV bytes or byte-seconds.

- 9.8a Native pinned-DRAM KV crossover (#187, G5 F4a): this gate is closed by
  retained schema-v2 Qwen3-32B TP4/TP8 evidence; the procedure below remains
  the reproducibility contract. Use one standalone clean clone of the exact
  implementation commit and build one immutable image from it; an older image
  is invalid. TP4 must exit successfully and release all GPUs before TP8
  starts. Do not overlap the runs, reuse a tag that resolves to another image,
  substitute a dirty bind mount, or invent a crossover when the retained
  samples have no stable passing suffix.

  The completed schema-v1 FlashInfer TP4/TP8 collection passed correctness and
  provenance checks but produced no stable restore-winning suffix. Subsequent
  review found that its cold-recompute arm split the final prompt token into an
  additional model invocation, biasing the comparison toward restore. The v1
  raw is diagnostic only: it cannot seed runtime policy, be replayed as v2, or
  supply any sample to this run.

  Fresh raw and profiles use schema v2. Every rank must report the exact
  `cuda-pinned-dram-fragment-major-torch-copy-v1` backend: one NUMA-attested
  pinned owner viewed as `[fragment, slot, bytes]`, with one copy submission
  per fragment and jointly contiguous extent. The raw, profile, and runtime
  policy bind that versioned backend identity and fail closed on an old or
  different implementation. Restore includes checksum validation, H2D, the
  final prompt-token query, and sampling. Cold recompute uses production
  prompt chunks through the final prompt token and samples its hidden state
  directly, without an additional one-token model call.

  The model volume contains a `qwen3-32b` subdirectory, so mount that exact
  volume subpath. Run as the host UID/GID, leave Docker's default hostname in
  place so it remains the container-ID prefix, and give the container only a
  read-only source clone, read-only full-container-ID metadata directory, and
  its own read-write output directory. `memlock=-1`, host IPC, Docker's
  default bridge network, loopback Gloo, and an explicit writable Triton cache
  and FlashInfer workspace are part of the retained runtime contract. Without
  the vendor-specific FlashInfer workspace, a non-root container falls back to
  Torch when the library tries to create `/.cache`; the formal operator rejects
  that fallback before loading the model. Host networking is invalid here: it
  replaces the container-ID hostname with the host name and breaks the provenance
  binding. `Dockerfile.cuda` keeps the revision label in a metadata-only final
  stage so changing the source commit does not invalidate the large dependency
  payload.

  ```bash
  set -euo pipefail
  COMMIT=$(git rev-parse HEAD)
  SESSION_ROOT=$(mktemp -d /tmp/kairyu-f4a.XXXXXX)
  SOURCE_ROOT="$SESSION_ROOT/source"
  RUN_ROOT="$SESSION_ROOT/evidence"
  HOST_UID=$(id -u)
  HOST_GID=$(id -g)

  git clone --no-hardlinks . "$SOURCE_ROOT"
  git -C "$SOURCE_ROOT" switch --detach "$COMMIT"
  test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"

  IMAGE_REF="kairyu-f4a:$COMMIT"
  docker build --build-arg "KAIRYU_VCS_REF=$COMMIT" \
    -t "$IMAGE_REF" -f "$SOURCE_ROOT/Dockerfile.cuda" "$SOURCE_ROOT"
  IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
  test "$(docker image inspect --format \
    '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
    "$IMAGE_ID")" = "$COMMIT"
  mapfile -t REPO_DIGESTS < <(docker image inspect --format \
    '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE_ID")
  REPO_ARGS=()
  for digest in "${REPO_DIGESTS[@]}"; do
    if test -n "$digest"; then
      REPO_ARGS+=(--repo-digest "$digest")
    fi
  done

  mkdir -p "$RUN_ROOT/tp4" "$RUN_ROOT/tp8" \
    "$RUN_ROOT/metadata/tp4" "$RUN_ROOT/metadata/tp8" \
    "$RUN_ROOT/artifact"
  docker image inspect "$IMAGE_ID" > "$RUN_ROOT/metadata/image-inspect.json"

  TP4_CID=$(docker create \
    --name "kairyu-f4a-tp4-${COMMIT:0:12}" \
    --gpus '"device=0,1,2,3"' \
    --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    --ulimit memlock=-1:-1 --ipc=host \
    -e GLOO_SOCKET_IFNAME=lo \
    -e TRITON_CACHE_DIR=/evidence/triton-cache \
    -e FLASHINFER_WORKSPACE_BASE=/evidence/flashinfer-workspace \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount \
      type=volume,src=kairyu-qwen3-32b_qwen3-32b,dst=/models/qwen3-32b,volume-subpath=qwen3-32b,readonly \
    --mount "type=bind,src=$RUN_ROOT/tp4,dst=/evidence" \
    --mount \
      "type=bind,src=$RUN_ROOT/metadata/tp4,dst=/run/kairyu-meta,readonly" \
    "$IMAGE_ID" \
    bench/dram_kv_tier_qwen.py run \
      --tp 4 --model-path /models/qwen3-32b \
      --output /evidence/dram-kv-tier-qwen3-32b-tp4-raw.jsonl \
      --image-id "$IMAGE_ID" \
      --container-id-file /run/kairyu-meta/container-id \
      "${REPO_ARGS[@]}" --max-num-batched-tokens 2048 --timeout-s 1800)
  test "${#TP4_CID}" -eq 64
  test "$(docker inspect --format '{{.Config.Hostname}}' "$TP4_CID")" \
    = "${TP4_CID:0:12}"
  test "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$TP4_CID")" \
    != host
  printf '%s\n' "$TP4_CID" > "$RUN_ROOT/metadata/tp4/container-id"
  docker inspect "$TP4_CID" \
    > "$RUN_ROOT/metadata/tp4/container-inspect-created.json"
  docker start -a "$TP4_CID" || {
    docker inspect "$TP4_CID" \
      > "$RUN_ROOT/metadata/tp4/container-inspect-exited.json"
    exit 1
  }
  test "$(docker inspect --format '{{.State.ExitCode}}' "$TP4_CID")" = 0
  test "$(docker inspect --format '{{.State.Running}}' "$TP4_CID")" = false
  test -s "$RUN_ROOT/tp4/dram-kv-tier-qwen3-32b-tp4-raw.jsonl"
  docker inspect "$TP4_CID" \
    > "$RUN_ROOT/metadata/tp4/container-inspect-exited.json"

  TP8_CID=$(docker create \
    --name "kairyu-f4a-tp8-${COMMIT:0:12}" \
    --gpus '"device=0,1,2,3,4,5,6,7"' \
    --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    --ulimit memlock=-1:-1 --ipc=host \
    -e GLOO_SOCKET_IFNAME=lo \
    -e TRITON_CACHE_DIR=/evidence/triton-cache \
    -e FLASHINFER_WORKSPACE_BASE=/evidence/flashinfer-workspace \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount \
      type=volume,src=kairyu-qwen3-32b_qwen3-32b,dst=/models/qwen3-32b,volume-subpath=qwen3-32b,readonly \
    --mount "type=bind,src=$RUN_ROOT/tp8,dst=/evidence" \
    --mount \
      "type=bind,src=$RUN_ROOT/metadata/tp8,dst=/run/kairyu-meta,readonly" \
    "$IMAGE_ID" \
    bench/dram_kv_tier_qwen.py run \
      --tp 8 --model-path /models/qwen3-32b \
      --output /evidence/dram-kv-tier-qwen3-32b-tp8-raw.jsonl \
      --image-id "$IMAGE_ID" \
      --container-id-file /run/kairyu-meta/container-id \
      "${REPO_ARGS[@]}" --max-num-batched-tokens 2048 --timeout-s 1800)
  test "${#TP8_CID}" -eq 64
  test "$(docker inspect --format '{{.Config.Hostname}}' "$TP8_CID")" \
    = "${TP8_CID:0:12}"
  test "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$TP8_CID")" \
    != host
  printf '%s\n' "$TP8_CID" > "$RUN_ROOT/metadata/tp8/container-id"
  docker inspect "$TP8_CID" \
    > "$RUN_ROOT/metadata/tp8/container-inspect-created.json"
  docker start -a "$TP8_CID" || {
    docker inspect "$TP8_CID" \
      > "$RUN_ROOT/metadata/tp8/container-inspect-exited.json"
    exit 1
  }
  test "$(docker inspect --format '{{.State.ExitCode}}' "$TP8_CID")" = 0
  test "$(docker inspect --format '{{.State.Running}}' "$TP8_CID")" = false
  test -s "$RUN_ROOT/tp8/dram-kv-tier-qwen3-32b-tp8-raw.jsonl"
  docker inspect "$TP8_CID" \
    > "$RUN_ROOT/metadata/tp8/container-inspect-exited.json"
  ```

  Keep both stopped containers and their inspect records until assembly and
  verification finish. Run the offline steps from the same clean source and
  immutable image; they need no GPU. `assemble --assert-gate` is allowed to
  fail when the measured grid has no stable restore-winning suffix—that is an
  honest open gate, not an environment skip or a value to fill manually.

  ```bash
  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence" \
    "$IMAGE_ID" bench/dram_kv_tier_qwen.py assemble \
      --tp4-raw /evidence/tp4/dram-kv-tier-qwen3-32b-tp4-raw.jsonl \
      --tp8-raw /evidence/tp8/dram-kv-tier-qwen3-32b-tp8-raw.jsonl \
      --output-dir /evidence/artifact --assert-gate

  cp -a "$RUN_ROOT/metadata" "$RUN_ROOT/artifact/container-metadata"

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence" \
    "$IMAGE_ID" bench/dram_kv_tier_qwen.py verify \
      --artifact-dir /evidence/artifact --assert-gate

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence,readonly" \
    "$IMAGE_ID" bench/dram_kv_tier_qwen.py replay \
      --tp4-raw /evidence/artifact/dram-kv-tier-qwen3-32b-tp4-raw.jsonl \
      --tp8-raw /evidence/artifact/dram-kv-tier-qwen3-32b-tp8-raw.jsonl \
      --assert-gate
  ```

  Retain both raw shards, the manifest, both runtime profiles, image inspect,
  full container IDs, and created/exited inspect records together before
  publishing a crossover. Only then copy the complete artifact into
  `bench/results/g5-f4a-dram-kv-tier-qwen3-32b-<gpu>-<date>/` and update F4a.
  After that retained copy independently verifies, the two named measurement
  containers may be removed explicitly. Never use a broad Docker prune as
  part of this procedure. Build and collect only after focused, unit, and real
  GPU validation is green and the complete implementation is committed; the
  detached source clone must resolve to that exact commit.

  The binding 2026-08-01 run used clean commit
  `edd535f7018695fc03c479a86fbd690174cca5ef`, immutable image
  `sha256:25543ae9cbc9d2e80f1b4be2193d138486adb91c89a02cdbd0be0e62a1cc67be`,
  and separate default-bridge containers
  `69254c1819b1a0203cbfc0f74d3ae4e2eb4cdda6c6949332cdf7b2fec7ef9c9b`
  (TP4) and
  `c6b5acde2e366e08fa4d983dff629c2bc13e119f8eada8e00a233f3ba2226c8b`
  (TP8). TP4's stable passing suffix begins at 1,024 tokens: 512 failed
  honestly at a 1.021531 median paired ratio and 2/9 restore wins, then 1,024
  passed at 0.975449 and 8/9 and every larger cell passed. TP8 passed all ten
  cells from 16 through 8,192 tokens with 9/9 restore wins, placing its
  crossover at or below the measured 16-token lower bound. Assembly,
  verification from the retained repository copy, and independent raw replay
  all passed. The artifact is retained at
  `bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`;
  its manifest SHA-256 is
  `0680333d06bf6d06ea91fbd12ef5b88732c936d1446a06d631674dcb15946fd6`,
  and the TP4/TP8 raw SHA-256 values are
  `609ff6bb1b951a7d4f70bb5948e1f9dcd68e55b0523e2993effa76a3b750cf01`
  and
  `2947d27dcb51a227ec9e21c72a4e435e693dbb675ff683632dd285cfc86d6611`.

- 9.8b Agentic DRAM-tier on/off trace (#188, G5 F4b): run four fresh Qwen3-32B
  TP4 arms from one detached clean source commit and the exact immutable
  runtime image used to calibrate the retained F4a TP4 profile. Cohort A is
  physical GPUs 0--3 and cohort B is physical GPUs 4--7. Execute, in order,
  round-0 off on A, round-0 on on B, round-1 on on A, and round-1 off on B.
  Every container must exit and release its four GPUs before the next starts;
  never run any pair concurrently. This AB/BA arm order and cohort swap
  controls for both elapsed-time order and a fixed four-GPU cohort effect.

  The trace and gates are fixed before measurement: 16 sessions by eight
  turns, a 2,048-token fleet-shared prefix, 512 new session-history tokens per
  turn, and exactly 32 output tokens per request. The tier-on arms must use the
  retained F4a Qwen3-32B TP4 profile, without editing it; tier-off arms must not
  receive a profile. Assembly independently recomputes pooled and per-cohort
  engine cached-token rates, request-level nearest-rank
  `stream-terminal-token-v1` TPOT p99, and the pooled plus cohort-geometric-mean
  1.10 noninferiority gates. Synchronous `EngineLoop` step-boundary evidence
  must show no post-first-token offload/restore counter movement and must see
  the free-page decrease from output 16 to 17. That page-boundary control makes
  the decode-critical-path exclusion check non-vacuous.

  Cross-arm free-running output equality is diagnostic, not a correctness
  gate. A different cache-prefill shape can resolve a BF16 near-tie to another
  greedy token; positions after that first token then no longer share an input
  prefix. Instead, after the timing run, execute a timing-nonbinding quality
  companion on cohort A: one fresh tier-off container followed by one fresh
  tier-on container, both requesting top-64 logprobs. Before the first
  divergence, selected-token logprobs must differ by at most 0.25 nat. At the
  first divergence, both selected tokens must appear in the reciprocal top-64
  view and each cross-arm difference must be at most 0.25 nat. Never compare
  positions after divergence. The quality runs must exactly reproduce the
  corresponding cohort-A performance arm's prompts, outputs, cached-token
  usage, runtime, and tier behavior, and retain full Docker lifecycle records.
  Logprob-enabled timestamps are never substituted for the performance TPOT.

  Commit the implementation and focused tests before measurement. Reuse the
  content-addressed F4a image below; rebuilding the same package versions is
  not equivalent because the retained profile binds the compiled attention
  runtime identity. The image's OCI revision records the source used to build
  that calibrated runtime, while the read-only source mount independently
  records and supplies the Kairyu code under measurement. Do not use a dirty
  source bind mount, a moving image tag, another image or revision, an F4a
  profile from a different checkpoint/TP/layout/backend, host networking, or a
  shared arm cache. The model checkpoint, container provenance, measured
  source, calibrated image, and F4a profile identities must remain stable
  across all four arms.

  ```bash
  set -euo pipefail
  COMMIT=$(git rev-parse HEAD)
  SESSION_ROOT=$(mktemp -d /tmp/kairyu-f4b.XXXXXX)
  SOURCE_ROOT="$SESSION_ROOT/source"
  RUN_ROOT="$SESSION_ROOT/evidence"
  HOST_UID=$(id -u)
  HOST_GID=$(id -g)

  git clone --no-hardlinks . "$SOURCE_ROOT"
  git -C "$SOURCE_ROOT" switch --detach "$COMMIT"
  test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"

  F4A_PROFILE=/workspace/kairyu/bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/dram-kv-tier-qwen3-32b-tp4-profile.json
  test -s "$SOURCE_ROOT/${F4A_PROFILE#/workspace/kairyu/}"

  IMAGE_ID=sha256:25543ae9cbc9d2e80f1b4be2193d138486adb91c89a02cdbd0be0e62a1cc67be
  IMAGE_REVISION=edd535f7018695fc03c479a86fbd690174cca5ef
  test "$(docker image inspect --format '{{.Id}}' "$IMAGE_ID")" = "$IMAGE_ID"
  test "$(docker image inspect --format \
    '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
    "$IMAGE_ID")" = "$IMAGE_REVISION"
  mapfile -t REPO_DIGESTS < <(docker image inspect --format \
    '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE_ID")
  REPO_ARGS=()
  for digest in "${REPO_DIGESTS[@]}"; do
    if test -n "$digest"; then
      REPO_ARGS+=(--repo-digest "$digest")
    fi
  done

  mkdir -p "$RUN_ROOT/metadata" "$RUN_ROOT/artifact"
  docker image inspect "$IMAGE_ID" > "$RUN_ROOT/metadata/image-inspect.json"

  run_arm() {
    label=$1
    arm=$2
    round=$3
    cohort=$4
    physical_gpus=$5
    mkdir -p \
      "$RUN_ROOT/$label/triton-cache" \
      "$RUN_ROOT/$label/flashinfer-workspace" \
      "$RUN_ROOT/metadata/$label"

    PROFILE_ARGS=()
    if test "$arm" = on; then
      PROFILE_ARGS=(--profile "$F4A_PROFILE")
    fi

    CID=$(docker create \
      --name "kairyu-f4b-${label}-${COMMIT:0:12}" \
      --gpus "\"device=$physical_gpus\"" \
      --user "$HOST_UID:$HOST_GID" \
      --entrypoint /app/.venv/bin/python \
      --ulimit memlock=-1:-1 --ipc=host \
      -e GLOO_SOCKET_IFNAME=lo \
      -e TRITON_CACHE_DIR=/evidence/triton-cache \
      -e FLASHINFER_WORKSPACE_BASE=/evidence/flashinfer-workspace \
      -w /workspace/kairyu \
      --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
      --mount \
        type=volume,src=kairyu-qwen3-32b_qwen3-32b,dst=/models/qwen3-32b,volume-subpath=qwen3-32b,readonly \
      --mount "type=bind,src=$RUN_ROOT/$label,dst=/evidence" \
      --mount \
        "type=bind,src=$RUN_ROOT/metadata/$label,dst=/run/kairyu-meta,readonly" \
      "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py run \
        --arm "$arm" --round "$round" --cohort "$cohort" \
        --model-path /models/qwen3-32b \
        "${PROFILE_ARGS[@]}" \
        --output "/evidence/$label-raw.jsonl" \
        --image-id "$IMAGE_ID" \
        --container-id-file /run/kairyu-meta/container-id \
        "${REPO_ARGS[@]}")
    test "${#CID}" -eq 64
    test "$(docker inspect --format '{{.Config.Hostname}}' "$CID")" \
      = "${CID:0:12}"
    test "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$CID")" \
      != host
    printf '%s\n' "$CID" > "$RUN_ROOT/metadata/$label/container-id"
    docker inspect "$CID" \
      > "$RUN_ROOT/metadata/$label/container-inspect-created.json"
    docker start -a "$CID" || {
      docker inspect "$CID" \
        > "$RUN_ROOT/metadata/$label/container-inspect-exited.json"
      exit 1
    }
    test "$(docker inspect --format '{{.State.ExitCode}}' "$CID")" = 0
    test "$(docker inspect --format '{{.State.Running}}' "$CID")" = false
    test -s "$RUN_ROOT/$label/$label-raw.jsonl"
    docker inspect "$CID" \
      > "$RUN_ROOT/metadata/$label/container-inspect-exited.json"
  }

  run_arm round0-off off 0 A 0,1,2,3
  run_arm round0-on  on  0 B 4,5,6,7
  run_arm round1-on  on  1 A 0,1,2,3
  run_arm round1-off off 1 B 4,5,6,7
  ```

  Keep all four stopped containers until sealing and replay finish. Assemble,
  verify, and independently replay inside the same immutable image; these
  offline commands use no GPU. A failed gate is a measured failure, not an
  environment skip, and must not be edited into a pass.

  ```bash
  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence" \
    "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py assemble \
      --raw /evidence/round0-off/round0-off-raw.jsonl \
      --raw /evidence/round0-on/round0-on-raw.jsonl \
      --raw /evidence/round1-on/round1-on-raw.jsonl \
      --raw /evidence/round1-off/round1-off-raw.jsonl \
      --metadata-dir /evidence/metadata \
      --output-dir /evidence/artifact --assert-gate

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence,readonly" \
    "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py verify \
      --artifact /evidence/artifact --assert-gate

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence,readonly" \
    "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py replay \
      --raw /evidence/artifact/agentic-kv-tier-f4b-raw.jsonl \
      --assert-gate
  ```

  Run and seal the quality companion only after the performance artifact
  passes. A previously retained performance artifact may be extended from a
  later clean harness commit without rerunning its timing arms only when the
  verifier proves that the engine and every other bound runtime/source file
  are byte-identical; only this benchmark script may differ. The quality
  outputs and cached-token usage must still exactly reproduce the matching
  parent arms.

  ```bash
  PERFORMANCE_ARTIFACT="$RUN_ROOT/artifact"
  PERFORMANCE_RAW="$PERFORMANCE_ARTIFACT/agentic-kv-tier-f4b-raw.jsonl"
  PERFORMANCE_RAW_SHA256=$(sha256sum "$PERFORMANCE_RAW" | awk '{print $1}')
  QUALITY_METADATA="$RUN_ROOT/quality-metadata"
  mkdir -p "$QUALITY_METADATA"
  docker image inspect "$IMAGE_ID" > "$QUALITY_METADATA/image-inspect.json"

  run_quality_arm() {
    label=$1
    arm=$2
    mkdir -p \
      "$RUN_ROOT/$label/triton-cache" \
      "$RUN_ROOT/$label/flashinfer-workspace" \
      "$QUALITY_METADATA/$label"
    PROFILE_ARGS=()
    if test "$arm" = on; then
      PROFILE_ARGS=(--profile "$F4A_PROFILE")
    fi

    CID=$(docker create \
      --name "kairyu-f4b-${label}-${COMMIT:0:12}" \
      --gpus '"device=0,1,2,3"' \
      --user "$HOST_UID:$HOST_GID" \
      --entrypoint /app/.venv/bin/python \
      --ulimit memlock=-1:-1 --ipc=host \
      -e GLOO_SOCKET_IFNAME=lo \
      -e TRITON_CACHE_DIR=/evidence/triton-cache \
      -e FLASHINFER_WORKSPACE_BASE=/evidence/flashinfer-workspace \
      -w /workspace/kairyu \
      --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
      --mount \
        type=volume,src=kairyu-qwen3-32b_qwen3-32b,dst=/models/qwen3-32b,volume-subpath=qwen3-32b,readonly \
      --mount "type=bind,src=$RUN_ROOT/$label,dst=/evidence" \
      --mount \
        "type=bind,src=$QUALITY_METADATA/$label,dst=/run/kairyu-meta,readonly" \
      "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py quality-run \
        --arm "$arm" --cohort A \
        --performance-raw-sha256 "$PERFORMANCE_RAW_SHA256" \
        --model-path /models/qwen3-32b \
        "${PROFILE_ARGS[@]}" \
        --output "/evidence/$label-raw.jsonl" \
        --image-id "$IMAGE_ID" \
        --container-id-file /run/kairyu-meta/container-id \
        "${REPO_ARGS[@]}")
    test "${#CID}" -eq 64
    printf '%s\n' "$CID" > "$QUALITY_METADATA/$label/container-id"
    docker inspect "$CID" \
      > "$QUALITY_METADATA/$label/container-inspect-created.json"
    docker start -a "$CID" || {
      docker inspect "$CID" \
        > "$QUALITY_METADATA/$label/container-inspect-exited.json"
      exit 1
    }
    test "$(docker inspect --format '{{.State.ExitCode}}' "$CID")" = 0
    test "$(docker inspect --format '{{.State.Running}}' "$CID")" = false
    test -s "$RUN_ROOT/$label/$label-raw.jsonl"
    docker inspect "$CID" \
      > "$QUALITY_METADATA/$label/container-inspect-exited.json"
  }

  run_quality_arm quality-off off
  run_quality_arm quality-on on

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence" \
    "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py seal-quality \
      --performance-artifact /evidence/artifact \
      --quality-raw /evidence/quality-off/quality-off-raw.jsonl \
      --quality-raw /evidence/quality-on/quality-on-raw.jsonl \
      --quality-metadata-dir /evidence/quality-metadata \
      --output-dir /evidence/artifact-with-quality --assert-gate

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence,readonly" \
    "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py verify-quality \
      --artifact /evidence/artifact-with-quality --assert-gate

  docker run --rm --user "$HOST_UID:$HOST_GID" \
    --entrypoint /app/.venv/bin/python \
    -w /workspace/kairyu \
    --mount "type=bind,src=$SOURCE_ROOT,dst=/workspace/kairyu,readonly" \
    --mount "type=bind,src=$RUN_ROOT,dst=/evidence,readonly" \
    "$IMAGE_ID" bench/agentic_kv_tier_f4b_bench.py replay-quality \
      --performance-raw \
        /evidence/artifact-with-quality/agentic-kv-tier-f4b-raw.jsonl \
      --quality-raw \
        /evidence/artifact-with-quality/agentic-kv-tier-f4b-quality-raw.jsonl \
      --assert-gate
  ```

  Assembly validates the image inspect and all four full-container-ID plus
  created/exited inspect sets, embeds normalized lifecycle descriptors in the
  combined raw, and copies those exact files to
  `artifact/container-metadata/`; do not copy or rewrite them manually.
  `replay` validates the embedded lifecycle from the combined raw, while
  `verify` additionally rehashes every retained metadata file against those
  descriptors. Quality sealing performs the same checks for its two
  containers and retains them under `quality-container-metadata/`. Retain the
  six input raw JSONL files, both sealed combined raw files and manifests, F4a
  profile identity, and both complete metadata trees together. Copy
  `artifact-with-quality/` to
  `bench/results/g5-f4b-agentic-kv-tier-qwen3-32b-<gpu>-<date>/` only after the
  retained copy itself passes `verify-quality` and `replay-quality`. Then
  remove only the six named measurement containers; never use a broad Docker
  prune here.

  The retained closure artifact is
  `bench/results/g5-f4b-agentic-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`.
  Its unchanged performance raw SHA-256 is
  `63ee8bc89bd19e331354419e1f1511428b90b60c2785f71c218c7df113637e05`;
  its quality raw SHA-256 is
  `aaa989e790aa3c857048fd4ab6d6be9e47a1e05c61f136774a8a182f07492109`.
  The pooled prefix-hit-rate gain was 12.4397 percentage points, the pooled
  TPOT p99 ratio was 1.03721, and the cohort-ratio geometric mean was 1.04488.
  Quality replay compared 3,968 aligned positions with a 0.195256-nat maximum
  difference and four reciprocal first-divergence pairs with a 0.213124-nat
  maximum difference. Seal, retained-copy verification, and independent
  replay all passed.

- 9.9 G4 E-KV FP8 cache correctness bake (#170): run from a clean commit on
  one SM120 GPU with the exact reviewed Qwen3-32B checkpoint. The current
  retained production image predates E4M3 attention AOT and contains no
  `nvcc`, so this one closure run must honestly use the host's pinned venv,
  CUDA 13.3 compiler, and already compiled FlashInfer 0.6.14 cache mounted
  into that container. This is not evidence that the old image was rebuilt.
  The source-JIT/compiler paths, compiler hash/version, FlashInfer shared
  object hashes, image ID, source commit/files, checkpoint shards, GPU, and
  environment are all retained and must be stable across the run.

  ```bash
  IMAGE=kairyu-qwen3-32b-kairyu:latest
  IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
  RESULT_ROOT=$(mktemp -d /tmp/kairyu-g4-ekv.XXXXXX)
  mkdir -p \
    "$RESULT_ROOT/artifact" \
    "$RESULT_ROOT/home" \
    "$RESULT_ROOT/torch-extensions" \
    "$RESULT_ROOT/triton"
  test -d /tmp/kairyu-issue170-flashinfer/.cache/flashinfer

  docker run --rm --gpus device=7 \
    --entrypoint /runtime/.venv/bin/python \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e KAIRYU_ATTENTION_BACKEND=flashinfer \
    -e CUDA_HOME=/usr/local/cuda \
    -e HOME=/runtime-state/home \
    -e TORCH_EXTENSIONS_DIR=/runtime-state/torch-extensions \
    -e TRITON_CACHE_DIR=/runtime-state/triton \
    -e FLASHINFER_WORKSPACE_BASE=/tmp/kairyu-issue170-flashinfer \
    -e GIT_CONFIG_COUNT=1 \
    -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/workspace \
    -e PATH=/usr/local/cuda/bin:/runtime/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -v "$PWD:/workspace:ro" \
    -v "$PWD/.venv:/runtime/.venv:ro" \
    -v /usr/local/cuda-13.3:/usr/local/cuda:ro \
    -v /usr/bin/x86_64-linux-gnu-gcc-13:/usr/bin/gcc:ro \
    -v /usr/bin/x86_64-linux-gnu-g++-13:/usr/bin/g++:ro \
    -v /usr/libexec/gcc/x86_64-linux-gnu/13:/usr/libexec/gcc/x86_64-linux-gnu/13:ro \
    -v /usr/lib/gcc/x86_64-linux-gnu/13:/usr/lib/gcc/x86_64-linux-gnu/13:ro \
    -v /usr/include/c++/13:/usr/include/c++/13:ro \
    -v /usr/include/x86_64-linux-gnu/c++/13:/usr/include/x86_64-linux-gnu/c++/13:ro \
    -v /usr/bin/git:/usr/bin/git:ro \
    -v /usr/lib/git-core:/usr/lib/git-core:ro \
    -v kairyu-qwen3-32b_qwen3-32b:/models:ro \
    -v "$RESULT_ROOT:/runtime-state" \
    -v /tmp/kairyu-issue170-flashinfer:/tmp/kairyu-issue170-flashinfer \
    -w /workspace \
    "$IMAGE" \
    bench/fp8_kv_g4_ekv_bench.py measure \
    --model-path /models/qwen3-32b \
    --image-id "$IMAGE_ID" \
    --output-dir /runtime-state/artifact \
    --assert-gate

  .venv/bin/python bench/fp8_kv_g4_ekv_bench.py verify \
    --artifact "$RESULT_ROOT/artifact" --assert-gate
  .venv/bin/python bench/fp8_kv_g4_ekv_bench.py replay \
    --artifact "$RESULT_ROOT/artifact" --assert-gate
  ```

  The binding workload is one prompt at each exact length 8,192, 16,384,
  and 32,768, native FlashInfer ragged prefill in 2,048-token chunks, then
  16 greedy tokens through tensor decode. BF16 and explicit unit-scale E4M3
  KV arms must have exact output IDs/stopping, finite common-prefix selected
  logprobs with maximum absolute delta 0.25, complete
  BF16-input/SATFINITE E4M3 write
  audits, bit-exact stored bytes, and dequantization error no greater than
  `max(abs(input)/16, 2^-10)`. Fixed samples from 8 layers × 16 positions per
  prompt require NRMSE at most 0.05 and cosine at least 0.99. Timing is
  diagnostic only. A process/model/runtime error still writes raw JSONL and a
  manifest with verdict FAIL; a missing GPU/runtime is not a skip or PASS.

  Recorded 2026-07-31: the unit-scale candidate produced a retained **FAIL**.
  The environment, source, checkpoint, and FlashInfer shared-object identity
  were stable. All 7,522,091,008 audited K/V values passed the SATFINITE write
  contract, but 8K and 32K outputs diverged, the common-prefix selected-logprob
  maximum was 0.3099/0.4518/0.2656 at 8K/16K/32K, and cache NRMSE reached
  0.1047. Keep public FP8 KV disabled; do not relabel this as PASS. Evidence:
  `bench/results/g4-ekv-fp8-kv-qwen3-32b-sm120-fail-2026-07-31/`
  (raw SHA-256
  `f759fa3308f90f70c26e04e51ebf82515a2891d1b183ef3a8bbfa67acbada305`;
  manifest SHA-256
  `4c213ebfb7376755e98bddb7c16ad508ee8ac56ef88fd69c57971e78c2224a64`).

- 9.10 G6 P-C4 Open WebUI image chat (#203): run from a clean implementation
  commit on all eight GPUs. The opt-in overlay pins stock vLLM, the
  Qwen3-VL-32B-Instruct revision, TP8, one image, video disabled, and the exact
  Qwen processor pixel range. Keep its model-cache volume between runs:

  ```bash
  WEBUI_VLM_KEEP_STACK=1 ./scripts/webui_vlm_smoke.sh
  ```

  The gate must pass every phase: deterministic RED and BLUE images produce
  the corresponding different answers through Kairyu; unary and SSE responses
  include positive exact processor usage inside the 8,192-token complete
  context; a metadata-service URL returns `400 invalid_image` without upstream
  media I/O; and the pinned Playwright browser creates/signs into Open WebUI,
  selects `qwen3-vl-32b`, uploads the RED PNG through the real file input and
  `/api/v1/files/` ownership path, sends its file ID/type/url in the normal
  chat request, and renders RED. Any missing runtime/GPU, omitted usage,
  incorrect semantic answer, upload shortcut, skip, or upstream error is a
  failure rather than a closure result.

  The retained clean-commit run on 2026-07-31 passed all phases on
  8× RTX PRO 6000. See
  `bench/results/issue-203-vlm-image-chat-qwen3-vl-32b-tp8-2026-07-31.json`.
