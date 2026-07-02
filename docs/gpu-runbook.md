# GPU-Day Runbook (H100/A100)

Purpose: everything CPU-verifiable is done (177 tests, main). This is the ordered,
command-level plan for the first GPU session. Est. scope: 2–4 focused days.

## 0. Environment (30 min)

```bash
# Ubuntu 22.04+, CUDA 12.4+, driver >= 550
git clone <repo> && cd rLLM
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
