# M12 Design: Real Model Zoo (Dense) — Llama/Qwen, Multi-Layer Paged KV, Parity

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (2-reviewer agent panel with
empirical verification against transformers 5.12.1 / torch 2.12.1, 2026-07-03;
amendments applied — §6 is the binding record for items not rewritten inline).
Milestone: M12 (roadmap Track E1 local-complete half; the first of the
GPU-path-implemented-locally milestones M12–M19)
Date: 2026-07-03
Depends on: M8 (Sampler seam, StepOutput protocol, CheckpointReader,
quant detection, tokenizer seam), M9 (chat templates for real prompts).
Consumed by: M13 (AttentionBackend seam extraction), M14 (QuantizedLinear
loader hook), M15 (MoE/MLA archs), M16 (per-rank sharded load), M17 (hidden
states for EAGLE).

## 1. Goal

Replace the toy compute with real dense decoder architectures, implemented in
pure torch and **CPU-runnable at tiny sizes** so correctness is locally
provable: config-driven Llama-3.x / Qwen2/2.5 / Qwen3 models, a multi-layer
paged KV pool, and a `PagedModelRunner` behind the existing `ModelRunner`
protocol. The parity gate: byte-exact greedy agreement with `transformers`
through the FULL engine (chunked prefill, radix reuse, paging, sampler).

## 2. Key design decisions and rationale

### D1 — `ModelConfig`: pure config.json parsing, one dataclass for the dense family

`kairyu/models/config.py`: frozen `ModelConfig` parsed from an HF
`config.json` dict (same style as `quant_config.py`): `architecture`
(`architectures[0]`), `hidden_size`, `num_hidden_layers`,
`num_attention_heads`, `num_key_value_heads`, `head_dim` (explicit field when
present — Qwen3 decouples it from hidden/heads — else derived),
`intermediate_size`, `vocab_size`, `rms_norm_eps`, `rope_theta`,
`rope_scaling` (llama3 type: factor, low/high_freq_factor,
original_max_position_embeddings), `attention_bias` (Qwen2 qkv bias),
`qk_norm` (implied by `Qwen3ForCausalLM`), `tie_word_embeddings`,
`torch_dtype`. Unknown architectures fail fast with the supported list.

### D2 — One decoder implementation, config-switched; HF module names

`kairyu/models/layers.py` (RMSNorm, RoPE, SwiGLU MLP) +
`kairyu/models/attention.py` (GQA over the paged pool) +
`kairyu/models/llama.py` (`DenseDecoder` covering Llama-3.x, Qwen2/2.5
(attention_bias), Qwen3 (per-head q/k RMSNorm)):

- **RoPE uses HF's `rotate_half` convention exactly** (not interleaved) —
  checkpoints assume it; this is the single most load-bearing numeric detail.
  Llama-3 rope_scaling implemented per HF's `_compute_llama3_parameters`.
- Module tree mirrors HF names (`model.embed_tokens`,
  `model.layers.N.self_attn.{q,k,v,o}_proj`, `.mlp.{gate,up,down}_proj`,
  `.input_layernorm`, `.post_attention_layernorm`, `model.norm`, `lm_head`)
  so the loader is a 1:1 name map with zero renaming tables.
- Forward contract (paged, incremental): `forward_tokens(token_ids,
  positions, kv_pool, page_table, seq_len) -> hidden` writes each layer's KV
  into the pool at the given positions and attends over `seq_len` entries via
  the page table; **hidden is returned for ALL chunk positions,
  post-final-norm** (M17's EAGLE tap); `logits(hidden) -> [*, vocab]` stays a
  separate method. Attention lives in ONE function with the M13-extractable
  signature `attention(query, kv_pool, layer, page_table, seq_len,
  chunk_start)` (it takes the pool + page table, not pre-gathered K/V).
  **Chunk mask (amended, verified)**: `is_causal=True` is WRONG for a chunk
  over a cached prefix (top-left aligned; measured maxdiff 2.22) — build the
  boolean mask `mask[i, j] = (j <= chunk_start + i)` of shape
  `[chunk_len, seq_len]` and pass it as `attn_mask` (SDPA `enable_gqa=True`
  verified working on CPU torch 2.12).
- **RoPE numerics contract (amended, verified)**: cos/sin computed once per
  forward at model level from the chunk's absolute positions, in fp32
  regardless of model dtype, then cast; `inv_freq` from the exact
  `arange(0, dim, 2, int64)/dim` expression; HF `attention_scaling` equals
  1.0 for `default` and `llama3` rope types (no-op here; load-bearing only if
  yarn/longrope archs land later). Qwen3 q/k RMSNorm is per-head over
  head_dim, applied BEFORE RoPE.
- Computation dtype: fp32 on CPU for parity tests; the module honors
  `torch_dtype` for load (bf16 weights cast to fp32 compute on CPU — recorded;
  GPU keeps native dtype).

### D3 — `PagedKVPool`: layer-major, one tensor per K and V

`kairyu/engine/core/kv_pool.py`: `k`/`v` shaped
`[num_layers, num_pages, page_size, num_kv_heads, head_dim]` — layer-major so
M18's `KVTransport` fragments (per-layer × per-shard) slice contiguously.
API: `write(layer, page_table, positions, keys, values)` and
`gather(layer, page_table, seq_len) -> (K, V)`; a `bytes_per_token` property
(KV budget arithmetic); dtype/device constructor args.
**Sizing (amended)**: `RadixKVCache` gains public `num_pages`/`page_size`
properties (retiring the scheduler's `getattr(_page_size)` smell); the pool
is built via `PagedKVPool.for_cache(cache, config, ...)` and
`PagedModelRunner.__init__` validates pool/cache agreement, failing fast —
a silent mismatch is wrong-slot corruption. The same page ids index every
layer; `RadixKVCache`/`Scheduler` stay untouched.

### D4 — `PagedModelRunner`: the real ModelRunner; TinyAttentionLM stays the oracle

`kairyu/engine/core/model_runner.py`: implements the m8 `ModelRunner`
protocol (`execute(scheduled, states) -> StepOutput`) over `DenseDecoder` +
`PagedKVPool` + the m8 `Sampler`:

- Reads the same state fields the toy runner reads (`request`,
  `allocation.pages`, `decode_pages`, `computed_prompt`, `outputs`) — page
  table = `allocation.pages + decode_pages`, identical slot math
  (`position // page_size`, `% page_size`).
- Prefill chunk: forward `prompt[end-num_tokens : end]` at positions
  `[end-num_tokens, end)` in ONE batched token pass per request (not
  token-at-a-time — chunked prefill is the perf shape the GPU keeps); on
  prompt completion, sample from the last position's logits via the Sampler
  (grammar/logprobs/seeds all inherited from m8).
- Decode chunk: single-token forward at the absolute position, sample.
- Requests are processed sequentially within a step (CPU correctness first;
  cross-request batching is an M13/GPU concern — recorded non-goal).
- Cached-prefix skip: positions `< computed_prompt - chunk` are never
  recomputed — exactly the scheduler's compute-skip contract; radix-reused
  pages hold valid KV from the earlier request (the parity suite pins this
  with a shared-prefix trace).
- **KV-write skip (amended, BLOCKING)**: the runner MUST skip pool writes for
  positions `< allocation.num_cached_tokens` (compute Q only — the pool
  already holds valid KV there). Verified: radix sharing is page-granular and
  tail pages are private, so cached-prefix chunk writes are safe by
  construction EXCEPT the recompute-last-token position of a fully-cached
  page-aligned prompt, which lands in a shared page's last slot — benign
  under deterministic CPU fp32, a data race once M13/GPU removes determinism.
- **Decode input-token rule (amended)**: input token = `state.outputs[p-1]`
  (`prompt[-1]` at p==0) read from the PASSED state at execute time — this is
  what makes `SpeculativeRunner`'s overlay-state mechanism work unchanged;
  KV write-before-read at every decode position per layer (rejected-draft
  stale KV is always overwritten before any read). Sampler `position`: 0 for
  the prompt-completing sample, output index for decode (seed idempotence).
- **State-access contract (amended, canonical)**: the runner reads exactly
  `request.{prompt_token_ids, request_id, sampling, eos_token_id}`,
  `allocation.pages`, `allocation.num_cached_tokens`, `decode_pages`,
  `computed_prompt`, `prefill_done`, `outputs` (values). `RequestSnapshot`
  is extended NOW with `outputs`, `sampling`, `num_cached_tokens` and a
  pages accessor so M16 is non-breaking; `build_engine_loop` REJECTS
  `model_path` + `tensor_parallel_size > 1` ("arrives in M16") — the current
  CPU TP path duplicates one runner instance across ranks, which is
  incompatible with a stateful pool/sampler runner.

### D5 — Loader + registry: checkpoint dir → runner

`kairyu/models/loader.py`: `load_model(path) -> (DenseDecoder, ModelConfig)`
using m8's `CheckpointReader` (index.json/sharded/single-file) — iterate
`named_parameters()`, fetch by HF name, `tie_word_embeddings` maps `lm_head`
to `embed_tokens`, dtype policy from config. Quantized checkpoints
(`detect_quantization() != NONE`) fail fast with "arrives in M14" (the hook
point is a `linear_factory` argument defaulted to `torch.nn.Linear` — M14
swaps `QuantizedLinear` in without touching the loader body). `get_slice`
sharding hook stays unused until M16.
`kairyu/models/registry.py`: `architectures[0]` → builder fn
(`LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`).
`KairyuBackend(model_path=...)` convenience wiring (amended details):
`tokenizer` defaults to `None` — resolved to the model dir when `model_path`
is given, else `"toy"` (an explicit tokenizer alongside model_path is allowed
— tests need it; fixture dirs carry no tokenizer.json); `runner=` and
`model_path=` are mutually exclusive; `validate_tp_degree` uses the config's
real `num_key_value_heads` (not the hardcoded 8); EOS may be a LIST in
generation_config.json (Llama-3 Instruct) — first entry becomes
`eos_token_id`, the rest `stop_token_ids`; tokenizer vocab larger than the
model's `vocab_size` fails fast. The `kairyu-proc` service gains `model_path`
(picklable str) and reports its port BEFORE building the loop so model-load
time doesn't eat the spawn timeout.

### D6 — Parity harness: transformers is the oracle; no committed weights

- **Primary (offline, deterministic)**: fixtures build tiny transformers
  models (2 layers, hidden 64, 4 heads / 2 kv heads, vocab 256; Qwen3 with
  its decoupled head_dim) with `save_pretrained(tmp_path,
  safe_serialization=True)` — exercising our REAL config.json + safetensors
  load path — then assert (a) fp32 logits max-abs diff < 1e-4 on random token
  sequences, (b) **32-step greedy exact-match through the full Kairyu engine**
  (chunked prefill with a small token budget, radix reuse via a second
  same-prefix request, decode paging) vs `model.generate(do_sample=False)`.
  Covered archs in M12: `LlamaForCausalLM` (rope_scaling on and off),
  `Qwen2ForCausalLM` (attention_bias), `Qwen3ForCausalLM` (qk-norm,
  explicit head_dim).
- **Secondary (opt-in, networked)**: `@pytest.mark.hf_hub` +
  `scripts/parity_real_model.py` — Qwen2.5-0.5B-Instruct greedy on CPU vs
  transformers (run manually before deploy day; catches config-parsing gaps
  tiny synthetic configs miss).
- **Cross-cutting mechanics land here** (used by all of M12–M19):
  pytest markers registered in pyproject (`gpu`, `hf_hub`, `dist`) with
  `addopts` deselecting `gpu`/`hf_hub` by default **+ `--strict-markers`**
  (typo'd marks must not silently always-run); `pytest -m hf_hub` documented
  as the escape hatch. Coverage omit globs already exist (m8 D6).
  **Oracle pinning (amended, verified)**: transformers v5 honors checkpoint
  dtype — force `dtype=torch.float32` (v5 kwarg is `dtype`, NOT torch_dtype)
  and pin `attn_implementation` (measured eager-vs-sdpa delta 1.5e-7 in fp32;
  chunked-vs-full 1.8e-7 — two orders under the 1e-4 gate). One bf16-saved
  fixture loaded to fp32 by both sides proves the upcast path. Tied weights:
  the safetensors file genuinely OMITS lm_head.weight — the tie mapping in
  the loader is mandatory, not optional. Fixtures are session-scoped per arch
  (build+save once; HF oracle and loaded DenseDecoder read-only) to keep the
  suite under ~30s.

## 3. Non-goals (recorded)

- Cross-request batched attention within a step (M13 seam + GPU).
- FlashInfer / any CUDA kernel (M13 adapters).
- Quantized weight loading (M14; loader hook only).
- MoE / MLA (M15). TP sharding (M16). Draft heads (M17).
- Prompt-logprobs; sliding-window attention (neither target arch needs it at
  these sizes; recorded for Qwen2's config flag — rejected loudly if set).
- Performance on CPU (correctness only; tiny models keep the suite fast).

## 4. Phasing (each green: pytest + ruff, cov ≥ 80%)

1. pyproject markers + `ModelConfig` (+ tests).
2. Layers/attention/decoder + direct-logits parity vs transformers (tiny).
3. `PagedKVPool` + `PagedModelRunner` + full-engine greedy parity.
4. Loader/registry through real safetensors + backend wiring (`model_path=`).
5. hf_hub opt-in script/test.

## 5. Verification

- fp32 logits < 1e-4 vs transformers for all three archs; 32-step full-engine
  greedy exact match incl. chunked prefill (budget smaller than prompt),
  radix-reuse second pass (identical output + cached_tokens > 0), and decode
  crossing page boundaries.
- Loader round-trip: save_pretrained → our loader → parity (proves name map,
  dtype policy, tied embeddings).
- Sampler integration: seeded sampling reproducible on the real model;
  logprobs finite; stop strings/EOS via the real tokenizer's eos_token_id.
- Full existing suite green (471 baseline); `pytest -m hf_hub` documented as
  the pre-deploy manual gate.

## 6. Review record

2-reviewer panel (numerics/HF-compat with empirical runs; engine-integration),
2026-07-03 — both APPROVE-WITH-AMENDMENTS, all applied inline above:

- Numerics: dual config.json generations (transformers-5 fixtures write
  rope_parameters/dtype — CRITICAL); Qwen2 bias derived from architecture
  (no attention_bias field exists — CRITICAL); rectangular chunk mask spec
  (is_causal measured wrong, 2.22 maxdiff); oracle dtype/attn pinning;
  cached-position KV-write skip; RoPE fp32 contract; sampler position pin;
  fixture dirs lack tokenizer.json (engine parity drives token ids directly).
- Integration: full 9-field state-access contract made canonical;
  model_path×TP rejected in M12 + RequestSnapshot extended NOW (outputs,
  sampling, num_cached_tokens) so M16 is non-breaking (BLOCKING); decode
  input-token rule stated for SpeculativeRunner overlay compat; pool/cache
  sizing single-sourced with fail-fast validation (BLOCKING pair with the
  write-skip); backend wiring details (tokenizer default None, mutual
  exclusion, port-before-build, per-config TP validation, EOS lists);
  addopts/-m verified safe against CI + --strict-markers; session-scoped
  fixture strategy; forward contract pinned for M17 (all-position post-norm
  hidden) and M13 (attention takes pool+page_table).
