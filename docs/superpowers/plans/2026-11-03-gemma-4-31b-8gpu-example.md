# Gemma 4 31B + EmbeddingGemma 2 example (`gemma-4-31b-8gpu`)

Status: **Plan approved 2026-11-03 (owner), including the verification plan.**
Based on `main` only.

Amendment (2026-11-03, owner decision after the CPU implementation): Kairyu's
`/v1/responses` rejected `reasoning.effort: none`, so the Chat UI's thinking-off
choice could not work; Responses now passes `none` to an engine or pool model as
thinking off, as Messages has since 2026-10-19 (m11 D4 Responses thinking-off
amendment).

## Request

- A new example: Gemma 4 **31B dense** (owner's choice) for chat, EmbeddingGemma 2
  for embeddings, both on vLLM (L1).
- Its Chat UI offers every feature of `deepseek-v4.1-flash-1017-8gpu`'s Chat UI.
- The replica counts are found by measurement, including the embedding trade-off.
- L2 load-balances over however many replicas are chosen; L3 makes this embedding
  model the one that is used automatically.
- Like every example on main, it owns its Chat UI (compose, provisioning, plugins,
  gates); nothing is shared with another example.

## Facts

1. `/v1/embeddings` serves only in-process CPU backends (`mock`, `fastembed`,
   `kairyu/deploy/builder.py:58-61`): no upstream forwarding, no replicas. The 1017
   Chat UI embeds with bge-m3 inside its own container and never calls Kairyu.
2. Chat replicas are already balanced by `pools:` (`ReplicaPool`: least outstanding,
   prefix score, ejection, probe restore, placement log; `qwen3.8-27b-dp8-8gpu`).
   The prober, `/readyz` and the pool metrics are written against `ReplicaPool`.
3. vLLM v0.32.0 ships `EmbeddingGemma2Model` (`--runner pooling`), the Gemma 4
   tool/reasoning parsers and `tool_chat_template_gemma4.jinja`; Responses maps
   `reasoning.effort` to Gemma's `enable_thinking` (`none` = off) and request
   kwargs override server defaults. v0.33.0 (2026-11-02) adds Gemma 4 fixes.
4. Open WebUI v0.12.3 has `RAG_EMBEDDING_ENGINE=openai`, query/content prefixes,
   async/concurrent embedding and per-connection `model_ids`.
5. Gemma 4 31B: 60 layers, 50 sliding-window (1,024) and 10 full-attention. The
   Quyet example (same base weights, batch-invariant vLLM v0.31.0) had a
   36,226-token KV cache on one GPU at 95 %; whether one GPU holds 262,144 tokens
   depends on vLLM's hybrid KV manager and is measured.
6. Checkpoints (ungated): `google/gemma-4-31B-it@842da379`,
   `google/gemma-4-31B-it-assistant@627c5ec1`, `google/embeddinggemma-2@f9972ac2`
   (768 dims, 8,192 tokens, bf16 only, query/document prefixes).

## Structural changes (approved)

- **S1 framework**: `/v1/embeddings` forwards to upstream embedding servers and L2
  balances their replicas.
- **S2 new example** `examples/gemma-4-31b-8gpu/` (fifth model family).
- **S3** the 1017 eight-GPU stack is stopped during GPU work and restarted after.

"Used automatically" means: Kairyu publishes `embeddinggemma-2` on `/v1/embeddings`
and `/v1/models` and places each batch on an embedding replica; the Chat UI's
knowledge bases (document and query embeddings) always go through it, `run.sh up`
fails closed otherwise, and a gate shows Kairyu's embedding pool receiving them.
EmbeddingGemma's task prefixes are added by the Chat UI, which knows query from
document. The embedding model is hidden from the chat model picker.

## S1: framework

| Layer | File | Change |
|---|---|---|
| L1 | `kairyu/engine/embedding.py` | `OpenAIEmbeddingBackend` (one upstream `/embeddings`, no `dimensions`, reorder by index, shared vector validator, exact upstream usage, 4xx = `UpstreamClientError`) |
| L2 | `kairyu/orchestration/replica.py` | `_record_placement` extracted from `_place_prepared`; `ReplicaPool.embed` (synchronous least-outstanding choice, shared failure/ejection/finish); unanimous `dimensions` / `reports_exact_usage` |
| L3 | `kairyu/deploy/spec.py` | `backend: openai` embeddings: `model`, `dimensions`, `replicas[{base_url, health_url}]`, `api_key_env`, `timeout_s`, `unhealthy_after`, `probe_interval_s`, `placement_log_path` |
| L3 | `kairyu/deploy/builder.py` | adapters + pool + `HealthProber`; placement-log uniqueness covers embedding pools |
| L3 | `kairyu/entrypoints/server/health.py`, `app.py` | `/readyz` 503 for an empty embedding pool; pool metrics |

Admission (`.claude/rules/framework-boundary.md`): (1) an engine-served embedding
model cannot be published through L3 and has no placement or health; (2) the
embedding factory builds one object from one section and `ReplicaPool` was
chat-only, while a separate pool would copy eligibility, drain, membership and
probe logic; (3) any GPU embedding server (vLLM, TEI, llama.cpp) behind Kairyu's
auth, metering and tenancy; today `backend: openai` is rejected at load and a
dead embedding replica is never ejected; (4) one adapter, one pool method, one
config section, two one-line hooks; model, replica count, dimensions, vLLM flags,
prefixes and concurrency stay in the example.

## S2: example

| Layer | Content |
|---|---|
| L1 | vLLM v0.33.0 (v0.32.0 if it fails): chat N replicas, embedding M replicas; official BF16 only (no FP8 KV or third-party quantization). Chat: `gemma4` tool and reasoning parsers, the image's `tool_chat_template_gemma4.jinja`, `--default-chat-template-kwargs '{"enable_thinking": true}'`, 262,144 context, images. Embedding: `--runner pooling`, text only, bf16, 8,192. No own template; an SM120 overlay only if needed |
| L2/L3 | `kairyu.yaml`: pool `gemma-4-31b` (N, `queue_depth_threshold: 0`, `prefix_index: true`, placement log), embeddings `embeddinggemma-2` (`backend: openai`, M, vLLM `/health`, placement log), `upstream_chat/responses/messages_models` |
| Chat UI | Open WebUI v0.12.3 + docling-serve (1017 pins): Docling OCR `iso:ja`, bge-reranker-v2-m3 in the Chat UI (pinned, hashed), chunks 300/30, BM25 0, top_k 20/5 as 1017; embeddings via Kairyu `embeddinggemma-2` with the two prefixes, batch 32, async with concurrency >= M; Responses connection with `model_ids` = chat model; Pyodide code interpreter and `file_download` on by default; thinking levels `default` (on) / `none` (off); API 8020, Chat UI 3020, bind 0.0.0.0, Chat UI sign-in (admin) |

Files (all example-owned): `README.md`, `MEASUREMENTS.md`, `example.json`,
`model-manifest.json`, `kairyu.yaml`, `compose.yaml`, `control.py`, `run.sh`,
`webui-responses-filter.py`, `webui-file-download-tool.py`,
`webui-browser-smoke.mjs`, `verify.sh`, `verification.py`, `benchmark.py`, `tune.py`.
Storage `/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/`.

## Replica search (`tune.py`, about 8 h)

Fixed conditions: every chat replica holds at least one 262,144-token context;
knowledge budgets (ingest <= 300 s, query <= 10 s) hold under chat load; at
least two embedding replicas.

1. TP1 and TP2: vLLM's KV capacity and 262,144-token concurrency; hybrid KV on?
2. L1 options one at a time (>= 5 % rule): MTP assistant 4 / 8 / off, async
   scheduling, max-num-batched-tokens.
3. Embedding alone: memory, 300-character chunk throughput, query latency;
   co-located with a chat replica, the chat slowdown with and without ingestion.
4. Layouts that survive (about 1 h each), e.g. TP1x8 + 2 co-located embeddings,
   TP1x7 + one embedding GPU, TP2x4 + 2 co-located: the 1017 serving rows (8K in,
   256 out, c1/8/16/32/64 x 64) with three ingestion streams running.
5. Pick the highest c32/c64 aggregate throughput; within 5 %, the faster c1.
   Record in `MEASUREMENTS.md`; fix the layout in `compose.yaml` / `kairyu.yaml`.

## Tests (CPU)

`tests/server/test_serve_builder.py`, three A-to-C cases (YAML -> `/v1/embeddings`
-> two fake upstreams): concurrent batches spread; a 500 ejects its replica,
later batches succeed on the other, `/readyz` is 503 while none is eligible and a
probe restores it; a 400 is not a failure. No adapter-only or static-config
tests. Example tests only for behaviour that would hurt the product. Locally
ruff and the touched tests; base/head collection counts reported.

## Records

m5 D4 embedding amendment, m11 D4 upstream-embeddings amendment and A20, m7 D4
readiness note, FN-D9 Gemma amendment in `docs/design/frontier-native-runtime.md`,
`docs/deployment.md`, `examples/README.md`, the example's README/MEASUREMENTS,
`PROGRESS.md`.

## Verification plan (stop at the first failure with a fix plan; passed gates are not re-run)

| Gate | Checks | Pass | Budget |
|---|---|---|---|
| search (1-5) | layout | at least one layout meets the conditions; reason recorded | 8 h |
| `l1` | thinking on/off renders, tool call, image, finite logprobs; embeddings vs sentence-transformers bf16 cosine >= 0.999, norm ~1 | all | 30 min |
| `serving` | c1/8/16/32/64, every chat replica used, largest <= 1.25x mean at c >= 8 | all complete, no skew | 40 min |
| `completed` | generic and coding, c1/8/32 | every reply has content | 30 min |
| `tool-calling` | every chat replica | all pass | 10 min |
| `vision` / `reasoning` / `cancellation` | colour, thinking on/off, disconnect | 1017 criteria | 20 min |
| `long-context` | 32K / 128K retrieval; ~250K accepted and ended normally (owner amendment 2026-11-04: official sampling only, the 250K answer is model accuracy) | all three | 30 min |
| `embeddings` (new) | Kairyu = vLLM vectors; spread over M; one embedding replica stopped: at most one failed request before ejection, then all succeed, `/readyz` ready; restored after restart | all | 20 min |
| `restart` | `run.sh up` Chat UI checks all match | ready | 20 min |
| `api-equivalence` | every case incl. `image-tools` | L1 = L3 | 10 min |
| `docling` / `knowledge` | OCR; 20-page scan key chunk in top 5, ingest <= 300 s, query <= 10 s; Kairyu embedding pool reached on every replica | 1017 + via Kairyu | 20 min |
| `chat-ui` | 323 -> 324, Red, `#` knowledge answer cited, no embedding model in the picker | all | 10 min |
| `knowledge-serving` | c1/c8 during ingestion >= 95 % of the same run's unloaded rows | 1017 criteria | 15 min |
| `office` | docx/xlsx/pptx read and edit saved from the link, then an image | all three | 15 min |
| `file-download` | desktop, after reload (same SHA-256), iPhone 13 | three saves | 20 min |

## Risks

- If one GPU cannot hold 262,144 tokens, TP2+ and at most four chat replicas.
- Gemma 4 31B may not handle the Office edits or the download tool; reported as found.
- Co-located embeddings may slow chat during ingestion; measured in the search.
- A replica dying mid-ingestion fails that one upload once (Open WebUI does not
  retry); a retry on another replica is out of scope.
- All embedding replicas down makes the whole gateway not ready (as FastEmbed today).
