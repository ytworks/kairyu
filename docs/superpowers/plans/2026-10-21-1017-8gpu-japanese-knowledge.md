# Japanese knowledge bases for the 1017 eight-GPU Chat UI

Status: **Approved 2026-10-21 (owner).**

Accepted scope (owner, 2026-10-21):

- The Chat UI (Open WebUI v0.12.3) of `examples/deepseek-v4.1-flash-1017-8gpu/`
  ingests Japanese text PDFs and scanned PDFs into knowledge bases, finds the
  right passage for a Japanese question, and the chat answers from it. The
  configuration is part of the example, so it survives `run.sh down` / `up`.
- Only this example changes. `kairyu/`, the other examples (including
  `deepseek-v4.1-flash-1017-6gpu`) and kairyu-bench are not changed. The L1
  (vLLM) and L2 (Kairyu) settings are not changed; the change is the L3
  Chat UI and one Docling service next to it.

## Facts

1. Today the Chat UI embeds with `sentence-transformers/all-MiniLM-L6-v2`
   (English, 256-token window), has no OCR (image-only PDFs yield no text),
   and `RESET_CONFIG_ON_START: true` resets Admin Panel changes at every
   start, so settings must come from `compose.yaml`.
2. Open WebUI's BM25 builds terms with `text.split()`
   (`open_webui/retrieval/utils.py`), i.e. on whitespace. Japanese has no
   spaces between words, so BM25 cannot match Japanese terms. Hybrid search
   with `RAG_HYBRID_BM25_WEIGHT=0` runs vector search followed by the
   reranker. The knowledge tools the model calls (`query_knowledge_files`,
   `query_knowledge_bases`) go through the same `query_collection` /
   hybrid path, so they use the same configuration.
3. Open WebUI v0.12.3 sends documents to Docling at
   `POST {DOCLING_SERVER_URL}/v1/convert/file` with `DOCLING_PARAMS` as form
   fields. docling-serve accepts `do_ocr`, `force_ocr`, `ocr_lang` (BCP-47)
   and an OCR preset (`ocr_engine` is deprecated). In Docling, RapidOCR maps
   `ja` to the PP-OCRv6 `japan` model.
4. The docling-serve image prefetches only RapidOCR's default (`ch`)
   checkpoints into its artifacts path, and with an artifacts path Docling
   does not download missing checkpoints at run time.
   `docling-tools models download rapidocr --rapidocr-backend-lang
   onnxruntime:ja` prefetches the Japanese set. The image's Tesseract has
   only the English language pack.
5. Open WebUI accepts an absolute local path as `RAG_EMBEDDING_MODEL` /
   `RAG_RERANKING_MODEL`. `OFFLINE_MODE=true` turns off model downloads. The
   Chat UI container failed an IPv6-first fetch from Hugging Face, so models
   are fetched on the host side.
6. The DeepSeek L1 container's cpuset covers all 128 host CPUs; Docling OCR
   and CPU embedding/reranking compete with it unless limited.

## Changes

1. **Docling service** in `compose.yaml`: base
   `quay.io/docling-project/docling-serve-cpu:v1.37.0` pinned by digest,
   built through the example's own `docling-ja.Dockerfile`, which prefetches
   RapidOCR's Japanese checkpoints and fails the build if they are missing.
   `DOCLING_SERVE_MAX_SYNC_WAIT=3600` (default 120 s would cut long PDFs),
   `cpus: 16`, a health check, no published port.
2. **Open WebUI extraction**: `CONTENT_EXTRACTION_ENGINE=docling`,
   `DOCLING_SERVER_URL=http://docling:5001`, `DOCLING_PARAMS` with OCR on,
   Japanese OCR language and `force_ocr` off (text layers are kept; only
   bitmaps are OCRed). The exact parameter names are fixed against v1.37.0
   in gate V1.
3. **Embedding** `BAAI/bge-m3` (multilingual, 8,192-token window, no query
   prefix) and **reranker** `BAAI/bge-reranker-v2-m3` (multilingual), pinned
   revisions in `example.json`, downloaded into `webui-data/rag-models/` and
   hashed by `control.py` with the same attestation program as the
   checkpoint (unneeded files such as ONNX exports excluded). Open WebUI gets
   absolute paths and `OFFLINE_MODE=true`.
4. **Search**: `ENABLE_RAG_HYBRID_SEARCH=true`, `RAG_HYBRID_BM25_WEIGHT=0`,
   `RAG_TOP_K=20` (reranker candidates), `RAG_TOP_K_RERANKER=5` (chunks given
   to the model).
5. **Chat UI** `cpus: 16` (embedding and reranking run on CPU).
6. **Startup validation**: `run.sh up` reads Open WebUI's retrieval
   configuration (as `_check_chat_ui_connection` reads the connection) and
   fails unless it reports Docling, the two model paths and BM25 weight 0.
7. **Docs**: README section on Japanese knowledge bases (how to create
   them, Admin Panel changes are reset, re-ingest after changing the
   embedding or OCR); MEASUREMENTS rows; a short addendum to the 1017
   amendment in `docs/design/frontier-native-runtime.md` (this example's L3
   differs from its siblings); one PROGRESS.md Change Log entry.
8. **Tests**: no pytest is added or removed. The configuration is guarded
   by the startup validation and the behavior by the GPU gates (test
   policy). Base and head collection counts are reported.

Chat history in `webui-data` is kept. There are no knowledge bases today,
so nothing needs re-indexing.

## Verification

Every gate checks behavior. Stop at the first failure, fix all known risks
together, re-run only the affected gates.

| # | Gate | Pass | Budget |
|---|---|---|---|
| V0 | Local CPU | ruff on changed files; `tests/unit/test_deepseek_v41_1017_8gpu_example.py` passes; pytest collection base = head | 2 min |
| V1 | Docling alone (no GPU, running stack untouched) | The example's image builds; started with `--network none`, it converts a three-page image-only Japanese PDF (rendered with a pinned Noto Sans JP) and each page's planted Japanese key sentence is extracted; a text-layer PDF keeps its own text | ≤ 30 s per page |
| V2 | Restart | `run.sh down`, then `run.sh up` with the external URL settings passes, including the startup validation (the Chat UI is down about 7 min) | 10 min |
| V3 | `verify.sh knowledge` (new) | Through Open WebUI's API: a knowledge base with a 20-page scanned PDF and 20 Japanese text documents; the key sentence is in the OCR text; a Japanese paraphrase query returns its chunk in the top 5; the knowledge base is deleted afterwards | ingestion ≤ 5 min, query ≤ 10 s |
| V4 | Chat UI in a browser (example's own Playwright script) | The existing `323` / `324` / `Red` pass, plus a Japanese question with the knowledge base selected by `#` answered from the key sentence with a citation; Kairyu `/v1/responses` increases, `/v1/chat/completions` stays 0 | 10 min |
| V5 | Serving during ingestion | `serving` c1/c8 rows while a 20-page OCR ingestion runs: ≥ 95 % of the recorded 151.2 / 378.8 tok/s; on failure lower the Docling CPU limit and re-run V5 only | 10 min |
| V6 | Persistence and hand-back | `run.sh down` / `up` again with the external URL settings; the configuration is restored from the environment and V3's query passes again; handed back at `http://61.206.39.14:3018` | 10 min |

Afterwards: MEASUREMENTS, PR status, push, CI, report. Leaving draft and
merging are the owner's decisions.
