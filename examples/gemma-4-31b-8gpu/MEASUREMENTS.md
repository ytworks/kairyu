# Measurements: Gemma 4 31B + EmbeddingGemma 2 on eight RTX PRO 6000 GPUs

Nothing is measured yet: the GPU stage follows the review of the CPU
implementation (plan `docs/superpowers/plans/2026-11-03-gemma-4-31b-8gpu-example.md`).

## Images and checkpoints

| Item | Value |
|---|---|
| vLLM | overlay `local/vllm-openai:gemma-4-31b-8gpu-v0.33.0` (`vllm-sm120.Dockerfile`, one edit) on `vllm/vllm-openai:v0.33.0@sha256:d825baf9…84c6e`; fallback the same overlay on v0.32.0 `@sha256:befccf0e…5b34` |
| Chat checkpoint | `google/gemma-4-31B-it@842da379`, tree `cab440f8…ccbd9` |
| MTP assistant | `google/gemma-4-31B-it-assistant@627c5ec1`, tree `bb22c750…d5556` |
| Embedding checkpoint | `google/embeddinggemma-2@f9972ac2`, tree `287cbbd3…03661` |
| Reranker | `BAAI/bge-reranker-v2-m3@953dc6f6`, tree `5c3ea0eb…9fba` |
| Open WebUI / Docling | v0.12.3 / docling-serve-cpu v1.37.0 (pinned digests in `example.json`) |

The tree hashes were computed from the Hugging Face tree API (LFS SHA-256)
and the small files' own SHA-256 on 2026-11-03; the reranker's equals the
1017 example's recorded hash. `run.sh up` re-hashes every downloaded file.

## Replica search

To be recorded: TP1 / TP2 KV capacity and 262,144-token concurrency; L1
candidates; embedding throughput and interference; layouts under ingestion;
the selected layout and why.

## Gates

To be recorded per gate, with run ids.
