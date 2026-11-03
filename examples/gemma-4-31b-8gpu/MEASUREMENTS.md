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

Host: eight RTX PRO 6000 Blackwell Server Edition (SM120, 97,887 MiB, PCIe
only), vLLM overlay on v0.33.0.

### L1 start-up findings (2026-11-04)

- EmbeddingGemma 2 did not start on stock v0.33.0 or v0.32.0: the Triton
  encoder attention needs 160 KiB of shared memory per block for its
  256-dimensional heads and SM120 has 99 KiB (`OutOfResources`).
  `--attention-backend FLEX_ATTENTION` failed on CUDA-graph reuse (also with
  `--enforce-eager`) and `FLASHINFER` has no encoder-only attention. The
  example's overlay (`patch_sm120.py`) uses 64-row tiles there; the replica
  then starts and returns 768-dimensional unit vectors.
- A TP2 chat replica failed CUDA-graph capture in vLLM's custom all-reduce
  (`custom_all_reduce.cuh:164 'invalid argument'`); chat replicas run with
  `--disable-custom-all-reduce` (NCCL), as the 1017 examples do.

### Step 1: TP1 and TP2 facts (run `20261103T151811Z-tune-layout`, `20261103T152225Z-tune-layout`)

| Chat TP | Weights / GPU | KV cache | 262,144-token requests held | Result |
|---|---|---|---|---|
| TP1, `gpu_memory_utilization` 0.92 | 58.83 GiB | 26.54 GiB (33.29 GiB needed for one request; 173,504 tokens max) | 0 | excluded: vLLM refuses to start |
| TP2, 0.92 | 30.14 GiB | 55.36 GiB per GPU, 871,782 tokens | 3.33 | kept |

TP1 cannot hold one full-context request even at 0.98 (about 32.3 GiB of
KV), so every layout uses TP2 chat replicas (at most four).

One TP2 replica (GPUs 0-1), 8K-token prompts, at most 256 output tokens, 64
requests per row (thinking on, as served):

| Concurrency | Succeeded | Output tok/s | TTFT p50 / p99 | TPOT p50 | Request p50 |
|---|---|---|---|---|---|
| 1 | 64/64 | 33.2 | 1.39 / 1.47 s | 24.8 ms | 7.7 s |
| 8 | 64/64 | 117.1 | 7.42 / 10.1 s | 39.4 ms | 17.5 s |
| 16 | 64/64 | 142.6 | 8.71 / 20.1 s | 78.0 ms | 28.5 s |
| 32 | 64/64 | 161.5 | 9.86 / 39.9 s | 157.8 ms | 50.2 s |
| 64 | 64/64 | 172.4 | 44.6 / 80.0 s | 188.6 ms | 93.9 s |

### Step 2: L1 options on one TP2 replica (run `20261103T154313Z-tune-l1`)

Output tok/s of the same rows (8K in, at most 256 out, 64 requests each; every
request succeeded). A change is kept only if it gains at least 5 %.

| Candidate | c1 | c8 | c16 | c32 | c64 | 262,144-token requests held | Decision |
|---|---|---|---|---|---|---|---|
| baseline | 33.2 | 116.7 | 142.6 | 161.4 | 172.3 | 3.33 | reference |
| baseline-repeat | 33.2 | 116.9 | 142.8 | 161.2 | 172.0 | 3.33 | run-to-run within 0.2 % |
| MTP assistant, 4 tokens | 67.8 | 149.5 | 167.0 | 176.5 | 179.0 | 3.23 | **kept**: +104 / +28 / +17 / +9 / +4 % |
| MTP assistant, 8 tokens | 61.4 | 149.5 | 163.6 | 170.0 | 173.0 | 3.23 | never above MTP 4 |
| no async scheduling | 33.1 | 117.0 | 143.0 | 161.8 | 172.6 | 3.33 | within 0.3 %; async kept |
| max-num-batched-tokens 4096 | 33.2 | 117.5 | 143.0 | 162.2 | 171.6 | 4.11 | within 0.7 % |
| max-num-batched-tokens 16384 | 33.4 | 116.8 | 142.3 | 160.8 | 172.6 | 2.40 | within 1 % |
| max-num-seqs 128 | 33.2 | 117.0 | 143.0 | 161.7 | 172.5 | 3.31 | within 0.3 % |

MTP 4 accepted 3.0-3.1 tokens per step on average (per-position acceptance
about 0.75 / 0.56 / 0.41 / 0.32). Speculative decoding verifies every drafted
token against Gemma 4 31B, so the output distribution is the target model's.

### Step 3: embedding throughput and interference (run `20261103T180542Z-tune-embedding`)

Layout `tp2x4-embed-shared` (four TP2 chat replicas with MTP 4 at 0.86, two
EmbeddingGemma 2 replicas at 0.06 on GPUs 1 and 7). Through Kairyu:

- Open WebUI's ingestion shape (32 chunks of 300 characters per batch, four
  batches in flight): 517 texts/s, batch p50 0.24 s, max 0.33 s. A single
  query: p50 15 ms, max 19 ms.
- Chat rows with the embedding replicas idle, then while four threads send
  such batches without pause (5,462 batches, no error):

| Concurrency | Idle tok/s | Loaded tok/s | Change | TTFT p50 idle / loaded |
|---|---|---|---|---|
| 1 | 66.9 | 59.4 | -11 % | 1.38 / 1.55 s |
| 8 | 380.7 | 340.6 | -11 % | 1.44 / 2.61 s |

Saturating embedding load slows the chat replicas that share a GPU with an
embedding replica (a TP2 replica runs at the pace of its slower GPU).
Knowledge-base ingestion is paced by Docling's OCR, so step 4 measures the
layouts under real ingestion.

### Step 4: full layouts under ingestion (run `20261103T182136Z-tune-layout`)

Both layouts with MTP 4, while three Chat UI ingestion streams of fresh
20-page Japanese scans keep Docling busy and a retrieval query runs every
20 s against a reference knowledge base. Output tok/s (TTFT p50):

| Layout | c1 | c8 | c16 | c32 | c64 | Requests held / replica |
|---|---|---|---|---|---|---|
| `tp2x4-embed-shared`: 4 x TP2 at 0.86, embeddings on GPUs 1 and 7 | 67.3 (1.38 s) | 383.4 (1.42 s) | 503.1 (2.65 s) | **584.7** (6.43 s) | **649.9** (12.9 s) | 2.89 |
| `tp2x3-embed-dedicated`: 3 x TP2 at 0.92, embeddings on GPUs 6 and 7 | 66.9 (1.38 s) | 318.6 (1.43 s) | 387.9 (2.66 s) | 454.4 (7.66 s) | 494.0 (16.7 s) | 3.23 |

Every row: 64/64 requests succeeded. Knowledge budgets (ingestion <= 300 s,
query <= 10 s, ingestion covering the whole measured window):

| Layout | Window | Coverage | Ingestions | Longest ingestion | Queries | Slowest query | Errors |
|---|---|---|---|---|---|---|---|
| `tp2x4-embed-shared` | 377 s | 100 % | 10 | 197.6 s | 17 | 4.77 s | 0 |
| `tp2x3-embed-dedicated` | 413 s | 100 % | 11 | 207.5 s | 18 | 4.17 s | 0 |

Embedding batches during the load: 573 texts/s for both (batch p50 0.17 /
0.21 s, query p50 14 / 18 ms).

### Selection

`tp2x4-embed-shared` (committed in `example.json`, `compose.yaml`,
`kairyu.yaml`): 29 % more output than `tp2x3-embed-dedicated` at c32 and
32 % at c64 under ingestion, within every knowledge budget; c1 is the same.
Each replica holds 2.89 full-context requests. Sharing GPUs 1 and 7 with the
embedding replicas costs at most 11 % on two replicas only while embeddings
saturate (step 3); real ingestion did not measurably slow chat (c1 67.3 vs
66.9 tok/s idle). L1: MTP assistant 4 tokens, async scheduling,
`max-num-batched-tokens` 8192, `max-num-seqs` 64, custom all-reduce off.

## Gates

To be recorded per gate, with run ids.
