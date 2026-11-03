# Measurements: Gemma 4 31B + EmbeddingGemma 2 on eight RTX PRO 6000 GPUs

GPU-verified 2026-11-04: the replica search chose four TP2 chat replicas with
the MTP assistant and two embedding replicas sharing GPUs 1 and 7, and all 17
gates pass (plan `docs/superpowers/plans/2026-11-03-gemma-4-31b-8gpu-example.md`).

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

Layout and L1 as selected above (overlay image `sha256:af9e746b…`), every gate
through Kairyu unless it says otherwise. Run ids are directories under
`model-volumes/gemma-4-31b-8gpu/verification-results/`.

| Gate | Run | Result |
|---|---|---|
| `l1` | `20261103T185141Z-l1` | PASS, 14 cases: each chat replica holds 2.89 x 262,144 tokens; checkpoint sampling defaults; thinking on/off render as the template defines; 48 + 48 exact `17 * 19` probes with finite log-probabilities; both embedding replicas vs sentence-transformers 6.1.0 min cosine 0.99997, unit norm; tool call, image (Red) and an embedding batch through Kairyu |
| `serving` | `20261103T185214Z-serving` | PASS: 8K in, at most 256 out, 64 requests per row, all succeeded. Output tok/s 68.1 / 384.7 / 502.5 / 588.6 / 651.4 at c1/8/16/32/64; TTFT p50 1.38 / 1.42 / 2.65 / 5.22 / 12.9 s; placements 16/16/16/16 on the four replicas at c8-c64 (largest 1.0 x mean; c1 stays on the warm idle replica by design) |
| `completed` | `20261103T185830Z-completed` | PASS: generic and coding tasks, 32 per row at c1/8/32, thinking on, every reply has visible content (192/192). Output tok/s generic 123.2 / 735.4 / 1,575.8, coding 136.9 / 911.5 / 1,949.8; request p50 7.5-12.3 s, p99 at most 23.1 s |
| `tool-calling` | `20261103T191058Z-tool-calling` | PASS: 8 automatic tool calls, a tool-result turn and a streamed call (10/10 HTTP 200 with function calls); all four replicas placed tool requests |
| `vision` | `20261103T191100Z-vision` | PASS: 6/6 answers "red" |
| `reasoning` | `20261103T191103Z-reasoning` | PASS: default (thinking on) reasons (142 characters) and answers 323; `none` returns no reasoning and answers 323 in 0.09 s |
| `cancellation` | `20261103T191104Z-cancellation` | PASS: after a disconnect mid-stream the L1 request is released in 0.81 s and a follow-up completes |
| `long-context` | `20261103T191107Z-long-context` | **FAIL** at ~250K: 32K and 128K returned the key (8.9 / 61.2 s); at 250,018 prompt tokens the reply stopped at the 8,192-token output cap with no content: the reasoning copied the "filler" haystack (57,087 characters) |

`long-context` diagnosis (same 250,018-token request, 2026-11-04): with
thinking on at the checkpoint's sampling (temperature 1.0) the reasoning
falls into copying the one-word haystack in 2 of 4 samples (the gate run and
one of two replays through Kairyu); one replay and one request sent to vLLM
directly answered correctly. Thinking off answered correctly (182 s, first
request), and with temperature 0 both thinking on (179 tokens) and thinking
off returned the key. KV capacity, Kairyu forwarding and the replica are not
involved; the verdict depended on sampling chance.

Owner decision (2026-11-04): gates use the official sampling only, and judging
the ~250K answer was an excessive test. The gate now judges the key at 32K and
128K, and at ~250K that the request is accepted with the server's prompt-token
count at the target and ends normally.

| Gate | Run | Result |
|---|---|---|
| `long-context` | `20261103T215321Z-long-context` | PASS: 32,786 / 131,090 / 250,018 prompt tokens, each ending with `stop`; keys correct at 32K and 128K (5.9 / 42.0 s); the ~250K request ended normally in 165.3 s (its reply was the key, not judged) |
| `embeddings` | `20261103T215656Z-embeddings` | PASS: Kairyu equals L1 (min cosine 1.0000), exact upstream usage (2,062 tokens); 8 concurrent batches on both replicas; with one replica stopped, 1 of 70 requests failed before ejection, all later ones succeeded and `/readyz` stayed 200; after its restart the replica rejoined and 8 batches spread over both again |
| `restart` | `20261103T215806Z-restart` | PASS: every service restarted, ready after 131.7 s, `run.sh up` checks all match |
| `api-equivalence` | `20261103T220018Z-api-equivalence` | PASS: 8 cases (text, stream, effort, tools, image, image-tools, cap, Open WebUI) with no L1/L3 difference; L1 received 20 Responses and 2 Chat requests, no Messages |
| `docling` | `20261103T220041Z-docling` | PASS: a 3-page Japanese scan OCRed in 12.0 s (4.0 s/page) with the key on every page; the text-layer PDF in 4.0 s with no missing line |
| `knowledge` | `20261103T220128Z-knowledge` | PASS: 20-page scan ingested in 67.0 s (distractors 22.6 s), 13,298 OCR characters with the key; query 3.89 s, key chunk rank 1 (reranker 0.993, next 0.487); 46 embedding requests through Kairyu on both replicas, the query embedded by the pool |
| `chat-ui` | `20261103T220306Z-chat-ui` | PASS in Chromium, signed in: only the chat model offered; 323, then 324 in the same chat; Red for the PNG; a `#` knowledge-base answer citing the scan (20.7 s); 13 Responses and 3 embedding requests, no Chat Completions |
| `knowledge-serving` | `20261103T220338Z-knowledge-serving` | FAIL in the gate, not the product: the unloaded c1 reference row (64/64 requests succeeded) counted 65 placements because a request it did not send (the Chat UI had just finished the chat-ui gate's chats) landed in the pool log. Fixed in `c8d6f56b`: rows are judged on their own requests' placements (`X-Request-ID`); replayed, the row has its 64 and the earlier serving rows keep their verdicts |
| `knowledge-serving` | `20261103T220935Z-knowledge-serving` | PASS: with three ingestion streams keeping Docling at its 4-CPU cap for the whole 292.5 s window (8 ingestions, no error, no stuck worker), c1 66.3 tok/s = 98.3 % and c8 382.9 tok/s = 100.9 % of the same run's unloaded rows (67.5 / 379.5); TTFT p50 unchanged (1.38 / 1.42 s) |
| `office` | `20261103T222216Z-office` | **FAIL** (model, official sampling): docx edited and saved from the link (replace, append, sentence and table kept); xlsx total row `=SUM(B2:B7)` saved from the link; pptx: the code interpreter ran and the reply said the file could be downloaded, but the model never called `provide_download`, so the reply had no link |

Owner decision (2026-11-04): a reply that says a file is ready without calling
`provide_download` is the model's miss at its official sampling, recorded and
not judged. `office` now requires every format to be read and edited by the
code interpreter, every linked result to be correct and at least one link;
`file-download` asks in up to three new chats per device until a reply links
the file.

| Gate | Run | Result |
|---|---|---|
| `office` | `20261103T222708Z-office` | PASS, no model miss: docx (replace, append, sentence and table kept), xlsx (`=SUM(B2:B7)` total row, rows kept) and pptx (titles kept, "まとめ" slide with three bullets added) each read, edited by the code interpreter in 17-21 s and saved from the reply's link; an image in that chat after the tool calls answered Red; 32 Responses and 11 embedding requests, no Chat Completions |
| `file-download` | `20261103T222900Z-file-download` | FAIL in the gate, not the product: the first desktop reply wrote `{provide_download(...)}` as text (model miss, recorded), the second linked `business_plan_format.xlsx` (4 sheets, 222 filled cells), saved as desktop.xlsx; after the reload Open WebUI's "what's new" dialog covered the link and the click timed out. Fixed in `29638c81`: the reload step closes the notices as every new chat does |
| `file-download` | `20261103T223114Z-file-download` | FAIL in the gate: the three desktop replies were counted as having no link. Two of them had called `provide_download`, which attached the stored file to the reply (file item above the text), without repeating the link in the text; the third called no tool. |

Owner decision (2026-11-04): a reply delivers a file when the tool attached it
or its text links it; the gate saves it as a user does, from the text link or
from the file item's dialog (`a784eb84`). Checked first on that run's second
reply: the same 8,766-byte workbook saved by click on a desktop and by tap on
an iPhone 13.

| Gate | Run | Result |
|---|---|---|
| `file-download` | `20261103T223719Z-file-download` | PASS: desktop, first reply delivered `Annual_Business_Plan_Template.xlsx` (attached file, saved from its dialog: 4 sheets, 289 filled cells); after a reload the same link saved the same file (same SHA-256); iPhone 13: two replies delivered no file (model misses, recorded), the third linked it and a tap saved it (4 sheets, 277 filled cells); 21 Responses requests, no Chat Completions |

All 17 gates pass on `a784eb84` (2026-11-04): `l1`, `serving`, `completed`,
`tool-calling`, `vision`, `reasoning`, `cancellation`, `long-context`,
`embeddings`, `restart`, `api-equivalence`, `docling`, `knowledge`,
`chat-ui`, `knowledge-serving`, `office`, `file-download`. Gates passed before
a later gate-only fix were not re-run: each fix changed only the gate that
failed (`long-context`, the serving rows' placement attribution, the office
and download delivery check), and the earlier serving rows were replayed with
the new attribution (same verdicts). Model misses observed at the official
sampling: the ~250K long-context reasoning loop (2 of 4 samples); one pptx
edit and four Excel requests (of nine) that delivered no file; three Excel
replies that attached the file without repeating its link.
