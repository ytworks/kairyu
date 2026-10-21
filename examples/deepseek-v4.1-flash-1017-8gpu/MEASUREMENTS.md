# deepseek-v4.1-flash-1017-8gpu evidence

Status: **all final gates PASS on the committed configuration (2026-10-19).**

Hardware: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB,
SM120, PCIe); 1 TiB host memory, 4 NUMA nodes with GPU pairs
(0,1)(2,3)(4,5)(6,7) each on one node.
Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
Artifacts live under
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-1017-8gpu/`.

## Checkpoint against 0909

Compared with `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a`
(the 0909 examples' pin): `config.json`, `tokenizer_config.json`,
`inference/config.json`, the 96,085 tensor names of
`model.safetensors.index.json` and its `total_size` (510,286,023,000 bytes)
are identical; the 48 weight shards and `encoding/` changed, and the
technical report PDF, `assets/` and `evaluation/` are no longer shipped.

`encoding/encoding.py` changes (the model card's "minor update"):

- `find_last_user_index` skips a user message whose content blocks hold
  only tool results: without a tools declaration, reasoning before the
  latest *user request* is dropped, and the current turn keeps its
  reasoning across tool calls (0909 dropped it at the first tool result).
  With a tools declaration nothing is dropped, as before.
- tool messages may carry content blocks (images in tool results), user
  content may be a list, and images are collected after tool results are
  merged into prompt order.

The checkpoint (the six-GPU 1017 example's download, hard-linked) hashes to tree SHA-256
`e25bddd997c2e7b959f444bb70813e210b813b8c528bf25e7a0e25b1d89bb807` (83 files,
`model-manifest.json`); every size and all 49 LFS SHA-256 values equal
Hugging Face's. Identical to 0909: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors.index.json`, the encoder test
fixtures and most of `inference/`.

The reference `inference/model.py` changed in two places: only an index-key
owner layer (`kv_source_layers`) publishes the shared index keys, and the
attention output `o` is FP8 quantized and dequantized before the `wo_a`
einsum (the production FP8 `wo_a` GEMM's numerics); `tilelang` moves to
0.1.12. vLLM keeps its own implementation of both; nothing is ported.

vLLM's encoder copy (`vllm/tokenizers/deepseek_v41_encoding.py`) is the same
file in the 2026-09-30 nightly (`ac68c3087`) and v0.32.0 and keeps the 0909
rule. vLLM's tokenizer wrapper flattens list content to text with image
placeholders before encoding, so only the turn rule changes what vLLM
renders; `patch_sm120.py` ports it. On the host: both vLLM copies fail the
checkpoint's own 1017 test case before the edit ("the current turn's
reasoning is dropped at its tool result") and pass it after, as the 1017
reference encoder does.

## Runtime image

| Image | vLLM / FlashInfer / torch | Overlay build | Kernel gate (`check_sm120_kernels.py`) |
|---|---|---|---|
| `vllm/vllm-openai:v0.32.0@sha256:befccf0e…` + this example's overlay → `local/vllm-openai:deepseek-v41-1017-8gpu` `sha256:7237421efe0a2eb901560ff550600c02e90310fda100caded27ff8475bf19b0c` | 0.32.0 / 0.7.0.post1 / 2.13.0+cu130 | all nine anchors patched; encoder checks pass (efforts 50/75/100, `xhigh` 100, default 75, 1017 tool-turn case) | PASS 28/28 (`kernel-checks/kernels-v032-r2.log`; the overlay accepts only v0.32.0, the six-GPU 1017 example's S1 selection) |

The image follows the six-GPU 1017 example's S1 selection: v0.32.0 over the
2026-09-30 nightly with the same overlay (c32 +7.6 %, c1/c8 within noise;
`deepseek-v4.1-flash-1017-6gpu/MEASUREMENTS.md`).

## L1 selection

Rule (the six-GPU amendment's): one change at a time; adopted only with
≥ 5 % output throughput at c1, c8 or c32 and no > 5 % loss elsewhere; the
replica shape is chosen by c8 throughput (the DeepSWE run's concurrency),
then c32. Rows (`tune.py`): 32 requests of ≈ 8K unique input tokens through
Kairyu's `/v1/responses`, capped at 256 output tokens, thinking high.
Cells: output tok/s / model TTFT p50 ms. Single trials.

### `l1` on DP8 / EP8

Run `20261018T204321Z-l1`: **PASS 17/17.** Rendered efforts default 75,
low 50, high 75, max 100, `xhigh` 100, chat closes thinking; vLLM's prompt
tokens equal the checkpoint encoder's for the 1017 tool-turn case and the two
golden chat fixtures; 12 concurrent `17 * 19` probes per variant all exact
with finite log-probabilities, answered by all eight DP engines; L1 default
sampling `{'temperature': 1.0, 'top_p': 1.0}`; tool call and image pass.
Weights 47.55 GiB per GPU (48 of 384 experts per rank), 25.03 GiB KV,
13,799,928 KV tokens per DP engine; idle GPUs at 87,633 MiB.

### S1: replica shape

Run `20261018T204356Z-tuning` on v0.32.0, starting from DP8 / EP8 (the
committed command at the time, with the six-GPU 1017 levers).

| Candidate | c1 | c8 | c32 | KV per GPU / tokens per engine | Result |
|---|---|---|---|---|---|
| DP8 / EP8, Engram in shared host memory (running) | 101.0 / 726 | 291.7 / 1,216 | 620.2 / 4,319 | 25.03 GiB / 13,799,928 | PASS |
| DP8 / EP8, restarted | 98.7 / 719 | 288.9 / 1,378 | 655.8 / 4,331 | 25.03 GiB / 13,799,928 | PASS |
| DP8 / EP8 + `embedding_across_dp` (Engram sharded in HBM) | — | — | — | — | **FAIL** at start: 1.51–1.84 GiB KV per GPU, below the 1.9 GiB one 1,048,576-token request needs |
| **TP8 / EP8**, Engram TP-sharded in HBM, NCCL (`--disable-custom-all-reduce`) | **152.2 / 470** | **376.2 / 1,515** | **458.2 / 7,017** | 15.70 GiB / 8,613,322 | PASS |

Against the mean of the two DP8 rows (c1 99.9, c8 290.3, c32 638.0), TP8
is +52 % at c1, +30 % at c8 and −28 % at c32. By the plan's rule (c8, the
concurrency of the DeepSWE runs, decides; c32 second) **TP8 / EP8 is
adopted**; DP8 keeps the higher aggregate rate at c32. Weights are 61.16 GiB
per GPU on TP8 (47.55 GiB on DP8 with the tables in host memory). For
comparison, `deepseek-v4.1-flash-8gpu` (0909 weights, TP8 on its 0909 image,
chat rows of exactly 256 tokens) measured 126.8 / 275.4 / 326.8 tok/s at
c1/c8/c32.

### S2: levers on TP8

Run `20261018T212205Z-tuning` after `l1` passed on TP8 (`20261018T212139Z-l1`:
17/17, weights 61.16 GiB per GPU, 15.70 GiB KV, 8,613,322 KV tokens, every
probe exact from the single TP8 engine). Percentages against the mean of the
two TP8 rows (S1 and the running baseline here: c1 151.5, c8 378.3, c32
463.4).

| Candidate | c1 | c8 | c32 | KV tokens | Result |
|---|---|---|---|---|---|
| TP8 baseline (running) | 150.8 / 471 | 380.3 / 1,280 | 468.6 / 7,005 | 8,613,322 | PASS |
| `--max-cudagraph-capture-size 1024` | 151.8 / 469 (+0.2 %) | 365.0 / 1,104 (−3.5 %) | 466.6 / 7,004 (+0.7 %) | 6,748,667 | rejected |
| **batch 8K** | 152.5 / 461 (+0.7 %) | 379.7 / 866 (+0.4 %) | **488.7 / 6,789 (+5.5 %)** | 6,803,944 | **adopted** |
| batch 16K | 152.8 / 440 (+0.9 %) | 386.0 / 822 (+2.0 %) | 482.3 / 6,505 (+4.1 %) | 3,731,439 | rejected (below 5 %) |
| memory 0.90 | 149.7 / 471 (−1.2 %) | 371.5 / 912 (−1.8 %) | 465.1 / 7,002 (+0.4 %) | 7,566,052 | rejected |

Batch 8K is the only lever over the line (c32 +5.5 %; the two TP8 baseline
rows differ by 2.3 % at c32); it also lowers c8 TTFT p50 by about a third.
It keeps 6.5 × a 1,048,576-token request of KV. Not measured on TP8: the
DP-only `--prefill-schedule-interval` and adaptive verification (the V4.1
indexer backend rejects it on every shape, six-GPU S2).

## Selected L1

vLLM v0.32.0 + this overlay, TP8 / EP8 (48 experts per rank), NCCL
all-reduce, TP-sharded GPU-resident Engram tables,
`FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer, 64-token pages, Marlin
MoE, DSpark 5 (full verification), `--max-num-batched-tokens 8192`,
`--gpu-memory-utilization 0.92`, `--max-num-seqs 128`, prefix caching,
temperature 1.0 / top_p 1.0, thinking high by default.

`use_thp` (recipe B200 TP2 arm) is not a candidate: v0.32.0's EngramConfig
requires `dp_shared_memory=false` for it, i.e. one private pinned copy of
the 183 GiB tables per DP rank (8 × 183 GiB exceeds the host's 1 TiB).

## Final gates on the committed configuration

Run 2026-10-19 06:53–07:17 JST, image `sha256:7237421e…`, TP8 / EP8 with
8,192 batched tokens, served-config SHA-256
`e8226cff9ee8a22df88cfa504a188c2aeb166a759201420d4632826aecf57fb1`, every
gate `--no-start` in order (`final-gates-20261019.log` in the NVMe example
directory).

| Gate | Result | Evidence |
|---|---|---|
| `l1` | PASS 17/17 (9 s) | efforts 75/50/75/100, `xhigh` 100; encoder parity (1017 tool-turn case, two golden fixtures); probes exact from the TP8 engine |
| `serving` | PASS (255 s) | c1/8/16/32/64: 151.2 / 378.8 / 434.5 / 484.7 / 511.1 tok/s; model TTFT p50 461 / 845 / 937 / 2,116 / 13,171 ms (p99 478 / 3,262 / 6,470 / 12,971 / 26,652); 64/64 per row (`20261018T215333Z-serving`) |
| `completed` | PASS 192/192 (875 s) | generic c1 218.4 tok/s, completion p50 2.4 s (p99 17.4 s); c8 680.7, 4.4 s (21.1 s); c32 895.1, 7.9 s (33.6 s); coding c1 266.7, 6.6 s (46.8 s); c8 975.6, 14.7 s (63.6 s); c32 1,323.4, 31.5 s (111.2 s) (`20261018T215748Z-completed`) |
| `tool-calling` | PASS 5/5 | concurrent calls, tool-result turn, streamed call, low and max effort |
| `vision` | PASS 6/6 | completed red answers |
| `reasoning` | PASS 4/4 | default/low/high/max all `323` with reasoning |
| `cancellation` | PASS | released 0.23 s after the disconnect; follow-up completed |
| `long-context` | PASS 4/4 (86 s) | 32,800 / 131,104 / 262,176 / 1,039,904 prompt tokens, 2.2 / 6.4 / 11.4 / 65.9 s, exact keys |
| `restart` | PASS | healthy and answering 194 s after the L1 restart |
| `api-equivalence` | PASS (9 s) | |
| Chat UI | PASS | Chromium (Playwright 1.60.0) on the example's Open WebUI: `323`, then `324` from the same chat, and `Red` for an attached red PNG; Kairyu `/v1/responses` 555 → 561, `/v1/chat/completions` 0 → 0 |

Open WebUI v0.12.3 stored and showed only the message items of these answers
although Kairyu streams their reasoning items (as in the six-GPU 1017
example; the L3/UI is the V4.1 examples', unchanged).

For orientation (same host): `deepseek-v4.1-flash-8gpu` (0909 weights, TP8
on its 0909 image, fixed 256-token chat rows) measured 126.8 / 275.4 /
305.4 / 326.8 / 314.3 tok/s at c1/8/16/32/64; the six-GPU 1017 example
measured 101.8 / 278.0 / 364.2 / 579.0 / 803.7 on these Responses rows.

## Japanese knowledge bases (2026-10-21)

Branch `claude/1017-8gpu-japanese-knowledge` (PR #661), L1/L2 as above.
Fixtures are generated per run: Japanese report pages built from shared
templates (every page names other `KR-73xx` samples and storage
temperatures), rendered as an image-only A4 PDF at 150 dpi with Noto Sans
JP 2.004; the key sentence (`KR-7319`, `マイナス四十二度`, `冷凍庫で保管`) is
on page 13 of 20. Results are under
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-1017-8gpu/verification-results/`.

Final runs on the committed configuration (Docling 4 CPUs, Chat UI 8 CPUs,
2026-10-21 15:45-15:58 UTC):

| Gate | Result | Evidence |
|---|---|---|
| `docling` | PASS | 3/3 image-only pages carry the key sentence, 4.7 s per page; the Chromium text-layer PDF keeps all 18 lines; Docling on an internal network (no route out) (`20261021T154536Z-docling`) |
| `knowledge` | PASS | 20-page scan ingested in 80.2 s, 20 text distractors in 31.0 s; the key sentence is in the OCR text; the Japanese question's top chunk is the key one (reranker 0.984, next 0.533); query 3.52 s (`20261021T154625Z-knowledge`); second fixture 80.7 s, 0.993, 3.55 s (`20261021T155515Z-knowledge`) |
| `chat-ui` | PASS | Chromium: `323`, `324` in the same chat, `Red` for an attached PNG; with the knowledge base picked by `#`, "KR-7319は摂氏マイナス42度（−42℃）の冷凍庫で保管するよう決められており…" citing `kairyu-gate-scan.pdf`; Kairyu `/v1/responses` 133 → 146, `/v1/chat/completions` 0 → 0; cleanup left no gate base or file (`20261021T155715Z-chat-ui`) |
| `knowledge-serving` | PASS | three load streams; Docling busy (≥ 1 core) for 100 % of the 154 s of rows, observed 100 %, mean 3.79 cores of 4; c1 150.9 tok/s (99.8 % of 151.2), c8 375.9 (99.2 % of 378.8); the references are the final `serving` rows above (`20261018T215333Z-serving`) (`20261021T154830Z-knowledge-serving`) |

Restart checks (Docling 8 CPUs, Chat UI 16 CPUs; the services' limits do
not change startup): `run.sh up` with the new services and its retrieval
check, ready in about 5 min; `run.sh down` / `up` again in about 4 min, and
the kept knowledge base answered the same query with the same ranking
(0.998), 2.28 s (`20261021T143049Z-knowledge`).

Settings changed by these gates (one cause each, measured before the fix):

- **Chunk size 300 characters, overlap 30** (Open WebUI's default 1,000 / 100
  is sized for English). One 20-page fixture, `bge-m3` dense rank of the key
  chunk among all chunks: 1,000 characters 27 / 40, 500 48 / 80, 300 9 / 120,
  200 2 / 166. With 1,000 the key page missed the reranker's 20 candidates.
- **Chat UI `OMP_NUM_THREADS=16`**: torch sees 128 host CPUs and started 64
  threads under the 16-CPU limit. Reranking 20 real pairs: 64 threads 20.1 s,
  16 threads 7.9 s, 8 threads 10.5 s; a knowledge query took 17.4 s before.
- **Docling 4 CPUs, 2 threads; Chat UI 8 CPUs, 8 threads** (were 16 / 8
  and 16 / 16). With one load stream and Docling at 16 CPUs, c1 fell to
  142.8 tok/s against 148.9 idle the same day. The load gate then measured
  Docling's own CPU time (review): with two streams Docling at 8 CPUs gave
  c1 141.7 (93.7 %), at 4 CPUs 143.4 (94.9 %), with Docling busy only 91.9 %
  and 94.4 % of the rows because both streams embedded at once. Splitting
  the load, same day: idle c1 / c8 155.1 / 378.1; Docling alone at 4 CPUs
  (busy 100 %) 153.0 / 379.0; Chat UI embedding alone at 16 CPUs (two text
  streams) 135.9 / 358.4. The embedding was the cost, so the Chat UI went to
  8 CPUs and the gate to three streams (owner decisions).
- **OCR language `iso:ja`**: Docling 2.138 treats a bare `ja` as a RapidOCR
  engine code and fails the conversion. The image's prefetched PP-OCRv6
  recognizer is the same file `docling-tools` fetches for `iso:ja`
  (`PP-OCRv6_rec_small.onnx`, sha256 `6f327246…`), so the stock image is used.
- **Key fragments**: RapidOCR read the kanji `一` of `一回` as the long-vowel
  `ー` on two of three pages (identical glyphs), so the key check uses
  fragments without such twins.
