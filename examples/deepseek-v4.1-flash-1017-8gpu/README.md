# DeepSeek V4.1 Flash 1017 on eight RTX PRO 6000 GPUs

One `deepseek-ai/DeepSeek-V4.1-Flash-1017` replica on all eight 96 GB
RTX PRO 6000 Blackwell (SM120, PCIe) cards, behind the L2/L3 structure of
the V4.1 examples: one Kairyu `ReplicaPool` replica, the OpenAI-compatible
API, and Open WebUI.

```text
Open WebUI (:3018) -> Kairyu (:8018) -> ReplicaPool -> vLLM (TP8 / EP8 + DSpark, GPUs 0-7)
   |
   +-> Docling (OCR, CPU) for knowledge-base documents
```

Unlike its sibling examples, this one's Chat UI is set up for Japanese
knowledge bases (below).

1017 (2026-10-17) is DeepSeek's weight refresh of V4.1-Flash: `config.json`,
the tokenizer, the 96,085 tensor names and the 510.3 GB total are those of the
0909 checkpoint; the weights and `encoding/encoding.py` changed.

> **GPU-verified 2026-10-19.** TP8 / EP8 with 8,192 batched tokens; every
> gate, `api-equivalence` and the Chat UI pass succeed (`MEASUREMENTS.md`).

## Start

```sh
./run.sh up          # preflight, image build, model download + hash check, readiness
./run.sh status
./verify.sh l1 --no-start
./verify.sh serving --no-start
./verify.sh completed --no-start
./verify.sh tool-calling --no-start
./verify.sh vision --no-start
./verify.sh reasoning --no-start
./verify.sh cancellation --no-start
./verify.sh long-context --no-start
./verify.sh restart --no-start
./verify.sh api-equivalence --no-start
./run.sh down
```

`run.sh up` refuses to start when GPUs 0–7 are busy or when the host has
less than 64 GiB of available memory (the Engram tables are GPU-resident on
this shape). The checkpoint (≈476 GiB) is downloaded to
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-1017-8gpu/models` and
hashed against the pinned tree. To reuse a copy that is already on the same
NVMe filesystem, set `DEEPSEEK_MODEL_SEED=<path to deepseek-v4.1-flash-1017>`:
it is hard-linked (no extra space) and then re-hashed; nothing is trusted
from it.

API: `http://127.0.0.1:8018/v1`, model `deepseek-v4.1-flash-1017`.
Chat UI: `http://127.0.0.1:3018` (local, no authentication).

## Chat UI and API

Open WebUI talks to Kairyu over `/v1/responses` (`OPENAI_API_CONFIGS`
`api_type: responses`; m11 D7 amendment 2026-10-17). `run.sh up` installs
the Reasoning Effort filter (`webui-reasoning-effort-filter.py`) as a
global filter and checks the connection type. Besides the effort dropdown,
the filter turns Open WebUI's chat body into a strict Responses body: the
effort becomes `reasoning.effort`, Chat Controls fields the Responses API
lacks (`seed`, `top_k`, ...) are dropped, saved output items keep only their
Responses fields, and `store` is false because the UI resends the whole
conversation every turn.

Saved model settings: Open WebUI applies a model's saved Advanced Params
(Admin Panel or Workspace > Models) once more after the filter, just before it
sends the request, so the filter cannot normalize them. Save only Responses
fields there (temperature, top_p, max tokens; Open WebUI itself drops stop and
the penalties). A default effort goes in as the Custom Parameter `reasoning`
with the value `{"effort": "high"}`; it applies to requests that carry no
effort of their own. A saved `seed`, `top_k`, `min_p` or `reasoning_effort`
makes Kairyu answer 400 `Unknown parameter`, as OpenAI's Responses API does
(m11 D7 amendment 2026-10-18).

Kairyu sends each request to vLLM in its caller's API (`kairyu.yaml`
declares the model in `upstream_chat_models`, `upstream_responses_models`
and `upstream_messages_models`; Kairyu checks at startup that vLLM serves
`/v1/responses` and `/v1/messages`): chat as the caller's own messages,
Responses as Responses, Anthropic Messages as Messages. On the Responses
wire Kairyu spells L3 `max` as OpenAI `xhigh`, which `patch_sm120.py` maps to
the max budget (100); chat callers reach vLLM with `max` (100).

## Japanese knowledge bases

Open WebUI's knowledge bases (Workspace > Knowledge) are configured for
Japanese documents in `compose.yaml`, with every value pinned in
`example.json`:

- **Extraction**: documents other than plain text go to a Docling service
  (`docling-serve-cpu` v1.37.0, pinned by digest). Text layers are kept;
  bitmaps (scanned pages, images in a PDF) are read by RapidOCR with
  `ocr_lang` `iso:ja` (PP-OCRv6, one recognizer for Japanese, Chinese and
  English). Docling and the Chat UI are limited to 16 CPUs each so OCR and
  embedding do not slow the L1, whose cpuset spans every host CPU.
- **Embedding** `BAAI/bge-m3` and **reranking** `BAAI/bge-reranker-v2-m3`,
  both multilingual. `run.sh up` downloads them at pinned revisions into the
  Chat UI's data volume (`webui-data/rag-models`) and hashes them as it does
  the checkpoint; Open WebUI runs with `OFFLINE_MODE` and loads those copies.
- **Search**: hybrid search with BM25 weight 0. Open WebUI's BM25 splits
  text on whitespace, which cannot tokenize Japanese, so a query runs vector
  search for 20 candidates and the reranker keeps the best 5.

`run.sh up` fails unless Open WebUI reports exactly this configuration.
Because the Chat UI starts with `RESET_CONFIG_ON_START`, Admin Panel changes
to these settings are reset at the next start; change `example.json`
instead. Documents are embedded when they are added: after changing the
embedding model or the OCR settings, add the documents again (or use
Admin Panel > Settings > Documents > Reindex).

To use one: create it in Workspace > Knowledge with a name and a description
that says what it holds (the model chooses among knowledge bases by them),
add files, then either attach it to the model (Workspace > Models) or pick it
in a chat with `#`. Knowledge bases live in `webui-data` and survive
`run.sh down` / `up`.

## Why this configuration

The settings start from the official sources (1017 model card and encoder,
vLLM recipe, whose default checkpoint is 1017) and the measured selections of
`deepseek-v4.1-flash-1017-6gpu` and `deepseek-v4.1-flash-8gpu`, and change only what this hardware forces or
what a bounded measurement in `MEASUREMENTS.md` supports (one parameter at a
time; a change is adopted only for ≥ 5 % on c1, c8 or c32 throughput without
a > 5 % loss elsewhere).

| Setting | Value | Source / reason |
|---|---|---|
| Runtime | `vllm/vllm-openai:v0.32.0` pinned by digest + `patch_sm120.py` (selected over the 2026-09-30 nightly on six GPUs: c32 +7.6 %, c1/c8 within noise) | v0.32.0 adds V4.1 work after the nightly (faster CPU-offloaded Engram lookups, FP8 Engram exchange across DP, SWA replay trimming, SM120 split-K attention, DSpark fixes) but none of the SM120 edits below. |
| Replica shape | TP8 / EP8, 48 experts per rank, NCCL all-reduce | Measured against DP8 / EP8 (the six-GPU 1017 shape on eight ranks) on v0.32.0: c1 +52 % (152.2 vs 99.9 tok/s), c8 +30 % (376.2 vs 290.3), c32 −28 % (458.2 vs 638.0). Selected by c8, the concurrency of the owner's DeepSWE runs; DP8 serves more aggregate tokens at c32. |
| Engram | GPU-resident, TP-sharded (≈23 GiB per GPU) | As in `deepseek-v4.1-flash-8gpu`'s TP8. On DP8, `embedding_across_dp` in HBM left 1.5–1.8 GiB of KV per GPU, below one 1,048,576-token request (start refused); `use_thp` needs one private host copy per DP rank (8 × 183 GiB exceeds the host's 1 TiB). |
| Batched tokens / memory | 8192 / 0.92 | Measured on TP8 against 4096: c32 +5.5 % (16K: +4.1 % with half the KV; memory 0.90 no gain). |
| DSpark | 5-token block, probabilistic drafts, block rejection, full verification | Official method and block (lossless rejection sampling). Adaptive verification (the NVIDIA default) fails at start: the V4.1 indexer backend rejects it. |
| Attention | `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer | Official Blackwell settings. |
| Pages | 64-token blocks | SM120 requirement (FlashInfer SWA pages, DeepGEMM C1/C2 indexer pages). |
| MoE | Marlin | SM120 MXFP4 MoE path. |
| Sequences / graphs | 128; default CUDA graph sizes | Official Blackwell value. `--max-cudagraph-capture-size 1024` measured no gain on TP8. |
| Effort | default thinking high (75); low 50, max 100 (and `xhigh` 100) | The 1017 encoder (`encoding/encoding.py`). `xhigh` is Kairyu's Responses spelling of max. |
| Encoder | vLLM's `deepseek_v41` encoder + the 1017 tool-turn rule | 1017 no longer starts a turn at a user message holding only tool results, so without a tools declaration the current turn's reasoning survives its tool calls. vLLM's copy still has the 0909 rule; the overlay ports the 1017 one. |
| Sampling | temperature 1.0, top_p 1.0 | The 1017 model card. Applied at L1 with `--override-generation-config`; callers can still send their own. |

## SM120 runtime overlay

`vllm-sm120.Dockerfile` builds on the pinned vLLM image and runs
`patch_sm120.py`, which accepts only the vLLM versions it was written for,
edits exactly one known source anchor per change and fails the build if any
anchor is missing or repeated:

- the 1017 encoder rule: a user message that holds only tool results does
  not start a new turn;
- 64-token SWA / MLA / indexer pages and C2 (32-token) dual-cache prefill;
- MXFP4 indexer enabled on SM120 for V4.1 only;
- masked sparse-KV rows read a zero row instead of a recycled slot (without
  it every EP6 shape of the 0909 six-GPU work returned NaN log-probabilities
  or unrelated text);
- the split top-p cutoff keeps a candidate when a forced logit rounds the
  cutoff to the maximum;
- `xhigh` renders the max budget (100) instead of vLLM's 75;
- build-time checks that the encoder renders the model author's efforts,
  `xhigh` as max, and the checkpoint's own 1017 tool-turn test case.

Prebuilt FlashInfer JIT caches are removed so the patched sources are the
ones compiled. `check_sm120_kernels.py` is the numerical gate for these
kernels (including a NaN-poisoned masked slot):

```sh
docker run --rm --gpus device=0 -v "$PWD:/checks:ro" --entrypoint python3 \
  local/vllm-openai:deepseek-v41-1017-8gpu /checks/check_sm120_kernels.py
```

## Verification gates

Every gate that goes through Kairyu uses `/v1/responses`, the Chat UI's API,
including the readiness `17 * 19` probes (exact answer and finish; the
Responses API has no token log-probabilities). Checks that talk to vLLM
directly (`/tokenize` renders, L1 probes with finite log-probabilities) stay
on vLLM's chat API.

- `l1`: rendered efforts (default/low/high/max/chat, plus `xhigh` rendering
  the max budget); vLLM's prompt tokens equal to the checkpoint encoder's
  for the 1017 tool-turn case (no tools declared) and the checkpoint's two
  golden chat fixtures; rounds of 12 concurrent `17 * 19` probes per variant
  with exact answers and finite log-probabilities from the TP8 engine; the L1 default sampling, tool call, image, and memory.
- `serving`: 8K-in rows capped at 256 output tokens at c1/8/16/32/64 (64
  requests each), every request placed on the replica, GPU peaks. The
  Responses API has no `ignore_eos`/`min_tokens`, so each request ends
  naturally within the cap.
- `completed`: natural-completion generic and coding rows at c1/8/32 with
  the default effort; every request must complete with visible content.
- `tool-calling`, `vision`, `reasoning`, `cancellation` (natural completion
  within `max_output_tokens`, disconnected mid-stream), `long-context`
  (32K/128K/256K/≈1M retrieval), `restart`.
- `api-equivalence`: the same Responses requests sent to vLLM directly and
  to Kairyu agree, and vLLM's own request counters show Kairyu reached it on
  `/v1/responses` (`verification/product/correctness/upstream_api_equivalence.py`).
- `docling` (no GPU, the running stack untouched): the pinned Docling on an
  internal network converts a generated image-only Japanese PDF through Open
  WebUI's own loader, with each page's key sentence extracted, and a
  Chromium-printed text-layer PDF with every line intact.
- `knowledge`: through the Chat UI's API, a knowledge base of a 20-page
  scanned PDF and 20 Japanese text distractors; the key sentence is in the
  OCR text, a Japanese question returns its chunk in the top 5, within the
  ingestion and query budgets. `KNOWLEDGE_KEEP=1` keeps it for `chat-ui`;
  `KNOWLEDGE_REUSE=1` re-checks a kept one (after a restart).
- `chat-ui`: Chromium (`webui-browser-smoke.mjs`): `323`, then `324` in the
  same chat, `Red` for an attached PNG, and a Japanese question with the kept
  knowledge base picked by `#` answered from the scanned PDF with it cited;
  Kairyu sees `/v1/responses` only.
- `knowledge-serving`: the `serving` c1/c8 rows while the Chat UI OCRs
  scanned PDFs, at least 95 % of the recorded throughput.

L1 candidates are compared with `tune.py` (one parameter at a time, the
committed configuration restored afterwards; `dp8` is the other measured
shape). Results and every official deviation are
in `MEASUREMENTS.md`.

## Primary references

- [DeepSeek-V4.1-Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [1017 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/README.md)
- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
  (`vllm-project/recipes` `models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`, `date_updated: 2026-10-17`)
- [vLLM v0.32.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
