# DeepSeek V4.1 Flash 1017 on six RTX PRO 6000 GPUs

One `deepseek-ai/DeepSeek-V4.1-Flash-1017` replica on GPUs 0–5 of 96 GB
RTX PRO 6000 Blackwell (SM120, PCIe) cards, behind the L2/L3 structure of
the V4.1 examples: one Kairyu `ReplicaPool` replica, the OpenAI-compatible
API, and Open WebUI. GPUs 6–7 are not used.

```text
Open WebUI (:3017) -> Kairyu (:8017) -> ReplicaPool -> vLLM (DP6 / EP6 + DSpark, GPUs 0-5)
```

1017 (2026-10-17) is DeepSeek's weight refresh of V4.1-Flash: `config.json`,
the tokenizer, the 96,085 tensor names and the 510.3 GB total are those of the
0909 checkpoint; the weights and `encoding/encoding.py` changed.

> **GPU-verified 2026-10-19.** Every gate, `api-equivalence` and the Chat UI
> pass succeed on the committed configuration (`MEASUREMENTS.md`).

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

`run.sh up` refuses to start when GPUs 0–5 are busy or when the host has
less than 256 GiB of available memory (the Engram tables are pinned in host
memory). The checkpoint (≈476 GiB) is downloaded to
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-1017-6gpu/models` and
hashed against the pinned tree. To reuse a copy that is already on the same
NVMe filesystem, set `DEEPSEEK_MODEL_SEED=<path to deepseek-v4.1-flash-1017>`:
it is hard-linked (no extra space) and then re-hashed; nothing is trusted
from it.

API: `http://127.0.0.1:8017/v1`, model `deepseek-v4.1-flash-1017`.
Chat UI: `http://127.0.0.1:3017` (local, no authentication).

## Chat UI and API

Open WebUI talks to Kairyu over `/v1/responses` (`OPENAI_API_CONFIGS`
`api_type: responses`; m11 D7 amendment 2026-10-17). `run.sh up` installs
the L4 workspace's Responses filter (shown as Reasoning Effort) as a global
filter and checks the connection type. Besides the effort dropdown,
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

## Knowledge bases, Word, Excel and PowerPoint

The Chat UI is the shared L4 chat workspace
([`deploy/workspace`](../../deploy/workspace/README.md)), selected by this
example's `workspace` section in `example.json` (profile `ja-v2`). It gives
Japanese knowledge bases (Docling OCR for scanned pages, `BAAI/bge-m3`
embedding, then `BAAI/bge-reranker-v2-m3` reranking of the 50 nearest chunks,
from pinned, hashed copies shared by every example; 300-character chunks), Word / Excel / PowerPoint read
and edit through the browser code interpreter (on in every chat by default),
and download links for the files it writes. Docling runs at 4 CPUs and the
Chat UI at 8; `verify.sh knowledge-serving` checks that the serving rows keep
95 % of their unloaded throughput beside OCR load. The environment is the
Chat UI's configuration on every start, so Admin Panel changes do not
survive a restart. A knowledge base built before this example moved to the
workspace was embedded with another model: add its files again.

## Why this configuration

The settings start from the official sources (1017 model card and encoder,
vLLM recipe, whose default checkpoint is 1017) and the measured selection of
`deepseek-v4.1-flash-6gpu`, and change only what this hardware forces or
what a bounded measurement in `MEASUREMENTS.md` supports (one parameter at a
time; a change is adopted only for ≥ 5 % on c1, c8 or c32 throughput without
a > 5 % loss elsewhere).

| Setting | Value | Source / reason |
|---|---|---|
| Runtime | `vllm/vllm-openai:v0.32.0` pinned by digest + `patch_sm120.py` | Measured against the 2026-09-30 nightly with the same overlay: c32 +7.6 %, c1/c8 within noise. v0.32.0 carries none of the SM120 edits below. |
| Replica shape | DP6 / EP6 (TP1 per rank) | Official Blackwell "DEP" shape on the TP path's SM120 kernels (the DEP kernels are SM100-only). `deepseek-v4.1-flash-6gpu` measured it 44–47 % faster at c32 than TP2 × DP3. |
| Engram | pinned host memory, one copy shared by the DP ranks | 6 × 96 GB is below the checkpoint's official 614 GB minimum (recipe memory-bound arm; "output is unchanged"). `use_thp` is not a candidate: it needs one private host copy per DP rank (6 × 183 GiB exceeds the host's 1 TiB). |
| Batched tokens / memory | 4096 / 0.92 | Recipe memory-bound arm; needed to fit DSpark on DP6. 8K measured +2.5 % / +4.3 % / −0.5 % at c1/c8/c32 with 40 % less KV (not adopted). |
| DSpark | 5-token block, probabilistic drafts, block rejection, full verification | Official method and block (lossless rejection sampling). Adaptive verification (the NVIDIA default) still fails at start: the V4.1 indexer backend rejects it. |
| Attention | `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer | Official Blackwell settings. |
| Pages | 64-token blocks | SM120 requirement (FlashInfer SWA pages, DeepGEMM C1/C2 indexer pages). |
| MoE | Marlin | SM120 MXFP4 MoE path. |
| Sequences / graphs | 128 per DP engine; default CUDA graph sizes | Official Blackwell value. The recipe's `--max-cudagraph-capture-size 1024` and `--prefill-schedule-interval 64` measured slower (c8 −4.8 % / c32 −19.4 %). |
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
  it every EP6 shape returned NaN log-probabilities or unrelated text);
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
  local/vllm-openai:deepseek-v41-1017-6gpu /checks/check_sm120_kernels.py
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
  with exact answers and finite log-probabilities until all six DP engines
  have answered; the L1 default sampling, tool call, image, and memory.
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

L1 candidates are compared with `tune.py` (one parameter at a time, the
committed configuration restored afterwards; `image-nightly` builds the same
overlay on the 2026-09-30 nightly). Results and every official deviation are
in `MEASUREMENTS.md`.

## Primary references

- [DeepSeek-V4.1-Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [1017 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/README.md)
- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
  (`vllm-project/recipes` `models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`, `date_updated: 2026-10-17`)
- [vLLM v0.32.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
