# DeepSeek V4.1 Flash 1017 on eight RTX PRO 6000 GPUs

One `deepseek-ai/DeepSeek-V4.1-Flash-1017` replica on all eight 96 GB
RTX PRO 6000 Blackwell (SM120, PCIe) cards, behind the L2/L3 structure of
the V4.1 examples: one Kairyu `ReplicaPool` replica, the OpenAI-compatible
API, and Open WebUI.

```text
Open WebUI (:3018) -> Kairyu (:8018) -> ReplicaPool -> vLLM (DP8 / EP8 + DSpark, GPUs 0-7)
```

1017 (2026-10-17) is DeepSeek's weight refresh of V4.1-Flash: `config.json`,
the tokenizer, the 96,085 tensor names and the 510.3 GB total are those of the
0909 checkpoint; the weights and `encoding/encoding.py` changed.

> **L1 selection in progress (2026-10-19).** The replica shape (DP8 / EP8
> with host or HBM-sharded Engram tables, or TP8 / EP8) and the recipe
> levers are being measured on the image the six-GPU 1017 example selects
> (`MEASUREMENTS.md`); until then the image pins are empty and `run.sh up`
> serves only with `DEEPSEEK_ALLOW_UNPINNED_IMAGE=1`.

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
less than 256 GiB of available memory (the Engram tables are pinned in host
memory). The checkpoint (≈476 GiB) is downloaded to
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

## Why this configuration

The settings start from the official sources (1017 model card and encoder,
vLLM recipe, whose default checkpoint is 1017) and the measured selections of
`deepseek-v4.1-flash-1017-6gpu` and `deepseek-v4.1-flash-8gpu`, and change only what this hardware forces or
what a bounded measurement in `MEASUREMENTS.md` supports (one parameter at a
time; a change is adopted only for ≥ 5 % on c1, c8 or c32 throughput without
a > 5 % loss elsewhere).

| Setting | Value | Source / reason |
|---|---|---|
| Runtime | `vllm/vllm-openai:v0.32.0` pinned by digest + `patch_sm120.py` (the six-GPU 1017 example's selection decides; in progress) | v0.32.0 adds V4.1 work after the nightly (faster CPU-offloaded Engram lookups, FP8 Engram exchange across DP, SWA replay trimming, SM120 split-K attention, DSpark fixes) but none of the SM120 edits below. |
| Replica shape | DP8 / EP8 (TP1 per rank), 48 experts per rank (selection in progress against TP8 / EP8) | Official Blackwell "DEP" shape on the TP path's SM120 kernels (the DEP kernels are SM100-only). On six GPUs DP6 was 44–47 % faster at c32 than TP2 × DP3, and the DP6 example outran `deepseek-v4.1-flash-8gpu`'s TP8 from c8 up (586.7 vs 326.8 tok/s at c32). Selected by c8 throughput (DeepSWE's concurrency), then c32. |
| Engram | pinned host memory, one copy shared by the DP ranks (selection in progress against `embedding_across_dp` in HBM) | 8 × 96 GB clears the official 614 GB minimum only narrowly; the recipe's memory-bound arm keeps the 183 GiB tables in host memory ("output is unchanged"). `embedding_across_dp` shards one table copy over the eight ranks (≈23 GiB each) at the cost of a per-step DP exchange. `use_thp` is not a candidate: it needs one private host copy per DP rank (8 × 183 GiB exceeds the host's 1 TiB). |
| Batched tokens / memory | 4096 / 0.92 | The six-GPU selection; 8K/16K and 0.90 are re-measured with the larger per-rank memory. |
| DSpark | 5-token block, probabilistic drafts, block rejection, full verification | Official method and block (lossless rejection sampling). Adaptive verification is re-measured on the selected image. |
| Attention | `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer | Official Blackwell settings. |
| Pages | 64-token blocks | SM120 requirement (FlashInfer SWA pages, DeepGEMM C1/C2 indexer pages). |
| MoE | Marlin | SM120 MXFP4 MoE path. |
| Sequences | 128 per DP engine | Official Blackwell value. |
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
  with exact answers and finite log-probabilities until all eight DP engines
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
committed configuration restored afterwards; `embedding-across-dp` and `tp8`
are the shape candidates). Results and every official deviation are
in `MEASUREMENTS.md`.

## Primary references

- [DeepSeek-V4.1-Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [1017 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/README.md)
- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
  (`vllm-project/recipes` `models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`, `date_updated: 2026-10-17`)
- [vLLM v0.32.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
