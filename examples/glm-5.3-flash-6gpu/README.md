# GLM-5.3-Flash on six RTX PRO 6000 GPUs

One GLM-5.3-Flash replica on GPUs 0–5 of 96 GB RTX PRO 6000 Blackwell
(SM120, PCIe) cards, behind the same L2/L3 structure as
`deepseek-v4.1-flash-6gpu`: one Kairyu `ReplicaPool` replica, the
OpenAI-compatible API, and Open WebUI. GPUs 6–7 are not used.

```text
Open WebUI (:3016) -> Kairyu (:8015) -> ReplicaPool -> vLLM v0.31.0 (TP2 x DP3 / EP6 + MTP, GPUs 0-5)
```

GLM-5.3-Flash is Z.ai's 320B-total / 18B-active multimodal MoE (288 routed
experts, 8 per token; 34 KDA linear-attention layers and 11 NoPE sparse-MLA
layers; one MTP layer). It always thinks before it answers; the
`reasoning_effort` field selects `low`, `high` or `max` (default `max`).

## Start

```sh
./run.sh up          # preflight, image pull + digest check, model download + hash check, readiness
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

`run.sh up` refuses to start when GPUs 0–5 are busy. The official FP8
checkpoint (≈306 GiB) is downloaded to
`/mnt/nvme/kairyu/model-volumes/glm-5.3-flash-6gpu/models` and hashed
against the pinned tree; every cache (Hugging Face, Xet, torch.compile,
Triton, FlashInfer) stays below the same NVMe directory. To reuse a copy that
is already on the same NVMe filesystem, set
`GLM_MODEL_SEED=<path to glm-5.3-flash>`: it is hard-linked (no extra space)
and then re-hashed; nothing is trusted from it.

The API and the Chat UI listen on all interfaces; `run.sh up` prints their
external URLs (set `PUBLIC_HOST` to override the detected address, or
`API_BIND_ADDRESS` / `CHAT_UI_BIND_ADDRESS=127.0.0.1` to keep them local).
Model `glm-5.3-flash`; the Chat UI has no authentication.

The Chat UI talks to Kairyu over the OpenAI Responses API (`/v1/responses`,
Open WebUI `OPENAI_API_CONFIGS` `api_type: responses`), and Kairyu sends each
request to vLLM in the API its caller used: chat as chat, Responses as
Responses, Anthropic Messages as Messages (`kairyu.yaml` lists the model in
`upstream_chat_models`, `upstream_responses_models` and
`upstream_messages_models`). `run.sh up` installs the L4 workspace's effort filter, which
also leaves only Responses fields in the Chat UI's body (`reasoning.effort`,
no `seed`/`top_k`/`chat_template_kwargs`, saved output items reduced to their
Responses fields, `store: false`), and checks that the Chat UI's connection
is `api_type: responses`. GPU verification of this connection is pending
(`MEASUREMENTS.md`, 2026-10-17).

Saved model settings: Open WebUI applies a model's saved Advanced Params
(Admin Panel or Workspace > Models) once more after the filter, just before it
sends the request, so the filter cannot normalize them. Save only Responses
fields there (temperature, top_p, max tokens; Open WebUI itself drops stop and
the penalties). A default effort goes in as the Custom Parameter `reasoning`
with the value `{"effort": "high"}`; it applies to requests that carry no
effort of their own. A saved `seed`, `top_k`, `min_p` or `reasoning_effort`
makes Kairyu answer 400 `Unknown parameter`, as OpenAI's Responses API does
(m11 D7 amendment 2026-10-18).

## Knowledge bases, Word, Excel and PowerPoint

The Chat UI is the shared L4 chat workspace
([`deploy/workspace`](../../deploy/workspace/README.md)), selected by this
example's `workspace` section in `example.json` (profile `ja-v1`). It gives
Japanese knowledge bases (Docling OCR for scanned pages, `BAAI/bge-m3`
embedding and `BAAI/bge-reranker-v2-m3` reranking from pinned, hashed copies
shared by every example, 300-character chunks), Word / Excel / PowerPoint read
and edit through the browser code interpreter (on in every chat by default),
and download links for the files it writes. Docling runs at 4 CPUs and the
Chat UI at 8; `verify.sh knowledge-serving` checks that the serving rows keep
95 % of their unloaded throughput beside OCR load. The environment is the
Chat UI's configuration on every start, so Admin Panel changes do not
survive a restart. A knowledge base built before this example moved to the
workspace was embedded with another model: add its files again.

## Usage

- Effort: `"reasoning_effort": "low" | "high" | "max"` on chat,
  `"reasoning": {"effort": ...}` on Responses (Kairyu also maps
  `minimal` → `low`, `medium` → `high`, `xhigh` → `max`); omitted means `max`.
  The Chat UI has a dropdown under Chat Controls → Valves → Reasoning Effort.
  Kairyu sends `max` to vLLM's `/v1/responses` as `xhigh` (the OpenAI
  vocabulary has no `max`); the official template renders every effort
  other than `low`/`high` as Max, which `verify.sh l1` checks (`render_xhigh`)
  on GPU.
- Sampling: the checkpoint's `generation_config.json` (temperature 1.0,
  top_p 0.95), the model author's recommendation.
- `clear_thinking`: the official template's default (`false`) keeps earlier
  turns' reasoning for clients that send `reasoning_content` back (agents).
  The model author asks chat clients to pass `clear_thinking=true`, but
  Kairyu forwards no `chat_template_kwargs` for this model (chat rejects them
  with 400 and the Responses API has no such field), so neither the Chat UI
  nor API clients can set it here; the template default applies to every
  request.
- Images: up to 8 per request (8 MiB each); video is not accepted (Kairyu's
  public API carries image parts only).
- Tools: OpenAI tool calling, parsed by vLLM's `glm47` parser. Kairyu sends
  each conversation as it is (chat messages or Responses input items), so the
  official template renders an agent's earlier turns, tool calls and tool
  results.

## Why this configuration

The settings start from the official sources (model card, chat template,
vLLM recipe) and change only what this hardware forces or what a bounded
measurement in `MEASUREMENTS.md` supports (one parameter at a time; a change
is adopted for ≥ 5 % on c1 or c16 throughput without a > 5 % loss
elsewhere).

| Setting | Value | Source / reason |
|---|---|---|
| GPUs | six (0–5), one replica | The official FP8 checkpoint (306 GiB) does not fit one TP4 replica on four 96 GB cards with a 1M-token KV pool: 1.81 GiB of KV memory per GPU against 7.56 GiB needed. Planned on four GPUs; six was the owner-approved fallback. |
| Runtime | stock `vllm/vllm-openai:v0.31.0`, registry digest pinned | The first release with NoPE sparse MLA on the FlashInfer SM120 backend; no overlay or patch. |
| Replica shape | TP2 × DP3, EP6 (48 of 288 experts per GPU) | TP2 pairs share a NUMA node. Against DP6 / EP6 at the same 96 sequences: c1 +12 %, c16 +11 %, c64 +3 %, twice the KV pool. TP4 or TP6 cannot use six GPUs (64 heads). |
| MTP | the checkpoint's MTP layer, 3 draft tokens | c1 74 vs 37 tok/s (TPOT 9.7 vs 23.0 ms), c16 and c64 unchanged; exact answers behind a cached prefix. The recipe's 5 lost 6 % at c16; on DP6 no MTP depth leaves room for a 1M KV pool. |
| All-reduce | NCCL (`--disable-custom-all-reduce`) | vLLM's custom all-reduce fails while the drafter's CUDA graphs are captured on TP2; NCCL measured no different without MTP. |
| KV cache | FP8, pinned at 16.75 GiB per GPU: 2,028,392 tokens per engine | Recipe's Blackwell setting and the only layout the SM120 sparse-MLA backend accepts. Pinned so a cold compile cache cannot shrink the pool below one full context. |
| Context / batching | 1,048,576 tokens; 8,192 batched tokens; 0.95 memory | Model maximum. 4K batching lost 11 % at c64; 16K leaves no 1M KV pool. |
| Sampling / effort | temperature 1.0, top_p 0.95; effort default max | The checkpoint's `generation_config.json` and chat template, unchanged. |
| Parsers | `glm47` reasoning and tool-call parsers | Official recipe. |

Verification: `verify.sh l1 | serving | completed | tool-calling | vision |
reasoning | cancellation | long-context | restart | api-equivalence`. The
public gates run through Kairyu's `/v1/responses` (m11 D7 amendment
2026-10-17); the L1 render, `/tokenize` and log-probability checks stay on
vLLM's or Kairyu's chat API because the Responses API has no such fields.
`serving` rows are ~8K in and up to 256 out with natural completion (the
Responses API has no `ignore_eos`), and `cancellation` ends naturally within
its output cap, so their numbers are not comparable with the fixed-length
chat rows before. `api-equivalence` sends the same Responses requests (text,
stream, efforts low/high/xhigh, a tool round trip, an image, a length cap,
the Open WebUI body) to vLLM directly and through Kairyu, and checks from
vLLM's own request counters that Kairyu reached it on `/v1/responses`. GPU
verification of the Responses gates is pending. The previous chat
configuration passed all nine earlier gates (`MEASUREMENTS.md`, run
`20261011-r2`): 70.6 / 298.3 / 455.7 tok/s at c1 / c16 / c64 (~8K in, 256
out), the planted key found in 1,039,890 prompt tokens, 80 default-effort
answers completed.
