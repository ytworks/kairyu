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

## Usage

- Effort: `"reasoning_effort": "low" | "high" | "max"` (Kairyu also maps
  `minimal` → `low`, `medium` → `high`, `xhigh` → `max`); omitted means `max`.
  The Chat UI has a dropdown under Chat Controls → Valves → Reasoning Effort.
- Sampling: the checkpoint's `generation_config.json` (temperature 1.0,
  top_p 0.95), the model author's recommendation.
- `clear_thinking`: the official template's default (`false`) keeps earlier
  turns' reasoning for clients that send `reasoning_content` back (agents).
  The model author asks chat clients to pass `clear_thinking=true`, but
  Kairyu's chat path for this model forwards no `chat_template_kwargs` on
  text requests (it rejects them with 400), so neither the Chat UI nor API
  clients can set it here; the template default applies to every request.
- Images: up to 8 per request (8 MiB each); video is not accepted (Kairyu's
  public API carries image parts only).
- Tools: OpenAI tool calling, parsed by vLLM's `glm47` parser.

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
reasoning | cancellation | long-context | restart`; the committed
configuration passes all nine (`MEASUREMENTS.md`, run `20261011-r2`):
70.6 / 298.3 / 455.7 tok/s at c1 / c16 / c64 (~8K in, 256 out), the planted
key found in 1,039,890 prompt tokens, 80 default-effort answers completed.
