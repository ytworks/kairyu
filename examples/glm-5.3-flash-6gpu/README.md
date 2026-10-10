# GLM-5.3-Flash on six RTX PRO 6000 GPUs

One GLM-5.3-Flash replica on GPUs 0–5 of 96 GB RTX PRO 6000 Blackwell
(SM120, PCIe) cards, behind the same L2/L3 structure as
`deepseek-v4.1-flash-6gpu`: one Kairyu `ReplicaPool` replica, the
OpenAI-compatible API, and Open WebUI. GPUs 6–7 are not used.

```text
Open WebUI (:3016) -> Kairyu (:8015) -> ReplicaPool -> vLLM v0.31.0 (DP6 / EP6, GPUs 0-5)
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
- `clear_thinking`: the official template keeps earlier turns' reasoning
  (`false`, for agents that send `reasoning_content` back). The model author
  asks chat clients to pass `clear_thinking=true`; the Chat UI does, and an
  API client may send `"chat_template_kwargs": {"clear_thinking": true}`.
- Images: up to 8 per request (8 MiB each); video is not accepted (Kairyu's
  public API carries image parts only).
- Tools: OpenAI tool calling, parsed by vLLM's `glm47` parser.
