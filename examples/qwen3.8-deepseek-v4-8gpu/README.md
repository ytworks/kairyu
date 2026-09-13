# Qwen3.8 + DeepSeek V4.1 critical ensemble on 8 × RTX PRO 6000

This existing example now configures DeepSeek-V4.1-Flash on GPUs 0–5 and two
Qwen3.8-27B-FP8 TP1 replicas on GPUs 6–7. Four Qwen candidate roles share the
two replicas. The public API remains `:8003`, the Chat UI remains `:3000`, and
the public model IDs remain `kairyu-auto-max` and `embed-small`.

**Status: implementation candidate, not a validated six-GPU deployment.**
The V4.1 runtime and checkpoint come from the current
[standalone TP8 example](../deepseek-v4.1-flash-8gpu/README.md). Its successful
TP8 measurements do not validate this topology. A 32-head independent
quantized-arithmetic oracle now passes all 16 cases at the unchanged
tolerance. The original BF16-Q floating-reference comparison retains two
one-element failures; the saved outlier is reproduced by the intended FP8
Q arithmetic. Same-fixture eight-head and 32-head slices are bit-exact.
The full-model candidate has not started, and the original services remain
healthy. See [MEASUREMENTS.md](MEASUREMENTS.md) for the separate arithmetic
and fidelity results and [the approved implementation plan](../../docs/superpowers/plans/2026-09-13-v41-six-gpu-critical-ensemble.md)
for the remaining gates. Do not promote this configuration until six-GPU
startup, native AUTO contracts and live product gates close.

## Runtime and allocation

| Worker | Physical service | GPUs | Parallelism | Runtime |
|---|---|---|---|---|
| `tier1` | `qwen-0`, `qwen-1` | 6, 7 | TP1 × 2 | pinned upstream vLLM v0.23.0 |
| `tier2`, `tier2-direct` | `deepseek` | 0–5 | TP2, attention DP3, EP6 | existing V4.1 SM120 overlay |

The physical assignment preserves the existing six-plus-two host placement.
It is independent of the logical order of the roles. TP6 is not the candidate:
V4.1 has 64 attention heads and eight output groups, which are not divisible
by six. TP2 with DP3 and EP6 is source-feasible; its memory, startup, correctness
and performance still need actual model evidence.

The complete immutable identities live in [example.json](example.json):

- Qwen checkpoint revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`,
  upstream image digest `6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`.
  Its existing [Qwen template](qwen3.8-chat.jinja), FP8 KV, float16 Mamba cache,
  PIECEWISE graphs, 32K batching, 32 sequences and disabled MTP are retained.
- DeepSeek checkpoint revision `dba1be0a40aa45a94ad051997016db3960a90277`,
  image ID `027bf47b2bd6f0d0abe54b296e7e9e3d31ee103bb6e46fa0a9807117681c2359`,
  vLLM source `179dd0fa9`, FlashInfer revision
  `60b49158ab4fb81718aef486c2d3c89aec4c1901`.
  The build uses the standalone example's existing Dockerfile and
  `patch_runtime.py`; this example adds no runtime Python implementation.

DeepSeek keeps the main runtime's native `deepseek_v41` tokenizer, reasoning
and tool parsers; Marlin MoE; FP8 MLA KV; MXFP4 indexer; block size 64; and
DSpark five-token probabilistic drafting with block rejection and adaptive
verification disabled. This candidate changes TP8/EP8 to TP2/DP3/EP6, enables
CPU Engram offload and sets 32 sequences. It keeps 16K batching and 0.90 GPU
memory utilization. The Python frontend remains enabled with
`VLLM_USE_RUST_FRONTEND=0` so the pinned native image encoder is used.
CPU Engram offload trades GPU memory for host memory and transfer cost; no
six-GPU throughput or host-memory adequacy result is claimed.

## Native conversation and reasoning

Both DeepSeek aliases call the same native `/v1/chat/completions` service.
`deepseek-v4.1-flash-thinking` and `deepseek-v4.1-flash` are private pool names;
role effort determines thinking. The former V4 completion passthrough,
handwritten DeepSeek prompt templates and local gateway tokenizer mounts are
removed from the runtime configuration. Original system, developer, user,
assistant and tool messages, tool IDs, assistant reasoning and ordered image
parts travel through the typed native chat contract.

The ensemble's V4.1 service default is `{"thinking":false}`. A CPU check using
the exact pinned image and actual checkpoint exercised
`ChatCompletionRequest.build_chat_params` and the native tokenizer:

| Request effort | Native thinking | Native budget | Assistant suffix |
|---|---|---|---|
| omitted | disabled | none | closed `</think>` |
| `low` | enabled | 50 | open `<think>` |
| `high` | enabled | 75 | open `<think>` |
| `max` | enabled | 100 | open `<think>` |

This proves rendering, not completed GPU output. Requirements extraction
uses a `high` floor: omitted/low/high become high and explicit max remains
max. Other thinking DeepSeek roles inherit the caller's effort, with high
as the L2 default. Qwen thinking roles retain the existing medium behavior:
the unchanged template maps the DSL's `reasoning_effort: high` to medium.
Non-thinking Qwen roles retain their existing settings.

Both model families accept one image per request under the existing gateway
limits: 8 MiB, 2,097,152 pixels, maximum dimension 4096 and aspect ratio 200.
DeepSeek roles receive the original image natively. There is no
`image_description` replacement stage. V4.1's standalone eight-image limit
is not inherited by this product.

## Routing and primary workflow

The existing bounded Qwen judge chooses one of five profiles. Its serial
latency belongs to request TTFT. Invalid or failed verdicts fall back to
`primary`; direct-route traffic cannot substitute for an ensemble performance
result.

| Judge label | Profile | Generation policy |
|---|---|---|
| `QWEN` | `qwen_direct` | non-thinking Qwen; T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5 |
| `QWEN_THINK` | `qwen_think_medium` | Qwen medium; T=1, top_p=0.95, top_k=20 |
| `DEEPSEEK` | `deepseek_direct` | native non-thinking V4.1; T=1, top_p=1 |
| `DEEPSEEK_THINK` | `deepseek_think` | native V4.1 at inherited effort; T=1, top_p=1 |
| `ENSEMBLE` | `primary` | requirement extraction, five candidates, critical review, synthesis and audit |

[auto-max.yaml](auto-max.yaml) owns the prompts and dependencies. With the
existing level-synchronous scheduler the primary profile has these waves:

1. Qwen `head`, DeepSeek `requirements`, and independent `deepseek_candidate`.
   The candidate sees only the original conversation and response intent;
   requirements extraction produces a checklist rather than an answer.
2. DeepSeek `policies` derives four different approaches from the source and
   checklist.
3. Qwen `answer_1` through `answer_4` produce four complete candidates on the
   two replicas, each bound to its policy and the original source.
4. DeepSeek `review` checks all five candidates against the source and
   requirements, separating defensible criticism from unsupported objections.
5. DeepSeek `synthesis` writes the answer, and a separate DeepSeek `audit`
   checks the result. Bounded repair applies to failed minimum requirements;
   optional suggestions do not become new requirements.

The original conversation remains available to every dependent role.
Checklists, candidates and reviews are derived data and do not replace or
truncate it. The existing head conditions suppress the opening for tool
calls, structured responses, multiple choices and other headless contracts.
The audit gates the publishable answer; private stage output remains separate
from public answer content in the trace and Chat UI. The sandbox service
remains deployed and unreferenced by this DAG.

## MAX, context and validation limits

Public `max_tokens` / `max_completion_tokens` constrains each public answer
choice, including its reasoning and visible output. Private role budgets
remain separate. The Chat UI no longer injects the old implicit MAX 65,536;
an explicit user MAX remains a caller constraint. Qwen direct profiles keep
their 131,072 cap. DeepSeek direct profiles impose no inherited V4 393,216
cap; they use the caller allowance and context capacity.

The private ceiling is provisionally 131,072, including the complete Qwen
candidates and native DeepSeek private roles. This is an implementation
starting point, not evidence that every stage can finish within that limit.
The step budget is 19 for one choice plus 9 per additional choice: initial
DAG work, up to two repairs, two verdict attempts per candidate and one
thinking continuation. Required roles fail the request on missing, failed,
empty or truncated inputs; optional roles retain their existing fallback. Context fitting must use the actual native rendered request and keep
the input intact. A smaller output allowance cannot make an oversized input
valid. Final usage must include retries, discarded choices and audits.

The functional gate must validate native text, tools and image flows;
requirement extraction and review/audit outcomes; each public choice;
reasoning and output accounting; timeout/cancellation; and terminal errors.
It must reject missing or truncated required stages. Performance must include
forced-primary runs at concurrency 1/8/16/32 and a DeepSeek-direct baseline
from the same V4.1 candidate. Historical V4 latency fallback values have been
removed. Old measurements remain dated evidence for the old configuration.

## Operation

The existing entry points remain `run.sh`, `verify.sh` and `browser-smoke.sh`.
The commands below describe operation after the pending gates permit startup;
this revision was validated without restarting the running product.

Prerequisites are eight RTX PRO 6000 Blackwell Server Edition GPUs with at
least 90,000 MiB each, Docker Compose with NVIDIA GPU access, and NVMe storage
under `/mnt/nvme`. Model and cache data use these existing locations:

```text
/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/models/
/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-8gpu/models/
/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4-8gpu/compile-cache/
/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4-8gpu/webui-data/
```

`control.py` verifies the GPU inventory, discovers NUMA-local CPU affinity,
pulls the pinned Qwen image or builds the existing standalone V4.1 overlay,
checks the V4.1 image ID, and attests both checkpoint trees. It then starts
the Compose stack, validates readiness/routing/embeddings/tokenizer access,
and provisions the existing Reasoning Effort dropdown. The storage preflight
requires 600 GiB free for checkpoint preparation. `VERIFY_MODEL=1` rehashes
an already attested checkpoint. `HF_TOKEN`, when needed to download, is
forwarded by environment name and is not written to configuration.

Keep a stable `KAIRYU_RESPONSES_COMPACTION_SECRET` of at least 32 bytes across
gateway restarts. The existing deployment must retain its secret. Once the
gates close and that environment is configured:

```bash
./examples/qwen3.8-deepseek-v4-8gpu/run.sh up
./examples/qwen3.8-deepseek-v4-8gpu/run.sh status
./examples/qwen3.8-deepseek-v4-8gpu/run.sh logs
./examples/qwen3.8-deepseek-v4-8gpu/run.sh down
```

`NVME_STORAGE_ROOT`, `API_BIND_ADDRESS`, `API_PORT`, `CHAT_UI_BIND_ADDRESS`,
`CHAT_UI_PORT`, `PUBLIC_HOST` and `WEBUI_URL` retain their existing meanings.
API and UI bind to all interfaces by default. DeepSeek L1 port 8005 stays
loopback-only for exact tokenizer and baseline probes. The pinned Chat UI is
auth-disabled, lists only `kairyu-auto-max`, and keeps its separate
intermediate-processing display. `embed-small` remains the same offline
384-dimensional MiniLM embedding service, with concurrency two.
