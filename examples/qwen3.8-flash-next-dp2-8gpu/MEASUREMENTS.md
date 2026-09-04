# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The locked run measures the
tool-calling configuration (vLLM-owned chat rendering via the example-local
template, `--tool-call-parser qwen3_xml`, thinking off by default, image
input enabled) against an already-running stack, with row-unique prompt
prefixes so neither vLLM prefix caching nor Kairyu's prefix-aware placement
can inflate the matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0, PCIe-only interconnect (no NVLink paths).
- Model revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`
- Model tree SHA-256:
  `20d458ae01a9b053cf72ba2084ec29a055fdac08bddcd7b61df946db9fd0bb10`
- vLLM: upstream `27a94d1ce4e3fc100c4732439ccec10f8246a804` (digest-pinned
  nightly wheel) with FlashInfer `60b49158ab4fb81718aef486c2d3c89aec4c1901`
  overlaid from source (`vllm-sm120.Dockerfile`), image ID
  `sha256:b47e22101829b471e8d3b0702a50ab884f58390cc2dfd50cb83aaf43ae30e9d4`
- Served-config SHA-256:
  `ac3c8f9ff8a7fb1178bfdd797a8410b8ce6052e57f7bac86a9ffe27416d9af8f`
  (identical across the three gate runs below). Two earlier passes are
  superseded: the runs with the recipe's MTP k=3 enabled
  (`20260904T121052Z`/`121116Z`/`121132Z`, served-config `c8acee3c…`), whose
  vision gate returned `ductduct…` for 2 of 4 batched answers (see "Why MTP
  is off"), and the first MTP-off runs (`20260904T130031Z`/`130046Z`/
  `130058Z`, served-config `a1fc7b55…`), which measured replica 0 on a
  cold-boot 741K-token KV cache against replica 1's 3.47M (see "Why the KV
  budget is pinned"); their output rates were within 3% of the table below
  at every row (564 vs 582 tok/s at c32), with a higher c32 TTFT p50
  (4.3 s vs 3.3 s).
- L1: 2 x TP4 replicas (GPU 0-3, GPU 4-7), the official recipe's
  `rtx_pro_6000_4x` layout minus MTP: `max_num_batched_tokens=8192`,
  `max_num_seqs=16`, prefix caching, FlashInfer autotune off, full/piecewise
  CUDA Graphs, KV budget pinned with `--kv-cache-memory 46346979738`
  (43.16 GiB — the value vLLM derives for the recipe's 0.95 utilization on
  a warm start; vLLM then skips memory profiling); vLLM renders chat with
  `qwen3.8-flash-next-chat.jinja` (official template + L3 effort alias) and
  parses XML tool calls (`--enable-auto-tool-choice --tool-call-parser
  qwen3_xml`, `--reasoning-parser qwen3`, thinking off by default). KV cache
  per replica: 3,449,784 tokens (13.16x the 256K context), identical on both
  replicas, replica 0 booted with an empty torch.compile cache.
- L2: one pool, `prefix_index: true`, `queue_depth_threshold: 0`,
  `unhealthy_after: 1`; Kairyu forwards tools to /chat/completions
  (`legacy_chat_models`) and admits images under `image_input_policy`.

## Warm serving result

Each operating point completed 64/64 requests with approximately 8K input
tokens and exactly 256 output tokens per request (success rate 1.0 in every
row). Percentiles use the harness's nearest-rank method. The placement gate
requires exactly the row's 64 placements, both replicas served, and neither
above 1.25x the even share (40 of 64) at concurrency >= 8. The matrix stops
at c32 because `max_num_seqs=16` per replica (the recipe's verified value)
caps the pool at 32 concurrent sequences.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s | Placements (replica 0, 1) | Placement gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 727.74 | 764.58 | 2,955.00 | 8.732 | 0.34 | 86.58 | 64, 0 | reported only |
| 8 | 2,368.26 | 2,637.62 | 5,555.14 | 13.633 | 1.43 | 366.96 | 32, 32 | pass |
| 16 | 3,346.90 | 4,684.59 | 8,497.18 | 21.864 | 1.87 | 479.83 | 32, 32 | pass |
| 32 | 3,258.06 | 9,047.17 | 13,859.51 | 38.644 | 2.27 | 582.21 | 32, 32 | pass |

Run ID: `20260904T133019Z`.

## Tool-calling gate (same served configuration)

`./verify.sh tool-calling` run `20260904T132953Z`: **PASS, 6/6 cases** —
4 concurrent auto bash tool calls (the SWE-bench Pro mini-swe-agent request
shape) split 2/2 across the replicas, the `role: "tool"` follow-up turn, the
streamed variant, `reasoning_effort: high` with tools (aliased to the
official `medium` by the template; reasoning_content present, call parsed),
and the non-thinking default. `run.sh up` additionally fails closed on a
readiness tool-call probe.

## Vision gate (same served configuration)

`./verify.sh vision` run `20260904T133007Z`: **PASS, 2/2 cases** — 4
concurrent image requests (2 per replica; the same solid red PNG with a
row-unique "Vision case N: what single color fills this image? Answer with
one word." prompt) all named the colour ("Red" x3, "red"; the gate requires the
word, not just non-empty content), and the placement log correlated them
2/2 across the replicas by `x-request-id`. `run.sh up` additionally fails
closed on a readiness image probe.

## Why MTP is off

The recipe's `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
was enabled in the first GPU pass. Single requests were correct, but as soon
as requests were batched a fraction of answers came back as degenerate
repetition (`ductductduct…`, `duct Register Register…`) — greedy, text-only
requests included. Isolation on one replica, 2-12 concurrent, greedy,
16-token answers, unique prompts:

| Configuration | Corrupted answers |
|---|---|
| MTP k=3 + prefix caching (recipe) | 13 / 274 (4.7%) |
| MTP k=1 + prefix caching | 23 / 407 (5.7%) |
| MTP k=3 + prefix caching + `--no-async-scheduling` | 427 / 669 (63.8%) |
| MTP k=3, prefix caching off | 0 / 646 |
| MTP off, prefix caching on (this example) | 0 / 862, then 0 / 1,308 on the final stack |

This matches upstream
[vllm-project/vllm#53912](https://github.com/vllm-project/vllm/issues/53912)
(prefix caching + MTP corrupting hybrid GDN models, reproduced by others on
H100), so it is not SM120-specific. Prefix caching was kept: an 8K-context
follow-up turn starts in 194 ms with it and 1,340 ms without, and Kairyu's
prefix-aware placement has no L1 effect without it. The cost is decode speed
— 104 vs 175 output tok/s single-stream and 512 vs 687 tok/s at c8 in the
same ad-hoc comparison (512-token answers, short prompts).

## Why the KV budget is pinned

vLLM sizes the KV cache from a start-up memory profile. On the first boot of
a replica whose torch.compile cache is empty, that profile includes the
inductor compile scratch (35.3 GiB "peak activation" against 1.18 GiB on a
warm replica), so the first MTP-off pass served replica 0 with 741,471 KV
tokens (9.28 GiB) next to replica 1's 3,469,597 (43.4 GiB) — the same
command, the same image. `--kv-cache-memory 46346979738` reserves the
warm-start value up front and vLLM skips the profile; the locked run above
was taken with replica 0's compile cache deliberately removed, and both
replicas report 3,449,784 tokens.

## Reading the result

- **Placement is even.** At every concurrency >= 8 the two replicas received
  exactly 32/32 of the 64 requests; c1 sends all 64 requests to replica 0
  because least-outstanding ties resolve to the lowest replica id (reported
  only, by design).
- **TTFT is prefill-bound from c8.** An 8K prompt prefills in about 2 s on
  one TP4 replica with the current QSA indexer prefill kernels, and each
  replica handles 4/8/16 such prompts per row within its 8,192-token batch
  budget, so TTFT grows roughly linearly with per-replica concurrency
  (2.4 s / 3.3 s p50 at c8 / c16; the c32 p50 sits at 3.3 s because the
  16-sequence cap queues the rest, which shows up as the 9.0 s p99). Decode
  is not the limiter: mean TPOT stays under 40 ms at c32.
- **Two replicas scale to 6.7x the c1 output rate at c32** (582.2 vs 86.6
  tok/s) while both stay within the recipe's 16-sequence cap.

## Raw evidence identity

Artifacts are below
`/mnt/nvme/kairyu/model-volumes/qwen3.8-flash-next-dp2-8gpu/verification-results/20260904T133019Z/`
on the measurement host (`run.json`, per-row `*-serving.json`,
`placement.json`, and logs); the tool-calling gate is under
`…/20260904T132953Z/tool-calling.json` and the vision gate under
`…/20260904T133007Z/vision.json`.

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `650bb8a5dcc0ab56ff0fc17875598df3bb06ee3a2fd9b8182f0a639b7918605d` |
| Serving concurrency 8 | `efcb9678b24f728bc185fcfadf5f3750fb7e68a74bc4eff5201f5bf24672da74` |
| Serving concurrency 16 | `bd0b966fbba026b0869d9c915eee24e247619295fea724bfb2ec7a517696546c` |
| Serving concurrency 32 | `5bef15490ef6ff58f81bf725799f380b1a288fab6540416219750485ef5838a5` |
| Tool-calling gate | `24a031a4de8689e82bbea5d49616e43be18804983e4bf5761dc855d70f2f2a12` |
| Vision gate | `ce2fa61af4ed751a696d7f528153e0126f51778a9bc4a49898e280b18e64b524` |
