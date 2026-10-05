# Measurements: winnow-12b-q8-1gpu

Status: **GPU verified 2026-10-05.** `./verify.sh all` passes all six gates
on 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition (GPU 0).

- Run: `20261005-gpu1-1a38f7f`, 02:39:55–02:43:26 UTC.
- Served config SHA-256: `04ce07a0…ba64`.
- Kairyu at PR #620 head `1a38f7f` (after the second review: `n > 1`
  rejected before dispatch, the 11-row contract gate, and the stream gate
  that assembles the streamed tool call).

## Images

| Image | Digest |
|---|---|
| `local/winnow-inference:77d1458-gemma4req-sm120` | `sha256:c87eedcbfef9dceeeeaa8ece3345cb05d74c3bdd9c58b6d14903a354aa385c2f` |
| `local/kairyu:winnow-llamacpp-example` | `sha256:65bbf8821a7052fb38832493c5b02e30b961bdd24d6c69c367495c88515725b5` |

The `winnow-server` image is winnow-inference `77d1458` (llama.cpp `911f6cd`,
b11036, plus Winnow's four patches) with one more patch: llama.cpp's
`f072b10` from `winnow-patches/`.

## Gates

| Gate | Result |
|---|---|
| `attest` | PASS. `/props`: `build_info` `b11036-911f6cdc8`, 8 slots, 65,536-token slots, defaults temperature 1.0 / top_k 64 / top_p 0.95 / min_p 0 / repeat_penalty 1.0, tool-capable template, vision projector. |
| `contract` | PASS, all 11 rows of `l1.correctness.llamacpp_upstream_contract`. |
| `tool-calling` | PASS: auto, named, tool-result turn, and streaming (the streamed `get_weather` call is assembled and checked, with `finish_reason: tool_calls`). |
| `vision` | PASS: PNG answered "red", WebP (re-encoded as PNG by Kairyu) answered "blue". |
| `systemone` | PASS. The answers through Kairyu match a direct read. |
| `serving` | PASS, 3 complete rows (below). |

**Why the extra patch.** The first run (`20261005-gpu1`) used the image
without `f072b10` and failed two contract rows:

- `named_tool_choice`;
- `stream_usage_and_done`, which ran out of tokens.

It also failed `tool-calling/named` with HTTP 502 `tool_choice_not_satisfied`.

At b11036, the Gemma 4 grammar ignores `tool_choice: "required"`, and the
model answers in text. llama.cpp fixed this in `f072b10` (first in b11058).
With the patch, every gate passes. All other gates passed in both runs.

## Serving

Through Kairyu at `/v1/chat/completions`. Each row has 32 requests, a prompt
of about 1K tokens and exactly 256 output tokens (`ignore_eos`). Every row
has 8,192 completion tokens.

| Concurrency | TTFT p50 | TTFT p99 | TPOT mean | Per-request tok/s | Aggregate output tok/s |
|---|---|---|---|---|---|
| 1 | 212 ms | 242 ms | 12.1 ms | 82.4 | 77.3 |
| 4 | 856 ms | 1,118 ms | 29.2 ms | 34.3 | 124.7 |
| 8 | 1,561 ms | 2,125 ms | 24.3 ms | 41.5 | 265.7 |

At concurrency 4, per-request decode (34 tok/s) is slower than at 8
(43 tok/s). This reproduced in all three runs and on the DP8 example at 4 requests
per replica. It is the runtime's batch-4 decode speed, not interference.

## GPU memory (GPU 0, sampled every 2 s)

| Phase | Used |
|---|---|
| Model and 8 x 65,536-token chat slots loaded (q8_0 KV) | 21,297 MiB |
| After the first `/v1/systemone` read (decision context allocated by `--memory auto`) and during all serving rows | 24,229 MiB (peak) |

The card has 97,887 MiB. The committed geometry fits with about 73 GiB
spare. KV is allocated up front, so serving load does not raise usage. These
settings are kept as committed.
