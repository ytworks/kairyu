# Measurements: winnow-12b-q8-dp8-8gpu

Status: **GPU verified 2026-10-05.** `./verify.sh all` passes all six gates
on 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition (one replica per GPU).

- Run: `20261005-gpu8-1a38f7f`, 02:44:33–02:46:53 UTC.
- Served config SHA-256: `95e223ce…0854`.
- Kairyu at PR #620 head `1a38f7f` (after the second review: `n > 1`
  rejected before dispatch, the 11-row contract gate, and the stream gate
  that assembles the streamed tool call).

## Images

| Image | Digest |
|---|---|
| `local/winnow-inference:77d1458-gemma4req-sm120` | `sha256:c87eedcbfef9dceeeeaa8ece3345cb05d74c3bdd9c58b6d14903a354aa385c2f` |
| `local/kairyu:winnow-llamacpp-example` | `sha256:65bbf8821a7052fb38832493c5b02e30b961bdd24d6c69c367495c88515725b5` |

The `winnow-server` image is the 1-GPU example's: winnow-inference `77d1458`
plus llama.cpp's Gemma 4 `required` tool-grammar fix `f072b10`.

## Gates

| Gate | Result |
|---|---|
| `attest` | PASS on all 8 replicas. Each reports `/props` `build_info` `b11036-911f6cdc8`, 8 slots, 65,536-token slots, the Gemma 4 sampling defaults, a tool-capable template and vision. |
| `contract` | PASS on all 8 replicas, all 11 rows. |
| `tool-calling` | PASS: auto, named, tool-result turn, streaming (assembled `get_weather` call), and a 16-request burst that the placement log shows on all 8 replicas. |
| `vision` | PASS: PNG and WebP. |
| `systemone` | PASS. The answers through Kairyu match a direct read. |
| `serving` | PASS, 3 complete rows (below). |

The first run (`20261005-gpu8`) used the image without `f072b10`. It failed
exactly as the 1-GPU example did:

- two contract rows on every replica;
- `tool-calling/named`.

Every other gate passed, including `every_replica` and all serving rows.

## Serving

Through Kairyu and the ReplicaPool. Each row has 128 requests, a prompt of
about 1K tokens and exactly 256 output tokens (`ignore_eos`). Every row has
32,768 completion tokens.

| Concurrency | TTFT p50 | TTFT p99 | TPOT mean | Per-request tok/s | Aggregate output tok/s |
|---|---|---|---|---|---|
| 8 (1 per replica) | 219 ms | 264 ms | 12.1 ms | 82.3 | 615.1 |
| 32 (4 per replica) | 908 ms | 1,266 ms | 24.3 ms | 44.4 | 988.1 |
| 64 (8 per replica) | 1,764 ms | 2,476 ms | 23.9 ms | 42.1 | 2,101.0 |

Each row matches the 1-GPU row at the same per-replica concurrency. The pool
spreads load evenly. The relatively slow 4-per-replica row is the runtime's
batch-4 decode (see the 1-GPU example).

## GPU memory

After the run, each replica used 21,391 MiB (model and 8 x 65,536-token q8_0
chat slots). GPU 0 used 24,229 MiB: it served the `systemone` gate's direct
read, which allocated its decision context. The card has 97,887 MiB.

KV is allocated up front. The 1-GPU example's sampling shows usage stays
flat under serving load. These settings are kept as committed.
