# Measurements: quyet-1.0-large-1gpu

Status: **GPU verified 2026-10-10.** All nine gates pass on 1 x NVIDIA RTX PRO 6000
Blackwell Server Edition (GPU 0), run `20261010-s1-r2`, against one stack instance
started 2026-10-10 07:50 UTC at `452bee96`.

- `reference`, `attest`, `systemone`, `fanout`, `consistency`, `sdk` and
  `systemone-isolation` ran on `452bee96`.
- `jevbench` and `systemone-serving` ran on the same instance after two gate fixes
  (only `verification.py` and the verification section of `example.json` changed):
  JevBench never overwrites raw evidence, so each run now gets its own raw directory;
  the serving rule now accepts Kairyu's designed 429 above its forwarding limit (below).
  The ECE comparison moved from per split to all 231 items (owner decision).

## Images and checkpoint

| Item | Value |
|---|---|
| vLLM | `vllm/vllm-openai@sha256:c1c9f6fd…67c3` (v0.31.0), `VLLM_BATCH_INVARIANT=1` |
| Adapter | `local/quyet-systemone:1.0.2-vllm-v0.31.0`, image `sha256:50f7132d…ce1b` on this host (containerd store); its source labels match the tree |
| Kairyu | `local/kairyu:quyet-1gpu-example`, image `sha256:11519f15…f1f2` |
| Checkpoint | `chinhnc/Quyet-1.0-Large@9a0f0511`, 15 files, tree `8f287d92…2a05`; every file matches the publisher's `MANIFEST.sha256` and was re-hashed by `attest` |

vLLM reports a 36,226-token KV cache at `--max-model-len 8192`, 64 sequences and 95 %
of GPU memory. GPU 0 holds 93,619 of 97,887 MiB while serving.

## Gates

| Gate | Result |
|---|---|
| `reference` | PASS. The `quyet` CLI (transformers, bf16, GPU 0, stack down) answered all 279 requests in 98.7 s, four of them truncated. Its answers were identical across three runs. |
| `attest` | PASS, 10 cases: registry digest and adapter source labels on the running containers, vLLM 0.31.0 and settings (batch invariance included), checkpoint re-hash, adapter calibration (`quyet` 1.0.2, prompt version 2, temperatures), System One public with four names, no chat model (`/v1/chat/completions` answers 404). |
| `systemone` | PASS. All 279 requests had the official input token count, truncation and shape. 308 of the 309 official answers with TypeSafe confidence >= 0.5 kept their top option (99.7 %; the flip is a 3,290-token policy state). Over 1,172 probabilities the difference is 0.0001 at the median, 0.072 at p99 and 0.22 at most. Aliases and the documented refusals (11 options, images, think, unknown type, malformed body, unknown model) pass. |
| `jevbench` | PASS (table below). |
| `fanout` | PASS. One 1K-token state: 1 question 0.246 s, 8 questions 0.409 s, 32 questions 0.653 s (p50, 8 requests each); 32 questions take 2.65 times one. |
| `consistency` | PASS. The same request 10 times alone and 10 times while 128 other reads ran: identical probabilities (spread 0.0). |
| `sdk` | PASS. typesafe-sdk 0.7.4 read typed answers under `jev-latest` (its default), `quyet-latest` and `quyet-1.0-large-systemone`; an 11-option Choice raised `TypeSafeBadRequestError`. Recorded, not gated: `client.models.list()` fails validation because Kairyu's Jev model list has no `release_date`. |
| `systemone-serving` | PASS (table below). |
| `systemone-isolation` | PASS. 640 concurrent reads: 80 answered, 560 got Kairyu's 429 in 7.8 s, no 529 or 5xx; Kairyu stayed ready on a healthy model server and the next read answered. |

## JevBench (public items, JevBench `b6b8fff7`, `typesafe` adapter, sequential)

| Split | Correct (Kairyu / official) | Brier (Kairyu / official) | ECE (Kairyu / official) | p50 / p95 |
|---|---|---|---|---|
| original (72) | 72 / 71 | 0.0177 / 0.0163 | 0.039 / 0.026 | 0.089 / 0.091 s |
| easy (48) | 48 / 48 | 0.0001 / 0.0001 | 0.005 / 0.005 | 0.090 / 0.100 s |
| hard (111) | 91 / 90 | 0.2817 / 0.2845 | 0.098 / 0.101 | 0.201 / 0.805 s |
| all (231) | 211 / 209 | 0.1409 / 0.1418 | 0.038 / 0.042 | 0.099 s |

Every answer was valid. The official column is the `quyet` package's own answers on
the same items, scored by JevBench's scorer. On the 72-item split two near-even answers
(0.505/0.495, 0.497/0.503) crossed ECE bins, which moved ECE by 0.013; over all 231
items the difference is 0.004.

## System One throughput (cache-busted states, 3 questions, 64 requests per row)

| State | c1 p50 | c1 req/s | c16 req/s | c16 p50 | c32 | c64 |
|---|---|---|---|---|---|---|
| ~50 tokens | 0.161 s | 6.2 | 20.4 | 0.72 s | 20.6 req/s, all answered | 19.3 req/s, all answered |
| ~2,000 tokens | 0.932 s | 1.07 | 1.22 | 12.6 s | 1.22 req/s, all answered | 51 answered, 13 shed (429) |
| ~6,000 tokens | 2.730 s | 0.37 | 0.39 | 40.8 s | 50 answered, 14 shed | 26 answered, 38 shed |

The GPU is the limit: about 2,400 prompt tokens per second for a 31B dense model in
bf16 with batch-invariant kernels. Up to Kairyu's forwarding limit (16) every request
is answered; beyond it Kairyu queues for 30 s and then answers 429, which TypeSafe's
SDK retries with backoff.

## Earlier runs (not evidence for the final configuration)

- `20261010-jev-r1` (chat still published, no batch invariance): reference and attest
  passed; `systemone` failed the first rule (max difference 0.12 > 0.06, one flip of an
  official 0.56/0.44 answer).
- `20261010-s1-r1` (System One only, no batch invariance): `systemone` failed the p99
  rule (0.067 > 0.06); the official answers matched run `jev-r1` exactly while vLLM's
  moved by up to 0.14 between configurations.
- `20261010-s1-bi` / `20261010-s1-nobi`: batch invariance made 20 repeats identical
  (0.0026 spread without) at about 10 % latency (1 question 0.225 -> 0.251 s,
  32 questions 0.60 -> 0.66 s).
