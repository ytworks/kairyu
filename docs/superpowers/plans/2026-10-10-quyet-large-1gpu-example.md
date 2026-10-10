# Quyet-1.0-Large single-GPU example (chat + System One)

Status: **Approved 2026-10-10; implementation and GPU gates in progress.**

Accepted scope (owner, 2026-10-10): the Jev-family example shape
(`winnow-12b-q8-1gpu`, `openjev-diffusiongemma-26b-1gpu`) for
[chinhnc/Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) on one
RTX PRO 6000 Blackwell. Kairyu (`kairyu/`) is not changed.

## Sources (pinned)

1. Checkpoint `chinhnc/Quyet-1.0-Large` at `9a0f051112ba4301f6e4c8b87fc088ce00b11aae`:
   Gemma-4-31B-it with a merged rank-16 LoRA, bf16, 62.5 GB, Apache-2.0 (keep
   `NOTICE`). The publisher's `MANIFEST.sha256` lists every file's hash.
2. Runtime package `quyet` 1.0.2 (PyPI, Apache-2.0). It has no server. A decision
   is one forward pass per question: the options are lettered A..J in a fixed
   chat prompt (prompt version 2), and the answer is the softmax of those letters'
   next-token logits at a per-type calibration temperature (`quyet_config.json`:
   choice 1.3007, score 1.3159, noul 1.4957). Only the state is truncated
   (6,000 tokens; 8,000-token prompt). Text only, at most 10 options.
3. The model card: the checkpoint "serves with vLLM as a standard
   gemma-4-31B-it-architecture checkpoint, but the decision prompt and
   temperatures live in the package".
4. vLLM `v0.31.0` (`vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd…`, latest stable,
   2026-10-04): Gemma 4 support and `/v1/completions` `logprob_token_ids`.
5. Precedent: OpenJev serves JevK5 (also a merged-LoRA letter-readout model)
   exactly this way: vLLM returns the letters' logprobs, the prompt comes from the
   model's own package. OpenJev measured, against JevK5's own transformers run on
   231 items: same top answer on all 231, same input tokens, probability
   difference 0.0012 median and 0.055 max.

## Shape

```text
Open WebUI (:3014) -> Kairyu (:8014) -> vLLM: Quyet-1.0-Large bf16, one GPU       (chat)
Playground (:3015) -> Kairyu /v1/systemone -> quyet-systemone (CPU) -> same vLLM   (System One)
```

- **L1 vLLM** (stock image, no overlay): bf16, prefix caching, Gemma 4 tool and
  reasoning parsers, up to 4 images, the checkpoint's generation defaults
  (temperature 1.0, top_k 64, top_p 0.95). Context 65,536 unless vLLM's KV cache
  cannot hold 8 such sequences, then 32,768 (recorded in MEASUREMENTS.md).
- **L1 quyet-systemone** (example-owned, CPU only): `quyet` 1.0.2 with only the
  forward pass replaced. A subclass of `quyet.llm.runtime.LLMModel` overrides
  `__init__` (tokenizer and config, no torch model) and `_letter_logits` (vLLM
  `/v1/completions` with the exact prompt token IDs, one token,
  `logprob_token_ids` = the option letters). Prompt, truncation, calibration and
  the answer shape (`warnings`, `truncated`) are the package's own code. Jev
  wire contract as OpenJev's: 422 for a wrong shape, 400 for a question it cannot
  ask and for `images`/`think`/`samples`>1/`steps`>1/`sequential`, 529 when its
  queue is full, `Server-Timing`. Startup refuses a different `quyet` version or
  multi-token letters.
- **L3 Kairyu** (configuration only): pool `quyet-1.0-large` (one vLLM replica,
  legacy chat, multimodal admission) and `systemone: quyet-1.0-large-systemone`
  (aliases `quyet-latest`, `jev-latest`, `jev-preview`). Chat admission leaves
  vLLM room for reads; Kairyu forwards fewer System One requests than the
  adapter accepts, so callers get Kairyu's 429, never the adapter's 529.
- **UI**: Open WebUI for chat; a Jev-style playground (System One answers left,
  the chat model's answer to the same questions right).
- Chat quality is not claimed: the model is trained for decisions; gates prove
  that chat, tools and images work.

## Files (all example-owned)

`examples/quyet-1.0-large-1gpu/`: `README.md`, `MEASUREMENTS.md`, `example.json`,
`compose.yaml`, `kairyu.yaml`, `run.sh`, `verify.sh`, `control.py`,
`verification.py`, `quyet_systemone.py`, `quyet-systemone.Dockerfile`,
`systemone-reference.jsonl`, `playground/{index.html,nginx.conf}`.
Tests: `tests/unit/test_quyet_1gpu_example.py`; one entry in the example list of
`tests/unit/test_frontier_examplectl.py`. Docs: `examples/README.md`, FN-D9
Quyet one-GPU amendment in `docs/design/frontier-native-runtime.md`,
`PROGRESS.md`.

## Verification (rebuilt 2026-10-10)

The first gate list checked chat features (tool calls, images, chat throughput).
That is not how a Jev-family model is used, and the decision fine-tune does not
write Gemma 4 tool calls. On the owner's instruction the gates were rebuilt from
TypeSafe's documentation (docs.typesafe.ai): a System One model is a decision API
for software, with typed answers, calibrated probabilities, many questions per call,
consistent answers and an official SDK; JevBench is the independent benchmark for
Jev-compatible systems (Quyet-1.0-Large is first on its open-weights board).
Chat stays plain text for the playground; tools and images are not offered.

CPU: ruff and the changed tests only (adapter refusals and overload against a fake
vLLM, the kairyu.yaml / example.json contract, the parity comparison).

GPU gates, in order; stop and report at the first failure:

| Gate | Claim | Pass criteria |
|---|---|---|
| `reference` | official answers exist | stack down; the `quyet` CLI (transformers, GPU) answers the 48 authored requests and JevBench's 231 public items (pinned revision, hash-checked) |
| `attest` | the pinned stack runs | images, checkpoint re-hash, vLLM settings and version, sampling defaults, context rule, adapter calibration, Kairyu model lists |
| `systemone` | Kairyu's answers are the official package's | all 279: same input tokens and truncation; same top option where the official top-two gap is >= 0.05; probability difference median <= 0.005, max <= 0.06; aliases; error shapes |
| `jevbench` | Jev-standard quality and speed | JevBench's runner (typesafe adapter, sequential) on Kairyu: 100 % valid; per split correct within 1 item of official, Brier and ECE within 0.01; p50 <= 0.5 s |
| `fanout` | many questions per call | 1/8/32 questions on one 1K-token state all answered; 32 questions <= 4x one |
| `consistency` | same request, same answer | 10 alone + 10 under load: top answers stable, probabilities within 0.01 |
| `sdk` | the official client works | typesafe-sdk 0.7.4: typed answers under jev-latest / quyet-latest / full name; 11 options -> TypeSafeBadRequestError |
| `systemone-serving` | bulk decisions | states ~50/2,000/6,000 tokens, 3 questions, c1/16/32/64 x 64: all answered; req/s, p50/p95 |
| `systemone-isolation` | a decision burst does not break chat | 640 reads + 8 chats: reads 200 or 429 only, chats answer, replica healthy |

## Checklist

- [x] Owner approves this plan.
- [ ] Example files, CPU tests, lint.
- [ ] GPU gates; MEASUREMENTS.md; FN-D9 amendment; PROGRESS.md.
