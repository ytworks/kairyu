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

## Verification

CPU: ruff and the changed tests only. Tests cover the adapter's refusals
(unsupported options, 11 options, wrong shape) and overload (529) against a fake
vLLM, and that Kairyu's System One forwarding bound stays at or below what the
adapter accepts. Base/head collection counts are reported.

GPU gates, in order; stop and report at the first failure (about 50 minutes
after the download):

| Gate | Claim | Pass criteria |
|---|---|---|
| `reference` | official answers exist | before the stack starts, the `quyet` CLI (transformers, GPU) answers the 48 requests of `systemone-reference.jsonl` |
| `attest` | the pinned stack runs | vLLM version and image, every checkpoint file hash, served name, context, generation defaults, `quyet` version, temperatures, adapter image ID match `example.json` |
| `systemone` | Kairyu's answers match the official package | per request identical `usage.input_tokens` and `truncated` flags; same top option wherever the official top-two gap is at least 0.05; absolute probability difference median <= 0.005 and max <= 0.06; aliases and error shapes through Kairyu |
| `tool-calling` | chat tools work | auto, named, tool-result turn and streamed calls |
| `vision` | chat reads images | PNG and WebP colors named correctly |
| `serving` | chat completes under load | 1K-in / 256-out (`ignore_eos`) at c1/4/8, 32 requests each, all complete |
| `systemone-serving` | decisions are fast | cache-busted 1K-token states, 3 questions, c1/16/32/64, all valid; c1 p50 <= 1.0 s |
| `systemone-isolation` | a decision burst does not break chat | 640 reads with 8 chats: reads only 200 or 429, chats answer correctly, the chat replica stays healthy |

## Checklist

- [x] Owner approves this plan.
- [ ] Example files, CPU tests, lint.
- [ ] GPU gates; MEASUREMENTS.md; FN-D9 amendment; PROGRESS.md.
