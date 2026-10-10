# Quyet-1.0-Large single-GPU example (chat + System One)

Status: **Approved 2026-10-10; implemented; all nine GPU gates pass (run `20261010-s1-r2`).**

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

## Shape (rebuilt 2026-10-10: System One only)

Jev-family models are decision APIs for software (docs.typesafe.ai): a state and
typed questions in, calibrated probabilities out; no chat, no tools, no text. The
first version of this plan copied the chat surfaces of the Winnow/OpenJev examples;
the owner rejected that, and the example now serves System One only.

```text
apps / TypeSafe SDK / playground (:3015) -> Kairyu (:8014) /v1/systemone -> quyet-systemone (CPU) -> vLLM (internal)
```

- **L1 vLLM** (stock `v0.31.0`, registry digest pinned): bf16, 8,192-token context
  (Quyet's 8,000-token prompts), 64 sequences, prefix caching, text only,
  batch-invariant kernels. Internal.
- **L1 quyet-systemone** (example-owned, CPU): `quyet` 1.0.2 with only the forward pass
  replaced by vLLM letter logprobs; Jev error shapes as OpenJev; 529 past 16 running
  and 16 waiting requests; the image carries its source hashes as labels.
- **L3 Kairyu** (configuration only): `public_models` = the System One model (aliases
  `quyet-latest`, `jev-latest`, `jev-preview`); forwards 16, queues 64, answers 429.
  vLLM is a non-public pool so that readiness follows the model server (Kairyu requires
  one engine; `kairyu/` is not changed).
- **UI**: the Jev-style playground only. No Open WebUI.
- **Storage**: checkpoint, compile caches, logs and verification scratch on NVMe.

## Files (all example-owned)

`examples/quyet-1.0-large-1gpu/`: `README.md`, `MEASUREMENTS.md`, `example.json`,
`compose.yaml`, `kairyu.yaml`, `run.sh`, `verify.sh`, `control.py`,
`verification.py`, `quyet_systemone.py`, `quyet-systemone.Dockerfile`,
`quyet-requirements.txt`, `systemone-reference.jsonl`, `playground/{index.html,nginx.conf}`.
Tests: `tests/unit/test_quyet_1gpu_example.py`; one entry in
`tests/unit/test_frontier_examplectl.py`. Docs: `examples/README.md`, FN-D9 Quyet
amendment in `docs/design/frontier-native-runtime.md`, `PROGRESS.md`.

## Verification (rebuilt 2026-10-10)

From TypeSafe's documentation: typed answers, calibrated probabilities, many
questions per call, consistent answers, the official SDK, bulk throughput; JevBench
is the independent benchmark for Jev-compatible systems.

CPU: ruff and the changed tests (adapter refusals and overload against a fake vLLM,
the kairyu.yaml / example.json contract, the parity rule, reference reuse).

GPU gates, in order; stop and report at the first failure:

| Gate | Claim | Pass criteria |
|---|---|---|
| `reference` | official answers exist | stack down; the `quyet` CLI (transformers, GPU) answers 48 authored + 231 JevBench public items; reused only for the same request bodies, checkpoint and adapter sources |
| `attest` | the pinned stack runs | registry digest and source labels, checkpoint re-hash, vLLM settings and version, adapter calibration, System One public, no chat model |
| `systemone` | Kairyu's answers are the official package's | all 279: same input tokens, truncation and shape; >= 99 % of official answers with TypeSafe confidence >= 0.5 keep their top option; probability difference median <= 0.005; aliases; error shapes |
| `jevbench` | Jev-standard quality and speed | JevBench's runner (typesafe adapter, sequential) on Kairyu: 100 % valid; per split correct within 1 item of official and Brier within 0.01; ECE over all 231 within 0.01; p50 <= 0.5 s |
| `fanout` | many questions per call | 1/8/32 questions on one 1K-token state all answered; 32 questions <= 4x one |
| `consistency` | same request, same answer | 10 alone + 10 under load: top answers stable, probabilities within 0.01 |
| `sdk` | the official client works | typesafe-sdk 0.7.4: typed answers under jev-latest / quyet-latest / full name; 11 options -> TypeSafeBadRequestError |
| `systemone-serving` | bulk decisions | states ~50/2,000/6,000 tokens, 3 questions, c1/16/32/64 x 64: all answered up to Kairyu's forwarding limit (16), only answers or Kairyu's 429 above it; req/s, p50/p95, shed |
| `systemone-isolation` | overload is shed cleanly | 640 reads: 200 or 429 only, never 529; ready and answering right after |

Owner decisions (2026-10-10):

- After run `20261010-jev-r1` (median difference 0.0001, p99 0.043, max 0.12; one flip
  of an official 0.56/0.44 answer): judge the top option only where the official
  answer clears TypeSafe's 0.5 confidence floor.
- After runs `20261010-s1-r1` and `20261010-s1-bi`: the tail difference is vLLM-kernel
  noise (p99 0.043 / 0.067 / 0.066 across runs, max up to 0.21 on 3K-token states;
  the official package is identical run to run), so the p99 bound is dropped; at least
  99 % of confident official answers must keep their top option (0/309 and 1/309
  flipped). vLLM runs batch-invariant (`VLLM_BATCH_INVARIANT=1`): 20 repeats alone and
  under load gave identical probabilities (0.0026 spread without), at about 10 %
  latency (1 question 0.225 -> 0.251 s, 32 questions 0.60 -> 0.66 s).
- After run `20261010-s1-r2`: ECE is compared over all 231 public items, not per split.
  On the 72-item split two near-even answers (0.505/0.495, 0.497/0.503) crossing bins
  moved ECE by 0.013 while accuracy (72 vs 71) and Brier (0.0177 vs 0.0163) matched;
  over all 231 the served ECE is 0.038 against the official 0.042.

## Checklist

- [x] Owner approves this plan.
- [x] Example files, CPU tests, lint.
- [x] GPU gates; MEASUREMENTS.md; FN-D9 amendment; PROGRESS.md.
