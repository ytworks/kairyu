# Frontier model native runtime and example boundary

Status: **DeepSeek native EP/Attention-DP and SM120 packed-FP4 execution are
implemented; full-checkpoint native production gates remain open; the example
surface is superseded by FN-D8** (2026-08-11)

This document amends FZ-D1 in `frontier-model-zoo.md`. It records what the
frontier example rebuild may claim before full-checkpoint GPU evidence exists.

## FN-D1 — Production selection is explicit and fail-closed

`execution_mode: native` is mandatory in the new Qwen3.6 and DeepSeek V4
Kairyu example configs. `execution_mode: reference` is retained only for CPU,
small fixtures, and diagnostics. A frontier architecture cannot silently enter
the generic paged-KV runner, use vLLM inside Kairyu, enable a draft decoder, or
reduce `max_model_len` when a capability or memory check fails.

The native single-rank runner advances only new tokens through the official
architecture implementation's `forward_cached` contract. Kairyu, rather than
that model wrapper, owns admission, scheduler lifetime, prefix identity,
sampling, cancellation, and cache rollback.

## FN-D2 — CacheDescriptor is the scheduler-facing ABI

`CacheDescriptor` and `CacheHandle` expose a model-specific composite cache
without pretending all state is KV:

- Qwen: FP32 gated-DeltaNet recurrent/conv state plus BF16 paged KV for the
  full-attention layers.
- DeepSeek: block-256 HCA and CSA state, 4/128 compression metadata, sparse
  top-k/indexer state, FP4-indexer-cache provenance, and mHC state.
- Prefix reuse stores only complete state snapshots in a byte-bounded LRU.
  A generic token-prefix hit alone never skips recurrent/compressed work.
- Transactions clone opaque state before commit and restore it on rollback.
  Nested transactions are rejected.

The current runner keeps opaque addresses model-owned and therefore remains
eager. CUDA Graph pointer stability and model-specific speculative
commit/rollback require their separate GPU gates before they can be enabled.

## FN-D3 — Checkpoint and parser trust boundary

Qwen3.6 and DeepSeek V4 are loaded through pinned Transformers architecture
classes with remote code disabled in the Kairyu process. The DeepSeek loader
validates every official checkpoint header, shards only routed experts,
preserves packed E2M1/UE8M0 experts and block-FP8 nonexperts, and disables
remote code. The pinned fine-grained Triton kernel executes FP8 activations
against the checkpoint's FP4 bytes directly on SM120; single-GPU kernel and
two-rank NCCL dispatch smokes are green. Full-checkpoint numerical and 1M
evidence remains a separate gate.

The L3 API normalizes OpenAI-style reasoning_effort aliases
(minimal/low→low, medium/high→high, xhigh/max→max), preserves
reasoning_content in complete and streamed responses, and parses the pinned
DeepSeek DSML tool-call envelope. OpenAI-compatible replica gateways render the
checkpoint chat template before sending an identity-wrapped request; Kairyu
does not use legacy role concatenation.

## FN-D4 — DeepSeek native distributed execution

The native worker supports request-owned Attention-DP with EP2/4/8. Each rank
retains its own sliding/HCA/CSA state, while every prefill/decode phase agrees
its forward count and pads missing-rank work before entering expert
collectives. Routed experts use equal-capacity NCCL all-to-all dispatch and
combine; ragged rank token counts require no host-derived split vectors and
top-k contributions are restored in deterministic slot order before one BF16
cast.

Two EP4 replicas remain the default example. A separate one-replica EP8 Compose
profile is selected only by the committed topology gate. No EP8 topology lock
is generated until real-checkpoint EP4/EP8 quality, 1M context, stability and
SLO-goodput evidence passes, with EP8 at least 2% ahead. CUDA Graph, DSpark,
30-minute soak, failure recovery and full-checkpoint 1M results remain open.

## FN-D5 — Orchestration policy

The L2 DSL can load a SHA-256-pinned calibrated router artifact. Artifacts
below a 0.99 quality-ratio confidence lower bound are rejected. `auto-max`
maps to three Tier1 proposals plus Tier2 synthesis. Tier1 direct failures retry
Tier2 once; a stream retries only before any output has been emitted, avoiding
mixed answers. Inputs are never truncated by the router or gateway.

The checked-in router is an all-Tier2 structural baseline. It is safe by
construction but does not claim Tier1 goodput. A measured train/holdout
artifact may replace it only after the benchmark calibration gate passes.

## FN-D6 — Rebuilt examples and evidence

`examples/` contains only the shared controllers and the Qwen 1-GPU,
DeepSeek 8-GPU, and combined 8-GPU environments. Model revisions, external
images, CUDA bases, contexts, GPU counts, VRAM, disk, SM120 capability, and
NUMA-local CPU sets are fail-closed. The first download hashes every model file
and subsequent starts mount the same volume read-only with offline mode.

Each environment exposes `run.sh` for lifecycle management and `verify.sh` for
serving verification. CLI enumeration, shell syntax, Compose expansion, and
report mechanics are CPU/static gates; measured performance remains a GPU gate.
Model and product evaluation is invoked separately through `python -m evals`.

## FN-D7 — Enablement gates

- Qwen MTP stays off until greedy equality, sampling-path invariants, and at
  least 5% SLO-goodput improvement pass.
- DeepSeek DSpark stays off under the checkpoint-declared 5-token gate.
- `PROGRESS.md` must not claim production frontier support until the real
  Qwen 262K and DeepSeek EP4/EP8 1M GPU runs, 30-minute soak, OOM/worker-failure
  recovery, and vLLM comparison all close.

## FN-D8 — The user-facing example is one measured vLLM deployment

This decision supersedes FN-D4's default-example topology and FN-D6's three
environment surface; it does not remove or weaken the native-engine gates.
`examples/` now contains exactly one deployment for the available 8 x RTX PRO
6000 Blackwell Server Edition host: Open WebUI calls Kairyu L3, and Kairyu calls
one vLLM L1 using all eight GPUs. The checkpoint's exact prompt encoder remains
owned by Kairyu and is preserved through an identity template at vLLM.

The committed default is selected only after same-host topology and feature
measurements. Its verification report records TTFT and output throughput. Model
and product evaluation is a separate checkout-only workflow and is not a
prerequisite embedded in the serving-performance runner. Public heterogeneous
figures are comparison context, not a substitute for local measurements.

**Layered-product amendment (2026-08-13, EO-D2..EO-D5).** The measured vLLM
services may remain transitional L1 workers while the tiered example proves
its direct L2-to-L1 object boundary, bounded verifier loop, one-model public
inventory, and separate model-attributed intermediate-output UI. That
structural pass does not satisfy the native production gate: the default may
be called native Kairyu L1 only after the full-checkpoint gates in FN-D7 pass.
The binding example contract is `example-layered-orchestration.md`.

**Native L1 rendering amendment (2026-10-17, owner decision).** The
checkpoint's prompt encoder no longer lives in Kairyu.
`deepseek-v4-flash-0731-8gpu` takes the L1 settings of FN-D9's tool-calling
amendment from the dp2 example: vLLM renders every request with the
checkpoint's own `deepseek_v4` encoder, separates reasoning, and parses DSML
tool calls (`--enable-auto-tool-choice --tool-call-parser deepseek_v4`), with
thinking off by default (`--default-chat-template-kwargs`). The identity
template, the Kairyu `chat_templates` entry and the templated passthrough are
removed, and the model is listed in `upstream_chat_models`,
`upstream_responses_models` and `upstream_messages_models` (m9 D2 amendment),
so each request reaches vLLM in its caller's API. Kairyu sends L3 `max` on the
Responses wire as `xhigh` (m11 D4 amendment) and the pinned tokenizer folds
`xhigh` to high, so an example-owned overlay (`vllm-effort.Dockerfile`,
`patch_effort.py`; the build fails on a missing or repeated anchor) maps
`xhigh` to max; its image ID is recorded after the GPU build. Why: the
conversation reaches L1 as it is, in the caller's API, and the example serves
OpenAI tool calls. The 2026-08-11 measurements belong to the previous
configuration; serving rows now run on `/v1/responses` and end naturally
within 256 tokens, so they are not comparable with the fixed-length rows. GPU
verification (`verify.sh serving`, `verify.sh api-equivalence`) is pending;
the FN-D7 gates are not weakened.

## FN-D9 — Replica-pool scale-out examples (amendment, 2026-09-01)

Status: accepted; implemented; GPU-verified 2026-09-01/02 (placement gates
green at c8–c64; runs `20260901T133331Z` Qwen, `20260902T005136Z` DeepSeek
after the tool-calling amendment below; `verify.sh tool-calling` green on both
— see the examples' `MEASUREMENTS.md`).

This amends the FN-D6/FN-D8 example surface: `examples/` gains two
environments in which Kairyu L2 does **no orchestration** — it is only the
`ReplicaPool` spreading one public model over identical vLLM L1 replicas on
the eight-card host:

- `qwen3.8-27b-dp8-8gpu`: Qwen3.8-27B-FP8 as 8 × TP1 replicas (one per GPU),
  each carrying the single-GPU example's measured L1 envelope.
- `deepseek-v4-flash-0731-dp2-8gpu`: DeepSeek-V4-Flash-0731 as 2 × TP4+EP4
  replicas (GPU 0-3, 4-7), each carrying the tiered example's measured Tier2
  envelope (DSpark-5, 16K batch, 32 sequences).

Placement policy (both pools): `prefix_index: true`, `queue_depth_threshold: 0`,
`unhealthy_after: 1`. A warm prefix is reused only while its replica is idle;
otherwise strict least-outstanding (m5 D4 / m10 D6 semantics), so concurrent
traffic spreads one-per-replica before any replica takes a second request.
The pool's `placement_log_path` is the evidence surface: `verify.sh serving`
reads the per-row JSONL delta and fails a row at concurrency ≥ 8 unless every
replica received traffic and none took more than 1.25× the even share; c1 is
reported only (least-outstanding ties resolve to the lowest replica id).
Checkpoints, templates, images, and L1 flags are shared by reference with
the sibling examples; no product code changed. Existing FN-D7 gates are not
weakened: these are vLLM-backed L1 deployments, not native-engine claims.

**Tool-calling amendment (2026-09-02, PR #584 review).** The served DP2
example returned `tool_calls: null` (DSML markup leaked into `content`), so
the official SWE-bench Pro mini-swe-agent failed every turn — a served example
that cannot drive tool agents does not satisfy this decision. The
Kairyu-rendered `/completions` passthrough shape is abandoned for these
examples: Kairyu never forwards `tools` on that path, and Kairyu's DSML parser
accepts only a whole-completion DSML block, which the prose-plus-call agent
format never satisfies. Both replica examples now use the Qwen examples'
layering — vLLM owns the chat rendering (for DeepSeek via the checkpoint's own
`deepseek_v4` encoder, which also renders DSML tools and merges `tool` turns)
plus `--enable-auto-tool-choice --tool-call-parser {qwen3_coder|deepseek_v4}`,
and Kairyu (`legacy_chat_models`) forwards tools to `/chat/completions` and
normalizes the parsed calls. Thinking defaults off in both via
`--default-chat-template-kwargs` (`thinking`/`enable_thinking: false`);
`reasoning_effort` re-enables it. For Qwen the flag is required even though
the Kairyu-owned template already renders non-thinking prompts: vLLM's `qwen3`
reasoning parser otherwise assumes thinking and files a plain answer (no
`</think>`) as `reasoning_content`, leaving `content` empty — caught by the
gate's non-thinking case on the first GPU run. The
example contract now includes fail-closed tool-calling evidence: a readiness
probe in `run.sh up` and the `verify.sh tool-calling` gate (auto call on every
replica, tool-result turn, streaming, thinking, non-thinking default).

**Vision replica amendment (2026-09-04).** Two more replica-pool examples
apply the same layering to the newly released vision-language checkpoints,
each as 2 × TP4 replicas (GPU 0-3, 4-7) — the only eight-card split that fits
either checkpoint (neither fits one 96 GB card; Qwen's 128-wide FP8 blocks
reject TP8 and pipeline parallel is unsupported):

- `deepseek-v4-flash-vision-exp-dp2-8gpu`: DeepSeek-V4-Flash-Vision-Exp
  (revision `6821d6ad`) with the official recipe's TP4+EP, FP8 KV, 256-token
  blocks, DSpark k=3 probabilistic drafting, `deepseek_v4` parsers, 1M
  context; SM120 pins `--moe-backend marlin` and disables DSpark adaptive
  verification. Effort levels `low/high/max` are the encoder's own vocabulary.
- `qwen3.8-flash-next-dp2-8gpu`: Qwen3.8-Flash-Next-FP8 (revision `236dfdf2`)
  with the official recipe's verified `rtx_pro_6000_4x` layout (16 sequences,
  8K batch tokens, 0.95 memory, prefix caching, `qwen3_xml`/`qwen3` parsers),
  256K context, **without the recipe's MTP k=3**: on `vllm@27a94d1c` prefix
  caching + MTP corrupts batched answers on hybrid GDN models
  (vllm-project/vllm#53912; reproduced 13/274 at 2-12 concurrent, 0/1,508
  with either feature off, 63.8% with `--no-async-scheduling`). Prefix
  caching stays because Kairyu's prefix-aware placement and multi-turn
  traffic depend on it; the price is single-stream decode 104 vs 175 tok/s.
  The KV budget is pinned (`--kv-cache-memory` 43.16 GiB, the warm-start
  value for 0.95 utilization) because a cold torch.compile cache inflates
  vLLM's start-up memory profile and a first boot otherwise serves 741K KV
  tokens instead of 3.45M. Kairyu L3 normalizes the wire vocabulary `medium→high`,
  `xhigh→max`, and the official template rejects anything outside
  `low/medium/xhigh` (HTTP 400), so the example-local template aliases
  `high→medium`, `max→xhigh` at the top and is otherwise byte-identical.
  Selecting an effort in the Chat UI also drops the pinned instruct sampling
  (T 0.7 / top_p 0.8 / presence 1.5) so vLLM applies the checkpoint's
  thinking `generation_config` — the two official sampling modes, not a
  blend.

Both are vision-capable (`allow_prompt_kinds: [multimodal]` paired with an
`image_input_policy`, Kairyu built with the `vision` extra), run the Chat UI
without login and provision the tiered example's Reasoning Effort dropdown
(fail-closed enum check) from `run.sh up`, and add a `verify.sh vision` gate
plus an image readiness probe (the first image request is where the SM120
sparse-MLA path failed on FlashInfer 0.6.18). vLLM image: no release or
official tag carries the merged support, so both share one overlay image —
upstream's digest-pinned nightly of `vllm@27a94d1c` plus FlashInfer
`60b49158` (#4802) with the stale AOT module cache removed — and `run.sh up`
refuses to serve if the built image ID differs from the pinned
`container_image_digest`. The vision gate requires the answer to name the
probe colour (a non-empty check let the corrupted `ductduct…` output pass
once). Status: GPU-verified 2026-09-04 on 8 × RTX PRO 6000 (tree hash /
image ID pinned; all three gates PASS for both examples; results in each
example's `MEASUREMENTS.md`).

### V4.1 Flash single-replica amendment (2026-09-11)

Status: GPU-verified on SM120; final evidence in the example's MEASUREMENTS.md.

`examples/deepseek-v4.1-flash-8gpu` uses one TP8 replica across GPUs 0–7,
retaining the V4 vision example's ReplicaPool, legacy OpenAI chat/tool path,
image admission, and Chat UI. The owner's revised allocation supersedes the
initial two-replica proposal. TP/EP, DSpark, batch limits, CUDA graphs, and
Engram CPU/GPU placement are L1 measurement choices on the SM120 host.

Omitted effort means thinking `high`. The model-author encoder at the pinned
checkpoint maps `low/high/max` to `50/75/100`; the initial vLLM image instead
maps `high` to 50. An image-local encoder adjustment aligns those aliases,
with the Python frontend selected and rendered-prefix checks at build time.
Kairyu's L2/L3 effort normalization remains the existing contract.

The official V4.1 image also needs SM120 page compatibility: 64-token
manager blocks in BLHNC, 64-token SWA pages on the SM120 subclass, and the
existing FlashInfer dual-cache prefill template instantiated for C2 pages
of 32 tokens (C1 uses 64). Exact-anchor image patches fail on source drift;
the 16-case packed-cache GPU numerical gate passes using upstream DSV4
tolerances. A V4.1-only SM120 indexer subclass also selects 64-token
manager blocks to satisfy DeepGEMM's 32/64 compressed-page envelope. SM120
FP8 indexer decode supports only 64-token pages, so this example selects
MXFP4 indexer Q/K. The existing MXFP4 kernels pass four real-writer and
prefill/decode tests against independently unpacked PyTorch logits (maximum
absolute error 2.4e-7); dtype enablement is limited to V4.1 on SM120.
Adaptive DSpark verification is disabled because the indexer backend rejects
it. Bounded GPU comparisons select TP8/EP8, DSpark 5, 16K batched tokens,
64 sequences, memory utilization 0.90 and NCCL. Retain GPU Engram and
breakable CUDA graphs. EP-off runs out of KV memory under the same limits;
PCIe IPC stalls during autotuning, and 8K batching shows no throughput gain.
These trials do not establish a global optimum. The final 320-request matrix,
thinking/tool/vision/cancellation gates, normal restart, and four retrieval
smokes through 1,039,909 actual prompt tokens pass on the pinned configuration.

Fixed-length performance rows record first model output (reasoning or
content) separately from first visible content, which stays null if no
content was emitted. Completed-answer/tool/image gates are independent.
Only SHA-bound measurements in the example's `MEASUREMENTS.md` establish
the final runtime and performance claims.

### V4.1 Flash six-GPU amendment (2026-09-30)

Status: accepted by the owner (plan
`docs/superpowers/plans/2026-09-30-deepseek-v41-flash-6gpu-example.md`);
selection and gates in the example's `MEASUREMENTS.md`.

`examples/deepseek-v4.1-flash-6gpu` serves one replica on GPUs 0–5 with the
8-GPU example's L2/L3 structure (one ReplicaPool replica, legacy OpenAI
chat/tool path, image admission, Open WebUI). GPUs 6–7 are not used. Every
script, overlay file and test is the example's own; no file of another
example is shared (owner instruction).

The L1 starts from the official sources and deviates only where SM120 or
the 576 GB of HBM forces it or a bounded one-parameter comparison supports
it (≥ 5 % on c1 or c32 throughput, no > 5 % loss elsewhere). Six 96 GB GPUs
are below the checkpoint's official 614 GB minimum, so the recipe's
memory-bound (8 × H100) arm applies: Engram tables in pinned host memory,
4,096 batched tokens and memory utilization 0.92. The replica is the
official Blackwell DEP shape — attention DP6 with EP6 (384 experts, 64 per
rank) — but on the TP path's SM120 kernels, because the recipe's DEP
kernels and `indexer_sparse_logits` are SM100-only. DP6 beats the official
TP2 degree (TP2 × DP3 on NUMA-local pairs) by 44–47 % at c32. DSpark runs
with its trained 5-token block and full verification (the V4.1 indexer
backend rejects adaptive verification); its 128 draft experts do not divide
EP6, which the fused-MoE path accepts. DSpark adds 77 % at c1 and 20 % at c32
over DP6 without it. TP2 × DP3 with DSpark fails in CUDA-graph capture of
the TP2 custom all-reduce.

The pinned nightly already renders the model author's efforts (low 50,
high 75, max 100, default high); the overlay checks that and adds the
SM120 edits the 8-GPU example needed, plus a zero row for masked sparse-KV
candidates and the split top-p guard that the earlier six-GPU attempts
showed necessary (without them EP6 returned non-finite output). Prebuilt
FlashInfer JIT caches are removed so the patched kernels are the ones
compiled; a NaN-poisoned masked slot in the kernel gate catches a shadowed
kernel. Only SHA-bound rows in the example's `MEASUREMENTS.md` establish
runtime and performance claims.

### OpenJev DiffusionGemma one-GPU amendment (2026-10-01)

Status: accepted by the owner (plan
`docs/superpowers/plans/2026-10-01-openjev-diffusiongemma-1gpu-example.md`);
GPU-verified 2026-10-01 (every gate in the example's `MEASUREMENTS.md`).

`examples/openjev-diffusiongemma-26b-1gpu` serves one replica on one selected
GPU. The L1 is a third-party server, OpenJev (DiffusionGemma 26B-A4B NVFP4
on vLLM `1b3b88ec`, behind OpenJev's OpenAI-style chat endpoint). L2/L3 are
the single-replica structure: one ReplicaPool replica, the legacy OpenAI
chat/tool path, image admission, and Open WebUI. The chat replica needs only
configuration:

- `health_url` set to OpenJev's `/health`, because Kairyu's default is
  `/readyz`;
- the fields OpenJev drops silently listed in `deny_sampling_fields`;
- `max_concurrency` set to OpenJev's generations in flight plus queued
  (40; 32 + 8 after the L1-1 measurement, OpenJev's default is 8 + 32).
  OpenJev answers 529 above that, and Kairyu counts a 529 as a replica
  failure.

Every chat completion thinks first, with a thought of at most 512 tokens
that callers cannot disable or resize. vLLM's `thinking_token_budget` cannot
enforce this: DiffusionGemma replaces the sampler with `DiffusionSampler`,
which never applies `ThinkingBudgetState`. The example therefore applies
OpenJev's own System One `think` method to chat, as an L1 overlay that the
example owns (a reasoning budget and a runtime adaptation, both example
policy):

- A thought pass continues an assistant prefill `<|channel>thought\n` with
  `max_tokens: 512` and a stop on `<channel|>`.
- An answer pass continues after the closed thought.
- One generation slot covers both passes.
- The overlay publishes only the capped first-pass thought as reasoning.
- A failure after the thought has streamed aborts the stream rather than
  ending it, because Kairyu ignores mid-stream error chunks.
- A request that only the answer pass would refuse is refused with a 400
  before the thought. Today that is a required or named `tool_choice`, which
  needs structured outputs. Otherwise any client could make a stream abort,
  and the abort would eject the only replica.

The checkpoint's chat template strips thoughts from every assistant message,
so `continue_final_message` cannot continue a prefill. The example's template
renders only a final assistant message verbatim. With the checkpoint
tokenizer, every other conversation renders byte for byte like the stock
template. At startup the overlay refuses to run unless vLLM uses that
template and it continues both prefills, both as a string and as the
text-part list that vLLM's `openai` content format sends (the first GPU start
showed the list form stripping the prefill). Only SHA-bound rows in the
example's `MEASUREMENTS.md` establish runtime and performance claims.

System One amendment (2026-10-01, accepted by the owner). OpenJev's System
One API is served through Kairyu's `/v1/systemone` (m11 D8), not as a
ReplicaPool member: Kairyu forwards at most 256 requests and queues 256 more,
below OpenJev's 529 point of 512 waiting requests, so a burst gets Kairyu's
429 and never ejects the chat replica. The example's UI follows Jev: a
System One playground on :3011 (a static page behind nginx, on Kairyu's
origin) shows each answer's distribution, confidence, tokens, latency and
`Server-Timing`, next to the think-first chat answer for the same state;
Open WebUI stays on :3010 for chat.

### Quyet-1.0-Large one-GPU amendment (2026-10-10)

Status: accepted by the owner (plan
`docs/superpowers/plans/2026-10-10-quyet-large-1gpu-example.md`); GPU-verified
2026-10-10 (all nine gates, run `20261010-s1-r2`, in the example's `MEASUREMENTS.md`).

`examples/quyet-1.0-large-1gpu` serves `chinhnc/Quyet-1.0-Large` (Gemma-4-31B-it
with a merged decision LoRA, bf16) on one selected GPU as a System One model only, the
way Jev-family models are used: Kairyu publishes `/v1/systemone` (m11 D8) and no chat
model. The stock vLLM `v0.31.0` image holds the weights as an internal, non-public
pool (`public_models`), so Kairyu's readiness follows the model server; it runs
vLLM's batch-invariant kernels so a request's answer does not depend on concurrent
load. `kairyu/` is not changed.

Quyet's runtime, the `quyet` package, has no server. The example's System One adapter
(example-owned L1 code, like OpenJev's JevK5 backend) keeps the package's prompt,
truncation, calibration and answer code and replaces only the forward pass: vLLM's
`/v1/completions` returns the option letters' logprobs (`logprob_token_ids`) for the
exact prompt token IDs the package built. The adapter follows OpenJev's Jev error
shapes, refuses the Jev options Quyet lacks (images, think, samples, steps,
sequential) with a 400, and answers 529 past 16 running and 16 waiting requests;
Kairyu forwards at most 16 and queues 64, so callers get Kairyu's 429. The gates
follow TypeSafe's documented usage: the served answers are checked against the
package's own transformers run on the same GPU (same prompts, confident decisions
kept, median probability difference bounded; vLLM's kernels differ in the tail), JevBench's runner scores Kairyu beside
that run, and fan-out, consistency, TypeSafe's SDK, throughput and overload are
checked through Kairyu. Only SHA-bound rows in the example's `MEASUREMENTS.md`
establish runtime and performance claims.

### GLM-5.3-Flash six-GPU amendment (2026-10-11)

Status: accepted by the owner (plan
`docs/superpowers/plans/2026-10-10-glm-5.3-flash-example.md`; the owner allowed six
GPUs if four did not fit); GPU-verified 2026-10-11 (all nine gates, run
`20261011-r2`, in the example's `MEASUREMENTS.md`).

`examples/glm-5.3-flash-6gpu` serves `zai-org/GLM-5.3-Flash` (320B-total / 18B-active
multimodal MoE with KDA linear attention and NoPE sparse MLA, official FP8
checkpoint) as one replica on GPUs 0-5 with the `deepseek-v4.1-flash-6gpu` L2/L3
structure: one public model behind one `ReplicaPool` replica, the OpenAI-compatible
API, and Open WebUI with an effort dropdown (low/high/max, the template's own
vocabulary). `kairyu/` is not changed and no file is shared with another example.
The plan targeted four GPUs; the fit probe left 1.81 GiB of KV memory per GPU on one
TP4 replica against 7.56 GiB for a 1M-token request, so the example uses six.

L1 is the stock vLLM `v0.31.0` image pinned by registry digest, with the recipe's
FP8 KV and `glm47` parsers. Measured deviations from the recipe: TP2 x DP3 with EP6
instead of TP4 (six GPUs; +12 % / +11 % at c1 / c16 over DP6 at equal capacity),
MTP drafting 3 tokens instead of 5 (c1 x2; 5 lost 6 % at c16), NCCL instead of the
custom all-reduce (which fails while the drafter's CUDA graphs are captured on
TP2), and a pinned 16.75 GiB KV pool per GPU (2,028,392 tokens per engine). The
model author asks chat clients to pass `clear_thinking=true`; Kairyu's legacy chat
path rejects `chat_template_kwargs` on text requests, so the Chat UI sends only the
effort and the template default applies. Only SHA-bound rows in the example's
`MEASUREMENTS.md` establish runtime and performance claims.

### Responses connection amendment (2026-10-17, owner decision)

Status: implemented; GPU-verified 2026-10-17/18 (PR #656 final test) in every
remaining example below: all gates, `api-equivalence`, the Chat UI pass and
the cancellation probe pass (each example's `MEASUREMENTS.md`; plan
`docs/superpowers/plans/2026-10-17-upstream-api-forwarding.md`).

Applies to the vLLM-backed examples of this record except the Jev ones
(OpenJev, Quyet-1.0-Large; the Winnow examples are governed elsewhere):
`qwen3.8-27b-1gpu`, `deepseek-v4-flash-0731-8gpu` (also FN-D8's native L1
rendering amendment), the four FN-D9 replica examples, and the two V4.1 Flash
and the GLM-5.3-Flash single-replica examples.

- Each served model leaves `legacy_chat_models` for `upstream_chat_models`,
  `upstream_responses_models` and `upstream_messages_models` (m9 D2
  amendment): chat reaches vLLM as the caller's own turns instead of one
  flattened user message, and Responses/Messages callers reach vLLM's own
  `/v1/responses`/`/v1/messages`. vLLM keeps rendering and parsing.
- Open WebUI uses `api_type: responses` with an example-owned filter that
  leaves only Responses fields (m11 D7 amendment); the four sign-in examples
  install it, and switch a saved chat connection to Responses, with
  `WEBUI_ADMIN_EMAIL`/`WEBUI_ADMIN_PASSWORD`, and warn without them.
  Qwen3.8-Flash-Next's UI loses `presence_penalty` 1.5 (no Responses field).
- L3 `max` travels as the Responses `xhigh`: the DeepSeek V4 examples add an
  overlay mapping it to max, the V4.1 overlays map the encoder's
  `xhigh` (75) to 100, GLM's template and Qwen's templates need no change
  (GPU checks). Rebuilt images have no recorded image ID until the GPU build;
  `run.sh up` stops unless the example's allow-unpinned variable is set.
- Every gate through Kairyu uses `/v1/responses` (including the readiness
  `17 * 19` probes, which lose their log-probability check there; the L1
  gates keep it); L1-direct checks stay on vLLM's chat API. Serving and
  cancellation rows end naturally within `max_output_tokens`, so they are not
  comparable with the earlier fixed-length rows. `tool_result_turn` uses PR
  #647's wording. `verify.sh api-equivalence` runs
  `verification/product/correctness/upstream_api_equivalence.py`.
- GPU final test fix (2026-10-18): `qwen3.8-27b-1gpu` adds
  `--default-chat-template-kwargs '{"enable_thinking": false}'`, as
  `qwen3.8-27b-dp8-8gpu` already did. With vLLM rendering the native
  messages, vLLM v0.23.0's `qwen3` parser treated a request without
  `enable_thinking: false` as thinking and filed the direct answer or tool
  call as reasoning (`api-equivalence`: the L1 returned only a reasoning item;
  through L3 a unary answer was empty). The template already rendered those
  requests non-thinking, so prompts are unchanged; an explicit effort still
  sets `enable_thinking` per request ahead of the default.
- GPU final test fix (2026-10-18): `deepseek-v4.1-flash-8gpu`'s overlay (the
  older `deepseekv41-flash-0909` base) lets the V4.1 tokenizer read the
  Responses `input_text`/`output_text` parts as text, as the nightly used by
  the other V4.1 examples already does; its image probe through Kairyu
  answered HTTP 400 ("supports text and image content only; got
  'input_text'"). The rebuilt overlay is pinned again.

### Example removal amendment (2026-10-18, owner decision)

Status: implemented in PR #656.

The owner keeps only the examples that serve DeepSeek-V4.1, Qwen3.8-27B,
GLM-5.3-Flash or Quyet. Removed from `examples/` with their own tests:
`deepseek-v4-flash-0731-8gpu` (FN-D8), the FN-D9 replica examples
`deepseek-v4-flash-0731-dp2-8gpu`, `deepseek-v4-flash-vision-exp-dp2-8gpu`
and `qwen3.8-flash-next-dp2-8gpu`, and `openjev-diffusiongemma-26b-1gpu`
(also the Winnow examples of `llamacpp-upstream.md` and the DeepSeek-V4
tiered example of `example-dual-track-orchestration.md`). Their decisions and
evidence above stay as history. FN-D9's replica pattern continues in
`qwen3.8-27b-dp8-8gpu`; the V4.1 Flash, GLM-5.3-Flash and Quyet amendments
are unchanged. No framework mechanism is removed: DeepSeek-V4 model support,
System One and the llama.cpp upstream stay in `kairyu/`.
Why: owner decision 2026-10-18; the removed examples' model storage (about
723 GiB on the host's NVMe) was needed for the remaining examples.

### Qwen3.8 vLLM v0.31.0 amendment (2026-10-18, owner decision)

Status: implemented; `qwen3.8-27b-1gpu`'s `run.sh up` and its probes passed
on v0.31.0 (GPU, 2026-10-18); every other GPU gate of the three examples
below is pending on the new image.

Applies to the examples that serve Qwen3.8-27B: `qwen3.8-27b-1gpu`, the
FN-D9 replica example `qwen3.8-27b-dp8-8gpu`, and the Qwen pools of
`qwen3.8-deepseek-v4.1-8gpu` (DTO-D16 amendment of the same date; its
DeepSeek overlay is unchanged).

- Qwen's L1 moves from the official `vllm/vllm-openai:v0.23.0` image
  (`sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`,
  source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, FlashInfer 0.6.12) to
  the official `v0.31.0` image the GLM-5.3-Flash amendment pins
  (`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`,
  source `db9527a46873454610df6dbedf79a36d6bf1a7f6`, FlashInfer 0.7.0.post1),
  in each example's `compose.yaml` default, `example.json` and `kairyu.yaml`
  `container_image_digest`. The L1 flags, the measured envelope
  (`qwen3.8-27b-1gpu/MEASUREMENTS.md`) and the launchers are unchanged.
- Why: on v0.23.0 every request that carries `extra_args` prefills about 3x
  slower, and v0.23.0's Responses serving always sets `extra_args={}`, so
  every Responses request Kairyu forwards to Qwen (Responses connection
  amendment) prefilled slowly. Measured on GPU with the V4.1 ensemble's
  `profile_judge` request, 16 concurrent on one Qwen3.8 replica: prefill
  1.43 s on chat against 4.33 s on Responses and 4.33 s on chat with an empty
  `vllm_xargs`; on v0.31.0 1.26 s / 1.27 s / 1.26 s. In the V4.1 ensemble the
  slowdown pushed the 5 s route judge past its timeout on 13 of 32 coding c32
  requests and doubled the generic c32 judge p50 (1,982 → 4,468 ms).
- The v0.23.0 findings recorded above and in the examples stay as v0.23.0
  evidence: the envelope's tuning rows (and its failed `FULL_AND_PIECEWISE`
  candidate), and the `qwen3` parser behaviour behind the 2026-10-18
  `--default-chat-template-kwargs` fix, which stays. The m9 D2 wire rules
  that cite v0.23.0 are unchanged, because the backend does not know the
  vLLM version.
- GPU consequences: every gate of the three examples runs again on v0.31.0.

### V4.1 Flash 1017 amendment (2026-10-19)

Status: accepted by the owner (plan
`docs/superpowers/plans/2026-10-19-deepseek-v41-flash-1017-examples.md`);
GPU-verified 2026-10-19: both examples pass every gate, `api-equivalence`
and the Chat UI pass (each example's `MEASUREMENTS.md`). Selected: vLLM
v0.32.0 on both (over the nightly: c32 +7.6 %); six GPUs keep the 0909
six-GPU levers (no recipe lever reached 5 %); eight GPUs serve TP8 / EP8
(c1 +52 %, c8 +30 %, c32 −28 % against DP8 / EP8) with 8,192 batched tokens
(c32 +5.5 %).

`examples/deepseek-v4.1-flash-1017-6gpu` (GPUs 0–5) and
`examples/deepseek-v4.1-flash-1017-8gpu` (GPUs 0–7) serve
`deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision `1bae0e10`) with the V4.1
examples' L2/L3 unchanged (one ReplicaPool replica, the Kairyu API, Open
WebUI over `/v1/responses`). Every script, overlay file and test is the
example's own; `kairyu/` and the 0909 examples are not changed.

1017 is a weight refresh: `config.json`, the tokenizer, the 96,085 tensor
names and the 510.3 GB total equal 0909's. Its encoder no longer starts a
turn at a user message holding only tool results, so without a tools
declaration the current turn's reasoning survives its tool calls; vLLM's
encoder copy (nightly 2026-09-30, v0.31.0, v0.32.0) keeps the 0909 rule, so
the overlay ports the 1017 one and the `l1` gate compares vLLM's prompt
tokens with the checkpoint's own encoder. Sampling follows the 1017 card
(temperature 1.0, top_p 1.0; 0909 allowed 0.95). Efforts stay the model
author's (low 50, high 75, max 100, default high) with `xhigh` mapped to 100.

The L1 is selected by the six-GPU amendment's rule (one change at a time,
≥ 5 % at c1, c8 or c32 without a > 5 % loss elsewhere, correctness first):
the vLLM v0.32.0 release (V4.1 Engram, SWA replay and SM120 attention work
after the nightly, none of the SM120 edits) against the 2026-09-30 nightly,
both with the same overlay; then the recipe's levers. On eight GPUs the
replica shape is measured too: DP8 / EP8 with host or HBM-sharded
(`embedding_across_dp`) Engram tables, or the 0909 eight-GPU example's
TP8 / EP8, selected by c8 throughput (the concurrency of the DeepSWE run on
the six-GPU example), then c32.

#### Eight-GPU Chat UI for Japanese knowledge bases (2026-10-21)

Status: accepted by the owner (plan
`docs/superpowers/plans/2026-10-21-1017-8gpu-japanese-knowledge.md`, PR
#661); GPU-verified 2026-10-21: `docling`, `knowledge`, `chat-ui`,
`knowledge-serving` and the restart check pass (the example's
`MEASUREMENTS.md`).

`examples/deepseek-v4.1-flash-1017-8gpu` alone departs from the V4.1
examples' L3: its Open WebUI extracts knowledge-base documents through a
pinned docling-serve (CPU; text layers kept, bitmaps read by RapidOCR with
`iso:ja`), embeds with `BAAI/bge-m3` and reranks with
`BAAI/bge-reranker-v2-m3` from pinned, hashed copies in its data volume
(offline), and runs hybrid search with BM25 weight 0, because Open WebUI's
BM25 splits terms on whitespace and cannot tokenize Japanese. Chunks are
300 characters (the English-sized default of 1,000 diluted the asked
passage out of the reranker's 20 candidates). Docling is capped at 8 CPUs
(16 cost c1 serving 4.1 % during OCR) and the Chat UI at 16 with 16 torch
threads. L1, L2, `kairyu/` and the other examples are unchanged.
