# Goal G6: Product Surface — Truthful API, Fugu-Class Orchestrated Product, Competitive Proof (Roadmap Track P)

Status: Goal defined (2026-07-03). P-A, all of P-B, P-C2, P-C3, and P-C4 are
green. P-C's scoreboard needs real engines for the Kairyu column but runs
against frontier APIs immediately.
Depends on: M1/M7 server stack; real token accounting quality gates depend on
Track E1 (real tokenizer). See `docs/roadmap.md` §4 Track P.
Date: 2026-07-03

## 1. Goal

Make Kairyu's serving surface (a) a credible OpenAI-compatible API for developers and
(b) a Fugu-class orchestrated product for end users, with a continuously produced
benchmark artifact backing the "beats Claude/GPT on TTFT/TPOT/goodput" claim.

### Competitive frame (research, 2026-07)

Sakana Fugu (GA 2026-06) sells a "multi-agent system as a model": `fugu` /
`fugu-ultra` on one OpenAI-compatible endpoint (Chat Completions + Responses API),
internal orchestration disclosed only as `orchestration_input/output_tokens` in
usage, cached-input discount on the public price sheet, developer console but **no
chat UI**, and measured latency of ~7–8 s (light) to 11–269 s (Ultra). Kairyu's
differentiation: **Fugu-class orchestration at direct-call latency** (owned GPUs,
radix KV, streaming synthesizer) and **orchestration transparency** (opt-in trace:
route decision, role DAG, verifier verdicts — Fugu is a black box). Sources in
`docs/roadmap.md` §7.

## 2. Acceptance gates

### Stage P-A — Truthful API core (MUST; unblocks billing, quality, benchmarks)

| Gate | Target | Where proven |
|---|---|---|
| P-A1 (usage truth) | `usage` computed from the real tokenizer (not whitespace split); `prompt_tokens_details.cached_tokens` populated from radix KV; `stream_options.include_usage` supported; second identical-prefix request shows cached_tokens >0 | `tests/server/` |
| P-A2 (chat templates) | HF Jinja `apply_chat_template` with per-model override in `DeploymentSpec` and tool schemas in-template; Llama/Qwen golden transcripts byte-match the HF reference | golden tests |
| P-A3 (sampling surface) | `logprobs`/`top_logprobs` returned (plumbing exists in `sampling_params.py`/`outputs.py` — surface it); `/v1/completions` served; `n>1` verified incl. streaming indices; OpenAI SDK round-trips all of it | `tests/server/` |
| P-A4 (structured outputs) | `response_format: json_schema` enforced through `engine/core/structured.py` (not `extra_args` passthrough): 100% schema-valid on a 50-schema suite | `tests/server/` |
| P-A5 (serving evidence) | `verification/l1/performance/serving_bench.py` records auth-aware token-granularity TTFT/TPOT and throughput; new per-run JSON defaults below `verification/results/` | verification run |

### Stage P-B — Fugu-class product (MUST)

| Gate | Target | Where proven |
|---|---|---|
| P-B1 (streaming orchestrator) | `Orchestrator.run_chat(messages, tools, stream=True)`: route decided fast, final synthesizer/worker stage streamed token-by-token, keep-alive status events on long Conductor runs; `kairyu-auto` TTFT ≤1.5× the underlying engine's TTFT on the direct-route path | bench + tests |
| P-B2 (orchestration usage + trace) | `usage.orchestration_input/output_tokens` on every auto request (Fugu parity, billing necessity); opt-in `X-Kairyu-Trace` returns route/DAG/verifier verdicts (the transparency differentiator) | `tests/server/` |
| P-B3 (chat UI — COMPLETE) | Open WebUI shipped as a compose service against the gateway; a fresh user chats with `kairyu-auto` and per-model endpoints, streaming, after one `docker compose up`. Custom UI work is limited to an orchestration-trace viewer | pinned Playwright compose smoke |
| P-B4 (tiered auto models) | `kairyu-auto` and `kairyu-auto-max` are both discoverable; alternating direct/AUTO serving requests keep AUTO TTFT within 1.5x of the direct path | verification + tests |
| P-B5 (tenancy v1) | Key→tenant map in `DeploymentSpec`; per-key token-bucket limits in-gateway; append-only usage ledger + `/admin/usage`; two keys get isolated 429s; ledger reconciles with Prometheus counters to <0.1% | `tests/server/` |

**Tiered-example amendment (2026-08-13).** The quality-first Chat UI profile
exposes exactly one orchestration model. L2 borrows its deployment-owned L1
pools directly and runs an explicit verifier-gated DAG with at most two
refinements. The profile opts in to model-attributed completed intermediate
outputs via `reasoning_content`, which Open WebUI presents under a separate
expandable reasoning control; the publisher answer remains separate
`content`. Direct models and policy candidates are confined to a loopback-only
benchmark profile. See `docs/design/example-layered-orchestration.md`.

P-B2 is CPU-green as of 2026-07-27 (issue #196): direct, Conductor, and
MoA unary/streaming paths expose cumulative internal usage; retry, fallback,
partial-failure, cancellation, structured-trace, and privacy cases are fixed
server tests.

P-B3 is closed on the pinned Open WebUI v0.11.0-slim surface (issue #197).
One normal Compose command starts healthy Kairyu and Open WebUI services with
the direct `default` and orchestrated `kairyu-auto` models. A profile-scoped,
digest-pinned Playwright v1.60.0 runner creates the first user through the UI,
selects and streams both models, reloads the conversation, observes a real
gateway-outage error, and succeeds again after only Kairyu is restarted. The
mandatory browser job is independent of the general Compose integration job.
The orchestration-trace viewer remains the only custom UI work and is explicitly
tracked in §3.

Direct/AUTO OpenAI request parity is GPU-green as of 2026-07-28 (issue #208).
An immutable per-request intent carries sampling, `n`, logprobs, tools/tool
choice, and response format through direct, Conductor, and MoA routes without
shared-orchestrator mutation. Private stages keep scalar sampling but use
single unstructured/tool-free generations; the final boundary preserves every
public choice and logprob and enforces tools/schema. Qwen3-32B TP8 produced two
schema-valid choices with choice-scoped logprobs plus the required named tool,
then completed a plain request with healthy readiness. The reproducible raw
artifact is
`bench/results/auto-params-qwen3-32b-tp8-2026-07-28.json`.

P-B4 serving performance was revalidated after request-intent propagation on
2026-07-28 (issues #198/#208). One Qwen3-32B TP8 deployment exposed direct,
standard AUTO, and max AUTO endpoints. Twelve alternating direct/standard pairs
measured 1.0123x p50 and 0.7666x p99 TTFT ratios. The retained runner now records
only this direct-versus-AUTO serving envelope; model evaluation belongs outside
the verification gate.

P-B5 is CPU-green as of 2026-07-27 (issue #199): the supported
`DeploymentSpec` path proves isolated two-key 429s and exact (0% error)
ledger-versus-Prometheus reconciliation. The same usage counters cover all
supported execution modes and restore from the ledger after restart, including
malformed-tail recovery and shutdown drain.

### Stage P-C — Competitive proof + developer completeness

| Gate | Target | Where proven |
|---|---|---|
| P-C1 (MUST — the headline artifact) | `verification/product/performance/frontier_compare.py`: multi-target (Kairyu, Anthropic, OpenAI, DeepSeek), identical prompt sets, TTFT/TPOT/goodput/$-per-Mtok + small quality eval; nightly unattended run publishing a dated scoreboard + methodology (prompts, sampling, region, time-of-day, provider cache state) to `bench/results/` | scheduled run |
| P-C2 (Responses API — COMPLETE) | `/v1/responses` developer surface (`input`, canonical streaming events, flat/namespace tool calls, `previous_response_id` server-side state): OpenAI SDK sync/async clients and a Codex-class agent work unmodified (Fugu parity) | `tests/server/test_responses_api.py`, Qwen3-32B TP8 Codex smoke |
| P-C3 (embeddings — COMPLETE) | `/v1/embeddings` (+optional rerank) as a new engine-backend kind; Open WebUI RAG works end-to-end against Kairyu alone | compose smoke |
| P-C4 (vision) | Content-parts (`[{type:"text"|"image_url"}]`) through template + engine; image chat works in Open WebUI against a VLM replica | manual + tests |
| P-C5 (pricing signals) | Per-tenant cached-token discount fields in the ledger + price-sheet config; invoice-grade CSV export distinguishes cached vs uncached input | `tests/server/test_pricing_invoice.py` |

P-C2 is CPU- and GPU-green as of 2026-07-28 (issue #201). The official OpenAI
Python SDK parses unary, sync stream, and async stream responses, including
gapless text/function events and completed/incomplete/failed terminals.
Function calls and outputs round-trip across `previous_response_id`; stored
state is bounded and tenant scoped. Codex namespace tools retain stable public
IDs while using the shared chat template/tool enforcement internally. An
unmodified Codex CLI completed a Responses-wire turn and namespace tool-result
loop against Qwen3-32B TP8 on all eight RTX PRO 6000 GPUs. The exact Codex
version, HTTP attempts, usage, command result, and unsupported hosted-search
negative gate are recorded in
`bench/results/responses-codex-qwen3-32b-tp8-2026-07-28.json`.

P-C3 is CPU-green as of 2026-07-31 (issue #202). The production `fastembed`
backend loads a revision- and SHA-pinned all-MiniLM-L6-v2 ONNX snapshot
offline, validates 384-dimensional normalized output, reports exact tokenizer
usage, bounds concurrent work, and participates in startup, readiness,
liveness, and shutdown. The pinned Open WebUI Compose topology sends both
embedding and chat traffic only to Kairyu. Its mandatory smoke proves a
two-input embedding request, document ingestion, vector retrieval of a
query-only canary, a citation-bearing Kairyu answer that requires the retrieved
context, visible gateway outage, and retrieval/answer recovery after restarting
only Kairyu. Optional reranking remains explicitly disabled and deferred; it
does not weaken the required retrieval gate.

P-C4's production path is implemented as of 2026-07-31 (issue #203). Ordered
OpenAI role/content parts cross Kairyu as a typed multimodal prompt into one
pinned stock-vLLM Qwen3-VL-32B-Instruct TP8 replica; vLLM alone owns the model
processor and chat template. Kairyu accepts only locally decoded and verified
inline PNG/JPEG/WebP data URLs under explicit count, byte, pixel, dimension,
aspect-ratio, and complete-context bounds. Full media work completes off the
event loop before admission or SSE headers, and exact processor usage is
mandatory. The GPU-only Compose overlay and `scripts/webui_vlm_smoke.sh` bind
RED/BLUE output semantics, unary/stream usage, remote-URL rejection, and the
normal Open WebUI owned-file upload path. The clean `b8971cb` TP8 gate passed:
RED and BLUE produced their corresponding distinct answers, unary and stream
each reported exact 1,060 input / 2 output tokens, the remote URL failed with
`400 invalid_image`, and the real browser owned-file path rendered RED. The
retained result is
`bench/results/issue-203-vlm-image-chat-qwen3-vl-32b-tp8-2026-07-31.json`.

## 3. Non-goals

- Audio endpoints; fine-tuning API; marketplace/OpenRouter distribution mechanics.
- Building a full custom chat frontend (Open WebUI integration first; custom work is
  the trace viewer only; revisit after real user feedback).
- Payment processing — the ledger exports billing data; invoicing is external.
- Per-request model attribution *pricing* (Fugu-style blended rate is the model;
  attribution appears in the trace, not the bill).

## 4. Seams (informative, non-binding)

- API gaps land in `entrypoints/server/app.py` + `protocol.py`; new route families
  (responses, embeddings, admin) as sibling route modules like `batch_routes.py`.
- `chat_template.py` is replaced by an HF-template layer; per-model template config
  rides `DeploymentSpec` engines.
- Orchestrator streaming extends `orchestration/orchestrator.py` / `conductor.py`;
  internal token accounting reuses `budget.py`'s existing spend tracking.
- Tenancy extends `server/settings.py` + `middleware.py`; the ledger reuses the
  `batch/store.py` atomic-file pattern; quota state feeds G5 F5 admission later.
- `verification/product/performance/frontier_compare.py` grows out of `verification/l1/performance/serving_bench.py` (P-A5 first).

## 5. Evidence and reporting rules

G2 §8 carries forward. Scoreboard claims (P-C1) never compare across sessions:
every published comparison ran the same prompts in the same session window, and the
Kairyu column states engine phase (mock/CPU/GPU) until Track E makes it real.

## 6. Human sign-off checklist (blocking)

- [ ] P-A gates green (CPU)
- [ ] P-B gates green (CPU/compose)
- [ ] P-C1 scoreboard producing nightly artifacts
- [ ] P-C2–C5 green
