# Goal G6: Product Surface — Truthful API, Fugu-Class Orchestrated Product, Competitive Proof (Roadmap Track P)

Status: Goal defined (2026-07-03). P-A and most of P-B are CPU-verifiable now;
P-C's scoreboard needs real engines for the Kairyu column but runs against
frontier APIs immediately.
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
| P-A5 (bench honesty) | `bench/serving_bench.py` gains auth headers and token-granularity TPOT (tokens, not SSE chunks — providers coalesce differently); per-run JSON to `bench/results/` | bench run |

### Stage P-B — Fugu-class product (MUST)

| Gate | Target | Where proven |
|---|---|---|
| P-B1 (streaming orchestrator) | `Orchestrator.run_chat(messages, tools, stream=True)`: route decided fast, final synthesizer/worker stage streamed token-by-token, keep-alive status events on long Conductor runs; `kairyu-auto` TTFT ≤1.5× the underlying engine's TTFT on the direct-route path | bench + tests |
| P-B2 (orchestration usage + trace) | `usage.orchestration_input/output_tokens` on every auto request (Fugu parity, billing necessity); opt-in `X-Kairyu-Trace` returns route/DAG/verifier verdicts (the transparency differentiator) | `tests/server/` |
| P-B3 (chat UI) | Open WebUI shipped as a compose service against the gateway; a fresh user chats with `kairyu-auto` and per-model endpoints, streaming, after one `docker compose up`. Custom UI work is limited to an orchestration-trace viewer | compose smoke |
| P-B4 (tiered auto models) | `kairyu-auto` (latency-biased routing) and `kairyu-auto-max` (Conductor/MoA depth) both in `/v1/models`; auto ≤1.5× direct-call latency, auto-max quality-wins on a fixed eval set | bench |
| P-B5 (tenancy v1) | Key→tenant map in `DeploymentSpec`; per-key token-bucket limits in-gateway; append-only usage ledger + `/admin/usage`; two keys get isolated 429s; ledger reconciles with Prometheus counters to <0.1% | `tests/server/` |

### Stage P-C — Competitive proof + developer completeness

| Gate | Target | Where proven |
|---|---|---|
| P-C1 (MUST — the headline artifact) | `bench/frontier_compare.py`: multi-target (Kairyu, Anthropic, OpenAI, DeepSeek), identical prompt sets, TTFT/TPOT/goodput/$-per-Mtok + small quality eval; nightly unattended run publishing a dated scoreboard + methodology (prompts, sampling, region, time-of-day, provider cache state) to `bench/results/` | scheduled run |
| P-C2 (Responses API) | `/v1/responses` subset (`input`, streaming events, tool calls, `previous_response_id` server-side state): OpenAI SDK `responses.create` and a Codex-class agent work unmodified (vLLM gap, Fugu parity) | `tests/server/` |
| P-C3 (embeddings) | `/v1/embeddings` (+optional rerank) as a new engine-backend kind; Open WebUI RAG works end-to-end against Kairyu alone | compose smoke |
| P-C4 (vision) | Content-parts (`[{type:"text"|"image_url"}]`) through template + engine; image chat works in Open WebUI against a VLM replica | manual + tests |
| P-C5 (pricing signals) | Per-tenant cached-token discount fields in the ledger + price-sheet config; invoice-grade CSV export distinguishes cached vs uncached input | `tests/server/` |

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
- `bench/frontier_compare.py` grows out of `bench/serving_bench.py` (P-A5 first).

## 5. Evidence and reporting rules

G2 §8 carries forward. Scoreboard claims (P-C1) never compare across sessions:
every published comparison ran the same prompts in the same session window, and the
Kairyu column states engine phase (mock/CPU/GPU) until Track E makes it real.

## 6. Human sign-off checklist (blocking)

- [ ] P-A gates green (CPU)
- [ ] P-B gates green (CPU/compose)
- [ ] P-C1 scoreboard producing nightly artifacts
- [ ] P-C2–C5 green
