# Tiered Example Layered Orchestration

Status: **Accepted; implementation in progress** (2026-08-13).
Applies to: `examples/qwen3.8-deepseek-v4-8gpu/`.

## Goal

The tiered Chat UI must expose one chat product model whose request crosses
Kairyu L3 once, executes a bounded multi-model workflow in L2 against
deployment-owned L1 engines, and returns one final L3 answer. The public API
also exposes a separately typed embedding capability. The UI may reveal
completed intermediate work only when the product policy explicitly enables
it, and must keep that work visually separate from the final answer.

## Decisions

### EO-D1 — L2 borrows deployment-owned L1 engines

An orchestration worker may name an `engine_ref` resolved from the deployment's
`engines:` or `pools:` registry. The resolved object is passed directly to L2;
L2 does not call the deployment's public L3 HTTP endpoint. Borrowed engines are
started and stopped by the deployment, while factory-created standalone DSL
workers remain owned by their orchestrator.

### EO-D2 — The product model registry is explicit

The product deployment sets a fail-closed `public_models` allowlist. Public
model discovery and generation routes use that allowlist; lifecycle,
readiness, metrics, and internal orchestration continue to cover all runtime
resources. This product example does not register direct L1 models or policy
candidates as orchestrators; it keeps one orchestration YAML and one public
chat model. A public embedding model is not a chat policy candidate and remains
absent from `/routing`.

### EO-D3 — The quality path is an explicit bounded DAG

> **Superseded (2026-08-15):** the role graph below was replaced by the
> head-streamed, execution-gated nine-role coding DAG of
> `example-coding-orchestration.md` (ECO-D2). The bounded-DAG principle and
> the "verify a draft before publishing" rule survive there; the
> pre-verified public stream is now split into a committed head prefix plus
> a verified continuation (ECO-D4). EO-D1/D2/D4/D5/D6 remain in force.

The UI-visible model uses this role graph instead of the single-pass MoA
shortcut:

```text
planner (Tier2)
  -> proposal_a / proposal_b / proposal_c (Tier1, parallel)
  -> draft_synthesis (Tier2)
  -> verifier (Tier2)
       FAIL -> draft_synthesis -> verifier (at most two refinements)
  -> publisher (Tier2)
```

The policy sets `moa_samples: 0`, `max_refine_depth: 2`, and `max_steps: 11`.
The draft is verified before publishing because streamed publisher deltas
cannot be retracted.

### EO-D4 — Intermediate work is policy-visible, attributed, and separate

Intermediate generated output remains hidden by default. An immutable
orchestration policy flag may expose completed planner, proposal, synthesis,
verifier, and refinement outputs for this example. Each section identifies its
role, attempt, engine, and model. Kairyu carries the sections in OpenAI
`reasoning_content`, while the publisher answer remains in `content`.

The pinned Open WebUI version already renders `reasoning_content` as a separate
expandable reasoning item, so this contract needs no UI fork. This visibility
is an operator-selected product feature, not a request-controlled escape hatch.
Trace v2 stays metadata-only and never becomes a second store for generated
text. API keys, backend URLs, system prompts, and exception details are never
included.

### EO-D5 — Structural and native-L1 evidence are distinct

The existing vLLM services may prove the L3/L2/L1 object boundary and UI
behavior. They do not prove the roadmap's native-Kairyu L1 requirement. The
example remains transitional until the same full-checkpoint product path passes
the native Qwen3.6 and DeepSeek V4 correctness, recovery, soak, and performance
gates.

### EO-D6 — The public embedding capability is pinned and truthful

The example publishes `embed-small` through the existing production FastEmbed
backend. Its image contains the immutable MiniLM repository revision, model
digest, and provenance digest, and runtime startup is offline. Public model
discovery therefore contains `kairyu-auto-max` and `embed-small`, while
`/routing` contains only the chat product. The launcher must pass a real
two-input embeddings request and validate model identity, ordered indices, two
finite 384-dimensional vectors, and positive exact usage. Open WebUI manually
allows only the chat model on its connection so the embedding ID cannot appear
as a chat choice.

## Acceptance

- `/v1/models` on the product gateway lists exactly the product orchestrator
  and `embed-small`; `/routing` and Open WebUI list only the product chat model,
  and guessed internal model names fail on public generation APIs.
- `/v1/embeddings` returns two ordered finite 384-dimensional vectors from the
  pinned offline bundle and reports non-empty exact usage.
- A request follows `L3 -> L2 -> L1... -> L2 -> L3` without a second L3 ingress
  and without duplicate lifecycle ownership.
- Scripted verifier failures cause real bounded synthesis retries, with exact
  internal usage and metadata trace accounting.
- Open WebUI shows model-attributed intermediate output behind its reasoning
  disclosure control and shows the publisher answer separately.
- An assistant response containing `reasoning_content` can be appended to the
  next Chat Completions request through the pinned LiteLLM message shape;
  nullable provider metadata is ignored without weakening non-null validation.
- The product URL is reported from the actual bound host after a live browser
  and API smoke, and the native-L1 limitation remains explicit until closed.
