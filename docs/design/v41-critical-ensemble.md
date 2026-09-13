# V4.1 critical ensemble

Status: implementation started from main `99c5eadb`; GPU gates pending.
Accepted plan: `docs/superpowers/plans/2026-09-13-v41-six-gpu-critical-ensemble.md`.

## V41C-D1 — Scope and ownership

The owner accepted the six-GPU V4.1 plus two TP1 Qwen replica plan on
2026-09-13 and requested a draft PR with incremental commits and pushes.
The existing tiered example owns the four policies, four Qwen candidates,
independent DeepSeek candidate, Requirement extraction, critical comparison,
reconstruction, DeepSeek audit and bounded repair policy. Keep its DSL and
existing operational entry points; do not add example Python files or an
alternative orchestration/input/HTTP/lifecycle stack.

Only the Requirement specification from closed PR #595 is inherited:
the JSON array, R1-based IDs, five string fields, minimum/optional criteria,
source attribution and per-requirement audit. The extractor uses DeepSeek
with a high floor and explicit max inheritance. Closed implementations and
their measurements do not establish this implementation's correctness.

The current main only measures V4.1 on TP8/EP8. Six-GPU runtime selection,
native conversation and image propagation, effort/default forwarding,
all-choice audit, capacity fitting and over-context processing must be
established independently. Mandatory primary measurements include the real
route judge and use fresh, completed six-GPU native baselines. No 113-task
rerun or general model-quality improvement campaign is required.

## V41C-D2 — Configured internal budgets are independent of public MAX

Shared contract: `OrchestrationRequest.internal_sampling_params`, consumed
by both Conductor and MoA, must distinguish the configured private generation
budget from the caller's final completion allowance. Main takes their minimum,
so a short public answer allowance also truncates a planner, proposal or
verifier. Role sampling cannot fix this because Conductor clamps each role
against that already-reduced value.

The smallest change uses an explicitly supplied internal ceiling as the
private ceiling, preserving caller sampling and final output parameters.
Calls without a configured private ceiling retain their existing fallback.
The example still supplies all model/role limits; the shared method knows
nothing about Requirement, candidates or audits. Existing orchestration
request tests exercise actual Conductor and MoA dispatch and check final
intent separately, including the case where public MAX is smaller.

This change alone does not fit a request to the model context, reserve
visible output after thinking, or solve the head-exhaustion and n-choice
continuation paths. Those remain separate work.

## V41C-D3 — Preserve terminal upstream request failures

Shared contract: a failed ordinary AUTO publisher must retain its actionable
backend failure through Conductor, Orchestrator and L3. Main records an
exception type and can replace a rejected upstream request with an unrelated
empty-output 502. Example configuration cannot recover that lost cause.

Carry the terminal error using the existing execution-result/error path.
Preserve safe upstream request validation status/code and retry classification,
without exposing raw upstream bodies, URLs or credentials. Upstream
authentication/infrastructure failures remain distinct from caller input
errors. Preserve the intentional best-so-far policy for other roles; the
example's mandatory-stage requirements do not authorize globally disabling it.
Regressions exercise ordinary one-role AUTO requests and public unary/SSE
errors, independent of the ensemble's model names and workflow.

Terminal rejection after a committed head follows the existing streamed
partial-result error contract: retain the published head and known usage,
then report failure. This also applies to the deferred verifier path; an
actual backend rejection must not become a successful head-only response.
Empty continuation without a backend exception retains its existing policy.

## V41C-D4 — Native chat is a typed backend input

Shared contract: an ordinary `GenerationRequest` sent to a chat-capable
backend must preserve message roles, nullable assistant content, reasoning
history, tool-call arguments/IDs, tool results and ordered image parts.
Main's string input becomes one user message. `MultimodalPrompt` preserves
roles and media parts but requires media and cannot carry tool metadata;
using it for text-only chat would weaken its existing contract. A rendered
string or JSON transcript is not equivalent to a native conversation.

Add `ChatPrompt` and `ChatMessage` to the existing prompt union, tagged wire
format and capability validation. Metadata is an immutable request-local
JSON snapshot; tool argument strings remain verbatim. Reuse the existing
media items, validation, preparation cache, transport and cancellation
lifecycle. Text-only chat does not imply image support. Engines without
a native chat renderer reject the input instead of flattening it.

The independently reusable use is an ordinary OpenAI-compatible backend
call with an assistant/tool conversation, with or without an image. Tests
cross the prompt wire and backend dispatch boundaries and assert the
actual upstream message payload. Existing image safety, usage and cleanup
cases cover both carriers. Token accounting must not replace native chat
with a whitespace estimate when the renderer's usage is unavailable.

This slice does not change L3 routing, select model templates, derive role
inputs or implement the ensemble DAG. Those policies and derivations remain
separate work. Admission work estimates are not exact rendered token counts;
per-dispatch capacity fitting remains open.

## Open implementation conditions

- Establish a supported six-GPU topology from the pinned main runtime and
  checkpoint; confirm draft expert divisibility and memory before selection.
- Keep the exact native message/tool/image structure through AUTO derivation
  and token accounting, including assistant continuation and tool metadata.
- Express V4.1 non-thinking and Requirement's high floor via existing
  extension points or an independently justified minimal request contract.
- Verify each final choice and keep its repair, continuation, usage and
  publication state separate. Existing n>1 audit/floor bypass is insufficient.
- Identify a minimal shared input-capacity contract for MoA and verifier
  overflow. Main has no lossless range traversal/editing facility; a generic
  name or similarity of overflow is not admission of a document runner.
- Preserve all mandatory stages without silently weakening shared best-so-far
  semantics. Record unresolved policy expression instead of adding a
  workflow-specific framework switch.

## Validation record

The existing running deployment and old branch results are not evidence for
this branch. No six-GPU startup or performance gate has passed on these bytes.

### Terminal error propagation

- Final and verifier rejection preserve safe status/code through unary and
  SSE responses, without retrying the rejected request or exposing internal
  draft content. Tests retain the committed head where present and reconcile
  the known usage/ledger on failure.
- 184 related usage/trace, head and Orchestrator tests pass after the headed
  error correction. The earlier unheaded version passed 554 tests across
  the related Conductor and public Chat/Responses suites. Ruff is clean.
- The judge/preflight accounting fixture now explicitly configures its
  eight-token internal budget; it no longer relies on a public limit to
  shrink private work. Its tenant burst and accounting assertions remain.

### Independent private budgets

- Before the correction, all three new small-public-MAX cases failed: unary
  Conductor, streamed Conductor, and MoA proposals inherited 17 instead of
  their configured 64 tokens. The existing eight-token private limit passed.
- The configured private ceiling now applies in both directions while final
  completion MAX and caller intent remain unchanged. Existing tests also
  retain the no-explicit-private-ceiling fallback.
- 32 request/MoA tests pass. Combined validation with the final headed-error
  correction passes 588 tests across request, MoA, Conductor/head,
  Orchestrator, usage/trace and public Chat/Responses API suites. Ruff is clean.
- Reservations now reflect the larger configured private work when a caller
  requests a small public answer; a deployment must provision tenant limits
  for that work. Public MAX cannot be used to understate its private cost.

### Native chat carrier

- 545 related tests pass across prompt/OpenAI transport, backend validation,
  media preparation, native/mock engines, LLM/AsyncLLM compatibility and
  tenant metering. The literal unary/SSE transcript assertions also pass.
- New coverage protects native tool history and exact argument strings
  through wire serialization and dispatch, image part ordering and the
  existing image rejection/usage contracts, plus missing-usage accounting
  on failed or cancelled chat streams. No static configuration-list tests
  or example-specific workflow tests were added.
- Independent review found no actionable regression in this slice. Ruff
  and whitespace checks pass. AUTO derivation and exact input-capacity
  accounting are still pending.
