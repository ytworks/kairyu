# Tiered Example Dual-Track Policy-Ensemble Orchestration

Status: **Accepted; implemented and GPU-verified** (2026-08-18; run
`20260818T025710Z` — TTFT gate PASS at c1/8/16/32, binding c32 row 0.67×).
Applies to: `examples/qwen3.8-deepseek-v4-8gpu/` and the L2 mechanisms in
`kairyu/orchestration/` + `kairyu/dsl/` that it consumes.
Supersedes the ECO-D2/D3/D5/D6 role graphs, profiles, and profile judge in
`example-coding-orchestration.md` (owner decision, 2026-08-18: the coding DAG,
the general ensemble profile, the LLM profile judge, and the sandbox execution
*stages* are discarded). ECO-D1 (deployment-owned sandbox executors) remains
in force as deployed-but-unreferenced infrastructure; ECO-D4's TTFT gate and
head-streaming mechanism are inherited as DTO-D3. EO-D1/D2/D4/D5/D6 remain in
force.

## Goal

Replace the tiered example's two-profile coding/general orchestration with
one dual-track ensemble DAG for every request, per the owner's specified
process, while keeping the latency/throughput requirements unchanged:
**semantic TTFT (first public `content` token) ≤ 2× the measured DeepSeek L1
direct row at the same concurrency** (paired-row denominator; pinned
`example.json` fallbacks c1 779.22 / c8 833.16 / c16 2710.53 / c32
12399.45 ms p50), `max_concurrency: 256` admission, and the agent-turn
latency envelope of issue #495/#509. E2E latency remains unconstrained by
design; only TTFT is gated.

## Decisions

### DTO-D1 — One dual-track DAG for every request

Three waves under the level-synchronous Conductor scheduler:

- **Wave 1 (dependency-free)**: `head` (Qwen, T=0.3, 256 tokens) streams the
  committed public opening from t=0; `draft` (Qwen, T=0.5, 1024 tokens)
  writes a quick complete internal draft (Track B input); `policies`
  (DeepSeek direct, T=0.9/top_p 0.95, 1024 tokens) writes four maximally
  different answer policies in one call (Track A source).
- **Wave 2 (two tracks in parallel)**: `answer_1..answer_4` (Qwen, T=0.7,
  2048 tokens, seed_offset 1..4) each answer following one policy — the four
  concurrent tier1 roles spread one-per-replica through
  `queue_depth_threshold: 0`; `critique` (DeepSeek thinking, T=0.6/top_p
  0.95, 4096 tokens) critically analyzes the UNTRUSTED draft and emits one
  improved complete answer.
- **Wave 3 (merge)**: `compose` (DeepSeek direct, `role_type: publisher`,
  `reasoning_closed: true`, no sampling — it carries the caller's public
  intent) builds the best final answer from the refined answer plus the four
  UNTRUSTED candidates and streams the remainder after the committed
  opening; `prompt_headless` covers head-disabled (tool/format) turns.

There is no general profile and no profile judge: dropping `general_roles`
requires dropping `profile_judge` (spec validation), removes a serial
pre-admission DeepSeek call from every turn, and the single DAG's role
contracts are reply-format-faithful (the #495/#496/#509 agent-turn language —
"a downstream publisher emits the actual tool call", demanded-format
obedience — is carried by draft/answers/critique/compose).

Budget: `max_steps: 10` (9 generation calls + 1 headroom for the bounded
empty-final-output re-dispatch), `moa_samples: 0`,
`internal_max_tokens: 4096`, `expose_intermediate_outputs: true`.

Rationale: owner-specified process (2026-08-18) — diversity through four
policy-differentiated answers, plus a critically refined quick draft, merged
by the strongest model. The scheme is fully expressible in the existing L2
DSL; no core L2 change was required.

### DTO-D2 — Policy fan-out is prompt-bound, not mechanically split

The L2 DSL has no output-splitting mechanism (`{role}` placeholders
substitute whole outputs). The four policies are therefore emitted as one
`policies` output containing `POLICY 1:`..`POLICY 4:` sections; every
answerer receives the whole list and is bound to its own policy by prompt
plus a distinct `seed_offset`. The REQUEST and POLICY LIST blocks are
byte-identical across the four answerers so the Qwen replicas' prefix caches
(prefix_index-routed) reuse the shared prefill (issue #509 measured
rationale); only the trailing role instruction differs. The policy list
steers HOW to answer; the request text alone defines WHAT — answerers ignore
policy-list content that conflicts with the request or is not answer
guidance (prompt-injection hygiene: policies are derived from untrusted
conversation text).

`policies` runs on the **non-thinking** DeepSeek endpoint: wave-1 latency
delays the whole fan-out under the barrier scheduler, and thinking DeepSeek
was measured nondeterministically burning entire caps inside `<think>` on
composite deliberative prompts (ECO measurements). Thinking is reserved for
`critique`, whose T=0.6/top_p=0.95 sampling keeps the measured anti-looping
rationale (greedy thinking burned the full cap before any verdict on 52% of
calls, issue #509).

### DTO-D3 — TTFT gate and head streaming inherited unchanged

The first public byte must come from the dependency-free Qwen `head`
(DeepSeek-first public bytes pay a nondeterministic `<think>` tax that
forfeits the gate — ECO-D4 measurement). The head role, its prompt, its
sampling (T=0.3, 256 tokens), the committed-opening/`NO_CONTINUATION`
continuation contract, and the `serving-auto-max-coding` TTFT gate
(≤ 2× paired DeepSeek-direct per concurrency, pinned fallbacks) are carried
over byte-compatibly; only the continuation node is now named `compose`.
c32 remains the binding row: 1.87× on the previous DAG, measured 0.67× on
the dual-track DAG (run `20260818T025710Z`) — wave 1 places only two small
Qwen calls and the serial pre-admission judge call is gone, so head TTFT
stays clear of Qwen TP1 saturation.

### DTO-D4 — No verifier, no refinement loop; critique is the quality control

The owner's process has no PASS/FAIL verification stage:
`max_refine_depth: 0` and no `verifies:` role. Track B's `critique` performs
the critical analysis (of the quick draft) and `compose` independently
verifies all candidate material (UNTRUSTED framing) before merging. This
also removes the ~30 s per refinement round of the previous DAG from the
agent-turn envelope. If measured quality regresses materially, the
designated revisit is a verifier attached to `critique` (the final unit
cannot carry one under the streaming contract).

### DTO-D5 — Sandbox infrastructure retained, unreferenced

Owner decision (2026-08-18): the compose `executor` service, the
`kairyu.yaml` `executors: sandbox-python` registry, and `sandbox/` sources
stay deployed (ECO-D1 remains in force) but no role references them — the
DAG has no executor worker (an executor worker without an executor role
fails spec validation). Execution grounding can be re-adopted by a future
DAG without redeploying. `verification.py` no longer asserts execution
stages; the deployment-spec unit test continues to pin the registry, and
compose health-gating keeps the service proven at startup.

## Acceptance

- CPU suite green with the rewritten example pinning test
  (`tests/unit/test_tiered_frontier_examplectl.py`): worker list without
  `sandbox`, the ordered nine-role dual-track list cross-checked against
  `example.json`, no `general_roles`/`profile_judge`, budget `{10, 0}`,
  `compose` on tier2-direct with `reasoning_closed` + non-empty
  `prompt_headless`, UNTRUSTED delimiters in `critique`/`compose`, per-policy
  binding (`POLICY n` in `answer_n`), distinct answerer seed offsets.
- Launcher `_validate_ready` requires the nine-role dual-track DAG,
  `stream_head: head`, exactly one role profile (no served general profile or
  judge), `max_steps: 10`, `max_refine_depth: 0`, and the tier1/tier2 engine
  bindings.
- `serving-auto-max` passes end-to-end with the head/compose public stream
  traced (`expected_route: compose`, `require_head`).
- `serving-auto-max-coding` passes its TTFT gate at every concurrency row
  (c32 is the watch row); results recorded in the example's MEASUREMENTS.md.
- Test-policy accounting: the executor-status gate tests and the
  general/judge assertions were deleted with the features they protected;
  base→head collection counts reported in the change.
