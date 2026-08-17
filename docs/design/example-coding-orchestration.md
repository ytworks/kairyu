# Tiered Example Coding Orchestration

Status: **Accepted; implementation in progress** (2026-08-15).
Applies to: `examples/qwen3.8-deepseek-v4-8gpu/` and the L2 mechanisms in
`kairyu/orchestration/` + `kairyu/dsl/` that it consumes.
Supersedes the EO-D3 role graph in `example-layered-orchestration.md`
(EO-D1/D2/D4/D5/D6 remain in force).

## Goal

Replace the tiered example's quality DAG with a coding-first orchestration
that approaches frontier coding accuracy from the unchanged L1 fleet
(4× Qwen3.8-27B-FP8 TP1 + DeepSeek-V4-Flash TP4+EP4) while keeping the
product's **semantic TTFT (first public `content` token) ≤ 2× the measured
DeepSeek L1 direct row at the same concurrency** (c1 denominator: 779.22 ms
p50). The constraint binds TTFT only; E2E latency is unconstrained, which the
design exploits by streaming a committed answer opening immediately and doing
the ensemble/execution/verification work behind it.

## Decisions

### ECO-D1 — Deployment-owned sandbox executors, borrowed by `executor_ref`

A deployment declares `executors:` (name → base_url/timeout/optional UDS),
parallel to the
`engines:`/`pools:` registry of EO-D1. An orchestration worker with
`executor_ref` borrows the deployment-constructed `HttpExecutionBackend`; the
deployment owns its lifecycle. Executors are internal orchestration
resources, never served models. `kairyu validate` reports
`schema.unknown_executor_ref` for dangling references.

The example's executor is a CPU-only compose service with
`network_mode: none`: it has no IP interface and Kairyu reaches its HTTP
protocol only through a shared Unix domain socket. The socket volume is
read-only in Kairyu; with no IP interface, hostile code has no callback path
back to the gateway. The container also has a read-only rootfs, noexec/nosuid
tmpfs workdirs, `user 65534`, `cap_drop: ALL`, `no-new-privileges`,
pids/memory/cpu limits, and no published ports. Inside,
each submission runs in a fresh workdir under a dedicated process group with
`setrlimit` (CPU/AS/NPROC/FSIZE/CORE), a cleared environment, an in-process
wall-clock SIGKILL, capped output pipes, and supervisor-owned descendant
cleanup. The runner is a child subreaper and tracks every live submission root;
as soon as a root exits, it repeatedly scans `/proc`, kills same-UID processes
in the runner tree that do not belong to another live submission, and reaps the
targeted PIDs. A child-created session/process group therefore cannot survive
normal completion or keep inherited pipes open past the wall timer.
Model-generated code is hostile input; both enforcement layers are required.
v1 executes Python/pytest only; other languages are rejected as `unsupported`
and take the LLM-only path.

Review amendment (2026-08-15): the original internal Docker bridge was
rejected because bridge membership is bidirectional; `internal: true`
prevents external routing but does not prevent executor-to-Kairyu callbacks.

Review amendment (2026-08-15): timeout-only process-group cleanup was rejected
because submitted code can call `setsid()` and survive a normally completed
shim. The subreaper plus `/proc` sweep is mandatory; active-root ancestry keeps
concurrent submissions out of another submission's cleanup set.

### ECO-D2 — One generalized coding DAG replaces the seven-role DAG

`kairyu-auto-max` keeps its single-DAG, always-`multi_agent` routing
(`router.json` and its sha256 pin are byte-identical) and replaces the role
graph with:

```text
head (Tier1, streams public opening from t=0)
  ∥ testgen / proposal_impl / proposal_edge (Tier1, parallel)
  -> exec_matrix (sandbox: each proposal × generated tests + consensus)
  -> draft_synthesis (Tier1, conditioned on committed head + candidates + matrix)
  -> exec_draft (sandbox, inline) -> verifier (Tier2 max-thinking)
       FAIL -> regenerate -> re-execute -> re-verify (≤2 refinements)
  -> continuation (Tier2-direct, streams the verified remainder after the head)
```

`max_steps: 15` (9 base + 2×3 refinement), `max_refine_depth: 2`,
`internal_max_tokens: 4096`. Non-coding traffic runs the same DAG: testgen
answers `NOT_APPLICABLE`, executor stages skip locally (zero steps, zero
sandbox calls), and the pipeline degrades to plan/propose/synthesize/verify.
Worker/policy selection was driven by live measurement of the DeepSeek
identity-template contract (see ECO-D4a below):

- **head and synthesizer → Qwen (tier1).** The Qwen vLLM service applies a
  real server-side chat template with thinking disabled, so both stages emit
  deterministically — the head at a small-prompt Qwen TTFT (~0.3 s measured
  at c1). Every DeepSeek policy paid a nondeterministic `<think>` tax on
  deliberative prompts: fatal for the head's TTFT gate, and the composite
  synthesis prompt was measured leaving an empty public draft on both the
  max-thinking (2048 and 4096 caps alike) and the ordinary policy.
  Synthesis quality is gated by the execution evidence and the thinking
  verifier rather than by DeepSeek drafting.
- **continuation → ordinary DeepSeek (tier2-direct)** with the inline
  `<｜Assistant｜></think>` scaffold: its restating task converges after at
  most a short think, which is harmless mid-stream (measured 32/32 non-empty
  under the gate workload).
- **verifier → max-thinking DeepSeek (tier2)** with an inline
  `<｜Assistant｜><think>` scaffold: deliberate-then-verdict converges
  reliably and the verdict rides the parser's content channel.

The private cap is 4096 (not the historical 2048) for thinking-stage
headroom — E2E latency is unconstrained by design.

### ECO-D4a — DeepSeek L2 prompts carry their chat scaffold; suffixes survive refinement

The DeepSeek vLLM service runs an identity chat template (Kairyu owns
templating) and its `deepseek_v4` reasoning parser classifies generated text
before an emitted `</think>` as private reasoning. A raw L2 role prompt
therefore reaches the model with no chat scaffold (near-base-model behavior)
and its output is nondeterministically hidden by the parser. DeepSeek role
prompts embed the scaffold inline — `<｜begin▁of▁sentence｜><｜User｜>` in
the body and the assistant marker as the new `RoleSpec.prompt_suffix`, which
the Conductor re-appends after every attempt (verifier refinements insert
their feedback before it, or refined generations would run against a
malformed prompt). Qwen roles need no scaffold: their vLLM service applies
the real Qwen chat template server-side.
Images continue to reach only the vision-capable Qwen roles; the head is
instructed never to guess unseen image content.

### ECO-D3 — Executor results are untrusted data; steps, not tokens; degrade, not fail

An executor role declares `executor: {code_from, tests_from, mode, limits}`.
Inputs are extracted as the first complete ```python fence (or a bare block
that parses); the structured report is rendered as machine JSON into
downstream prompts inside `--- UNTRUSTED ... ---` delimiters (the MoA
candidate-draft pattern). `mode: matrix` reports per-candidate results plus a
per-test consensus count, and synthesis/verifier prompts are explicitly
licensed to judge a test that every candidate fails against the request text
— generated tests must not be able to sink good proposals.

Budget semantics follow M1 D4: one step reserved strictly pre-await,
reconciled at cost 0; a stage with nothing runnable skips locally without
reserving; a sandbox outage/timeout produces an `unavailable` report and the
request continues (verifier may still PASS by inspection). Executor stages
report zero tokens (m9 truthful usage) and trace as `operation: "execution"`
with counts/status metadata only — never stdout/stderr text (trace v2 stays
metadata-only). An executor named in a verifier's `depends_on` is re-run
inline before every verdict so refinement never judges stale execution
evidence.

### ECO-D4 — Split public stream: head + continuation (mechanism EO-D7..D9)

Mechanism, owned by `kairyu/orchestration/conductor.py`:

- **EO-D7 (head/continuation).** At most one dependency-free
  `role_type: head` streams public `content` deltas from t=0 while the rest
  of the pre-final DAG runs; the selected final unit must transitively depend
  on the head and streams the remainder. Committed bytes are never retracted:
  only an exact verbatim repetition of the committed prefix is deduplicated;
  divergence flushes verbatim. Head sampling derives from the caller's public
  params (n=1 forced; tools/response_format/logprobs stripped). A caller
  intent incompatible with a split stream (tools, response_format, n>1,
  logprobs) disables the head for that call (`skipped:intent`) instead of
  rejecting. Failure contract: zero-byte head failure degrades to a
  continuation-only stream; post-byte failure commits the partial prefix;
  continuation failure raises `ConductorStreamError` whose result carries
  exactly the emitted public text; budget exhaustion returns the committed
  head as best-so-far. Head and continuation deltas are emitted bare
  (no completions) so the server owns public offsets across the seam; the
  result carries one merged public completion.
- **EO-D8 (incremental reasoning).** Completed pre-final stages stream as
  `reasoning` events the moment they finish, merged with head deltas through
  one bounded queue in a single producer task; the continuation tail stays
  pull-through (m11 D1). The orchestrator's multi-stage keep-alive now guards
  every inter-event gap, not only the first.
- **EO-D9 (per-role sampling).** `RoleNodeSpec.sampling` overrides
  temperature/top_p/max_tokens/stop and derives per-role seeds
  (`seed_offset`, MoA-style). Internal roles stay capped by
  `internal_max_tokens` (#208); the selected final unit rejects overrides —
  it carries the caller's public intent.

Review amendment (2026-08-16, issue #496): the head's role cap is clamped to
the caller's public output limit and the continuation receives the remaining
budget after the head's reported completion tokens — a finite caller limit is
one budget across the split stream, and a consumed budget skips the
continuation and reports `length` for the head-only answer. The `continuation`
role declares `reasoning_closed: true`: its `…</think>` scaffold is
prompt-side, so a direct answer that the DeepSeek reasoning parser classifies
entirely as reasoning is reclaimed as public text instead of surfacing an
empty `content`. It also declares a `prompt_headless` body rendered whenever
the head is disabled for the call (tools, `n>1`, logprobs, `response_format`):
with no committed opening to continue — exactly the post-tool-result turn —
the publisher writes the complete answer from the conversation context. The
head prompt gained an exact-answer branch (write exactly the requested fixed
output and nothing else) and the continuation prompt an "output nothing
further when the opening already answers" clause. An empty final unit gets one
bounded re-dispatch and then surfaces as an explicit error, never a silent
empty `stop` (m11 2026-08-16 amendment). These prompt/policy edits change the
measured artifact: `./verify.sh serving-auto-max` and
`serving-auto-max-coding` must be re-run before the example's gate status is
quoted again.

Review amendment (2026-08-16, issue #495 — agent tool-turn latency): a live
per-stage trace of a minimal tool-call turn measured 25.8 s, 81% of it three
thinking-verifier passes: the tools-blind internal roles cannot emit a tool
call, so the verifier deterministically FAILed the draft and burned the full
refinement budget on every agent turn. Changes, keeping the full DAG:
(1) the `draft_synthesis`/`verifier` prompts now state that the actual tool
call is emitted by the downstream publisher — a tool-turn draft is judged on
intended action and content, never on the absence of tool-call syntax; the
continuation prompt forbids commentary text around the answer, and a
publisher that finds the committed opening complete answers exactly
`NO_CONTINUATION`, which the conductor strips on both the streaming and
whole-text seams (only the exact sentinel is dropped; any longer output
flushes verbatim) — "output nothing" alone was measured unfollowable, the
model published its deliberation instead. (2) A verdict
that declares neither PASS nor FAIL (private deliberation consumed the role's
token cap) triggers one bounded re-verification demanding an immediate
verdict instead of counting as FAIL (`conductor.py`); the verifier cap is
also raised 2048 → 4096 for the same measured reason as
`internal_max_tokens` — its thinking span nondeterministically overran 2048,
truncating the verdict and burning the re-verification plus a refinement
round per occurrence. (3) The conductor's
KV-affinity `session_id` derives from a role-aware hash of the immutable
conversation opening (system/developer messages through the first user
turn; #507), so consecutive agent turns stay on the replica holding the
warm prefix while tasks sharing a long agent-harness preamble still map to
distinct keys — the earlier fixed 2048-char prompt-head hash collapsed such
tasks onto one replica and remains only as the legacy string-caller
fallback.
(4) The non-streaming AUTO handler cancels the in-flight DAG when the HTTP
client disconnects (499), releasing admission and replica capacity.
(5) Per-stage wall time is exported as `kairyu_conductor_stage_seconds`
(model/node/operation/status) and `/v1/models` advertises `max_model_len` so
LiteLLM-class clients stop guessing the context window. (6) The example
deployment bounds `admission_wait_timeout_s` (600 s) and upstream
`timeout_s` (1800 s) instead of the previous 86400 s no-op values.
(7) A rerun with (1)–(6) still timed out: Terminus-2 demands "format your
response as JSON" in plain text (no tools, no response_format), the enabled
head's mandatory prose opening corrupted the JSON, and single replies grew
to ~10K tokens (~11 minutes) while refinement rounds fired on 0.8 of
requests. The EO-D7 head-disable intent list therefore gains a fourth
member: a plain-text demand for a machine-parsed format anywhere in the
system/developer/user side of the conversation, detected at L3 and carried
as `OrchestrationRequest.structured_format_in_prompt` (agent loops state
the demand once in the first message while every later turn's latest
message is just command output). Detection matches command forms — format
as/in, respond/reply/answer as/in/with/using, must/should be, "in X
format" over json/jsonl/yaml/xml with a trailing-negation guard (#506) —
not bare format-word mentions, which would disable the TTFT-gated head on
ordinary chat that merely discusses those formats; and the example budget
drops `max_refine_depth` 2 → 1 (`max_steps` 15 → 12), keeping every DAG
role. Post-review refinements from the same round: a doubly inconclusive
verdict retains the best draft instead of burning a refinement round
(#503), the truncated first verification and its retry are separately
traced without double-counting stage timing (#502), a publisher whose
verified draft dedups to nothing beyond the committed opening is skipped
deterministically (#505, complementing the sentinel), and the model card
advertises the minimum engine context — the truthful bound when a
mandatory smaller worker sits in every route (#504). The official
900-second Terminal-Bench budget is unchanged; final verification run
`20260816T094355Z-0624822c` completed both previously affected tasks
inside the budget with reward 1.0 each. Prompt/policy edits again
invalidate the measured artifact until `./verify.sh serving-auto-max` and
`serving-auto-max-coding` are re-run.

Review amendment (2026-08-17, issue #509 — Terminal-Bench timeout root
causes): the interim run (89 trials) produced 12 `AgentTimeoutError`s
against the 900 s Terminus budget with public-request p50 63.9 s / p90
203.8 s and zero transport errors — pure per-turn cost. Two defects are
fixed without touching any contract: (1) the verifier ran greedy
(`temperature: 0.0`) on a thinking model, a documented cause of looping
deliberation (DeepSeek reasoning guidance: 0.5–0.7); 186/357 verdicts (52%)
burned the full 4096-token cap inside `<think>` before PASS/FAIL, each
triggering the bounded re-verification — the verifier (both profiles) now
samples at `temperature: 0.6, top_p: 0.95`, keeping caller-seeded runs
reproducible via the seed pass-through. (2) Qwen role prompts placed the
role instruction before the conversation, so the growing agent conversation
could never share prefill across roles or turns (measured 37.9% prefix-hit
rate on the Qwen pool vs 70.2% on DeepSeek); every Qwen role now leads with
a byte-identical `--- REQUEST ---` framing and puts its instruction after
the conversation, letting the prefix-indexed replicas reuse the
conversation KV. Both edits invalidate the measured artifact until the
`verify.sh` gates are re-run.

The example gate: `./verify.sh serving-auto-max-coding` runs a deterministic
self-contained Python-task matrix (c1/8/16/32), requires a valid trace with a
successful head and continuation in every sample, and counts a sample as
sandbox-executed only when **both** `exec_matrix` and `exec_draft` report
`execution_status` containing `ok` — the matrix joins per-candidate statuses
with commas, and one broken candidate is a model formatting slip the DAG
absorbs by design (consensus + verifier override), so `ok,setup_error` still
counts while a matrix with no ok candidate does not. Trace `status:
"success"` alone is NOT execution evidence: a degraded (`unavailable`)
sandbox call also traces as success. The executed fraction must be ≥90% per
row (review amendment 2026-08-15: the measured c32 row exposed both a
false-positive gate reading degraded stages as executed and executor slot
contention — the executor queue allowance is now explicit in deployment
configuration. The client budgets `wall_time + queue_wait + 5s`, shares one
absolute deadline across the initial call and retry, and sends the remaining
budget to the runner. The runner caps admission wait to that residual budget
and returns 429 instead of starting a job without enough time left for its
declared wall limit). The gate also measures the paired
DeepSeek-direct row on the same dataset via the loopback L1 endpoint and
fails unless `product semantic TTFT p50 ≤ 2.0 × direct p50` (pinned
`example.json` denominators are the fallback when the paired row is
unavailable).
`./verify.sh serving-auto-max` keeps the generic workload as the executor
skip-path regression envelope.

### ECO-D5 — Coding accuracy evidence is external and harness-independent

Product coding accuracy versus frontier APIs (Fable 5 / GPT 5.6) is owned by
the external `kairyu-bench` repository (which already owns tau2): a `coding/`
benchmark of LiveCodeBench-lite (post-training-cutoff subset) plus HumanEval+
and MBPP+, run pass@1 with identical extraction and kairyu-bench's own local
execution harness (never the product sandbox, so the serving path cannot game
the benchmark) against the L3 endpoint and the frontier APIs. The declared
acceptance target is pass@1 ≥ 0.95× the best frontier baseline per suite;
results are recorded with run IDs in the example's `MEASUREMENTS.md`. The
benchmark is requested as `ytworks/kairyu-bench#10`; no kairyu-bench
implementation work is part of this change. The removed Accuracy suite is
not resurrected; `verification/` continues to gate serving behavior only.

### ECO-D6 — General ensemble profile with automatic per-request selection (issue #509)

The coding DAG's role contracts are Python-module/pytest-specialized by
design (ECO-D2). The Terminal-Bench window showed those contracts idling on
agent-loop turns: Terminus demands shell-command batches in a JSON envelope,
testgen answered NOT_APPLICABLE 311/311 (so the sandbox never grounded a
draft), proposals stayed bound to a `solution` module the task never asked
for, and reply-format drift contributed to score-0 completions. Decision:
one served model (`kairyu-auto-max`) carries a second, equally full ensemble
DAG — `OrchestratorSpec.general_roles` — selected per request by a
deterministic rule computed identically at preview/preflight and execution
time (a pure function of the call, so the prepared-request contract cannot
diverge): tools, tools-in-prompt, or a plain-text structured-format demand
(the EO-D7 head-disable signals) select the general profile; otherwise a
code-task signal in the latest user turn (code fence or code vocabulary,
scanned on the caller's words, not the L2 envelope) keeps the coding
profile; otherwise general. Constraints, per the product's purpose: the
coding profile is unchanged; both profiles are full Conductor ensembles
under the same TTFT gate; there is no single-engine route in auto-max, and
no role is skipped inside either profile. `general_roles` is mutually exclusive
with `moa_samples > 0`; configuration fails closed instead of silently ignoring
both role DAGs in MoA mode. The example's general profile uses
every deployment model — Qwen head + two Qwen proposals + a thinking
DeepSeek deep proposal, Qwen synthesis, thinking DeepSeek verification
(format deviation is a FAIL), and the direct DeepSeek publisher with the
same `prompt_headless`/`reasoning_closed`/NO_CONTINUATION contracts as the
coding continuation. The chosen profile is recorded in the result trace
(`role profile: …`) and `/v1/route` reports both role sets; launcher
readiness (`control.py`) asserts the general role list like the coding one.

## Acceptance

- CPU suite: head streaming, dedup, error contract, per-role sampling,
  executor step/skip/degrade semantics, inline re-execution, deploy
  validation, and sandbox report summarization are covered by MockBackend /
  fake-executor tests; the old seven-role contract tests are updated in the
  same change.
- Launcher `_validate_ready` requires the nine-role DAG, `stream_head: head`,
  `max_steps: 15`, the executor binding, and the executor health probe.
- `serving-auto-max` passes end-to-end with executor stages present and the
  split head/continuation public stream traced.
- `serving-auto-max-coding` passes its TTFT gate at every concurrency row
  with real sandbox execution in every sample.
- The pinned Open WebUI browser smoke still renders interleaved
  reasoning/content correctly (R1); if it misrenders, a policy flag to
  suppress reasoning after first content is the designated fallback.
