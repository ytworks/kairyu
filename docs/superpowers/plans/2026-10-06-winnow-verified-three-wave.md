# Verified route as three waves (DeepSeek drafts, Qwen requirements, Winnow judgments)

Owner request, 2026-10-06; revised the same day with the owner's additions
(Qwen at low effort, DeepSeek at the caller's effort, a verification plan).
Changes the VERIFIED route of `examples/deepseek-v4.1-qwen3.8-winnow-8gpu`
(VCO-D18: one max-effort DeepSeek call). Routing (Winnow THINK/VERIFIED), the
THINK route, L1 models and GPUs, UIs and public models are unchanged.

## Goal (owner's design)

- Wave 1, in parallel:
  - DeepSeek-V4.1 writes five answer drafts to the request, from viewpoints
    as different from one another as possible.
  - Qwen3.8 lists the requirements an answer to the request must meet,
    necessary and sufficient, MECE.
- Wave 2: Winnow (Jev family, System One)
  - judges, per draft, whether it can be adopted as the final answer, with the
    request in the state;
  - judges, per requirement x draft, whether the draft meets the requirement.
- Wave 3: DeepSeek reads the request, the five drafts and Winnow's judgments,
  thinks critically, and writes the best answer (the published reply).

## Owner decisions (2026-10-06)

| Decision | Choice |
|---|---|
| Wave 1 parallel | yes, through framework change F1 |
| Five drafts | one DeepSeek call writes all five |
| Qwen effort | always low (owner corrected medium to low) |
| DeepSeek effort | the caller's effort (default high when none is sent), drafts and answer |
| GPU gates | all gates after the push, per the verification plan below |

## L2 DAG (`verified.yaml` primary profile = `verified-always.yaml`) — example

| Role | Worker | Wave | Output | Effort / cap |
|---|---|---|---|---|
| `drafts` | DeepSeek | 1 | JSON `{D1..D5: {viewpoint, answer}}` (fixed keys so Winnow's questions name each draft), one call | caller (`inherit`) / 131,072 (the DSL's internal maximum) |
| `requirements` | Qwen | 1 | JSON `{points: [{id: R1.., point}]}`, 1-16 points, MECE, necessary and sufficient, never the answer itself | low (thinking) / 65,536 |
| `judgments` | Winnow | 2 | checklist verifier, one System One request: 5 `adoptable` questions + 5 x N `meets` questions; state = the request and the drafts | Winnow default |
| `answer` | DeepSeek | 3 | final unit: request + drafts + requirements + judgments; critical comparison, then one best reply in the caller's format, with the caller's tools | caller (`inherit`) / 262,144 |

- Winnow never repairs (`max_refinements: 0`). Threshold 1.0 lists every
  judgment below certainty, with its probability, in the text the answer reads.
- Winnow unavailable or its 65,536-token decision context exceeded: the answer
  is written from the drafts and requirements without judgments
  (`on_unavailable: publish_unverified`); the request still completes.
- No `kairyu_verification` guarantee: the judgments inform the answer, they
  do not gate it. The answer page's panel stays as in VCO-D18.
- Budget: `max_steps: 6` (drafts, requirements, one Winnow read, answer,
  headroom); `max_refine_depth: 0`.

## Qwen low effort — example

The `requirements` role declares `reasoning_effort: low`. The example's Qwen
template already turns any explicit effort into thinking at low, so neither
the template nor the Qwen container changes.

## Framework (`kairyu/`) — F1, authorized by the owner

A verifier runs inline right after its target and could depend only on its
target's own dependencies (`conductor.py` validation), so judging `drafts`
against `requirements` forced one to wait for the other. Change: a verifier
may read a unit running beside its target; the target's verdict waits for it
(per-run settled events set when a unit has run, failed or been excluded).
Validation keeps it deadlock-free under the wave scheduler: the waited unit's
own dependencies must complete before the target generates, and it is not
the final unit. Documented as the m1 D8 amendment of 2026-10-06. Tests: one
new conductor test (both branches start together, the verdict reads both);
the old rejection test is rewritten to the surviving contract (a wait that
could never end is rejected).

## Files

- Example: `verified.yaml`, `verified-always.yaml`, `kairyu.yaml` (comments), `verification.py` (effort, verified-route, stage
  report), README, MEASUREMENTS (after the gates).
- Framework: `kairyu/orchestration/conductor.py`.
- Tests: `tests/unit/test_conductor_checklist.py` (+1),
  `tests/unit/test_conductor.py` (1 rewritten),
  `tests/unit/test_deepseek_v41_qwen38_winnow_example.py` (the VERIFIED test
  checks the three waves, efforts and what the answer reads).
- Docs: VCO-D19 in `docs/design/example-verified-checklist-orchestration.md`,
  m1 D8 amendment, `PROGRESS.md`, `examples/README.md`.
- Local checks before push: `uv run ruff check .`, changed-path tests with
  `CUDA_VISIBLE_DEVICES=`; CI runs the full suite.

## Verification plan (GPU)

Deploy: plain `./run.sh` (rebuilds the Kairyu image, recreates the gateway
with the new configs, waits for health, runs its L1 probes). Then the gates in
`verification.py` GATES order; stop and report on the first failure. Every
gate writes per-request evidence (latency, TTFT, tokens, tok/s, route,
efforts, stage times) to `model-volumes/<env>/results/`.

| # | Gate | Claim protected | Pass criteria | Budget |
|---|---|---|---|---|
| 1 | l1 | every L1 serves; one verified answer end to end | every DeepSeek DP rank (thinking, chat JSON), Qwen chat, Winnow chat and System One answer; the verified probe answers "Paris" | 30 min |
| 2 | routing | Winnow's route choice is unchanged | VERIFIED miss rate < 10 % on the calibration and held-out halves | 30 min |
| 3 | think-route | everyday requests stay on THINK | routed to `deepseek_think`, streamed, default effort high | 30 min |
| 4 | effort | the efforts the owner set | THINK: DeepSeek at the caller's effort (none→high, low, high, max). VERIFIED: drafts and answer at the caller's effort (none→high); Qwen at low every time | 90 min |
| 5 | verified-route | the three waves run as designed | 6 VERIFIED conversations x unary/streamed, caller effort cycling none/low/high/max: 200, non-empty answer; drafts, requirements, judgments, answer all succeed; requirements starts before drafts ends (wave 1 parallel); one Winnow read with 5 + 5 x N items (N ≥ 1); efforts as in gate 4 | 120 min |
| 6 | fallback | Winnow down does not break requests | Winnow stopped: 200 on THINK; Winnow back: routed again | 60 min |
| 7 | serving | `kairyu-verified-always` under load | c1/c4/c8/c16 with 8/16/16/32 InFoBench requests: every request 200; report p50/p95 latency, TTFT, tokens, tok/s, per-stage times, judgment success rate | 6 h |
| 8 | serving-routed | `kairyu-verified` under load | same plan on the routing set: every request 200; route mix, per-route latency, tokens, tok/s, judge time | 6 h |
| 9 | browser | both UIs answer | the answer page and Open WebUI answer for both models | 15 min |

Reported, not gated (no latency threshold, as in VCO-D18): verified latency,
how often Winnow could not read the judgments, draft and answer token counts.
Expected cost: a verified request now carries five drafts plus the answer, so
its latency will be several times VCO-D18's (p50 113 s at max); the serving
gates are the long pole (estimated 8-12 h for all nine gates).

Progress reports during the run use the full format: gate table with times,
per-route counts (judge success / fallback), stage times, tokens, tok/s,
TTFT/E2E, live worker load; times in JST.

## Redo plan after DeepSWE r1 (2026-10-07 JST) — awaiting owner decision

### Causes (confirmed by replaying recorded r1 turns through the example's
### own config; Qwen requests sent to the live Qwen)

1. **The conversation carries Kairyu's own stage report back.** With
   `expose_intermediate_outputs: true` every answer's `reasoning_content` is
   the stage report (five drafts, Qwen's reasoning, judgments). mini-swe-agent
   returns it in the next request's history, and the L2 conversation keeps
   assistant `reasoning_content` (m11 assistant-history amendment 2026-08-14),
   so every role re-reads every earlier report: 75-88 % of the conversation
   characters (14:31 turn: 389,264 of 536,279; 14:42: 776,783 of 887,879).
   Responses AUTO already drops replayed reasoning for this reason (m11 D4:
   "stage output would grow every L2 prompt"); Chat Completions does not.
   Replay without it: Qwen's prompt 148,455 → 51,387, 180,616 → 55,431,
   ≥ 196,609 (rejected) → 46,725 tokens, and Qwen returns a valid list (11-13
   points) on all three.
2. **Over-long turns are rejected by Qwen.** Live replay of the 14:42 turn:
   vLLM 400 "maximum context length is 262144 tokens … 65536 output tokens and
   your prompt contains at least 196609 input tokens" (Qwen's access log is
   off, so the 400 was invisible in its log).
3. **Two empty lists (14:31, 14:38).** Qwen itself ended generation after
   188 / 342 tokens in its reasoning (vLLM returned those usage counts with no
   list). The same inputs replayed four times all returned valid lists, so it
   is not deterministic; it happened only on the report-polluted inputs.
4. **Even without the report, long DeepSWE conversations exceed Qwen.** The
   09/12 DeepSeek-only run reached 714 messages / 925,539 content characters
   (p90 437 messages); at the measured ~2.5 JSON characters per Qwen token
   that is ~370K tokens.

### Fix plan (2026-10-07, revision of the F2 + F3 plan) — awaiting approval

Causes (confirmed by live replay of r1 turns):

1. The L2 conversation re-reads every earlier turn's `reasoning_content`
   (Kairyu's stage report echoed by the client): 189,953 of 240,650 Qwen
   tokens on the 14:42 turn; 75-88 % of the characters.
2. Qwen (262,144) rejects the `requirements` request: on all 67 failed turns
   its input was 199,082-325,598 tokens + the 65,536 output cap (vLLM 400).
3. Even without (1), a long agent conversation's own content can exceed Qwen
   (09/12 DeepSeek-only run: up to 714 messages / 925,539 characters).

Unchanged: Kairyu emits each turn's stage report in `reasoning_content`;
DeepSeek and Winnow roles, prompts, caps and the DAG.

Changes:

| # | Layer | Owner | What | Solves |
|---|---|---|---|---|
| F2 | L3→L2 | framework | orchestration Chat Completions renders the L2 conversation without assistant `reasoning_content` (still accepted on the wire; direct engines keep it) — the rule Responses AUTO already follows (m11 D4) | 1, 2 |
| F3 | L2 | framework | role option `max_conversation_chars`: `{conversation}` rendered with the existing `bounded_conversation` (first messages incl. the task, then the newest that fit, omitted count shown), as the route judge and checklist state already do | 3 |
| E | L2 | example | `requirements` sets `max_conversation_chars: 400000` (~160K Qwen tokens at the worst measured 2.5 chars/token; + 65,536 < 262,144) | 3 |

Measured with F2 (live `/tokenize` of the exact rendered request): the 67
failed turns become 30,330-96,837 tokens; all fit, so F3 never truncates them
(it only acts above 400,000 characters).

Tests: one chat-input test (F2: assistant `reasoning_content` absent from the
L2 prompt, content and tool calls kept); one conductor test (F3: the bounded
role reads the task and the newest messages with the omitted count).

Docs: m11 assistant-history amendment (F2), m1 (F3), VCO-D19 note, PROGRESS.

Verification:

1. CPU: ruff; changed-path tests.
2. Live before redeploying: the 67 failed turns' requirements requests
   rendered by the new code sent to Qwen — all accepted, valid lists; one
   synthetic > 400,000-character conversation (the 09/12 longest) — accepted,
   task kept, valid list.
3. Redeploy (`./run.sh`); all nine gates and `long-conversation` (five r1
   turns whose old Qwen input exceeded 262,144, through
   `kairyu-verified-always`; pass = requirements and judgments succeed).

