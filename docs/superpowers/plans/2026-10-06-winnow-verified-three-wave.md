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

## Follow-up

Multi-turn context fix: `2026-10-07-verified-requirements-context.md`.
