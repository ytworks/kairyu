# Verified route as three waves (DeepSeek drafts, Qwen points, Winnow judgments)

Owner request, 2026-10-06. Changes the VERIFIED route of
`examples/deepseek-v4.1-qwen3.8-winnow-8gpu` (VCO-D18: one max-effort DeepSeek
call) into a three-wave DAG. Routing (Winnow THINK/VERIFIED), the THINK route,
L1, UIs and public models are unchanged.

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

## L2 DAG (`verified.yaml` primary profile and `verified-always.yaml`)

| Role | Worker | Wave | Output | Effort / cap |
|---|---|---|---|---|
| `drafts` | DeepSeek | 1 | JSON `{drafts: [{id: D1..D5, viewpoint, answer}]}`, exactly 5, one call so each draft sees the others' viewpoints | max / 262,144 |
| `requirements` | Qwen | 1 | JSON `{points: [{id: R1.., point}]}`, 1-32 points, MECE, necessary and sufficient, never the answer itself | thinking / 65,536 |
| `judgments` | Winnow | 2 | checklist verifier, one System One request: 5 `adoptable` questions (foreach draft; the draft in the question context) + 5 x N `meets` questions (one static question per draft id, foreach requirement; drafts in the state) | 4 denoise steps as before |
| `answer` | DeepSeek | 3 | final unit: request + drafts + requirements + judgments; critical comparison, then one best reply in the caller's format, with the caller's tools | max / 262,144 |

- Winnow never repairs (`max_refinements: 0`); an unavailable or oversized
  judgment (`max_state_chars` within Winnow's 65,536-token decision context)
  sends the answer the drafts and requirements without judgments
  (`on_unavailable: publish_unverified`), so the request still completes.
- The judgments reach wave 3 as text: every item with its probability.
- No `kairyu_verification` guarantee is claimed (judgments are inputs, not a
  gate on the published answer); the answer page's panel stays as in VCO-D18.
- Budget: `max_steps` covers drafts, requirements, two Winnow reads, answer.

## Framework (`kairyu/`) — needs owner authorization

F1. Today a verifier runs inline right after its target and may depend only on
its target's own dependencies (`conductor.py` validation, "not available when it
runs inline"). Judging `drafts` against `requirements` therefore forces one of
them to wait for the other: wave 1 cannot run in parallel. Smallest shared
change: a checklist verifier may also read roles that run in parallel with its
target; the target's unit waits for them after generating and before the
verdict. Reusable contract: judging one branch against criteria produced by an
independent branch (also the old checklist DAG's shape). Test: a verifier over
two parallel branches starts both at once and judges after both finish.
Without F1: example-only, `requirements` runs after `drafts` (adds Qwen's time
to every verified request).

F2 (not proposed): a verifier's text output lists only failing items. The
example sets the judgment threshold to 1.0 so every item below certainty is
listed with its probability; no framework change.

## Gates (walk `verification.py` GATES; all re-run)

| Gate | Change |
|---|---|
| l1 | unchanged |
| routing, think-route, fallback | unchanged (routing untouched) |
| effort | VERIFIED efforts become drafts max + answer max |
| verified-route | a verified request runs drafts (5 drafts), requirements (Qwen), one Winnow judgment request with 5 + 5 x N items, and the answer at max; 200 and non-empty at each caller effort |
| serving, serving-routed | same c1/c4/c8/c16 plan; latency, tokens, tok/s reported |
| browser | unchanged |

## Files

- `examples/deepseek-v4.1-qwen3.8-winnow-8gpu/`: `verified.yaml`,
  `verified-always.yaml`, `verification.py` (effort, verified-route), README,
  MEASUREMENTS (after gates).
- `tests/unit/test_deepseek_v41_qwen38_winnow_example.py`: the routed DAG runs
  the three waves and the answer receives drafts, requirements and judgments
  (replaces the one-call assertion).
- F1 (if authorized): `kairyu/orchestration/conductor.py` + one conductor test.
- Design: VCO-D19 in `docs/design/example-verified-checklist-orchestration.md`;
  `PROGRESS.md`.
- Local checks: `uv run ruff check .`, changed-path tests with
  `CUDA_VISIBLE_DEVICES=`; CI runs the full suite.
