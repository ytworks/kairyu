# Verified tool route: tool-call routing and prompts tuned on DeepSWE (2026-10-09)

Status: approved 2026-10-09, implemented (VCO-D20). Branch: PR #641 (`claude/winnow-verified-three-wave`, reused).

## Context

Owner decisions 2026-10-09 after DeepSWE `deepswe-verified-3wave-nextstep-4w-20261007-r1`
(stopped at 75/113 scored, 35 official passes = 46.7%):

1. The verified route is replaced by the **verified tool route**; the old
   verified route (for any request) is removed. The normal route
   (`deepseek_think`) stays.
2. Winnow routes simply by whether the next reply needs a tool call.
3. One public model, `kairyu-verified-tool` (routed); `kairyu-verified` and
   `kairyu-verified-always` are removed.
4. The answer page (playground) is removed. Open WebUI stays.
5. The DSL structure stays (drafts beside requirements, judgments, answer;
   same workers and efforts). Accuracy rises by tuning the prompts on the
   benchmark's failures, kept generic for any tool-calling agent (no DeepSWE,
   git, test-runner or language specifics).
6. Work continues on PR #641.

## Evidence (failure analysis of the 75 scored tasks)

| Cause | Tasks | Example |
|---|---|---|
| Public surface differs from what the request names or implies ("same as X") | ~9 | cliffy 0/37: errors exported from a sub-module, test imports the root `mod.ts`; obsidian 0/41: rule named "Auto TOC" vs "Auto Table of Contents" |
| Own checks encode the agent's reading, and code is bent to the check | ~8 | fd: changed correct code to match its own wrong test; awilix: test initialized the root first though the request says "independently" |
| Verified at an internal layer, not the path/build/type-check the tests use | 5 | kysely 0/254: runtime scripts only, never type-checked; kgateway: helpers unit-tested, end-to-end golden tests never run |
| Finishing action batched with steps whose result was never seen | 2 | arktype 0/25: commit failed in the same turn as the submit, empty patch |
| Missing must-not behaviour (shows as broken existing tests) | 3 | helm: false warning on valid input; bandit: metric changed |
| Reply announces a tool call without making it | 9 replies | helm-unified: "issue the close-out command" with no call, then the 3-hour timeout |
| Infrastructure (verifier crash) | 1 | eicrud: mongod aborted |

Running tests does not separate pass from fail (32/36 vs 33/39 ran existing
tests); what is checked does. 74/75 submitted on their own once their own
checks were green.

## Changes (example only; no `kairyu/` change)

### Spec and models
- `verified.yaml` -> `verified-tool.yaml`; `verified-always.yaml` removed.
- `kairyu.yaml`: one orchestrator and public model `kairyu-verified-tool`.
- Route judge (winnow-route): "Does the next reply need to call one of the
  caller's tools?" Choices `TOOL` -> profile `primary` (verified tool route),
  `THINK` -> `deepseek_think`. The judge state already carries `tool_calling`
  (tools declared) and the conversation. Fallback stays `deepseek_think`.
- Labels, comments, docs: "verified route" -> "verified tool route".

### Prompts (same roles, workers, efforts, schemas)
- **Shared reply definition**: the reply is the next assistant message: the
  tool call(s) for the one move needed now, plus short text.
- **requirements**: what this next move must meet, from the request's literal
  words, only what applies at the current position:
  1. every identifier, path, export, entry point, signature, message and
     "same as X" clause in the request, honoured exactly and reachable from
     the public entry point a user would call;
  2. checks derived from the request's wording; when a check and the request
     disagree, the request wins (never bend a stated requirement to a check);
  3. verification through the same public path, build mode and type checks
     that existing tests of similar features use; "all X" covers more than one X;
  4. must-not behaviour: rejections, no new warnings on valid input, existing
     forms and tests unchanged;
  5. finishing only after earlier tool results show every delivery step
     succeeded, with the finishing action alone in its turn.
- **drafts**: five candidate moves, each with its tool_calls; at least one
  weighs verifying against the request before any finishing move.
- **judgments**: adoptable = the move needed now, calls fit the tools'
  schemas and the conversation's protocol, text matches its calls.
- **answer**: the reply contains the tool call it announces, never ends on a
  description; the finishing action goes alone, only after success is shown.

### Removed with the answer page
`playground/`, `playground-smoke.mjs`, the compose `playground` service and
its port, `example.json` playground entry, README section; `browser-smoke.sh`
keeps only Open WebUI.

## Verification plan

- CPU: ruff; example tests (one model, TOOL/THINK routing state, structured
  calls, no stage reports); frontier examples test.
- Datasets (committed, authored, generic tools such as shell, file read/write,
  HTTP fetch, calendar, search):
  - `tool-routing-set.json`: TOOL = mid-task agent turns whose next reply
    must call a tool; THINK = chats without tools (current 80 conversations)
    and tool-declared turns whose next reply is a final text answer.
  - `tool-turns.json`: agent turns for the verified tool route and serving.
- Replay before gates (host-local, not committed): saved DeepSWE turns
  including the failure turns (arktype batched commit+submit, helm-unified
  announce-without-call, fd check-vs-request) through `kairyu-verified-tool`;
  read requirements with stage reports on locally (as on 2026-10-07).
- GPU gates: every gate in `verification.GATES`, adapted:
  l1 (L1 probes and one tool-call answer), routing (tool-routing set, miss
  rate on TOOL < 10 % and THINK precision), think-route, effort,
  verified-tool-route (structured calls, three waves, judgments on
  winnow-judge), fallback (route judge down: THINK; judge down: answer without
  judgments), serving (tool turns at c1/c4/c8/c16), serving-routed (mixed
  set), browser (Open WebUI only).
- DeepSWE re-run only on the owner's instruction.
- Records: design amendment (VCO-D20), PROGRESS entry, MEASUREMENTS.
