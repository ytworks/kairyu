# Verified tool route: tool-call routing and prompts tuned on DeepSWE (2026-10-09)

Status: proposed, awaiting owner approval. Branch: PR #641 (`claude/winnow-verified-three-wave`, open).

## Context

Owner decision 2026-10-09 after DeepSWE `deepswe-verified-3wave-nextstep-4w-20261007-r1`
(stopped at 75/113 scored, 35 official passes = 46.7%):

1. Rename the verified route to the **verified tool route**.
2. Route simply by whether the next reply needs a tool call.
3. Keep the DSL structure (drafts and requirements, judgments, answer; same
   workers and efforts) and raise accuracy by tuning the prompts on the
   benchmark's failures. The route serves tool calls only, so the prompts are
   generic for any tool-calling agent (no DeepSWE, git, test-runner or language
   specifics).

## Evidence (failure analysis of the 75 scored tasks)

| Cause | Tasks | Example |
|---|---|---|
| Public surface differs from what the request names or implies ("same as X") | ~9 | cliffy 0/37: errors exported from a sub-module, test imports the root `mod.ts`; obsidian 0/41: rule named "Auto TOC" vs "Auto Table of Contents" |
| Own checks encode the agent's reading, and code is bent to the check | ~8 | fd: changed correct code to match its own wrong test; awilix: test initialized the root first though the request says "independently" |
| Verified at an internal layer, not the path/build/type-check the tests use | 5 | kysely 0/254: runtime scripts only, never type-checked; kgateway: helpers unit-tested, end-to-end golden tests never run |
| Finishing action batched with steps whose result was never seen | 2 | arktype 0/25: `git commit` failed in the same turn as the submit, empty patch |
| Missing must-not behaviour (shows as broken existing tests) | 3 | helm: false warning on valid input; bandit: metric changed |
| Reply announces a tool call without making it | 9 replies | helm-unified: "issue the close-out command" with no call, then 3-hour timeout |
| Infrastructure (verifier crash) | 1 | eicrud: mongod aborted |

Running tests does not separate pass from fail (32/36 vs 33/39 ran existing
tests); what is checked does. 74/75 submitted on their own when their own
checks were green.

## Changes (example only; no `kairyu/` change)

### Routing (`verified.yaml` profile_judge)
- Winnow route question: does the next reply need to call one of the caller's
  tools? Choices `TOOL` (verified tool route, profile `primary`) and `THINK`
  (`deepseek_think`). The judge state already carries `tool_calling` (whether
  tools are declared) and the conversation. Fallback stays `deepseek_think`.

### Rename
- Route/label/comments/docs: "verified route" -> "verified tool route",
  label `VERIFIED` -> `TOOL`; gate `verified-route` -> `verified-tool-route`.
- Public model names unchanged (`kairyu-verified`, `kairyu-verified-always`).
- The answer page sends to the routed model `kairyu-verified` (chat without
  tools takes the think route); `kairyu-verified-always` stays for gates.

### Prompts (same roles, workers, efforts, schemas)
- **Shared reply definition**: the reply is the next assistant message, made of
  the tool call(s) for the one move needed now plus short text.
- **requirements**: what this next move must meet, drawn from the request's
  literal words, covering only what applies at the current position:
  1. every identifier, path, export, entry point, signature, message and
     "same as X" clause named in the request, honoured exactly and reachable
     from the public entry point a user would call;
  2. checks derived from the request's wording; when a check and the request
     disagree, the request wins (never bend a stated requirement to a check);
  3. verification through the same public path, build mode and type checks
     that existing tests of similar features use; "all X" covers more than one X;
  4. must-not behaviour: rejections, no new warnings on valid input, existing
     forms and tests unchanged;
  5. finishing only after earlier tool results show every delivery step
     succeeded, with the finishing action alone in its turn.
- **drafts**: five candidate moves, each with its tool_calls; at least one
  draft considers verifying against the request before any finishing move.
- **judgments**: adoptable = the move needed now, its calls fit the tools'
  schemas and the conversation's protocol, and its text matches its calls.
- **answer**: the reply must contain the tool call it announces; never end on a
  description; the finishing action goes alone and only after success is shown.

## Verification plan

- CPU: ruff; example tests (route label, routing state, structured calls).
- Data: new routing set for the tool decision (saved DeepSWE agent turns =
  TOOL; the current 80 chat conversations without tools = THINK; tool-declared
  turns whose next reply is a final text answer = THINK).
- Replay (before gates): saved DeepSWE turns including the failure turns
  (arktype batched commit+submit, helm-unified announce-without-call, fd
  check-vs-request) through `kairyu-verified-always`; read the requirements with
  stage reports on locally (as on 2026-10-07).
- GPU gates: every gate in `verification.GATES`, adapted to tool calls:
  l1, routing (tool decision set), think-route, effort, verified-tool-route
  (saved agent turns: structured calls, next-step requirements), fallback,
  serving and serving-routed (tool-calling turns at c1/c4/c8/c16), browser.
- DeepSWE re-run only on the owner's instruction.
