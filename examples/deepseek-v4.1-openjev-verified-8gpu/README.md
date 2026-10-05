# Checklist-verified answers: DeepSeek-V4.1 (6 GPUs) + OpenJev (2 GPUs)

Every answer comes back with a guarantee flag. DeepSeek lists the points the
request asks for (explicit) and presupposes (implicit), one per issue,
together covering the request; OpenJev drops the implicit points the reply
does not need. After the answer is written, OpenJev checks each point against
it, then reads those results, the prompt and the answer and decides whether
the answer can be adopted as the reply the user expects. The flag is that
adoption decision: on when its p reaches the threshold (0.99), even if a
point check failed; a failed point check sends the point to a repair.
Otherwise the best available answer is returned with the flag off and the
reason. No rule-based check judges an answer: every judgment is a model
reading (VCO-D15).

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D1,
D7-D9, D15); framework mechanisms: m1 D8 (checklist verifiers), m1 D9 (the
System One route judge) and the m11 D8 replica amendment.

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). OpenJev (DiffusionGemma 26B-A4B NVFP4) on GPU 6 and GPU 7, published image unchanged, read only through System One. |
| L2 | `verified.yaml`: route judge, point extraction (explicit and implicit), history summary, adoption read, answer, coverage read, acceptance read, repair (at most 2), fallback. |
| L3 | Public models `kairyu-verified` (routed) and `kairyu-verified-always`; `kairyu_verification` on every verified answer; the answer page on :3013. |

## L2: how an answer is made

### 1. The DAG

Roles run in waves; roles in one wave run in parallel. DeepSeek writes,
OpenJev only returns probabilities through System One, and Kairyu L2 runs
every transition, the list edits and the threshold rule.

```text
request
  │
  ▼
profile_judge ── Jev, 1 request: THINK, VERIFIED_TOOL or VERIFIED? (kairyu-verified-always: VERIFIED_TOOL or VERIFIED)
  │   THINK ──► deepseek_think (DeepSeek) ─► answer, no flag
  │   VERIFIED_TOOL  ──► verified_tool (DeepSeek, caller's effort and tools) ─► Jev, 1 request:
  │                      four angles on the planned tool calls ─► miss → repair, at most twice
  │ VERIFIED
  ▼
┌─ wave 1 (DeepSeek, in parallel) ───────────────────────────────────────────────────┐
│ extract    explicit points   {"points": [{"id": "E1", "point": ...}]}  one per issue │
│ history    caller's effort: a summary of every message except the system /         │
│            developer messages and the latest user message (earlier turns)          │
└────────────────────────────────────────────────────────────────────────────────────┘
  │ explicit points, history
  ▼
┌─ wave 2 ───────────────────────────────────────────────────────────────────────────┐
│ implicit   implicit points   {"points": [{"id": "I1", "point": ...}]}  at most four │
│            reads the explicit points and lists none of what they already require   │
│            (the two lists stay mutually exclusive)                                 │
│   └─ adopt  Jev, 1 request: is each implicit point necessary now?                  │
│             (no implicit point → nothing to judge, no read)                        │
│             (implicit points only; explicit points always stay)                    │
│             p < 0.5 → the implicit point leaves its list                           │
└────────────────────────────────────────────────────────────────────────────────────┘
  │ adopted points
  ▼
┌─ wave 3 ───────────────────────────────────────────────────────────────────────────┐
│ answer     DeepSeek: the conversation plus the adopted points, written to meet     │
│            every point (the caller's tools and response_format apply)              │
│   └─ checklist  Jev, request 1: does the answer fully do each adopted point?       │
│                 Jev, request 2: given those results, the original prompt and the   │
│                 answer, can it be adopted as the reply the user expects?           │
│                 not adopted with a missed point → repair (DeepSeek), judged again, │
│                 at most twice; not adopted with every point met → no repair,       │
│                 published unverified                                               │
└────────────────────────────────────────────────────────────────────────────────────┘
```

A request that requires a tool call (an agent loop's turn, or a request to
act with the offered tools) goes to the verified-tool route (owner decision,
VCO-D18). DeepSeek writes the reply at the caller's effort with the caller's
tools. Jev then reads its planned tool calls (the calls in the reply and any
later steps its text states) in one request, from four angles:

1. `first_call`: given what is known and the tool results, is the first call
   an appropriate next move toward the final goal?
2. `order`: are the planned calls in an appropriate order?
3. `progress`: do they bring the work closer to the final goal?
4. `runs`: will they run without error and do what they are meant to?

When an angle fails, DeepSeek gets the draft and Jev's result and writes the
reply again. Jev reads the new reply, at most two repairs. The reply carries
`kairyu_verification` with the four angles.

A repair is the same reply written again in the draft's frame: the
conversation and every adopted point, then the draft and its missed points,
changing only what those points require.

### 2. The Jev requests

Each stage sends all of its questions in one System One request; every
question is a `noul` (yes/no) read that returns P(yes).

```text
adopt (necessity)                              checklist (coverage)
──────────────────────────────────────────    ──────────────────────────────────────────
state:                                         state:
  request: [system/developer messages,           request: the same, verbatim
            latest user message]  (verbatim)     history: the same summary
  history: the summary from `history`            answer:  the reply
questions, one per implicit point:             questions, one per adopted point:
  "Is this point necessary to answer the         "Does the answer fully and correctly do
   request?"                                      what this point requires?"
  yes: a reply missing it is not the reply        yes: every part met as the point states
       this conversation needs at this step       no:  missing, partial or incorrect
  no:  the reply needed now is complete and     threshold: τ_hi = 0.99894 (InFoBench)
       correct without it
threshold: drop below 0.5 (default)

acceptance (after coverage, same verdict)
──────────────────────────────────────────
state:
  prompt:    the request, verbatim
  answer:    the reply exactly as it will be sent
  checklist: [{id, point, p, passed}] from the coverage read
question: "Reading the prompt, the candidate answer and how well each point
           is met, can this answer be adopted as the reply the user expects?"
  yes: it can be adopted as is as the reply the user expects
  no:  it cannot be adopted as the reply the user expects
threshold: τ_accept = 0.99 (owner choice; InFoBench held-out: 7 of 46
           accepted answers miss a labelled requirement, 15.2 %, upper
           bound 26.7 %; no threshold meets α = 0.10, see MEASUREMENTS.md)
Jev reads (adopt, coverage, acceptance) use four denoise passes (steps: 4).
```

The coverage read's τ_hi only selects the missed points that the repair
must meet; the acceptance read decides the guarantee.

### 3. Outcomes

```text
checklist read
  ├─ acceptance p ≥ τ_accept ──────────────────────► guaranteed: true
  ├─ not accepted ──► repair (DeepSeek: original prompt, answer, ──► checklist read again
  │                   missed points; "rewrite to meet them")
  │                         └─ not accepted after 2 repairs ───► the last non-empty answer,
  │                                                              guaranteed: false (refinement_limit)
  └─ Jev unavailable, a list unreadable or ──────► the draft,
     no point left to judge
                                                    guaranteed: false (judge_unavailable /
                                                    checklist_unavailable)
```

## Run

```sh
./run.sh            # preflight, images, checkpoints, compose up, readiness probes
./verify.sh list    # GPU gates; evidence lands in model-volumes/<env>/results/
./browser-smoke.sh  # the answer page in a real browser
./run.sh down
```

`DEEPSEEK_MODEL_SEED` / `OPENJEV_MODEL_SEED` may name local copies of the
pinned checkpoints below `/mnt/nvme`; they are hard-linked and re-hashed
against the pins before serving. The DeepSeek image is the six-GPU example's
SM120 overlay (same Dockerfile and patch, copied here; pinned image ID).

## Using it

```sh
curl -s http://127.0.0.1:8013/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "kairyu-verified",
  "messages": [{"role": "user", "content": "List three primary colors, comma-separated."}]
}' | jq '{answer: .choices[0].message.content, verification: .kairyu_verification}'
```

`kairyu_verification`:

- `guaranteed: true`: Jev accepted the answer as the reply (`acceptance`,
  P(yes) of the acceptance read) after reading every adopted point's result.
  `requirements[]` lists each point with its `id` (`E…` explicit, `I…`
  implicit), `proposition` (the point), `tags.origin` (`explicit` or
  `implicit`), `p` and `passed`.
- `guaranteed: false` with `reason`:
  - `refinement_limit`: two repairs were still not accepted; the last
    non-empty answer is returned with its point results.
  - `not_accepted`: the answer met every point but was not accepted; with
    nothing to repair, it is returned with its point results.
  - `judge_unavailable`: neither OpenJev replica answered (down or
    overloaded); the first answer is returned as-is.
  - `checklist_unavailable`: a point list could not be read or the read
    exceeded OpenJev's input window; the draft is returned.
  - `budget`: the step budget ran out; the draft is returned.

The answer page (`http://<host>:3013`) shows the badge, the reason and the
point table; internal stages are folded below the answer.

## Limits

- A verified answer is not streamed before its checklist finishes.
- Only τ_hi is calibrated (InFoBench, α = 0.10); the 0.5 necessity cut of
  the adoption read is a default.
- The guarantee covers what the request needs (the adopted points), not the
  truth of every claim in the answer.
- An extractor cut off before its JSON closes leaves its list unreadable;
  the answer is then returned unverified (`checklist_unavailable`).
- The four tool angles' thresholds are placeholders (0.5) until they are
  calibrated on recorded DeepSWE turns (VCO-D18). Jev reads the tool calls
  as `<tool_call>` markup in the reply text.
- Latency: measured in `MEASUREMENTS.md`.
