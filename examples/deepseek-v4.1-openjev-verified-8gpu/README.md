# Checklist-verified answers: DeepSeek-V4.1 (6 GPUs) + OpenJev (2 GPUs)

Every answer comes back with a guarantee flag. The flag is on only when the
answer covers every point of a requirement set that is mutually exclusive
and collectively exhaustive (MECE) with respect to the request: DeepSeek
lists the points (one per issue, together covering the request), OpenJev
keeps only the points the request needs, and OpenJev confirms that the
answer contains each kept point. Otherwise the best available answer is
returned with the flag off and the reason. No rule-based check judges an
answer: every judgment is a model reading (VCO-D15).

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D1,
D7-D9, D15); framework mechanisms: m1 D8 (checklist verifiers), m1 D9 (the
System One route judge) and the m11 D8 replica amendment.

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). OpenJev (DiffusionGemma 26B-A4B NVFP4) on GPU 6 and GPU 7, published image unchanged, read only through System One. |
| L2 | `verified.yaml`: route judge, point extraction (explicit and implicit), history summary, generator, adoption read, coverage read, repair (at most 2), fallback. |
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
profile_judge ── Jev, 1 request: VERIFIED or THINK? ──THINK──► deepseek_think (DeepSeek) ─► answer, no flag
  │ VERIFIED (or kairyu-verified-always)
  ▼
┌─ wave 1 (parallel, DeepSeek) ───────────────────────────────────────────────────────┐
│ extract    explicit points   {"points": [{"id": "E1", "point": ...}]}  one per issue │
│ implicit   implicit points   {"points": [{"id": "I1", "point": ...}]}  at most four  │
└──────────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ wave 2 ─────────────────────────────────────────────────────────────────────────────┐
│ history    DeepSeek, no thinking: a summary of every message except the system /     │
│            developer messages and the latest user message (earlier turns, tool       │
│            calls, tool results)                                                      │
│   └─ adopt  Jev, 1 request: is each point necessary to answer the request?           │
│             p < 0.5 → the point leaves its list (extract or implicit)                │
└──────────────────────────────────────────────────────────────────────────────────────┘
  │ adopted points
  ▼
┌─ wave 3 (DeepSeek) ──────────────────────────────────────────────────────────────────┐
│ generator  the draft: the conversation plus the adopted points, written to meet      │
│            every point (the caller's tools and response_format apply)                │
└──────────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ wave 4 ─────────────────────────────────────────────────────────────────────────────┐
│ answer     = the generator's draft, unchanged (no model call)                        │
│   └─ checklist  Jev, 1 request: does the answer fully do each adopted point?         │
│                 a missed point → repair (DeepSeek), judged again, at most twice      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

With tools in the request (an agent loop), the answer is one assistant
message whose tool calls the caller runs before asking again. The extractors
then list what this one message must do now (call a tool instead of
describing it, build on the latest tool results, move the task forward),
never the completion of the whole task. The generator and every repair carry
the caller's tools, so a tool call is returned as a structured `tool_calls`
entry.

### 2. The two Jev requests

Each stage sends all of its questions in one System One request; every
question is a `noul` (yes/no) read that returns P(yes).

```text
adopt (necessity)                              checklist (coverage)
──────────────────────────────────────────    ──────────────────────────────────────────
state:                                         state:
  request: [system/developer messages,           request: the same, verbatim
            latest user message]  (verbatim)     history: the same summary
  history: the summary from `history`            answer:  the reply exactly as it will be
                                                          sent ({text, tool_calls} when it calls a tool)
questions, one per point of both lists:        questions, one per adopted point:
  "Is this point necessary to answer the         "Does the answer fully and correctly do
   request?"                                      what this point requires?"
  yes: an answer leaving it out would not         yes: every part met as the point states
       answer the request as asked                no:  missing, partial or incorrect
  no:  the request is answered fully without it threshold: τ_hi = 0.9895 (InFoBench)
threshold: drop below 0.5 (default)
```

### 3. Outcomes

```text
checklist read
  ├─ every point p ≥ τ_hi ────────────────────────► guaranteed: true
  ├─ some point missed ──► repair (DeepSeek, the missed points) ──► checklist read again
  │                         └─ still missed after 2 repairs ───► the last non-empty answer,
  │                                                              guaranteed: false (refinement_limit)
  └─ Jev unavailable or a list unreadable ────────► the draft,
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

- `guaranteed: true`: the answer contains every adopted point.
  `requirements[]` lists each point with its `id` (`E…` explicit, `I…`
  implicit), `proposition` (the point), `tags.origin` (`explicit` or
  `implicit`), `p` and `passed`.
- `guaranteed: false` with `reason`:
  - `refinement_limit`: two repairs still missed a point; the last
    non-empty answer is returned with the points it misses.
  - `judge_unavailable`: neither OpenJev replica answered (down or
    overloaded); the generator's draft is returned as-is.
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
- With tools, each assistant message is judged on its own points; whether
  the whole agent run solves the task is not part of the flag.
- Latency: measured in `MEASUREMENTS.md`.
