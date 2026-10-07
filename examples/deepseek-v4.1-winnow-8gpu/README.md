# Routed answers: DeepSeek-V4.1 (6 GPUs) + two Winnow-12B replicas (2 GPUs)

Winnow reads each conversation and picks one of two routes: the verified
route, three waves of DeepSeek and Winnow (VCO-D19), or the think
route, where DeepSeek answers at the effort the caller asked for.

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D18,
VCO-D19); framework mechanisms: m1 D8 (checklist verifiers, including the
2026-10-06 wait for a parallel branch), m1 D9 (the System One route judge)
and LCP-D1..D6 (llama.cpp upstream).

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). Two Winnow-12B Q8_0 replicas (each as in `winnow-12b-q8-1gpu`, chat and System One from one loaded model): `winnow-route` on GPU 6 answers only the route judge, `winnow-judge` on GPU 7 only the judgments. |
| L2 | `verified.yaml`: the Winnow route judge, the verified route and the think route. `verified-always.yaml`: the verified route only. |
| L3 | Public models `kairyu-verified` (routed) and `kairyu-verified-always`; Open WebUI on :3012; the answer page on :3013. |

## L2: how an answer is made

```text
request
  │
  ▼
profile_judge ── winnow-route (System One), 1 request: THINK or VERIFIED?
  │                (kairyu-verified-always skips the judge: VERIFIED)
  ├─ VERIFIED
  │    wave 1 ┬ drafts        DeepSeek, the caller's effort: five answers D1..D5 from
  │           │               viewpoints as different as possible (one call)
  │           └ requirements  DeepSeek, thinking at max: what the answer must meet,
  │                           necessary, sufficient, MECE (≤ 16)
  │    wave 2   judgments     winnow-judge, 1 request with the request: each draft
  │                           adoptable as the final reply? each draft x
  │                           requirement met? (5 + 5 x N probabilities)
  │    wave 3   answer        DeepSeek, the caller's effort: reads drafts, requirements and
  │                           judgments critically, writes the best reply
  └─ THINK ─────► deepseek_think_answer  DeepSeek, the caller's effort
                                          (default high), one call
```

- The judge sends the whole conversation (each message cut at 4,000
  characters, the whole at 120,000) with one choice question; the most
  probable label wins. The question and both criteria are the previous
  verified example's.
- When winnow-route does not answer within 10 s (down, overloaded,
  unreadable), the request takes the think route.
- The two Winnow replicas never share a queue: a burst of judgments cannot
  delay the next request's route decision.
- Both routes pass the caller's tools and response_format to the DeepSeek
  role that publishes (the verified route's answer).
- If winnow-judge cannot read the judgments (down, or the drafts exceed its
  65,536-token decision context), the answer is written without them.

## Run

```sh
./run.sh            # preflight, images, checkpoints, compose up, readiness probes
./verify.sh list    # GPU gates; evidence lands in model-volumes/<env>/results/
./browser-smoke.sh  # the answer page and Open WebUI in a real browser
./run.sh down
```

`run.sh` builds `winnow-server` from its pinned revision with the f072b10
llama.cpp backport (`winnow-patches/`) once for both replicas, and builds the
DeepSeek SM120 overlay when its pinned image is absent. `DEEPSEEK_MODEL_SEED`
may name a local copy of the pinned checkpoint below `/mnt/nvme`; it is
hard-linked and re-hashed against the pin before serving.

## Using it

```sh
curl -s http://127.0.0.1:8013/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "kairyu-verified",
  "reasoning_effort": "low",
  "messages": [{"role": "user", "content": "List three primary colors, comma-separated."}]
}' | jq -r '.choices[0].message.content'
```

The answer page (`http://<host>:3013`) sends every request to
`kairyu-verified-always`. Its guarantee panel is carried over unchanged and
stays empty until the guarantee is rebuilt.

## GPU gates (`verify.sh`)

| Gate | Checks |
|---|---|
| `l1` | every DeepSeek DP rank (thinking and chat JSON), chat and System One on each Winnow replica (L1), one verified answer |
| `routing` | `datasets/routing-set.json`: VERIFIED miss rate < 10 % on the calibration and held-out halves |
| `think-route` | everyday requests stream from the think route at the default effort |
| `effort` | the think route gets the caller's effort; the verified route's drafts and answer get it too (default high); the requirements always think at max; the route is judged on `winnow-route` |
| `verified-route` | a verified request runs the three waves: drafts beside requirements, one `winnow-judge` read of 5 + 5 x N items, then the answer |
| `fallback` | `winnow-route` down: think route, judgments still run; `winnow-judge` down: still routed, answer without judgments; both back: routed and judged |
| `serving` | `kairyu-verified-always` at c1/c4/c8/c16 (8/16/16/32 InFoBench requests) |
| `serving-routed` | `kairyu-verified` at c1/c4/c8/c16 on the routing set, per-route latency and tokens |
| `browser` | `browser-smoke.sh`: both UIs answer |

## Limits

- No answer carries a guarantee (`kairyu_verification` is absent): Winnow's
  judgments inform the answer, they do not gate it.
- The drafts share one call capped at 131,072 tokens (the DSL's internal
  maximum).
- Latency: measured in `MEASUREMENTS.md`.
