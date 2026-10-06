# Routed answers: DeepSeek-V4.1 (6 GPUs) + Qwen3.8-27B (1 GPU) + Winnow-12B (1 GPU)

Winnow reads each conversation and picks one of two routes: the verified
route, three waves of DeepSeek, Qwen and Winnow (VCO-D19), or the think
route, where DeepSeek answers at the effort the caller asked for.

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D18,
VCO-D19); framework mechanisms: m1 D8 (checklist verifiers, including the
2026-10-06 wait for a parallel branch), m1 D9 (the System One route judge)
and LCP-D1..D6 (llama.cpp upstream).

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). Qwen3.8-27B FP8 on GPU 6 (as in `qwen3.8-27b-1gpu`). Winnow-12B Q8_0 on GPU 7 (as in `winnow-12b-q8-1gpu`): chat and System One from one loaded model. |
| L2 | `verified.yaml`: the Winnow route judge, the verified route and the think route. `verified-always.yaml`: the verified route only. |
| L3 | Public models `kairyu-verified` (routed) and `kairyu-verified-always`; Open WebUI on :3012; the answer page on :3013. |

## L2: how an answer is made

```text
request
  │
  ▼
profile_judge ── Winnow (System One), 1 request: THINK or VERIFIED?
  │                (kairyu-verified-always skips the judge: VERIFIED)
  ├─ VERIFIED
  │    wave 1 ┬ drafts        DeepSeek, the caller's effort: five answers D1..D5 from
  │           │               viewpoints as different as possible (one call)
  │           └ requirements  Qwen, thinking at low: what the answer must meet,
  │                           necessary, sufficient, MECE (≤ 16)
  │    wave 2   judgments     Winnow, 1 request with the request: each draft
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
- When Winnow does not answer within 10 s (down, overloaded, unreadable),
  the request takes the think route.
- Both routes pass the caller's tools and response_format to the DeepSeek
  role that publishes (the verified route's answer).
- If Winnow cannot read the judgments (down, or the drafts exceed its
  65,536-token decision context), the answer is written without them.

## Run

```sh
./run.sh            # preflight, images, checkpoints, compose up, readiness probes
./verify.sh list    # GPU gates; evidence lands in model-volumes/<env>/results/
./browser-smoke.sh  # the answer page and Open WebUI in a real browser
./run.sh down
```

`run.sh` builds `winnow-server` from its pinned revision with the f072b10
llama.cpp backport (`winnow-patches/`), pulls the pinned vLLM release for
Qwen, and builds the DeepSeek SM120 overlay when its pinned image is absent.
`DEEPSEEK_MODEL_SEED` / `QWEN_MODEL_SEED` may name local copies of the pinned
checkpoints below `/mnt/nvme`; they are hard-linked and re-hashed against the
pins before serving.

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
| `l1` | every DeepSeek DP rank (thinking and chat JSON), Qwen chat, Winnow chat, Winnow System One (L1), one verified answer |
| `routing` | `datasets/routing-set.json`: VERIFIED miss rate < 10 % on the calibration and held-out halves |
| `think-route` | everyday requests stream from the think route at the default effort |
| `effort` | the think route gets the caller's effort; the verified route's drafts and answer get it too (default high); Qwen always thinks at low |
| `verified-route` | a verified request runs the three waves: drafts beside requirements, one Winnow read of 5 + 5 x N items, then the answer |
| `fallback` | Winnow down: 200 on the think route; Winnow back: routed again |
| `serving` | `kairyu-verified-always` at c1/c4/c8/c16 (8/16/16/32 InFoBench requests) |
| `serving-routed` | `kairyu-verified` at c1/c4/c8/c16 on the routing set, per-route latency and tokens |
| `browser` | `browser-smoke.sh`: both UIs answer |

## Limits

- No answer carries a guarantee (`kairyu_verification` is absent): Winnow's
  judgments inform the answer, they do not gate it.
- The drafts share one call capped at 131,072 tokens (the DSL's internal
  maximum).
- Latency: measured in `MEASUREMENTS.md`.
