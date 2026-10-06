# DeepSeek-V4.1 + Qwen3.8 + Winnow routed example (8 GPUs)

Owner-approved plan, 2026-10-06. Rebuilds
`examples/deepseek-v4.1-openjev-verified-8gpu` on main.

## Goal

One DeepSeek-V4.1 example with two routes, chosen per request by Winnow:
DeepSeek at the caller's effort, or the verified route. The verified route is,
for now, one DeepSeek call at max effort; the guarantee is rebuilt later.

## L1

| GPU | Model | Built as in |
|---|---|---|
| 0-5 | DeepSeek-V4.1-Flash DP6/EP6 | `deepseek-v4.1-flash-6gpu` |
| 6 | Qwen3.8-27B | `qwen3.8-27b-1gpu` |
| 7 | Winnow-12B Q8_0 (llama.cpp): chat and System One | `winnow-12b-q8-1gpu` |

Qwen is served as an internal pool and referenced by no route yet.

## L2

- Route judge: Winnow through System One, one choice over the whole
  conversation, THINK or VERIFIED, most probable wins. The question and both
  criteria are the old verified example's, minus the parts that existed only
  for the verified-tool route (its description in the question; "requests
  that need no tool call" in VERIFIED).
- THINK: `deepseek_think`, one DeepSeek call at the caller's effort (prompt
  unchanged).
- VERIFIED: one DeepSeek call at max effort.
- Both routes receive the caller's tools unchanged.
- Judge unavailable: THINK.
- Public models stay `kairyu-verified` (routed) and `kairyu-verified-always`
  (VERIFIED only).
- Removed: everything belonging to the verified-tool route (label, profile,
  `verified-tool-routing-set.json`, gates, tests, routing metric, docs); the
  checklist DAG (extract, history, implicit, adopt, answer, checklist,
  acceptance, repair); `calibrate.py`; `implicit-set.json`; both OpenJev
  services.
- No framework (`kairyu/`) change.

## L3 / UI

Open WebUI (:3012, effort dropdown and filter) and the answer page (:3013)
carry over unchanged. The answer page's guarantee panel stays; with no
guarantee computed it shows unverified or nothing. Accepted for now.

## GPU gates (old gates dropped; concurrency/count/budget conditions kept)

| Gate | Checks |
|---|---|
| l1 | every DeepSeek DP rank (thinking, chat); Qwen chat; Winnow chat and System One |
| routing | `routing-set.json`: VERIFIED miss rate < 10 % on calibration and held-out halves |
| think-route | everyday requests stream from THINK at the default effort |
| effort | THINK gets the caller's effort; VERIFIED always max |
| verified-route | a VERIFIED request is answered by one max-effort call |
| fallback | Winnow down: 200, THINK; Winnow back: normal routing |
| serving | `kairyu-verified-always` at c1/c4/c8/c16 (8/16/16/32 requests, same budget): latency, tokens, tok/s |
| serving-routed | `kairyu-verified` on `routing-set.json`, same plan: route mix, per-route latency, tokens, tok/s |
| browser | `browser-smoke.sh`: answers render in the answer page and Open WebUI for both models (guarantee panel content not checked) |

Dropped: calibrate, requirements, repair, structured, implicit (checklist
DAG), verified-tool-routing, verified-tool-route (verified-tool route).

## Other changes

- `git mv` the example to `examples/deepseek-v4.1-qwen3.8-winnow-8gpu`;
  rewrite compose, example.json, kairyu.yaml, control.py, run.sh, verify.sh,
  README, MEASUREMENTS.
- Design: VCO-D18 in `docs/design/example-verified-checklist-orchestration.md`
  (Winnow two-way judge, VERIFIED = one max call, verified-tool route removed,
  checklist temporarily removed, Qwen idle, UI kept); states that it
  supersedes VCO-D17. Update `PROGRESS.md` and `examples/README.md`.
- Tests: delete checklist and verified-tool tests. Keep: the gateway builds
  from the configs; GPU allocation matches compose; Winnow's choice maps to the
  right profile and effort; an unavailable judge routes to THINK. Report base
  vs head collection counts.
- Before push: `CUDA_VISIBLE_DEVICES= uv run pytest`, `uv run ruff check .`.
  Then run the GPU gates in the order above; stop and report on the first
  failure.
