# deepseek-v4.1-quyet-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. DeepSeek-V4.1
DP6/EP6 on GPUs 0-5 (image `sha256:119afb09…`, the six-GPU example's SM120
overlay); two Quyet-1.0-Large replicas (stock vLLM v0.31.0 bf16 plus this
example's System One adapter): `quyet-route` on GPU 6 (route judge only),
`quyet-judge` on GPU 7 (judgments and form check).
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-quyet-8gpu/results/`.

## Quyet layout and the rebuilt verified tool route (VCO-D21, VCO-D22)

GPU run 2026-10-11 04:38-05:46 JST (gates; UTC 2026-10-10 19:38-20:46) at
`6d19280e` (the judgments fix below), plan
`docs/superpowers/plans/2026-10-11-deepseek-v4.1-quyet-8gpu.md`. Storage prepared
by hard-linking the attested DeepSeek and Quyet checkpoints, then plain
`./run.sh`; `./verify.sh <gate>` in GATES order, run to the end. Logs
`run-20261011.log`, `pregate-20261011-r3.log`, `gates-20261011.log`; results
`*-20261010T19*`/`T20*`. **8 of 9 gates pass; `verified-tool-route` fails 2 of
12 on routing (`own-check` routed THINK); the pre-gate route replay misroutes 11
of 64 real turns (pass line 6).**

Fixed during the run: the first pre-gate run showed every judgment read refused
with 400 (`instructions must be a non-empty string`): the quyet package takes a
question's instructions only as one string, and the candidate went in as an
object (checklist `context`). The candidate now goes in the question text; the
example test's fake adapter applies the package's rule (5 of 11 cases fail on the
old spec).

### Before the gates (`pregate-20261010T192154Z.json`)

Route judge, 64 real DeepSWE turns that each need a tool call
(`deepswe-verified-3wave-nextstep-4w-20261007-r1`, the last and a middle turn of
32 runs; median 198,515 characters; the agent's `bash` tool given as a generic
`bash(command)` schema), one at a time: **11 routed THINK** (Winnow: 1 of 64 on
the same run), all of them last turns: the next move is the submit command after
tool results show the commit and tests done. The task message survived the
12,000-character bound in all 11. Reads 0.83 s median, 0 states cut.

TOOL route, six of those turns in process (real L1, every exchange recorded):
one was routed THINK; the other five ran 14-16 candidates, 84-96 judgment
questions in three parallel reads of 1.7-4.8 s, and passed the form check on the
first attempt (no fixes; F7 read 0.51-0.92, every other item >= 0.91). Final
replies made 2, 1, 4, 3 and 1 calls; 72-283 s per turn (candidates 59-174 s).
26 Quyet reads, 0 states cut.

### Gates

| Gate | Result |
|---|---|
| l1 | pass (29 s): every DeepSeek DP rank, each Quyet vLLM and System One, one structured tool call from the public model through the new route |
| routing | pass: TOOL miss 5 % (calibration) / 0 % (held-out); THINK precision 99.0 %, recall 99.0 % (tool-declared THINK 93.8 %); judge wall 11.1 s for 136 |
| think-route | pass: 6/6 THINK, streamed; p50 0.95 s, TTFT p50 0.76 s |
| effort | pass: THINK follows the caller (0.8-1.1 s); TOOL candidates and answer follow the caller, a structured `bash` call each time, form passed first time; 88-144 s, 15,277-25,657 output tokens, 174-185 tok/s |
| verified-tool-route | **fail**: 10/12 TOOL with 12-16 candidates, every judgment and form check on `quyet-judge`, form passed first time (no fixes), a structured call to a declared tool; `own-check` (unary and streamed) routed THINK (P(TOOL) 0.35) and answered with a `bash` call. p50 64.8 s, p95 114.3 s; 132,006 output tokens; one reply batched two calls |
| fallback | pass: `quyet-route` stopped: 2/2 THINK (`backend_error`, 0.8-1.5 s) with a call; `quyet-judge` stopped: TOOL, `judge_unavailable`, published unverified with a call (90.3 s); both back: judged and form-checked (134.6 s) |
| serving | pass: 72/72 answered, every reply a structured call (table below) |
| serving-routed | pass: 72/72 answered (table below) |
| browser | pass: Open WebUI answers |

serving (agent turns), Winnow layout p50 of 2026-10-09 in brackets:

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | TOOL n / p50 s | THINK n | Form: fixed / limit | Replies with 2+ calls |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| c1 | 8 | 74.0 (137.2) | 105.1 | 544 | 174 | 7 / 75.1 | 1 | 1 / 0 | 2 |
| c4 | 16 | 71.9 (131.5) | 120.0 | 360 | 494 | 15 / 79.0 | 1 | 1 / 0 | 7 |
| c8 | 16 | 103.7 (195.8) | 162.2 | 255 | 656 | 14 / 107.6 | 2 | 0 / 1 | 5 |
| c16 | 32 | 131.1 (208.6) | 200.6 | 364 | 1,042 | 30 / 138.3 | 2 | 0 / 1 | 13 |

The two replies published at the fix limit read F5 below 0.5 on one call (c8)
and F1 and F5 on two read-only calls with empty text (c16); both carried
structured calls. 342 Quyet reads (6,216 questions), 0 states cut; judgment
reads p50 2.1 s, max 6.3 s.

serving-routed (routing set):

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | TOOL n / p50 s | THINK n / p50 s | Route judge p50 s |
|---|---:|---:|---:|---:|---:|---|---|---|
| c1 | 8 | 15.8 | 56.0 | 216 | 157 | 2 / 59.8 | 6 / 12.2 | 0.13-0.15 |
| c4 | 16 | 29.6 | 74.6 | 148 | 458 | 6 / 61.3 | 10 / 3.1 | 0.14-0.15 |
| c8 | 16 | 46.6 | 114.3 | 135 | 645 | 6 / 77.9 | 10 / 16.2 | 0.19-0.39 |
| c16 | 32 | 49.6 | 182.9 | 256 | 759 | 10 / 123.7 | 22 / 12.2 | 0.14-0.70 |

Every TOOL reply here passed the form check first time. 168 Quyet reads, 0
states cut.

### Routing misses: what a TOOL floor would change (offline, recorded probabilities)

| P(TOOL) floor | Real turns misrouted | `own-check` | Routing set TOOL miss (cal / held-out) | THINK precision | Tool-declared text turns kept on THINK |
|---|---|---|---|---|---|
| none (served) | 11/64 | THINK | 5 % / 0 % | 99.0 % | 15/16 |
| 0.2 | 7/64 | TOOL | 5 % / 0 % | 98.9 % | 12/16 |
| 0.1 | 5/64 | TOOL | 5 % / 0 % | 98.9 % | 11/16 |

## Earlier layout: two Winnow-12B replicas (`deepseek-v4.1-winnow-8gpu`)

The sections below were measured before VCO-D21, when the example was
`deepseek-v4.1-winnow-8gpu`: two Winnow-12B Q8_0 replicas (winnow-server
`77d1458` + f072b10), `winnow-route` on GPU 6 and `winnow-judge` on GPU 7.
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-winnow-8gpu/results/`.
The verified tool route: `*-20261009T07*` to `T09*` (rerun, gate log
`gates-20261009-verified-tool-5.log`) and `*-20261009T04*` to `T06*` (gate log
`gates-20261009-verified-tool-4.log`). The two-Winnow sections below it:
`*-20261007T04*`, `T05*`, `T08*`, `T09*`, `browser-20261007T060041Z.json`, gate
logs `gates-20261007-1307.log` and `gates-20261007-1712-nextstep.log`.
Sections on the earlier Qwen layout
(`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-qwen3.8-winnow-8gpu/results/`,
`*-20261006T*.json` and `*-20261006T16*`/`T17*`; gate logs `gates-20261006.log`,
`gates-20261006-256k.log`, `gates-20261006-three-wave.log` and
`gates-20261007-requirements-without-reasoning.log`).

## GPU gates: rerun after the review fixes (2026-10-09 16:37-18:19 JST)

Commit `2b4ae850` (PR #641 review fixes: typed markerless DSML parameters,
validated messages passed as `OrchestrationRequest.conversation`, character
cost of a native conversation, orphan cleanup in `up`/`down`). `./run.sh`
removed no orphan (none left), then `./verify.sh <gate>` in GATES order. All
nine gates pass. Log `gates-20261009-verified-tool-5.log`, results
`*-20261009T07*` to `T09*`.

| Gate | Result |
|---|---|
| l1 | pass (1 s) |
| routing | TOOL miss 5 % on both halves; THINK precision 97.9 % (unchanged) |
| think-route | 6/6 THINK; p50 1.15 s, TTFT p50 0.91 s |
| effort | THINK 0.6-1.3 s; TOOL 111-152 s, 29,295-43,227 output tokens, 250-285 tok/s, structured `bash` call at every effort |
| verified-tool-route | 12/12 TOOL with a structured call; p50 122.1 s, p95 155.4 s; 346,246 output tokens; 211-271 tok/s |
| fallback | route down: 2/2 THINK with a call (0.85 s); judge down: TOOL, `judge_unavailable`, a call (167.0 s); recovered: judged (147.0 s) |
| serving | 72/72, every reply a structured call (table below) |
| serving-routed | 72/72 (table below) |
| browser | Open WebUI answers |

serving (agent turns):

| Level | Requests | p50 s | p95 s | Wall s | Output tokens | Output tok/s | TOOL n / p50 s | THINK n / p50 s |
|---|---:|---:|---:|---:|---:|---:|---|---|
| c1 | 8 | 137.2 | 166.5 | 1,075 | 255,436 | 238 | 8 / 137.2 | 0 |
| c4 | 16 | 131.5 | 205.5 | 541 | 321,438 | 594 | 12 / 160.3 | 4 / 1.6 |
| c8 | 16 | 195.8 | 305.3 | 472 | 401,993 | 852 | 13 / 213.3 | 3 / 1.1 |
| c16 | 32 | 208.6 | 344.9 | 491 | 766,927 | 1,562 | 27 / 217.8 | 5 / 2.7 |

serving-routed (routing set):

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | TOOL n / p50 s | THINK n / p50 s |
|---|---:|---:|---:|---:|---:|---|---|
| c1 | 8 | 10.1 | 126.7 | 355 | 184 | 2 / 131.3 | 6 / 9.3 |
| c4 | 16 | 13.4 | 130.8 | 268 | 455 | 5 / 125.1 | 11 / 2.5 |
| c8 | 16 | 97.7 | 213.5 | 304 | 649 | 6 / 185.1 | 10 / 17.8 |
| c16 | 32 | 57.6 | 238.7 | 276 | 1,115 | 11 / 193.7 | 21 / 15.1 |

## GPU gates: verified tool route (2026-10-09 13:30-15:15 JST)

Commit `8e85b30f` (PR #641, VCO-D20 and its amendments): one public model,
`kairyu-verified-tool`; Winnow routes TOOL (the next reply needs a tool call)
to the three waves and THINK to DeepSeek at the caller's effort; both
publishers receive the caller's conversation natively (m1 D8 amendment); the
route judge keeps an agent's task in long runs (m1 D9 amendment). `./run.sh`,
then `./verify.sh <gate>` in GATES order. All nine gates pass.

Before the gates (same day, committed code):
- 64 real DeepSWE agent turns (`deepswe-verified-3wave-nextstep-4w-20261007-r1`),
  each needing a tool call, through the route judge one at a time: 1 routed
  THINK (21 before the m1 D9 amendment and the TOOL criteria); slowest read
  6.8 s.
- The arktype, helm-unified and fd failure turns, twice each: 6/6 routed TOOL,
  each with one structured call; helm-unified made the submit call alone,
  arktype never batched commit and submit.

Earlier runs that day failed and led to the amendments: `routing` (TOOL miss
15 % on both halves), then `verified-tool-route` 11/12 (a DSML call written
without its markers) and 9/12 (calls written as JSON text).

| Gate | Result |
|---|---|
| l1 | DeepSeek every DP rank; chat and System One on both Winnow replicas; one structured tool call from the public model |
| routing | 136 conversations; TOOL miss 5 % on both halves; THINK precision 97.9 %, THINK recall 97.9 % (tool-declared THINK 87.5 %); judge wall 10.1 s |
| think-route | 6/6 THINK at high, streamed; p50 0.99 s, p95 2.93 s, TTFT p50 0.83 s |
| effort | THINK follows the caller (0.6-0.9 s); TOOL drafts/answer follow the caller, requirements at max, structured `bash` call each time; 126-269 s, 31,799-63,134 output tokens, 219-270 tok/s |
| verified-tool-route | 12/12 routed TOOL with a structured call to a declared tool (`bash`, `http_get`, `create_event`), all four stages, wave 1 parallel, judgments on `winnow-judge` (30-85 items); p50 131.7 s, p95 182.1 s; 385,506 output tokens; 180-306 tok/s; requirements end 74-184 s, answer 1-60 s |
| fallback | `winnow-route` stopped: 2/2 THINK (`backend_error`, 0.9-1.0 s) with a structured call; `winnow-judge` stopped: routed TOOL, judgments `judge_unavailable`, structured call (132.4 s); both back: routed and judged (170.0 s) |
| serving | 72/72 answered, every reply a structured call (table below) |
| serving-routed | 72/72 answered (table below) |
| browser | Open WebUI answers |

serving (agent turns, `datasets/tool-turns.json`):

| Level | Requests | p50 s | p95 s | Wall s | Requests/min | Output tokens | Output tok/s | TOOL n / p50 s | THINK n / p50 s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| c1 | 8 | 119.1 | 163.3 | 991 | 0.48 | 226,029 | 228 | 8 / 119.1 | 0 |
| c4 | 16 | 148.4 | 221.9 | 546 | 1.76 | 314,925 | 576 | 12 / 152.1 | 4 / 2.5 |
| c8 | 16 | 170.3 | 280.1 | 404 | 2.38 | 378,347 | 937 | 13 / 186.0 | 3 / 2.2 |
| c16 | 32 | 198.5 | 317.0 | 479 | 4.01 | 706,211 | 1,474 | 27 / 219.6 | 5 / 2.8 |

Four of the 24 agent turns (dedup-case, wrong-column, official-schedule,
code-blocks: a tool result that reads like an answer but needs a further
move) were routed THINK every time (12 of 72); their replies still carried a
structured call.

serving-routed (`datasets/tool-routing-set.json`):

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | TOOL n / p50 / p95 s | THINK n / p50 / p95 s | Route judge p50 s |
|---|---:|---:|---:|---:|---:|---|---|---|
| c1 | 8 | 35.6 | 92.5 | 396 | 201 | 2 / 126.1 / 92.5 | 6 / 15.7 / 48.5 | 0.08-0.09 |
| c4 | 16 | 9.4 | 133.5 | 247 | 513 | 5 / 130.6 / 133.5 | 11 / 2.3 / 36.9 | 0.08-0.09 |
| c8 | 16 | 76.9 | 189.9 | 216 | 733 | 6 / 166.5 / 189.9 | 10 / 17.2 / 111.4 | 0.10-0.12 |
| c16 | 32 | 70.2 | 242.9 | 282 | 1,136 | 11 / 185.8 / 242.9 | 21 / 19.5 / 124.6 | 0.09-0.17 |

## GPU gates: next-step replies, structured tool calls (2026-10-07 17:13-18:58 JST)

Commit `17242799` (PR #641): drafts, requirements and answer target the next
step on agent turns; each draft carries `tool_calls` and reads the caller's
tools; the judgments read the conversation and the tools; no stage reports in
`reasoning_content`. Kairyu restarted to load the spec, `./run.sh`, then
`./verify.sh <gate>` in GATES order. All nine gates pass. Log
`gates-20261007-1712-nextstep.log`, results `*-20261007T081*` to `T0958*`.

Before the gates, five saved DeepSWE turns (actionlint, abs-module, adaptix;
replayed reasoning removed) were sent to `kairyu-verified-always` twice: 10/10
returned a structured tool call and an empty `reasoning_content`, including the
actionlint turn whose answer had written its call as text. Run locally with
stage reports on, two turns' requirements targeted the next step (inspect the
unread `uses:` handling and do not edit yet; build and run a reproduction
before touching the loader), no draft wrote a call as text (0 of 10), and the
judge ranked the drafts that fit those requirements highest (p 0.889, 0.679).

| Gate | Result |
|---|---|
| l1 | DeepSeek every DP rank; chat and System One on both Winnow replicas; one verified answer (39 s) |
| routing | VERIFIED miss 0 % on both halves; everyday to THINK 96.9 %; judge wall 5.7 s |
| think-route | 6/6 THINK at high, streamed; p50 0.81 s, TTFT p50 0.70 s |
| effort | drafts/answer follow the caller, requirements at max; verified 30.5-52.5 s, 183-241 tok/s |
| verified-route | 12/12; judgments on `winnow-judge` (10-85 items, 0.6-1.7 s); p50 181.6 s, p95 267.8 s; 440,555 output tokens; 183-252 tok/s. Stage medians: requirements 92.5 s, drafts 80.3 s, answer 56.8 s. Streamed TTFT 80-297 s |
| fallback | `winnow-route` down: THINK fallback, judgments still on `winnow-judge` (85 items); `winnow-judge` down: routed, judgments `failed`; both back: routed and judged |
| serving | 72/72 answered (table below) |
| serving-routed | 72/72 answered (table below) |
| browser | the answer page and Open WebUI answer |

serving (`kairyu-verified-always`, InFoBench instructions):

| Level | Requests | p50 s | p95 s | Wall s | Requests/min | Output tokens | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 8 | 77.4 | 91.0 | 572 | 0.84 | 115,548 | 202 |
| c4 | 16 | 94.4 | 142.9 | 401 | 2.39 | 219,206 | 547 |
| c8 | 16 | 127.8 | 301.2 | 350 | 2.74 | 273,706 | 782 |
| c16 | 32 | 147.6 | 242.1 | 391 | 4.91 | 488,978 | 1,251 |

serving-routed (`kairyu-verified`, routing set):

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | VERIFIED n / p50 / p95 s | THINK n / p50 / p95 s | Route judge p50 s |
|---|---:|---:|---:|---:|---:|---|---|---|
| c1 | 8 | 82.7 | 138.1 | 679 | 196 | 7 / 89.5 / 138.1 | 1 / 0.5 / 0.5 | 0.07-0.08 |
| c4 | 16 | 116.8 | 213.2 | 560 | 473 | 10 / 173.6 / 213.2 | 6 / 1.8 / 4.5 | 0.07-0.08 |
| c8 | 16 | 141.4 | 180.4 | 312 | 693 | 11 / 151.7 / 180.4 | 5 / 1.7 / 3.7 | 0.08 |
| c16 | 32 | 86.8 | 430.5 | 465 | 1,000 | 16 / 261.7 / 437.2 | 16 / 2.8 / 9.8 | 0.08-0.23 |

Against the section below (same layout, before this change): verified-route
p50 rose from 141.6 s to 181.6 s and streamed TTFT from 24-69 s to 80-297 s;
serving p50 fell at c1/c4/c16 (98.3/103.0/151.4 → 77.4/94.4/147.6 s) and rose
at c8 (116.8 → 127.8 s); serving-routed VERIFIED p50 fell at c4/c8/c16 and
rose at c1 (77.9 → 89.5 s).

## GPU gates: DeepSeek max requirements, two Winnow replicas (2026-10-07 13:08-15:00 JST)

Commit `256d0ed5` (PR #641, with the verdict-wait validation fixes of the
review): `requirements` on DeepSeek at max, no Qwen, `winnow-route` judges
the route and `winnow-judge` the drafts. `./run.sh` (storage prepared by hard
links from the Qwen layout), then `./verify.sh <gate>` in GATES order. All
nine gates pass; `browser` passed on its rerun after the smoke script learned
to close Open WebUI's first-run "What's New" dialog (fresh `webui-data`).

| Gate | Result |
|---|---|
| l1 | DeepSeek every DP rank; chat and System One on both Winnow replicas; one verified answer (24 s for the gate) |
| routing | 80 conversations; VERIFIED miss 0 % on both halves; everyday to THINK 96.9 %; judge wall 5.7 s on `winnow-route` |
| think-route | 6/6 THINK at high, streamed, route judged on `winnow-route` (~0.07 s); p50 0.74 s, TTFT p50 0.62 s |
| effort | THINK and VERIFIED drafts/answer follow the caller (none→high, low, high, max), `requirements` at max every time; verified 42.4-66.9 s, 199-243 tok/s |
| verified-route | 12/12, all four stages succeed, wave 1 parallel, judgments on `winnow-judge` (10-85 items in 0.4-1.0 s); p50 141.6 s, p95 240.1 s; 398,540 output tokens; 181-235 tok/s. Stage ranges: requirements 23.8-124.2 s (median 73.1), drafts 38.9-109.8 s (median 79.5), answer 10.1-145.5 s. Streamed TTFT 24-69 s |
| fallback | `winnow-route` stopped: 2/2 routed requests on THINK (`backend_error`), the always model still judged on `winnow-judge` (15 items, 22.4 s); `winnow-judge` stopped: routing intact (0.18 s), the always model answers with judgments `failed` (18.4 s); both back: routed and judged |
| serving | 72/72 answered (table below) |
| serving-routed | 72/72 answered; every request judged on `winnow-route` (table below) |
| browser | the answer page and Open WebUI answer |

serving (`kairyu-verified-always`, InFoBench instructions):

| Level | Requests | p50 s | p95 s | Wall s | Requests/min | Output tokens | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 8 | 98.3 | 139.8 | 765 | 0.63 | 150,324 | 196 |
| c4 | 16 | 103.0 | 182.1 | 504 | 1.91 | 278,339 | 553 |
| c8 | 16 | 116.8 | 178.8 | 320 | 3.00 | 247,367 | 774 |
| c16 | 32 | 151.4 | 239.9 | 434 | 4.42 | 518,775 | 1,195 |

serving-routed (`kairyu-verified`, routing set):

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | VERIFIED n / p50 / p95 s | THINK n / p50 / p95 s | Route judge p50 s |
|---|---:|---:|---:|---:|---:|---|---|---|
| c1 | 8 | 69.8 | 90.3 | 651 | 200 | 7 / 77.9 / 90.3 | 1 / 0.5 / 0.5 | 0.08 |
| c4 | 16 | 114.1 | 238.2 | 623 | 490 | 10 / 185.5 / 238.2 | 6 / 1.4 / 1.6 | 0.07-0.08 |
| c8 | 16 | 169.0 | 207.8 | 369 | 688 | 11 / 183.5 / 207.8 | 5 / 2.5 / 3.2 | 0.08-0.09 |
| c16 | 32 | 56.3 | 409.2 | 496 | 948 | 16 / 282.2 / 434.8 | 16 / 2.9 / 6.5 | 0.07-0.26 |

Against the Qwen layout (section below): verified-route p50 fell from
162.7 s to 141.6 s on the routing set's VERIFIED conversations; on short
InFoBench instructions serving p50 rose (c1 51.5 → 98.3 s, c16 118.6 →
151.4 s), since DeepSeek at max now writes the requirements beside the drafts.

## GPU gates: Qwen requirements without replayed reasoning (2026-10-07 01:27-02:52 JST)

Commit `0c9e8c2d` (PR #641): Qwen `requirements` reads
`{conversation_without_reasoning}`; DeepSeek and Winnow unchanged. `./run.sh`
(only the Kairyu container recreated), then `./verify.sh <gate>` in GATES
order. All nine gates pass.

| Gate | Result |
|---|---|
| l1 | every L1 service and one verified answer (60 s for the gate) |
| routing | 80 conversations; VERIFIED miss 0 % on both halves; everyday to THINK 96.9 % |
| think-route | 6/6 THINK at high, streamed; p50 0.80 s, TTFT p50 0.68 s |
| effort | THINK and VERIFIED drafts/answer follow the caller (none→high, low, high, max), Qwen low every time; verified 29.9-53.3 s, 176-185 tok/s |
| verified-route | 12/12, all four stages succeed, wave 1 parallel, one Winnow read of 20-40 items in 0.5-1.2 s; p50 162.7 s, p95 272.9 s; 340,828 output tokens; 161-191 tok/s. Stage ranges: requirements 34-90 s, drafts 40-149 s, answer 12-159 s. Streamed TTFT 39-54 s |
| fallback | Winnow stopped: 2/2 routed requests on THINK (`backend_error`); the always model answers with judgments `failed` (18.4 s); Winnow back: routed again |
| serving | 72/72 answered (table below) |
| serving-routed | 72/72 answered; Winnow judged every request (table below) |
| browser | the answer page and Open WebUI answer |

serving (`kairyu-verified-always`, InFoBench instructions):

| Level | Requests | p50 s | p95 s | Wall s | Requests/min | Output tokens | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 8 | 51.5 | 86.5 | 467 | 1.03 | 71,900 | 154 |
| c4 | 16 | 67.8 | 113.3 | 304 | 3.15 | 159,065 | 523 |
| c8 | 16 | 95.2 | 169.5 | 268 | 3.58 | 169,509 | 633 |
| c16 | 32 | 118.6 | 188.7 | 353 | 5.45 | 333,188 | 945 |

serving-routed (`kairyu-verified`, routing set):

| Level | Route | Requests | p50 s | p95 s | Output tokens | Judge p50 s |
|---|---|---:|---:|---:|---:|---:|
| c1 | think | 1 | 0.39 | 0.39 | 41 | 0.073 |
| c1 | verified | 7 | 46.0 | 73.4 | 63,945 | 0.077 |
| c4 | think | 6 | 1.36 | 2.99 | 919 | 0.074 |
| c4 | verified | 10 | 106.6 | 200.0 | 193,596 | 0.082 |
| c8 | think | 5 | 1.74 | 3.02 | 653 | 0.239 |
| c8 | verified | 11 | 87.5 | 144.0 | 137,377 | 0.082 |
| c16 | think | 16 | 2.71 | 8.08 | 2,712 | 0.071 |
| c16 | verified | 16 | 174.4 | 330.5 | 290,733 | 0.220 |

## GPU gates: three-wave verified route (VCO-D19, 2026-10-06 21:17-23:03 JST)

`./run.sh`, then `./verify.sh <gate>` in GATES order; raw evidence
`*-20261006T12*` to `*-20261006T14*` and `gates-20261006-three-wave.log`.
All nine gates pass. VERIFIED = DeepSeek five drafts (caller's effort) beside
Qwen requirements (low), one Winnow judgment read, DeepSeek answer (caller's
effort).

| Gate | Result |
|---|---|
| l1 | every DeepSeek DP rank, Qwen chat, Winnow chat and System One; the three-wave verified probe answers "Paris" (39 s for the gate) |
| routing | 80 conversations; VERIFIED miss 0 % on both halves; everyday to THINK 96.9 %; P(VERIFIED) median 0.986-0.9996 for every accuracy category, 0.009 for everyday |
| think-route | 6/6 THINK at high, streamed; p50 0.77 s, TTFT p50 0.58 s |
| effort | THINK follows the caller (none→high, low, high, max); VERIFIED drafts and answer follow the caller (none→high), Qwen low every time; verified 30-50 s on a short prompt, wave 1 starts both roles at 0 s |
| verified-route | 12/12 (unary and streamed, every caller effort), all four stages succeed, wave 1 parallel, one Winnow read of 20-45 items in 0.5-1.3 s; p50 183.8 s, p95 267.5 s; 355,139 output tokens. Stage ranges: requirements 31-58 s, drafts 52-137 s, answer 7-146 s. Streamed TTFT 31-58 s (the exposed requirements) |
| fallback | Winnow stopped: routed requests answer on THINK (`backend_error`); the always model answers with judgments `failed` and the answer written without them (12.7 s); Winnow back: routed again |
| serving | below |
| serving-routed | below |
| browser | the answer page and Open WebUI answer for both models |

### serving (`kairyu-verified-always`, InFoBench instructions)

| Level | Requests | p50 s | p95 s | Wall s | Requests/min | Output tokens | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 8 | 61.8 | 89.1 | 502 | 0.96 | 83,891 | 167 |
| c4 | 16 | 67.9 | 91.6 | 295 | 3.25 | 151,960 | 515 |
| c8 | 16 | 83.8 | 156.8 | 269 | 3.57 | 172,812 | 642 |
| c16 | 32 | 119.2 | 208.2 | 319 | 6.02 | 341,267 | 1,070 |

Stage medians (min-max) in seconds:

| Level | Requirements (Qwen) | Drafts (DeepSeek) | Judgments (Winnow) | Answer (DeepSeek) | Judgment items |
|---|---|---|---|---|---|
| c1 | 31 (19-66) | 44 (17-63) | ≤ 1 | 15 (4-36) | 15-35 |
| c4 | 32 (18-61) | 43 (23-65) | ≤ 1 | 15 (2-58) | 15-70 |
| c8 | 36 (17-50) | 60 (21-207) | ≤ 2 | 20 (5-61) | 20-65 |
| c16 | 33 (13-62) | 76 (24-167) | ≤ 2 | 30 (1-116) | 10-55 |

Every request `stop`; Winnow judged all 72.

### serving-routed (`kairyu-verified`, routing set)

| Level | Route | Requests | p50 s | p95 s | Output tokens | Judge p50 s |
|---|---|---:|---:|---:|---:|---:|
| c1 | think | 1 | 0.38 | 0.38 | 29 | 0.076 |
| c1 | verified | 7 | 57.4 | 73.9 | 67,227 | 0.077 |
| c4 | think | 6 | 1.72 | 2.36 | 946 | 0.073 |
| c4 | verified | 10 | 83.0 | 191.9 | 195,544 | 0.081 |
| c8 | think | 5 | 1.77 | 3.15 | 644 | 0.079 |
| c8 | verified | 11 | 88.7 | 160.7 | 138,562 | 0.154 |
| c16 | think | 16 | 2.24 | 5.95 | 2,084 | 0.071 |
| c16 | verified | 16 | 181.2 | 424.1 | 576,601 | 0.191 |

Every request answered; Winnow judged every verified request. One c16
verified request (a medical-expense deduction question) took 1,543 s with
291,268 output tokens: its stages ended at 47 s (requirements), 181 s (drafts)
and 182 s (judgments), but the trace's last answer attempt starts at 1,442 s
and ends at 1,542 s with a 1,442-character reply. The answer's earlier
generation ran about 21 minutes (about 240K tokens) before that attempt; the
gate result keeps only stage summaries, so why it was re-run is not recorded.
This one request sets the c16 wall time (1,543 s); the next slowest took 424 s.

## Previous configuration (VCO-D18, one max-effort DeepSeek call)

Gates of 2026-10-06 (before VCO-D19):

All nine gates pass. Every output cap is 262,144 (the V4.1 card's `max_tokens` >= 256K):
verified-route, serving, serving-routed and browser were re-run after the gate
requests, the answer page (32,768) and Open WebUI (65,536) were raised to it; the
tables below are that re-run. l1, routing, think-route, effort and fallback ran
with the 32,768 request cap and are unaffected by it.

| Gate | Result |
|---|---|
| l1 | every DeepSeek DP rank (thinking and chat JSON), Qwen chat, Winnow chat, Winnow System One, one verified answer |
| routing | 80 conversations, Winnow read 5.7 s in all; VERIFIED miss 0 % on both halves; everyday to THINK 96.9 %; P(VERIFIED) median 0.986-0.9996 for every accuracy category, 0.009 for everyday |
| think-route | 6/6 THINK at high, streamed; p50 0.85 s, TTFT p50 0.69 s |
| effort | THINK follows the caller (none→high, low, high, max); VERIFIED is max every time |
| verified-route | 12/12 one DeepSeek call at max (unary and streamed, every caller effort), all `stop`; p50 113.2 s, p95 160.6 s, 177,281 output tokens, longest answer 25,355 tokens (at the earlier 32,768 cap one answer was cut, `length`) |
| fallback | Winnow stopped: 2/2 routed requests answered on THINK (fallback `backend_error`), the always model answered; Winnow restarted: routed again |
| serving | below |
| serving-routed | below |
| browser | the answer page and Open WebUI answer for both models |

#### serving (`kairyu-verified-always`, InFoBench instructions)

| Level | Requests | p50 s | p95 s | Output tok/s | Requests/min |
|---|---:|---:|---:|---:|---:|
| c1 | 8 | 23.5 | 47.5 | 131.7 | 2.46 |
| c4 | 16 | 12.6 | 41.4 | 424.3 | 13.56 |
| c8 | 16 | 21.9 | 79.3 | 445.9 | 8.05 |
| c16 | 32 | 24.6 | 68.0 | 736.8 | 21.86 |

No answer reached the cap (`length`: 0 at every level).

#### serving-routed (`kairyu-verified`, routing set)

| Level | Route | Requests | p50 s | p95 s | Output tokens | Judge p50 s |
|---|---|---:|---:|---:|---:|---:|
| c1 | think | 1 | 0.49 | 0.49 | 41 | 0.083 |
| c1 | verified | 7 | 11.6 | 28.5 | 18,280 | 0.077 |
| c4 | think | 6 | 1.46 | 2.78 | 1,007 | 0.072 |
| c4 | verified | 10 | 51.5 | 128.0 | 92,769 | 0.084 |
| c8 | think | 5 | 2.26 | 2.94 | 663 | 0.152 |
| c8 | verified | 11 | 41.4 | 70.7 | 43,412 | 0.077 |
| c16 | think | 16 | 3.29 | 6.69 | 2,317 | 0.092 |
| c16 | verified | 16 | 58.2 | 200.0 | 131,242 | 0.222 |

Every request answered at every level; none reached the cap.

The evidence of the checklist-verified configuration this example replaced
(OpenJev judge, checklist DAG) is in this file's history before VCO-D18, as
`examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md` at `df109a6b`.
