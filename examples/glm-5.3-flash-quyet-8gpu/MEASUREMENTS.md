# glm-5.3-flash-quyet-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. GLM-5.3-Flash
(official FP8) as one TP2 x DP3 / EP6 replica with MTP 3 on GPUs 0-5, stock
vLLM v0.31.0 (`sha256:c1c9f6fd…`); two Quyet-1.0-Large replicas on the same
image plus this example's System One adapter (`local/quyet-systemone:1.0.2-vllm-v0.31.0-glm53-8gpu`,
`sha256:aef3eff7…`): `quyet-route` on GPU 6 (route judge only), `quyet-judge`
on GPU 7 (the strict decision since VCO-D24; judgments and the strict form
check before).
Raw evidence: `/mnt/nvme/kairyu/model-volumes/glm-5.3-flash-quyet-8gpu/results/`
and `run-logs/`.

## One reply decided by Quyet (VCO-D24)

GPU run 2026-10-15 11:44-12:17 JST (UTC 02:44-03:17) at `06133907`: the
writer makes the move in one reply with every call needed now that is safe and
independent; Quyet reads it once, strict (nine questions: R1, S1-S3, F1-F5,
with S2 asking for one call first); a failing reply is written again once.
Plain `./run.sh` after `deepseek-v4.1-quyet-8gpu` went down, then the nine
gates in GATES order. Logs `run-logs/up-20261015.log`,
`run-logs/gates-20261015.log`; results `*-20261015T02*`/`T03*`. **All nine
gates pass.**

| Gate | Result |
|---|---|
| l1 | pass (3 s), including the strict name refusing a long state the plain name answers cut |
| routing | pass: TOOL miss 0 % held-out / 5 % calibration; THINK precision 98.9 %, recall 94.8 % (tool-declared THINK 68.8 %, 11/16); judge wall 11.1 s for 136, 0 cut |
| think-route | pass: 6/6 THINK, streamed; p50 4.9 s, TTFT p50 4.5 s (default effort max) |
| effort | pass: THINK and the answer follow the caller (default max, low, high, max); TOOL 1.8-10.4 s, one call each, decided first time |
| verified-tool-route | pass: 12/12 TOOL; the decision on `quyet_form` with nine questions passed 12/12 (11 first time, one after one fix); a structured call each; p50 1.95 s, p95 8.66 s, max 55.15 s (limit 900 s); 7,143 output tokens; 4 replies batched two calls |
| fallback | pass: `quyet-route` stopped: 2/2 THINK (`backend_error`, 1.7 s) with a call; `quyet-judge` stopped: TOOL, `judge_unavailable`, published unverified with a call (10.0 s); both back: decided and passed (7.1 s) |
| serving | pass: 72/72 answered, every reply a structured call (table below) |
| serving-routed | pass: 72/72 answered (table below) |
| browser | pass: Open WebUI answers |

serving (agent turns, default effort max; VCO-D23 layout p50 151.5 / 111.5 /
142.5 / 185.6 s, output tokens 121,961 / 156,871 / 178,601 / 383,830):

| Level | Requests | p50 s | p95 s | Max s | Wall s | Output tokens | Output tok/s | TOOL n / p50 s | THINK n | Decision passed | Fixes passed / failing | Replies with 2+ calls |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| c1 | 8 | 2.9 | 16.5 | 18.7 | 57 | 4,387 | 77 | 8 / 2.9 | 0 | 8 / 8 | 0 / 0 | 2 |
| c4 | 16 | 3.6 | 7.3 | 16.1 | 26 | 4,081 | 159 | 15 / 3.7 | 1 | 15 / 15 | 0 / 0 | 5 |
| c8 | 16 | 5.0 | 31.9 | 45.2 | 45 | 9,114 | 202 | 15 / 5.4 | 1 | 15 / 15 | 0 / 0 | 9 |
| c16 | 32 | 6.8 | 38.5 | 59.8 | 60 | 16,205 | 271 | 31 / 6.8 | 1 | 31 / 31 | 1 / 0 | 12 |

No reply was published failing its decision. The one-call misread of the old
F5 (`announced-no-call`) did not recur under S2.

serving-routed (the routing set, default effort max):

| Level | Requests | p50 s | p95 s | Max s | Wall s | Output tokens | Output tok/s | TOOL n / p50 s | THINK n / p50 s | Decision passed |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| c1 | 8 | 10.1 | 41.4 | 110.2 | 200 | 18,656 | 93 | 2 / 8.5 | 6 / 10.1 | 2 / 2 |
| c4 | 16 | 7.4 | 49.9 | 51.2 | 75 | 14,720 | 195 | 6 / 6.5 | 10 / 7.4 | 6 / 6 |
| c8 | 16 | 13.8 | 176.1 | 486.0 | 486 | 71,669 | 147 | 6 / 2.9 | 10 / 23.5 | 6 / 6 |
| c16 | 32 | 15.7 | 186.0 | 519.8 | 520 | 102,577 | 197 | 11 / 5.6 | 21 / 21.0 | 11 / 11 |

The longest requests are THINK answers at max effort (an everyday question
thought through for 486 s at c8).

Quyet in the gates:
- `quyet-route`: 310 reads.
- `quyet-judge`: 116 decision reads, 1,029 questions, p50 0.44-0.56 s, max
  1.13 s.
- No state was cut and no strict read was refused, apart from `l1`'s
  deliberate probe.

The gate datasets are short synthetic turns; DeepSWE measures real agent
turns.

## Final run: all nine gates pass

GPU run 2026-10-13 12:28-14:24 JST (UTC 03:28-05:24) at `90ab8016`: GLM pool
`unhealthy_after: 3`, F5 asks for one call first, the form check reports
without fixes (VCO-D23 amendments 2026-10-13). Storage prepared by
hard-linking the attested GLM and Quyet checkpoints; plain `./run.sh down` /
`./run.sh up`; `./verify.sh <gate>` in GATES order, stopping at the first
failure. Logs `up-20261013-r4.log`, `targeted-20261013.log`,
`gates-20261013-r4.log`; results `*-20261013T03*`/`T04*`/`T05*`.
**All nine gates pass.** GLM grammar errors 0, Kairyu errors 0; 759 Quyet
reads (9,849 questions), 0 states cut, 1 strict refusal (the `l1` probe).

Before the gates, the two turns whose fixes had failed earlier ran 10 times
each through the public model at the default effort (max, c4): 20/20 replies
carry a structured call to a declared tool, no fix ran; `reschedule` 4
replies made two calls and passed the form check, 6 made one call and
read F1 below 0.5 (published as written); `announced-no-call` 8 passed, 2
made two calls under the one-call protocol and failed F5 (published as
written). p50 267.7 s.

| Gate | Result |
|---|---|
| l1 | pass (70 s): every GLM DP rank, both Quyet replicas, the strict name refusing a 51,782-token state, a structured call from the public model |
| routing | pass: TOOL miss 5 % / 0 %; THINK precision 98.9 %, recall 94.8 % (tool-declared THINK 68.8 %); judge wall 11.1 s for 136 |
| think-route | pass: 6/6 THINK, streamed; p50 4.7 s, TTFT p50 4.4 s (default effort max) |
| effort | pass: THINK and TOOL follow the caller (default/low/high/max); TOOL 15.3-240.3 s, 1,180-23,397 output tokens, form passed each time |
| verified-tool-route | pass: 12/12 TOOL with a structured call, 10-15 candidates, judgments and the form check on `quyet-judge`, no fix; form passed 10, F1 (`reschedule`, low) and F6 (`finish-batched`, high) reported; p50 38.4 s, p95 202.5 s; 91,323 output tokens |
| fallback | pass: `quyet-route` stopped: 2/2 THINK (`backend_error`, 1.0-1.8 s) with a call; `quyet-judge` stopped: TOOL, `judge_unavailable`, published unverified with two calls (88.9 s); both back: judged and form-checked (173.2 s) |
| serving | pass: 72/72 answered, every reply a structured call (table below) |
| serving-routed | pass: 72/72 answered (table below) |
| browser | pass: Open WebUI answers |

serving (agent turns, default effort max):

| Level | Requests | p50 s | p95 s | Wall s | Output tokens | Output tok/s | TOOL n / p50 s | THINK n | Form passed | Replies with 2+ calls |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| c1 | 8 | 151.5 | 224.9 | 1,287 | 121,961 | 95 | 8 / 151.5 | 0 | 7/8 | 3 |
| c4 | 16 | 111.5 | 305.4 | 690 | 156,871 | 227 | 15 / 113.1 | 1 | 15/15 | 9 |
| c8 | 16 | 142.5 | 383.8 | 513 | 178,601 | 349 | 15 / 145.5 | 1 | 14/15 | 11 |
| c16 | 32 | 185.6 | 581.0 | 758 | 383,830 | 506 | 31 / 187.0 | 1 | 30/31 | 16 |

The three replies that failed the form check (F1 twice on `reschedule`, F7
once) were published as written, each with its calls. The THINK turn is
`dedup-case` (P(TOOL) 0.025); it made its call at every level of this run.
`deepseek-v4.1-quyet-8gpu` (DeepSeek at high effort): p50 74.8 / 65.4 / 108.7
/ 134.2 s, 175 / 433 / 751 / 955 tok/s at c1 / c4 / c8 / c16.

serving-routed (routing set):

| Level | Requests | p50 s | p95 s | Wall s | Output tok/s | TOOL n / p50 s | THINK n / p50 s | Route judge p50 s |
|---|---:|---:|---:|---:|---:|---|---|---|
| c1 | 8 | 18.9 | 99.5 | 333 | 89 | 2 / 114.1 | 6 / 17.9 | 0.13-0.15 |
| c4 | 16 | 32.8 | 151.6 | 257 | 240 | 6 / 68.1 | 10 / 8.1 | 0.14-0.18 |
| c8 | 16 | 66.5 | 209.0 | 378 | 259 | 6 / 90.4 | 10 / 24.4 | 0.19-0.39 |
| c16 | 32 | 68.7 | 254.4 | 653 | 297 | 11 / 148.3 | 21 / 21.1 | 0.17-0.70 |

## Earlier runs on the same day (before the amendments)

- Run 1 (01:25-03:08 JST, `85c172b7`): 8 of 9 gates pass; serving c16 21/32.
  vLLM ended one grammar-constrained `candidates` call with MTP ("grammar
  rejected tokens [`]}`, `<|user|>`]", once in about 100 candidates calls;
  reported upstream for MTP with `json_schema`, vllm#44006), and with `unhealthy_after: 1` the only GLM
  replica left placement, so the 10 requests waiting for it got 502. One
  `dedup-case` reply on the think route reported the task done without the
  call it still needed (1 of 3 that run). Owner decision: `unhealthy_after: 3`,
  both recorded as limits.
- Run 2 (08:46-10:08 JST, `6b45cc54`; l1, serving, serving-routed): no 502;
  serving c16 published one `announced-no-call` reply without a call: Quyet
  read F5 below 0.5 (0.16-0.23) on its one-call replies, two fixes ran and
  the last dropped the call. Replaying the 127 recorded TOOL replies through
  Quyet, an F5 that asks for one call first passed those replies and still
  failed the two-call reply under the one-call protocol. Owner decision: that
  F5.
- Run 3 (10:41-11:22 JST, `689d8874`; stopped during serving on the owner's order):
  `verified-tool-route` failed: on `reschedule` Quyet read the one-call
  reply's F1 below 0.5, and the second fix published GLM's reasoning about
  the check as text without a call. Across the runs 14 of 209 TOOL replies
  had failed their first check: fixes passed 4, left 8 unchanged at the
  limit (283-655 s) and dropped the call in 2; an F1 rewording did not move
  the misread on replay (0.098 -> 0.131). Owner decision: no fixes.
