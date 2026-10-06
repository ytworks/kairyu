# deepseek-v4.1-qwen3.8-winnow-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. DeepSeek-V4.1
DP6/EP6 on GPUs 0-5 (image `sha256:119afb09…`, the six-GPU example's SM120
overlay), Qwen3.8-27B FP8 on GPU 6 (vLLM v0.23.0), Winnow-12B Q8_0 on GPU 7
(winnow-server `77d1458` + f072b10).
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-qwen3.8-winnow-8gpu/results/`
(`*-20261006T*.json`), gate logs `gates-20261006.log`, `gates-20261006-256k.log` and
`gates-20261006-three-wave.log`.

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
