# deepseek-v4.1-qwen3.8-winnow-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. DeepSeek-V4.1
DP6/EP6 on GPUs 0-5 (image `sha256:119afb09…`, the six-GPU example's SM120
overlay), Qwen3.8-27B FP8 on GPU 6 (vLLM v0.23.0), Winnow-12B Q8_0 on GPU 7
(winnow-server `77d1458` + f072b10).
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-qwen3.8-winnow-8gpu/results/`
(`*-20261006T*.json`), gate logs `gates-20261006.log` and `gates-20261006-256k.log`.

## GPU gates (2026-10-06, `./run.sh` then `./verify.sh <gate>`)

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

### serving (`kairyu-verified-always`, InFoBench instructions)

| Level | Requests | p50 s | p95 s | Output tok/s | Requests/min |
|---|---:|---:|---:|---:|---:|
| c1 | 8 | 23.5 | 47.5 | 131.7 | 2.46 |
| c4 | 16 | 12.6 | 41.4 | 424.3 | 13.56 |
| c8 | 16 | 21.9 | 79.3 | 445.9 | 8.05 |
| c16 | 32 | 24.6 | 68.0 | 736.8 | 21.86 |

No answer reached the cap (`length`: 0 at every level).

### serving-routed (`kairyu-verified`, routing set)

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
