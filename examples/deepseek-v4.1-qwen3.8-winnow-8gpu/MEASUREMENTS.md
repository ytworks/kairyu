# deepseek-v4.1-qwen3.8-winnow-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. DeepSeek-V4.1
DP6/EP6 on GPUs 0-5 (image `sha256:119afb09…`, the six-GPU example's SM120
overlay), Qwen3.8-27B FP8 on GPU 6 (vLLM v0.23.0), Winnow-12B Q8_0 on GPU 7
(winnow-server `77d1458` + f072b10).
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-qwen3.8-winnow-8gpu/results/`
(`*-20261006T*.json`), gate log `gates-20261006.log`.

## GPU gates (2026-10-06, `./run.sh` then `./verify.sh <gate>`)

All nine gates pass.

| Gate | Result |
|---|---|
| l1 | every DeepSeek DP rank (thinking and chat JSON), Qwen chat, Winnow chat, Winnow System One, one verified answer |
| routing | 80 conversations, Winnow read 5.7 s in all; VERIFIED miss 0 % on both halves; everyday to THINK 96.9 %; P(VERIFIED) median 0.986-0.9996 for every accuracy category, 0.009 for everyday |
| think-route | 6/6 THINK at high, streamed; p50 0.85 s, TTFT p50 0.69 s |
| effort | THINK follows the caller (none→high, low, high, max); VERIFIED is max every time |
| verified-route | 12/12 one DeepSeek call at max (unary and streamed, every caller effort); p50 98.3 s, p95 177.1 s, 176,758 output tokens; one answer stopped at the 32,768-token request cap (`length`) |
| fallback | Winnow stopped: 2/2 routed requests answered on THINK (fallback `backend_error`), the always model answered; Winnow restarted: routed again |
| serving | below |
| serving-routed | below |
| browser | the answer page and Open WebUI answer for both models |

### serving (`kairyu-verified-always`, InFoBench instructions)

| Level | Requests | p50 s | p95 s | Output tok/s | Requests/min |
|---|---:|---:|---:|---:|---:|
| c1 | 8 | 18.2 | 45.2 | 124.8 | 2.38 |
| c4 | 16 | 10.5 | 25.5 | 386.0 | 14.75 |
| c8 | 16 | 24.9 | 70.9 | 380.8 | 6.92 |
| c16 | 32 | 25.6 | 71.8 | 652.1 | 15.97 |

### serving-routed (`kairyu-verified`, routing set)

| Level | Route | Requests | p50 s | p95 s | Output tokens | Judge p50 s |
|---|---|---:|---:|---:|---:|---:|
| c1 | think | 1 | 0.36 | 0.36 | 29 | 0.089 |
| c1 | verified | 7 | 14.0 | 35.0 | 17,277 | 0.077 |
| c4 | think | 6 | 1.26 | 1.63 | 791 | 0.072 |
| c4 | verified | 10 | 42.9 | 105.5 | 84,222 | 0.084 |
| c8 | think | 5 | 2.58 | 3.27 | 631 | 0.154 |
| c8 | verified | 11 | 41.5 | 55.0 | 42,991 | 0.075 |
| c16 | think | 16 | 3.05 | 4.55 | 2,032 | 0.071 |
| c16 | verified | 16 | 78.4 | 251.2 | 163,576 | 0.254 |

Every request answered at every level.

The evidence of the checklist-verified configuration this example replaced
(OpenJev judge, checklist DAG) is in this file's history before VCO-D18, as
`examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md` at `df109a6b`.
