# glm-5.3-flash-6gpu evidence

Status: **all nine final gates PASS on the committed configuration (2026-10-11, run `20261011-r2`).**

- Served-config SHA-256:
  `723c5c31cd5dfe5524a6233d80ade77e584146f49aacdd33b4f7836d325f8fef`

Hardware: 6 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB,
SM120, PCIe) out of an 8-GPU host; GPU pairs (0,1)(2,3)(4,5) each on one
NUMA node. Model: `zai-org/GLM-5.3-Flash` revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (official FP8, 73 files,
305.8 GiB), tree SHA-256
`ffc373d5dbd6f51183153bae96726694b913761480f2c0c43057b0383f051389`.
Runtime: stock `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd…` (vLLM 0.31.0,
FlashInfer 0.7.0.post1; `FLASHINFER_MLA_SPARSE_SM120` attention with the
`fp8_ds_mla` KV layout, DeepGEMM FP8 MoE). Artifacts live under
`/mnt/nvme/kairyu/model-volumes/glm-5.3-flash-6gpu/` (`run-logs/`,
`verification-results/`).

## Fit probe: four GPUs do not fit

The plan's rule: the official checkpoint fits when stock v0.31.0 with MTP
off starts at `--gpu-memory-utilization` ≤ 0.95, reports a KV pool of at
least 1,048,576 tokens, and runs 16 sequences.

| Shape | Weights / GPU | CUDA graphs / GPU | KV memory / GPU | Result |
|---|---|---|---|---|
| TP4, GPUs 0-3, 0.95 | 76.8 GiB | 4.28 GiB | 1.81 GiB | **does not fit**: a 1,048,576-token request needs 7.56 GiB (vLLM: maximum length 235,008); without CUDA graphs it would still be 6.1 GiB |
| DP6 / EP6, GPUs 0-5, 0.92 | 64.76 GiB | 4.33 GiB | 7.38 GiB | one full context needs 7.99 GiB (maximum length 966,144) |
| DP6 / EP6, GPUs 0-5, 0.95 | 64.76 GiB | 4.33 GiB | 10.29 GiB | **fits**: 1,347,016 tokens per engine (1.28 full contexts), 16 sequences; readiness passed |

Split of the checkpoint: routed experts 283.6 GiB (MTP layer experts 6.75
GiB), everything else 14.4 GiB, vision encoder 1.05 GiB. With experts split
six ways the replicated attention, KDA, shared and dense weights dominate
the per-GPU remainder. Logs: `run-logs/fit-probe-up.log`,
`fit-probe-6gpu-up.log`, `fit-probe-6gpu-up-095.log`. The owner-approved
fallback (six GPUs, one replica) is therefore used.

## L1 selection (one flag at a time)

Every candidate restarts only the vLLM service, must pass readiness (exact
`17 * 19` answers with finite log-probabilities from every DP rank, a tool
call, a red image) and the prefix-cache consistency probe (16 concurrent
exact multiplications behind one shared 4K-token prefix, cold then cached:
vllm#53912 corrupted prefix-cached MTP output on another hybrid model), then
fixed ~8K-in / 256-out rows of 64 requests. Cells: output tok/s / TTFT p50 /
TPOT p50. Selection rule: ≥ 5 % on c1 or c16 with no > 5 % loss elsewhere.

Stage 1 (`20261010T124026Z-tuning`, against DP6 / EP6, 16 sequences per engine):

| Candidate | c1 | c16 | c64 | KV / engine | Outcome |
|---|---|---|---|---|---|
| baseline | 33.1 / 1.19 s / 25.7 ms | 239.6 / 5.21 s / 43.6 ms | 401.3 / 12.89 s / 108.8 ms | 1,347,016 | reference |
| baseline-repeat | 33.2 / 1.17 s / 25.6 ms | 249.1 / 5.35 s / 42.4 ms | 428.5 / 12.87 s / 98.5 ms | 1,347,016 | noise: +0.3 % / +4.0 % / +6.8 % |
| MTP 3 | — | — | — | — | fails at start: 9.25 GiB needed for one full context, 7.6 GiB available (maximum length 842,240) |
| MTP 5 | — | — | — | — | fails at start: 9.66 GiB needed, 7.24 GiB available (734,720) |
| batch 4K | 33.2 / 1.16 s / 25.7 ms | 260.3 / 3.91 s / 44.9 ms | 368.0 / 12.18 s / 124.8 ms | 1,484,138 | rejected: c64 −11 % |
| batch 16K | — | — | — | — | fails at start: 4.03 GiB KV memory (maximum length 487,424) |
| TP2 × DP3 (16 sequences) | 37.1 / 1.05 s / 23.0 ms | 261.1 / 2.93 s / 49.6 ms | 365.7 / 13.81 s / 87.8 ms | 2,738,880 | not comparable: 3 × 16 = 48 slots queue at c64 |

Stage 2 (`20261010T141610Z-tuning`), TP2 × DP3 at the same total capacity
(3 engines × 32 = 96, as 6 × 16):

| Candidate | c1 | c16 | c64 | KV / engine | Outcome |
|---|---|---|---|---|---|
| TP2 × DP3, 32 sequences | 37.3 / 1.01 s / 23.0 ms | 272.4 / 2.90 s / 46.8 ms | 426.3 / 13.81 s / 93.6 ms | 2,713,714 | **adopted** over DP6: +12.5 % / +11.5 % / +2.7 % |
| + MTP 3 | — | — | — | — | fails at start: `custom_all_reduce.cuh:164 'invalid argument'` while the drafter's CUDA graphs are captured (the six-GPU DeepSeek example hit the same with DSpark on TP2) |
| + MTP 5 | — | — | — | — | same failure |

Stage 3 (`20261010T145331Z-tuning`), with vLLM's own switch
`--disable-custom-all-reduce` (NCCL):

| Candidate | c1 | c16 | c64 | KV / engine | Outcome |
|---|---|---|---|---|---|
| TP2 × DP3, NCCL | 37.3 / 1.03 s / 23.0 ms | 268.8 / 2.86 s / 47.9 ms | 425.4 / 13.81 s / 93.9 ms | 2,713,714 | NCCL costs nothing measurable (0 % / −1.3 % / −0.2 %) |
| + MTP 3 | 74.3 / 0.94 s / 9.7 ms | 271.7 / 2.18 s / 47.8 ms | 417.6 / 14.35 s / 89.5 ms | 2,032,690 | **adopted**: c1 +99 %, c16 +1 %, c64 −1.8 %; consistency 0 wrong |
| + MTP 5 | 68.5 / 0.92 s / 10.7 ms | 251.5 / 1.94 s / 54.8 ms | 443.8 / 15.16 s / 73.5 ms | 1,912,602 | rejected: c16 −6.4 % against MTP 3 |

These candidates ran under the names above against the DP6 command; each
run's `selection.json` keeps its exact command. `tune.py` now expresses
candidates as changes against the selected command (`no-mtp`, `mtp-5`,
`custom-all-reduce`, `dp6`, `batch-4k`, ...).

## Selected L1

TP2 pairs on NUMA-local GPUs (0,1)(2,3)(4,5), DP3, EP6 (48 experts per GPU),
32 sequences per engine (Kairyu admits 96), MTP drafting 3 tokens, NCCL
all-reduce, FP8 KV, `--max-num-batched-tokens 8192`,
`--gpu-memory-utilization 0.95`, prefix caching. Per GPU: weights 59.11 GiB,
CUDA graphs 5.83 GiB plus 0.79 GiB for the drafter, 16.79 GiB of KV memory
measured; the pool is pinned at 16.75 GiB (`--kv-cache-memory 17985175552`)
so a cold compile cache cannot shrink it: 2,028,392 tokens per engine (1.93
full contexts). Startup to ready ≈ 5 minutes with warm caches.

## Final gates on the committed configuration

Run `20261011-r2` (2026-10-11 02:00–02:42 JST; reasoning re-run 02:46 JST
after its criterion change), served-config SHA-256 `723c5c31…`, every gate
`--no-start` on the stack brought up by `run.sh up`.

| Gate | Result | Evidence |
|---|---|---|
| `l1` | PASS 16/16 | registry digest, `/version` 0.31.0, checkpoint attestation, KV pool 2,028,392 ≥ 1,048,576, sampling defaults from `generation_config.json` (temperature 1.0, top_p 0.95), `Reasoning Effort: Max/Low/High/Max` + open `<think>` for default/low/high/max, 16 exact probes per effort from all 3 engines, prefix-cache consistency 0 wrong (2 rounds × 16), tool call, red image |
| `serving` | PASS | ~8K in / 256 out, 64 requests per row, all placed: c1 70.6 tok/s (TTFT 0.94 s, TPOT 10.4 ms), c8 209.7 (1.29 s, 32.0 ms), c16 298.3 (2.28 s, 44.6 ms), c32 386.2 (3.55 s, 62.4 ms), c64 455.7 (15.06 s, 76.7 ms); TTFT p99 1.40 / 4.68 / 7.79 / 15.27 / 29.26 s |
| `completed` | PASS 80/80 | default effort (max), `max_tokens` 65536, every answer stops with content. generic c1 8/8 93.4 tok/s, end-to-end p50 8.1 s; generic c16 32/32 384.0 tok/s, 27.1 s; coding c1 8/8 103.0 tok/s, 65.4 s (content after 58.1 s); coding c16 32/32 577.0 tok/s, 169.7 s (p99 605.8 s); longest answer 38,867 tokens |
| `tool-calling` | PASS 7/7 | 32 concurrent auto calls, a tool-result turn carrying the previous reasoning, a streamed call (error events and a missing or repeated `[DONE]` fail), a call at each effort, the Chat UI filter's body |
| `vision` | PASS 9/9 | 8 concurrent red-image answers and one 8-image request |
| `reasoning` | PASS 4/4 | `323` streamed once and cleanly at default/low/high/max; reasoning at default (141 chars) and max (39 chars) |
| `cancellation` | PASS | L1 active within 15 s; Kairyu and L1 released 0.25 s after the disconnect; follow-up completed |
| `long-context` | PASS 4/4 | needle at 32,786 / 131,090 / 262,162 / 1,039,890 prompt tokens: exact key each time, 4.2 / 14.2 / 28.2 / 142.0 s |
| `restart` | PASS | L1 restart healthy and answering after 244 s |

The first full run, `20261011-r1` (served config `dd02886d…`), passed
`l1`, `serving`, `completed`, `vision`, `reasoning`, `cancellation` and
`restart` with numbers within run-to-run noise of the table above, and
failed three things, each changed with the owner's approval before `r2`:

- The Chat UI filter sent `chat_template_kwargs: {"clear_thinking": true}`
  (the model author's chat setting). Kairyu's legacy chat path rejects
  `chat_template_kwargs` on text requests: `HTTP 400 ... has no Kairyu chat
  template; chat_template_kwargs cannot be applied`, so every Chat UI text
  request failed. The filter now sends only `reasoning_effort`; the
  template's own `clear_thinking` default applies. `kairyu/` is unchanged.
- At low and high effort GLM answered the tool call without reasoning
  text (13 tokens: thinking closed at once). The case now requires the call
  at each effort; the effort reaching the template is the `l1` render check.
  The `reasoning` gate's first `r2` pass hit the same at low and high (4
  tokens, `323`, no reasoning) and now requires reasoning only at default
  and max.
- `long-context` at 131,090 tokens returned the right key inside a sentence
  (`The archive key is **K…**.`) and failed the exact-text check; the gate now
  requires the answer to name exactly the planted key as a whole identifier.

Only SHA-bound rows of run `20261011-r2` establish the claims above.
