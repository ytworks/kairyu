# DeepSeek V4.1 Flash 1017 examples on six and eight GPUs, and one DeepSWE pass

Status: **Approved 2026-10-19 (owner); implemented and GPU-verified 2026-10-19; DeepSWE 79/113 recorded in the kairyu-bench wiki.**

Accepted scope (owner, 2026-10-19):

- Two new examples for `deepseek-ai/DeepSeek-V4.1-Flash-1017`:
  `examples/deepseek-v4.1-flash-1017-6gpu/` (GPUs 0–5, API 8017, Chat UI
  3017) and `examples/deepseek-v4.1-flash-1017-8gpu/` (GPUs 0–7, API 8018,
  Chat UI 3018). Public model `deepseek-v4.1-flash-1017`.
- L2 and L3 are the existing V4.1 examples' structure, unchanged: one
  `ReplicaPool` replica, the Kairyu API, Open WebUI over `/v1/responses`
  with the effort filter, `upstream_*_models`. Only L1 is researched and
  tuned. The L1 must be the fastest measured setting without lowering model
  accuracy.
- Every script, overlay file and test is the example's own (no sharing
  with `deepseek-v4.1-flash-6gpu` / `-8gpu`, which stay unchanged).
  `kairyu/` and kairyu-bench are not changed.
- After both examples pass their gates: DeepSWE v1.1, all 113 tasks, one
  attempt, 8 workers, reasoning effort `max`, on the six-GPU example; the
  result goes to the kairyu-bench wiki (DeepSWE-Comparison).
- All storage (checkpoint, HF/xet cache, images, logs, benchmark data) is
  on NVMe under `/mnt/nvme`.

## Facts

1. 1017 has the 0909 architecture: `config.json`, tokenizer files, the
   96,085 tensor names and the 510.3 GB total are identical; only the
   weights and `encoding/encoding.py` changed (HF revision
   `1bae0e1088993b9b82de08f59de5be5a34e42170`).
2. 1017 encoder change: a user message holding only tool results no longer
   starts a new turn, so without a tools declaration the reasoning of the
   current turn survives tool calls; tool-result content may carry content
   blocks (images) and user content may be a list. vLLM's encoder copy
   (nightly 2026-09-30, v0.31.0, v0.32.0) still has the 0909 behavior.
   With a tools declaration (DeepSWE's bash tool) all reasoning is kept
   either way.
3. Effort: the 1017 encoder maps low 50, high 75, max 100, default high.
   vLLM adds `"xhigh": 75`; Kairyu spells L3 max as `xhigh` on the
   Responses wire (m11 D4 amendment 2026-10-17), so the overlay maps
   `xhigh` to 100 as in the existing examples. Chat callers (DeepSWE)
   reach vLLM with `max` (100).
4. Sampling: the 1017 model card recommends temperature 1.0, top_p 1.0
   (the 0909 card allowed 0.95 or 1.0; `deepseek-v4.1-flash-6gpu` uses 0.95).
5. vLLM v0.31.0 (2026-10-05) and v0.32.0 (2026-10-18) followed the
   2026-09-30 nightly. v0.32.0 adds faster CPU-offloaded Engram lookups
   (#59327), packed FP8 Engram exchange across DP (#59923), SWA bounded
   replay trimming (+3.6 %), SM120 split-K attention (#58482) and DSpark
   fixes. None of the six SM120 edits of `deepseek-v4.1-flash-6gpu`'s
   overlay is upstream in v0.32.0 (64-token SWA/MLA/indexer pages, the
   V4.1 MXFP4 indexer gate, FlashInfer C2 prefill pages, the masked
   sparse-KV zero row, the split top-p guard); their anchors exist in the
   v0.32.0 sources. FlashInfer stays 0.7.0.post1.
6. vLLM recipe (1017 is its default checkpoint): Blackwell uses
   `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer, 128 sequences;
   the memory-bound arm offloads Engram to host memory with 4,096 batched
   tokens and 0.92 memory; the DEP kernels are SM100-only (start failure
   recorded in `deepseek-v4.1-flash-6gpu`), but its
   `engram-config {"embedding_across_dp": true}` (Engram tables sharded
   over DP ranks in HBM) fits on eight GPUs. Also listed:
   `--prefill-schedule-interval 64`, CUDA graph capture 128 × (1 + 5) →
   1024, Engram `use_thp`.
7. Measured on this host (output tok/s c1/c8/c32/c64): the six-GPU DP6/EP6
   example 103.9 / 291.0 / 586.7 / 825.1; the eight-GPU TP8 example on the
   0909 image 126.8 / 275.4 / 326.8 / 314.3. DP8/EP8 was never measured.
8. Previous DeepSWE (0909, six GPUs, max, 8 workers,
   `deepswe-deepseek-v4.1-flash-6gpu-max-8w-20261013-r1`): 85 of 113 (two
   verifier timeouts count as not correct), about 15.4 h.

## Example contents (each example)

`control.py`, `run.sh`, `verify.sh`, `verification.py`, `tune.py`,
`benchmark.py`, `patch_sm120.py`, `check_sm120_kernels.py`,
`vllm-sm120.Dockerfile`, `compose.yaml`, `kairyu.yaml`, `example.json`,
`model-manifest.json`, `webui-reasoning-effort-filter.py`, `README.md`,
`MEASUREMENTS.md`; tests in `tests/unit/test_deepseek_v41_1017_<n>gpu_example.py`.

Overlay (`patch_sm120.py`, exact anchors of the selected vLLM version,
fail-closed): the six SM120 edits, `xhigh` → 100, and the 1017 encoder rule
(tool-result-only user messages do not start a turn; tool-result content
blocks; list user content). L1 defaults: temperature 1.0 / top_p 1.0;
default effort high (75).

Tests (test policy): only the existing contracts that apply to the new
examples — the three config files agree and the gateway validates them,
readiness answer checks, every-DP-engine coverage, rendered effort numbers
(max and xhigh 100), overlay anchors fail closed (including the encoder
rule). The shared Open WebUI filter test already covers every example's
filter.

## L1 selection (GPU)

Rule (as in the six-GPU amendment): one change at a time; adopt only with
≥ 5 % output throughput at c1, c8 or c32 and no > 5 % loss elsewhere;
correctness first. Rows: 32 requests of ≈ 8K unique input, exactly 256
output tokens, thinking high, at c1/c8/c32.

Six GPUs (DP6/EP6, Engram offload, 4K batch, 0.92, DSpark 5, Marlin, FP8
KV, MXFP4 indexer):

- S0: `check_sm120_kernels.py` 28/28 on each image.
- S1 image: A = vLLM v0.32.0 + overlay, B = the 2026-09-30 nightly +
  overlay. Both must pass `verify.sh l1` (exact `323` on every DP engine,
  finite log-probabilities, efforts 75/50/75/100 and xhigh 100, vLLM's
  rendering equal to the checkpoint's `encoding.py` on its test inputs and
  a tool-result conversation, tool call, image). Faster wins; within 5 %
  A is adopted.
- S2 on the adopted image: adaptive verification (only if it starts),
  `--prefill-schedule-interval 64`, `--max-cudagraph-capture-size 1024`,
  Engram `use_thp`, batch 8K.
- S3: the adopted levers together.

Eight GPUs (adopted image of S1):

- S1 shape: (a) DP8/EP8 + Engram offload, (b) DP8/EP8 +
  `embedding_across_dp`, (c) TP8/EP8; DSpark 5 in all. Selected by c8
  throughput (DeepSWE's concurrency), then c32; c1 recorded. TP2 × DP4 is
  not measured (TP2 × DP3 was 44 % slower on six GPUs and TP2 + DSpark
  failed CUDA-graph capture).
- S2: the applicable six-GPU S2 levers, batch 16K, memory 0.90/0.92.

## Gates (each example, final configuration, once)

`l1`, `serving` (c1/8/16/32/64, 64 requests each), `completed`
(generic/coding c1/c8/c32, 32 each, `max_tokens` 65,536), `tool-calling`,
`vision`, `reasoning`, `cancellation`, `long-context` (32K/128K/256K/≈1.04M),
`restart`, `api-equivalence`, Chat UI browser pass. Pass criteria as in the
existing examples. Stop at the first failure, fix every known risk
together, prove the fixes with targeted checks, then one full run.

## DeepSWE (six-GPU example)

- Pre-check: the previous run's second agent turn (bash tool, effort
  `max`) five times through Kairyu returns five structured calls; the
  Kairyu-to-vLLM body carries `reasoning_effort: max`, vLLM `/tokenize`
  renders `Reasoning Effort: 100` and counts the same prompt tokens.
- Run with existing kairyu-bench options only: workers 8, attempts 1,
  retries 3, API timeout 3600 s, effort `max`, all tasks, dedicated
  Docker-in-Docker daemon and usage relay as on 2026-10-13. Run ID
  `deepswe-deepseek-v4.1-flash-1017-6gpu-max-8w-<date>-r1`.
- Only tasks lost to infrastructure or network failures are re-run
  (`KAIRYU_BENCH_DEEPSWE_TASKS`); timeouts stay not correct.
- Wiki: results first ("DeepSeek-1017 6GPU max" next to "DeepSeek 6GPU
  max" 85/113), then API calls (L3/L1), tool calls, input and output
  (visible/thinking) tokens, per-reply averages and time, then conditions;
  counted as for the 2026-10-13 run.

## Order

1. This plan + draft PR.
2. Both examples, tests, docs (examples/README.md, FN-D9 1017 amendment,
   PROGRESS.md); ruff and the changed paths' CPU tests.
3. Checkpoint download once (476 GiB); the eight-GPU example's storage is
   hard-linked before its `run.sh` (`DEEPSEEK_MODEL_SEED`).
4. Six GPUs: L1 selection, gates, MEASUREMENTS.md.
5. Eight GPUs: L1 selection, gates, MEASUREMENTS.md.
6. DeepSWE on the six-GPU example, then the wiki.
7. Report; merge after CI is green, by the owner's decision.
