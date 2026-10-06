# Verified route: keep `requirements` inside Qwen's context

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641).
Follows `2026-10-06-winnow-verified-three-wave.md` (VCO-D19), GPU-verified on
single-turn requests. Status: awaiting owner approval.

## 1. Goal

On multi-turn agent conversations every VERIFIED turn runs all three waves:
Qwen always accepts the `requirements` request, so Winnow can judge the
drafts. Today long conversations exceed Qwen's 262,144-token context and the
turn is answered without requirements and judgments.

## 2. Owner requirements for this fix

- The cause is Qwen and its token limit; the fix is minimal and does not
  widen the problem.
- Kairyu keeps emitting each turn's stage report in `reasoning_content`, and
  the caller keeps sending earlier turns back; the conversation Qwen reads
  keeps that information.
- No framework (`kairyu/`) change: the example fixes its own deployment.
- Plan only; no benchmark work in this plan.

## 3. Facts (DeepSWE r1 `deepswe-verified-3wave-full-4w-20261006-r1`, recorded turns)

| Fact | Value | How measured |
|---|---|---|
| VERIFIED turns / with judgments / without | 132 / 61 / 71 | trace v2 of every recorded call |
| `requirements` failed (no list) | 67 turns, plus 2 empty lists | trace v2 |
| Qwen input on the 67 failed turns | 199,082-325,598 tokens (median 267,676) | Kairyu's exact rendered request, live Qwen `/tokenize` |
| Qwen's reply on those turns | HTTP 400 "maximum context length is 262144 tokens … 65536 output tokens … at least 196609 input tokens" | live replay (Qwen's access log is off, so it was not in its log) |
| Composition of one conversation (14:42 turn, 61 messages) | 240,650 tokens = earlier `reasoning_content` 189,953 + content 33,852 + tool calls 3,845 + framing | `/tokenize` of each part |
| Growth | 61 messages → 240,650 tokens; 95 messages → 320,460 tokens (~2.3K tokens per message) | `/tokenize` |
| `requirements` output when it succeeds | max 3,486 tokens, median 1,899 | trace v2 (61 calls) |
| The two empty lists (148K / 181K inputs) | Qwen ended inside its reasoning after 188 / 342 tokens; the same inputs replayed 4 times all returned valid lists | live replay |
| Qwen's context per its model card | 262,144 natively; "extensible up to 1,000,000 tokens" with YaRN; vLLM: `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `--hf-overrides` `rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 262144`, `--max-model-len 1000000`; static YaRN "potentially impacting performance on shorter texts" | `models/qwen/qwen3.8-27b-fp8/README.md` |
| DeepSeek's context in this example | 1,048,576 | `kairyu.yaml` |
| Qwen KV cache on GPU 6 | 1,791,840 tokens | Qwen startup log |

## 4. Cause

Qwen's context (262,144) is smaller than the conversation the example gives
it. The other roles run on DeepSeek with 1,048,576; Qwen is the only worker
whose context is below the conversation sizes this route produces, so it
fails first (from about turn 20 of a DeepSWE problem in r1).

## 5. Options inside the example

| Option | Against the requirements |
|---|---|
| Lower the `requirements` output cap (65,536 → 4,096) | Input alone exceeds 262,144 on 38 of 67 turns; fixes 29. Not enough |
| Stop exposing stage reports | Removes the information the requirement keeps. Rejected |
| Run `requirements` on DeepSeek | Changes the owner's design (Qwen lists requirements). Rejected |
| Add a summary stage for Qwen | Widens the DAG and drops information. Rejected |
| **Qwen at its documented 1M context (YaRN factor 4)** | Qwen reads the whole conversation; matches DeepSeek's 1M, so Qwen is no longer the first role to overflow; one service's flags. **Chosen** |
| YaRN factor 2 (524,288) | Covers the 67 recorded turns (max 325,598 + 65,536) but conversations keep growing (~2.3K tokens per message): overflow again near 150 messages, while recorded DeepSWE conversations reached 714. Not chosen |

## 6. Changes (example only)

| # | Layer | File | Change |
|---|---|---|---|
| 1 | L1 | `examples/deepseek-v4.1-qwen3.8-winnow-8gpu/compose.yaml` (service `qwen`) | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; `--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'`; `--max-model-len 1000000` (the model card's values; the other `rope_parameters` are the checkpoint's own) |
| 2 | L1 | `examples/deepseek-v4.1-qwen3.8-winnow-8gpu/kairyu.yaml` (pool `qwen3.8-27b`) | `max_model_len: 1000000` |
| 3 | docs | `README.md` (L1 row), `docs/design/example-verified-checklist-orchestration.md` (VCO-D19 note), `PROGRESS.md` | Qwen at 1M with YaRN, why |

No change to `kairyu/`, prompts, caps, efforts, the DAG, routing, Winnow or
the stage report.

## 7. Tests

None added or changed: the change is deployment configuration; its effect is
checked on GPU (test policy). `test_gateway_builds_from_the_example_configs`
already loads the edited `kairyu.yaml`.

## 8. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: `uv run ruff check .`; `tests/unit/test_deepseek_v41_qwen38_winnow_example.py` | green | 5 min |
| 2 | Redeploy with plain `./run.sh` (Qwen restarts with YaRN) | healthy; L1 probes pass; Qwen log reports max model len 1,000,000 | 20 min |
| 3 | Live replay of the 67 failed turns' `requirements` requests (exact rendered requests, 4 at a time) | all 67 accepted; 67 valid lists; 0 empty | 40 min |
| 4 | All nine GPU gates in GATES order (l1, routing, think-route, effort, verified-route, fallback, serving, serving-routed, browser) | each gate's existing criteria; verified-route also shows Qwen's lists unchanged in size on short requests (5 + 5 x N judgment items, N ≥ 1) | 2 h |
| 5 | New gate `long-conversation`: five r1 turns whose Qwen input was 199K-326K, through `kairyu-verified-always` | 200; requirements and judgments succeed on all five | 20 min |

Stop and report at the first failure.

## 9. Risks and limits

- Static YaRN may lower Qwen's quality on short inputs (model card). Qwen
  only lists requirements; the gates in step 4 run short requests through it.
- Four long `requirements` calls together can exceed the 1,791,840-token KV
  cache; vLLM then queues them (slower, not failing).
- Conversations still grow by every turn's report: at ~2.3K tokens per
  message, Qwen (1M with the 65,536 cap) and DeepSeek (1,048,576) both
  reach their contexts near 350 messages; beyond that the request exceeds
  both. Recorded DeepSWE conversations on this
  host reached 714 messages (DeepSeek-only run, smaller messages).

## 10. Decision requested

Approve changes 1-3 and the verification above.
