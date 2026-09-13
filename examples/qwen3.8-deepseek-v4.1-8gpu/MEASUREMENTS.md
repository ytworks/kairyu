# V4.1 ensemble evidence

Status: **GPU validation in progress.**

Current scope (V41E-D13): implementation behavior and existing serving
measurements. Model-answer quality, checklist completeness and audit accuracy
are not implementation completion gates. The remaining work is a bounded
image-rendering integration check, the public serving measurements, applicable
lifecycle checks and final documentation/CI. Reuse the completed effort/native
evidence when the relevant implementation is unchanged.

The user released all eight GPUs on 2026-09-13 (JST). GPU trials are recorded
below. The original CPU-only implementation and the sibling TP8 results do not
establish GPU correctness for this example.

## CPU and static evidence

Local validation on 2026-09-12: **202 tests passed** in one invocation:

```sh
python -m pytest tests/unit/test_v41_ensemble_*.py \
  tests/unit/test_tiered_frontier_examplectl.py \
  tests/unit/test_deepseek_v41_example.py \
  tests/unit/test_replica_examplectl.py --no-cov -q
ruff check .
python scripts/check_progress_size.py
```

After the GPU-driven implementation changes, the expanded selected suite passes
**289 tests** on 2026-09-13, including the native/L2 probe, capacity alias
collector, cancellation, candidate output reservation, kernel patch and runtime
attestation tests. Whole-repository Ruff,
progress-size and whitespace checks pass. Review reproduced and fixed a capacity
false pass where an altered context limit retained the original config-hash
environment variable; capacity now checks the actual entire DeepSeek command.

Docker Compose `config --quiet` passed using the lifecycle-generated environment
without starting containers. Configuration digest agreement and verification
command listing passed. An independent review's stale-deployment provenance
finding was fixed and re-reviewed; no actionable P1/P2 remained in that review.
The full repository test suite is left to CI; the tests above are the selected
local regression set. These are CPU/static checks only.

The subsequent separator and disconnect-ownership changes pass **506 selected
CPU tests** in one invocation (`test_v41_*.py`, `test_conductor*.py`, `test_dsl.py`,
`test_sse_response_cancellation.py`, `test_sse.py`, `test_orchestrator.py`).
Seven new ASGI tests include interrupted sends under ASGI 2.3/2.4 and pending
backend cleanup. Independent review also checks fragmented streaming/whole-text
separator equivalence. Whole-repository Ruff and progress/whitespace checks pass.

- The fixed existing V4.1 image and checkpoint source were inspected read-only.
  Exact image, source paths/hashes and constraints are in [L1-NOTES.md](L1-NOTES.md)
  and `example.json`. This is provenance and a static feasibility assessment.
- CPU contracts exercise deployment loading and Compose rendering; the real
  Conductor's effort/dependency/image/repair behavior with scripted engines;
  native OpenAI-compatible tool/image forwarding with an HTTP mock transport;
  role-specific mode/schema/budget middleware; lifecycle isolation and runtime
  attestation; stage coverage and measurement provenance.
- Requirement now applies a DeepSeek high floor and preserves canonical API max
  (V41E-D10); the earlier fixed-high evidence below remains historical. Qwen
  retains its existing effort mapping. The removed image-description stage is
  not part of the new DAG.

## GPU startup investigation

Hardware: 8 × RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, PCIe;
1 TiB host RAM. Initial trials use parent `027bf47b2bd6`; the selected child
image in the initial corrected runs is `8b8bcb200f3a` (full ID below).
The subsequent seeded top-p correction uses `18dad57d5b3d`, pinned in example.json.
The previous DeepSWE run completed before its TP8 service was stopped.

- `20260912-gpu-startup`: initial TP2/DP3/EP6, GPU-resident Engram, no DSpark,
  16K batching, 1M context and 0.90 memory utilization. Model loads at
  84.09 GiB/GPU, but sparse-indexer profiling fails allocating 512 MiB with
  only 377 MiB free. Full failure retained in `initial-worker.log`.
- `20260912-gpu-offload-startup`: only Engram CPU offload changes. GPU model
  memory becomes 52.62 GiB; two 15.74 GiB tables per rank consume about
  189 GiB pinned host RAM. Graph profiling completes and reports 21.42 GiB
  available KV/GPU and about 8.53M cache tokens per internal DP engine.
  API and correctness gates are still in progress; these capacity reports
  alone are not successful long-context requests.
- The offloaded service starts, but repeated short native requests expose NaN
  log probabilities, corrupt text and a tool grammar error. A first correct
  arithmetic answer was not sufficient evidence. Eager execution
  (`20260912-gpu-eager-startup`) and disabling custom all-reduce
  (`20260912-gpu-nccl-startup`) each reproduce the failure. Eager concurrent
  requests targeting all three DP ranks fail 10 of 12 cases.
- `20260912-gpu-finite-diagnostic`: a separate temporary image adds finite
  checks around Engram, Attention and MoE. The first nonfinite tensor is layer
  0's Attention output on both TP ranks of the second request. This diagnostic
  image is not the measured serving candidate.
- An independent GPU-6 sparse-attention oracle poisons unused KV rows with
  NaNs: all 12 cases containing masked `-1` indices produce NaNs, while all
  36 control cases remain finite. The independent reference remains finite
  in all 48 cases. Invalid indices are clamped to slot zero by the original
  gather; a zero attention weight cannot eliminate `0 * NaN` contamination
  in the value product. Raw diagnostics and failed builds are retained in the
  persistent kernel-investigation directory below.

## Masked-KV correction

Selected child image:
`sha256:8b8bcb200f3a77f289f2483526903df34c7f7b15f58691d8ab27c0ccb15e2eaf`.
The parent image and sibling example remain unchanged. Invalid gathers use an
aligned zero device row; valid addresses, transfer sizes and barriers are
unchanged. Dedicated decode, common gather/footer and prefill pointer paths are
covered. CUDA compilation exposed missing declaration visibility and a larger
DOTS3 row ABI in the first two builds; both failed builds are retained. They
were never accepted for serving verification.

- Paired oracle: **48/48 pass**, zero nonfinite outputs, **48/48 bitwise equal**
  to the original runtime's corresponding finite-cache outputs. The old runtime
  passes 36 controls and fails all 12 poisoned masked cases on the same inputs.
- Dimensions cover 8/32 heads (TP8/TP2), 1/22/48 query tokens, extra page size
  32/64, masked/unmasked and zero/NaN unused cache. Independent dequantization and
  attention use unchanged `atol=0.05, rtol=0.05`; maximum absolute error is
  0.0110161. These paired cases scale source amplitude by 0.25 to isolate the
  masked-cache failure from unrelated random FP8 tolerance outliers.
- The sibling's unchanged full-amplitude **16/16 numerical cases also pass**.
- SP communication diagnostic: 168 all-gathers and 168 reduce-scatters per rank
  match exactly. On this PCIe host these SP operations already use NCCL fallback;
  the test does not claim to exercise the custom communication implementation.

Raw kernel evidence is under `20260913-kernel-investigation/` relative to the
evidence root below (57 files, 11,891,234 bytes). Copies preserve all original
artifact hashes. This establishes the isolated kernel fix; full-model native,
composed L2, context and performance gates remain separate.

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `2a9c66eafd5bfba3e6127ebd19a7eaad5272c77ecb0750eca1e18ba87ad32d21` |
| `v41-masked-kv-fix-20260913/final-summary.json` | `843233132c42c2781e887d059b6aa18a65b61023526b6d44c8b6231730254c82` |
| `v41-masked-kv-fix-20260913/patched-v3-results.json` | `99f65990f6a7bb100e3ae34ed296a5ddd5139f59c34e820bdc8a3c5215cc7ad8` |
| `v41-masked-kv-fix-20260913/baseline-results.json` | `f7922e899d869059f8ff48a7545c8113be2c61013b279bcbf1250ce158948438` |

Raw trial directories are under
`/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4.1-8gpu/verification-results/`.
Each retains git state, hardware inventory, candidate spec/config hash and
complete worker logs.

## Full-model native and initial L2 evidence

`20260912-gpu-masked-kv-final-startup` runs the selected child with normal CUDA
graphs and the shipped collective settings. DeepSeek and both Qwen replicas
start successfully; Qwen loads 28.43 GiB per GPU. The native suite passes
**30/30 cases**: default/non-thinking on each DP rank, then default/low/high/max,
image identification, tool call/result, streaming, 12 Requirement effort
combinations, client cancellation/recovery and a forced thinking-budget case.
All 86,856 returned sample/top log probabilities are finite; no response hits
its total cap. This is the first successful full-model evidence after the
masked-KV fix. Client cancellation alone does not prove worker cleanup.

The 12 Requirement cases cross API omitted/low/high/max with nested
omitted/low/max. Each matches exactly one fixed-high hook record by message
SHA256 and its strict request time window. A separate native 16-token thinking
budget reports exactly 16 reasoning tokens and completed public content.

| Native artifact | SHA256 |
| --- | --- |
| `native-summary.json` | `a97bd034ba881f8b835736bde4e19ffd9c57f60c695f424cd52fb73654d71f25` |
| `requirements-effort-correlation.json` | `2fe35dcc9953e8b497865e4d4f38d819949eea5908ed325a260cbed289522eff` |

`20260913-l2-initial` returns 11/11 protocol successes and observes all five
routes. However, detailed candidate inspection **fails primary completion**:
the Qwen draft reaches 2048 tokens mid-sentence and answer_1 consumes all 4096
tokens in reasoning with an empty body. The final 208-word answer satisfies
this fixture's constraints and passes audit, which does not excuse the missing
peer. The revised hook reserves body tokens without changing Qwen effort, and
the revised probe requires completed nonempty draft and all three peers.
The headed response also lacks whitespace before its first section; this is
tracked separately from semantic correctness. The simple image and JSON cases
route directly, so they do not establish primary image/headless behavior.

The first native capacity run (`20260913-native-capacity`) reports 31/32 at c1.
Its failed row is a complete 256-token all-thinking response: native V4.1 uses
`delta.reasoning`, which the shared benchmark collector ignores. An example-local
adapter now recognizes this alias while preserving original raw SSE. The saved
row reparses correctly, but its old TTFT/TPOT values cannot be recovered from
untimed SSE and are excluded. A fresh run was required; the corrected rerun
is recorded below, and the original failed run is retained. Fixed-output throughput includes reasoning tokens and uses
`ignore_eos`; it is not a completed-answer quality measurement.

Open WebUI was exercised in a browser on port 3008: the effort valve displays
server default high and default/low/high/max options; default was restored after
inspection. A synthetic request displayed exactly `UI_READY_598`. This manual
observation establishes basic UI connectivity, not every effort's downstream
wire mapping.

## Measured native capacity and long-context retrieval

`20260913-native-capacity-reasoning` is the fresh run using the corrected
example-local reasoning-alias collector. The served runtime is recorded at
`a383b35f1b066004f27f6d152f23390cd0df3c5b`; the collector correction is committed
in `533f1172b1d276ff7a4343884d56126cc769eaa4`. The measured configuration SHA256 is
`17ecdd785c67d0059da78fff6595d2161b7fd1822e65c0fe473853cf4d44923c`.
Keep this distinction when replaying on another machine: these are measurements
of that attested native runtime, not evidence that later composed L2 changes
have passed. Paths below are relative to the persistent evidence root above.

All **128/128 fixed-output requests pass**, 32 requests at each concurrency.
Each generates exactly 256 tokens. These rows use `ignore_eos` and count both
reasoning and public tokens; an entirely thinking response is a valid capacity
sample. Model TTFT records the first reasoning or content delta. Public-content
TTFT remains absent when no public content appears, rather than substituting the
end of the stream. Throughput therefore does not measure completed-answer quality.

| Concurrency | Requests passed | Generated tokens/s |
| --- | --- | --- |
| 1 | 32/32 | 55.308 |
| 8 | 32/32 | 199.218 |
| 16 | 32/32 | 287.128 |
| 32 | 32/32 | 372.618 |

All **5/5 native retrieval probes pass** with the exact archive key and
`finish_reason=stop`. Prompt counts below are the actual processed counts
reported by the server, including its chat rendering, not raw dataset estimates.

| Retrieval case | Actual prompt tokens | Elapsed seconds |
| --- | --- | --- |
| 8K | 8,222 | 3.283 |
| 32K | 32,798 | 5.680 |
| 128K | 131,102 | 17.024 |
| 256K | 262,174 | 33.532 |
| Near 1M | 1,039,902 | 234.154 |

This establishes the tested native DeepSeek context sizes and fixed-output
concurrency envelope. It does not establish arbitrary long-context task quality,
public ensemble capacity at those lengths, or Qwen's configured 256K capacity.

| Capacity artifact | SHA256 |
| --- | --- |
| `20260913-native-capacity-reasoning/measurements.json` | `37c8e733c09f540d2914ec281a22e0105a9ac794786eba145dc49ea2aad88a55` |
| `20260913-native-capacity-reasoning/provenance.json` | `8fa72e415776c4322a93cd7060f9acc8d107254d193daecce957c05f6c22b44a` |

## Measured native cancellation cleanup

`20260913-native-cancellation` passes **3/3 targeted native DP-rank cases**.
For each case, public content was observed while the target rank's running
request gauge was 1. The client then closed the unfinished stream. All three
engines' running-request gauges reached 0, and a subsequent inference returned
exactly `323` with `finish_reason=stop`.

| Target DP rank | Observed cleanup milliseconds |
| --- | --- |
| 0 | 1,169.669 |
| 1 | 1,157.314 |
| 2 | 1,446.242 |

These times include the probe's one-second stable-zero check; they are not pure
engine abort latency. Unlike the earlier client-close-only smoke, this run has
worker-gauge evidence of native cleanup. It does not establish cancellation
propagation through public Kairyu/L2, nor completed-answer quality. The runtime
remained unchanged after the native runs, and the health endpoint returned 200.

The cancellation manifest records 29 evidence files. Preserve the raw metrics,
request/stream artifacts and recovery responses when copying the run elsewhere.

| Cancellation artifact | SHA256 |
| --- | --- |
| `20260913-native-cancellation/manifest.json` | `4e742a582a43314199376783cdf4d9ef95e85b57708046507f1cb69a36b1ef6b` |
| `20260913-native-cancellation/results.json` | `3315bbff7062b8647b388e848224e04cf74ffccd307c4110a072ef9d437210a0` |

## Reserved-candidate replay and follow-up defects

`20260913-reserved-startup` deploys `533f1172` with configuration
`9a834ec7b65307518ce2ab81071136bcecf6b91d64fc7859b77002a710be19ee`.
All services become healthy after normal recreation, and the private compaction
key fingerprint and mode remain unchanged. The restart-native probes pass,
including max API effort with nested low Requirement and forced budget closure.

`20260913-reserved-gates/l2-replay` observes all five routes. The primary replay
fixes candidate completion: draft uses 1405/2048 tokens, answer_1 2472/4096,
answer_2 2467/4096 and critique 2912/8192, all with nonempty complete bodies.
Exactly two policies use constraint-satisfaction and governance perspectives.
The final 248-word memo reaches the correct infeasibility decision, gives two
explicit conditional next steps and ends exactly `Decision: defer.`. One audit
PASS covers R1–R12; no repair attempt is exercised by this replay.

Detailed review nevertheless **rejects the head seam**: `constraint.Facts:`
has no whitespace. The Markdown-only probe falsely passes this plain-text
variant. Prompt instructions alone are insufficient; V41E-D5 adds an opt-in
deterministic separator and the probe gains the observed regression case.
Requirement R8 also ends its criterion at `Ends with exactly the literal `,
although the final/audit retain the required ending from original context.
This is a checklist-quality failure. A prompt candidate avoids decorative
quotation marks and literal checks now require the actual string in the
acceptance criterion, rather than only its source attribution.

Response SHA256: `9af9680dc7339595b509d3159e11374896098182e411690360a24c1d614ba205`.
The subsequent effort matrix was stopped by the operator at
`2026-09-12T18:07:50.493Z` before spending further time on the known seam
defect; its partial artifacts and `operator-stop.json` are retained.

That stop exposes a separate public cancellation failure: Qwen-0's audit,
started at 18:06:26, continues until 18:10:09, about 139 seconds after the
public connection ends. The all-worker idle gate refuses to start a subsequent
context test and records 70 snapshots. Native Qwen unary and exact-audit-hook
cancellation both clear in about 271–273 ms, with stable-zero confirmation by
1.05 seconds (`20260913-qwen-unary-cancel`). A CPU ASGI reproduction isolates
unclosed streaming-iterator ownership; a core fix and public GPU replay are
in progress. Native cancellation success must not be substituted for this gate.

## Requirement literal diagnostic

`20260913-requirement-literal-diagnostic` compares the original prompt and a
candidate that avoids decorative quotation marks, using native fixed-high
DeepSeek, the actual JSON schema, seed 10 and a 4096-thinking/8192-total cap.
This is a prompt-only diagnostic on the `533f1172` runtime, not proof of the
subsequent deployed hook. The candidate preserves `Decision: defer.` and
`Ready.` in complete criteria. The original decision case exhausts its total
cap with an empty body; both Ready cases pass. Both prompts fail the harder
literal `A|"B"` followed by a newline and `DONE.`: their valid JSON criteria
stop at `A|`, well below the output cap. These failures remain recorded rather
than being counted as checklist-quality passes. Parser/grammar isolation and
post-deployment literal checks follow separately.

CPU isolation subsequently identifies XGrammar 0.2.6's `minLength` lowering as
the cause: its generated character class excludes backslashes, so valid JSON
escapes fail both character-level and native-token matching. The installed
DeepSeek V4.1 parser preserves exact escaped strings in unary and fragmented
streaming paths. V41E-D7 removes `minLength` from the three free-text fields;
smoke/quality validators retain nonempty checks, while the runtime grammar now
permits empty strings. Post-deployment literal replay remains required.

| Literal diagnostic artifact | SHA256 |
| --- | --- |
| `results.json` | `0602b278644b957d5028d88543140a737b0b6531544828051dd23e538280778d` |
| `manifest.json` | `85054b81ed4d1086375a9fd5b6977d445e17aa06db5bbd1ac15508b3485c9cca` |

## Escaped-literal rollout and public cancellation

`20260913-escaped-startup` normally recreates the services with the separator,
iterator ownership and unconstrained-JSON-string fixes. Private state is
preserved. Checkout `3eb67062` is the startup revision; `9cbdf916` adds only the
executed cancellation probe timing guard. Its exact executed script is preserved
separately (SHA256 `d5ab6101249cab1b01993adb1ebbeabbe226309798e8b6698ffb1d33f33af05d`).
The native model image remains `8b8bcb200f3a` for these runs.

`20260913-escaped-literals` runs the actual mounted Requirement hook with API
max, nested low and disabled-thinking inputs. All three requests correlate
uniquely to fixed-high hook lines within their request windows. Quoted text plus
newline and the backslash path now pass with `stop`. The decision fixture still
fails at 8192 tokens with no public body, exposing the separate sampler defect
below. The old prompt-only success did not establish reliable budget closure.

Both public cancellation gates pass. Early parallel generation observes all
three workers active. Deferred audit observes a unique fresh Qwen-0 audit,
DeepSeek/other-Qwen idle, a later keepalive and another live audit snapshot
immediately before closing. Audit hash:
`66177df18c055a3d04d434619cac40653f00f14b1ee5d0e92484be66f67db1c0`.
Hook time is 18:52:57.322595Z, first activity 18:52:58.420340Z, later keepalive
18:53:09.461679Z, final activity 18:53:09.778392Z, close 18:53:09.778824Z,
on 2026-09-12 UTC. Every subsequent metrics sample retains all three native
DP ranks and both Qwen inventories and shows zero running/waiting requests;
both recovery requests return exact `323` with `stop`.

Cleanup is **1452.458 ms early-public / 1462.408 ms deferred-audit**, including
one second of stable-idle confirmation. This closes the prior 139-second public
cleanup failure for the tested Conductor path. The first wrapper's activation
value 900 was CLI-rejected before any request; `cancellation-driver2` corrects
it to the probe's 600-second bound and initializes all native rank metrics.

Each directory below contains `independent-review-manifest.json`.

| Independent evidence directory | SHA256 |
| --- | --- |
| `20260913-escaped-startup` | `d528a0a1f61fcded5339aa6b756a992a7bc6ebef3f4b0a939b590a7c18091e7b` |
| `20260913-escaped-literals` | `9ff8cf05ad3b73baa13329e940a9b55883c5b1d89ecaf4bd5c3cfd061c30e704` |
| `20260913-escaped-cancellation-driver2` | `bd2e8e0b9a64daf148ba84a1b5be608f4e0fde2e4918de2a81e1fb71a3d75046` |
| `20260913-escaped-public-cancel` | `f58e8ced4ebd515471dbecf1ced713a536b5907eadedf4e49dd9ee138c34493d` |
| `20260913-escaped-public-audit-cancel` | `1960ea30517402417e062ba796bb698eacb5afa1cffb190f58411db80f641871` |

## Seeded thinking-budget sampler investigation

`20260913-thinking-budget-pair` requests actual native token IDs with a bounded
128-token total and 16-token thinking allowance. Both structured/unstructured
requests produce 16 ordinary tokens followed by 112 token-zero IDs. Their
public bodies are empty. `20260913-thinking-budget-sampling` isolates sampling:
greedy and positive-temperature `top_p=1` correctly emit end token 128822 once
and begin a body; seeded `top_p=0.5` repeats the failure. These deliberately
short diagnostics are expected to truncate a complete checklist and do not
claim completed-answer success; the observed failure is missing termination.

The active V2 budget kernel forces a `1e9` end logit. The split top-p kernel
reconstructs its threshold as `log(pivot) + log(Z) + maximum`; FP32 rounds this
back to the maximum, and strict `>` excludes every token. The existing
monolithic sampler already handles this degenerate cutoff. `patch_top_p.py`
copies that safeguard to the split path and keeps the forcing kernel unchanged.
Input/output source hashes are in L1-NOTES and V41E-D8.

`20260913-thinking-sampler-oracle` independently reproduces the defect without
model inference. The exact committed patch passes **60/60 batch cases**:
batches 1/8/16/32/65 cross the actual split threshold of 64; each tests forced
1e9, normalized one-hot, finite-80 and ordinary logits at top-p .5/.95/1.
Every corrected forced row selects token 128822 and matches the CPU reference
probabilities exactly. All **45 control cases** retain bitwise masked logits
and same-seed token IDs versus the original. The original approximate top-p
algorithm's ordinary support/reference differences are retained, not claimed
as new numerical parity. The earlier 12-case single-row oracle also passes.

Independent preservation manifests retain the exact temporary diagnostic
scripts alongside raw events and source/image provenance. Each hash below is
for `independent-preservation-manifest.json` in the named directory.

| Sampler evidence directory | SHA256 |
| --- | --- |
| `20260913-thinking-sampler-oracle` | `d7ef8ae6d709c3533831aab2e1823349a3b9834dbba421b9afa664e2976974e9` |
| `20260913-thinking-budget-pair` | `820e7f68fbcdc11945a86c04c38ba4f3c7c6990a135c94d65dd7fb8b5f701a87` |
| `20260913-thinking-budget-sampling` | `a7238908f1a6f0759b54dbf16570bac34cae7eb5aa44bb5ba33654fbc89ce432` |

The expanded comparison repairs eight failing cases (114 all-masked rows) and
retains identical masks/samples in the other 52 cases. An independent image
check confirms the expected patched sampler and unchanged budget source in
vLLM `0.1.dev20904+g179dd0fa9`, Triton 3.7.1, Torch 2.13.0+cu130 and XGrammar 0.2.6.

New child: `local/vllm-openai:deepseek-v41-sm120-masked-kv-budget`, image
`sha256:18dad57d5b3d576797555e0e2c91f247ce4a39cba39ae20c79a4a2e09421a195`.
Parent and previous child bytes remain unchanged. Selected CPU regressions now
pass **518 tests**, and Ruff/progress/whitespace checks pass. Stochastic
forced-budget probes with/without grammar join the native suite. Fresh replay
on this child is recorded below; older native/context measurements retain
their original image provenance.

## Corrected-runtime native replay

`20260913-budget-startup` deploys source `05042b9b` and child `18dad57d5b3d`
with configuration
`8948a65760d0308e87a3397503f57f12035b7e50574f893bfc36fa4cb9dbe8b3`.
Normal startup succeeds and preserves the private key fingerprint/mode.
`20260913-budget-literals` passes all three actual-hook cases. The previously
empty decision checklist now completes in 71.138 seconds with 4908 generated
tokens, exactly 4096 reasoning tokens and `stop`. Independent semantic review
finds all R1–R13 criteria intact, including numeric limits and the complete
`Decision: defer.` ending; no rendering-scaffold requirement leaks into the list.
Decision response SHA256:
`314b4f4239ba4ae30bc1ff2e600f755933136c575aab702470de2210faf9120d`.

`20260913-budget-gates/native-restart` passes **28/28**, with no total-cap
failures. The three forced-budget probes (greedy, seeded nucleus, and seeded
nucleus with JSON grammar) each report exactly 16 reasoning tokens and return
public `437` with `stop`. All 14 Requirement calls (12 effort combinations plus
two escaped-literal cases) uniquely match fixed-high/8192-total/4096-thinking
hook records within their closed UTC request windows. These independent checks
use canonical message hashes and exact same-host timestamps.

All **89,730 raw returned logprob values** are finite. The normalized response
summary counts 89,724 because six wire values are lost by normalization; use
the raw-count addendum for the complete count. Native stream closure in this
suite remains client-close evidence; the separate all-worker public/native
cancellation probes establish cleanup at their recorded runtime revisions.

Files below are relative to `20260913-budget-gates/native-restart/`:

| Native replay artifact | SHA256 |
| --- | --- |
| `independent-native-manifest.json` | `0d38aba5417fae7bc7596baa84cef62956c49b11df1cf76dac01bb5149d73b7d` |
| `independent-hook-correlations.json` | `fbc938556a25cd0907d32e52908ee1de463587330488c278a846e38e58662ae1` |
| `independent-worker-window.log` | `a2860fab3b9c3d264a194eeda4a187f3ccda5f06c1da4485cb0d4d3388c3953d` |
| `independent-raw-logprob-count.json` | `67dab0a935d9b8fa24ecda518bbded2ce2465fe8f238574fc594b1382e2d76fc` |

## Corrected-runtime public replay

`20260913-budget-gates/l2-replay` passes **11/11** and observes all five
routes, including native-image, tool/result, headless JSON and stream recovery
cases. The primary request completes in 295.1 seconds. Its opening and body
have the intended `constraint.\n\n**Facts.**` boundary, and its two distinct
policies produce three complete, nonempty peers. Generated-token counts are
1372/2048 for the original draft, 2466/4096 and 2442/4096 for policy answers,
and 1449/8192 for the refined draft. None exhausts its cap.

Independent semantic review confirms the complete required ending, correct
arithmetic and infeasibility, two distinct conditional next steps, and a
253-word memo. The first audit passes after 4444 generated tokens; this case
does **not** exercise live repair or exhaustion. The public answer labels some
supplied constraints as assumptions and does not explicitly enumerate every
format requirement in its critical audit. These are recorded quality caveats,
not evidence of universal task compliance.

Artifacts below are relative to
`20260913-budget-gates/l2-replay/l2-route-primary/`:

| Primary replay artifact | SHA256 |
| --- | --- |
| `independent-semantic-review.json` | `4a520603c36a739c4e53e32ce86bffea2a99c98b7fedab21bdcda70ffbd8a5b8` |
| `response.json` | `8a65f377f779b9ebe3feac141272161a0f0d7b7ca0984f2de52e06d4da8e6a8f` |
| `result.json` | `bf4e6792ec4dd016e7b1db0881703e742b85bd9fd72f5c8cbce70bc8a24f8d40` |

## Qwen native near-limit retrieval

Qwen context evidence (`20260913-qwen-context-medium`) also passes on both
TP1 replicas at the reserved-candidate configuration. Each processes 260,706
prompt tokens with a 1024-token output allowance within the 262,144 limit.
Both return the exact key and `stop`: replica 0 in 102.006 seconds (179 generated
tokens), replica 1 in 98.657 seconds (114 tokens). Native medium uses the
unchanged Qwen template; tokenizer SHA256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
This is native c1 exact-key retrieval, not public ensemble long-context quality.

| Qwen context artifact | SHA256 |
| --- | --- |
| `20260913-qwen-context-medium/measurements.json` | `1b52214689764e182e5ea40a523937d4c95fd4c46a72ed14ec2641187c460dd9` |
| `20260913-qwen-context-medium/provenance.json` | `dadcf9ecf631fbdd8802d09a3d899ccf6dac2cc1393c76c9291d00e73b0e23b1` |
| `20260913-qwen-context-medium/manifest.json` | `017ef51366c0b7a7ebaf7da1f5335d7c7b64241f8dc62000fa54a61584a45800` |

## High-floor replay and remaining gates

**Latest contract amendment (V41E-D10):** the user now requires Requirement
to retain effort above high. The implementation inherits the canonical
top-level effort and applies a high floor: omitted/low/high become high,
max remains max, and nested template effort is overwritten consistently.
The 8192-token total, 4096-token thinking reservation and Qwen settings remain
unchanged. The amended CPU suite passes **560 tests**, with Ruff and progress/
whitespace checks passing. Native and four composed effort results are recorded
below; the current remaining work is summarized at the top of this file.

The preceding wordcount campaign completed high, max and low (347.7 seconds)
under the original fixed-high Requirement policy. The user amendment arrived
during omitted-effort execution, so that child was interrupted with SIGINT and
the parent stopped with code -2 before quality/performance/final cleanup.
`20260913-wordcount-gates/operator-effort-change/` records the operator reason
and all-worker stable idle afterward. This is an intentional interruption,
not a model failure, and the completed cases do not validate the new high floor.

### Fresh high-floor runtime and native evidence

`20260913-effort-floor-startup` normally restarts the clean GPU checkout
`f737e0b083200cb12ed194c79663d19471a908bd` with served configuration
`4216bf457bdf6082b8545c67783e1df3ed6344318efa258c14799fc8e061423f`.
The selected DeepSeek child remains
`sha256:18dad57d5b3d576797555e0e2c91f247ce4a39cba39ae20c79a4a2e09421a195`;
all worker image pins and private-state persistence pass independent review.
The serial driver and executed final-check script are copied into
`20260913-effort-floor-gates/` for reproducibility.

Initial per-rank checks pass **12/12** (four cases on each DP rank): all 546
returned logprob values are finite, and all six sampled/structured forced-budget
cases report exactly 16 thinking tokens, return `437`, and finish with `stop`.
Requirement replay then passes **14/14**: five API-max/literal cases preserve
max; nine omitted/low/high combinations use high, including conflicting nested
values. Each request hash uniquely matches the correct role hook inside its
closed UTC window. All 89,658 returned logprob values are finite (34,953 max;
54,705 high); all requests stop normally with complete JSON and no truncation.
The native hook retains the 8192 total / 4096 thinking reservation.

The table hashes each directory's `independent-review-manifest.json`, relative
to the persistent evidence root. These native contracts do not establish the
subsequent composed quality or performance gates.

| High-floor evidence directory | SHA256 |
| --- | --- |
| `20260913-effort-floor-startup` | `23a4e78a4c797bc8e400196f882b5bab7b3d046a17bb2df1846dfc96c40567d9` |
| `20260913-effort-floor-gates/native-rank-0` | `373e54b24fa6746bc63cb980683c46d5412e941e4121fc70ee6e46e866bf7b80` |
| `20260913-effort-floor-gates/native-rank-1` | `3220d965d9f6a4b3cf8ba86f083ecca5991e91fc04ae6db57ee70c3b732a124c` |
| `20260913-effort-floor-gates/native-rank-2` | `905b32c954d993bfbadf9aba3f32abb80f74e361d7888f0feeb3f1dfe11c1fae` |
| `20260913-effort-floor-gates/native-requirements-max` | `e367d997589c0aab30d5a4e9c8dfbf9c6ef11417e1c6c64d84db6e4afd490f24` |
| `20260913-effort-floor-gates/native-requirements-floor` | `8812e93ea46f5dac0846435ed437d540e8a78e39a98cdf0902cadfafcd0ae841` |

### Fresh composed effort replay

`20260913-effort-floor-gates/l2-max/l2-route-primary-max` passes in
**322.576 seconds**. Independent review matches all eight role executions to
exactly one hook inside that role's trace window: Requirement is now max with
8192 total / 4096 thinking, other thinking DeepSeek roles inherit max, and
Qwen retains its existing high-to-medium alias. All 16 extracted checklist
entries retain the requested constraints and literals.

The draft uses 1341/2048 tokens; the three synthesis peers use 2425/4096,
2357/4096 and 3145/65536. All are complete, with no deliberation spill in this
draft. The final memo has 248 whitespace-delimited words, preserves the clean
paragraph seam and exact `Decision: defer.` ending, and proposes distinct
conditional evidence-gathering actions for A's latency and B's cost while
retaining both hard limits. Audit returns its first PASS; **zero live repairs**
were exercised. Minor limitations remain: the draft omits the closing period,
some peers add unnecessary but labeled assumptions, and the audit loosely
paraphrases one criterion. The final avoids those stronger unsupported claims.
This case is a bounded semantic review, not proof of universal correctness.

The add-only `independent-max-review.json` beside the case has SHA256
`0657cf176978469e9d400ce54166ef997b21391a1616965dd6cd609757e6aed4`;
its source `response.json` has SHA256
`361c7cfea9ad60cb6f864b0df819f460d6de05ed75fe491fdf1bd5221d9c4a48`.
The separate `l2-other-efforts` directory completes low and omitted, but their
semantic review finds repeated source-status errors below. High is then stopped
intentionally before formal quality, public performance or final cleanup.

### Preserved semantic failures and source-status correction

At the same `f737e0b0` source/config, low completes in **332.992 seconds** and
omitted in **396.676 seconds**. Each has eight uniquely correlated role hooks:
Requirement high with 8192 total / 4096 thinking; other DeepSeek roles follow
low or default high; Qwen retains its existing medium alias. All draft/peer
bodies are complete and below caps, with no counting or private deliberation
spill. LOW's three peers use 2437/4096, 2429/4096, 1772/8192; omitted uses
1722/4096, 2583/4096, 4316/32768. Final memos are 249 and 236 whitespace words,
with clean seams, exact endings, correct numeric comparisons and neither option
claimed feasible. Each receives its first audit PASS; zero repairs occur.

These are **protocol/effort passes, not complete semantic passes**. Both final
memos classify the explicitly supplied hard constraints as assumptions, and
both audits incorrectly accept that classification. LOW additionally converts
"not supplied" into the unsupported categorical claim that no other options
or revisions exist; omitted correctly retains "are supplied" scope. The
Requirement checklists contain the original facts and constraints, so extraction
and forwarding are not the cause. Omitted audit's extra R15 memo-format row is
a redundant supported interpretation, permitted by the template's missing-ID
recovery; an extra ID alone is not an invented requirement.

| Independent report beside the case | SHA256 |
| --- | --- |
| `l2-other-efforts/l2-route-primary/independent-low-review.json` | `fc1ae6cdcd70e9c538218d946aa6a3bc9e1cf6814fd4534270555f837ec75543` |
| `l2-other-efforts/l2-route-primary/independent-low-semantic-addendum.json` | `89fa83815a343415e33994bf9138ba2eaac380218696c24336e23dd361ae30df` |
| `l2-other-efforts/l2-route-primary-omitted/independent-omitted-review.json` | `8b1ced0988816859dd765996ac376cd0b9913088ccc33bac75feb1227329ca1e` |

The source response hashes are
`65c6eb60e1f63a3605f075fa1f9ea34de91133ede121f4e1536dc6eb0849e30a`
(low) and `098f80e3accf7d9fe80f6cbaad7bc6a12b4f58561e7df5a610e64954f27f3e0d`
(omitted). Preserve the raw reports, including the initial LOW review's minor
classification; the stricter semantic conclusion above supersedes any implied
all-requirements pass.

V41E-D11 clarifies source status in both synthesis templates and audit, without
changing roles, effort, caps, schemas or fixtures. Existing actual-DAG/native
full-template-hook tests pass **148/148**. The new prompt needs fresh served
replay. `20260913-effort-floor-gates/operator-source-status-change/` records
SIGINT sent only to the active high-case child, the parent stop with code -2,
and subsequent all-worker stable idle. This partial high is an operator
interruption, not a model failure verdict. Quality/performance/final cleanup
were not started in this campaign. The new source-status campaign must retain
these failures and review the actual statements rather than a model's PASS.

### Source-status startup and native replay

The correction is committed as `e79dee15226b78e3bb39f8b9521f6e0b52df0296`,
with served configuration
`415382672aa77c425078118dea195b6dad3ce10ab070a39105ad5aa78ca4449c`.
`20260913-source-status-startup` completes a normal restart with private state
preserved. DeepSeek and Qwen pinned images remain unchanged; the rebuilt Kairyu
image is `sha256:20d1dc96b1f3369906ce20dd409557090e158b672842b48b642aab191fb21e7d`.
Independent comparison against `f737e0b0` confirms that the only served changes
are synthesis's headed/headless prompts and the audit prompt. Requirement's
native hook, template, effort policy and budgets are byte/structurally unchanged;
`20260913-source-status-gates/source-scope.json` records the scope check.

The new run passes **12 per-rank basic probes plus two native Requirement
conflict probes**. All 18,861 returned logprob values are finite: 549 from the
basic probes and 18,312 from Requirement. All six forced-budget basics use
exactly 16 thinking tokens and finish with `437`. The conflict requests each
uniquely match a hook in their closed UTC windows: low/nested-max becomes high;
max/nested-low remains max. Both complete JSON with `stop`. The full earlier
14-case Requirement matrix retains its `f737e0b0` evidence; these two new cases
are a scoped replay, not a renamed repeat of that full matrix.

The table hashes `independent-review-manifest.json` in each directory. Use the
conflict directory's corrected `independent-conflict-review.json` for its mixed
effort summary (SHA256
`b56ae6601938dfa3a118308d6d9a7484b49f6f6e5cfedf2ef366a5cb24cf248c`);
the original generic helper's final-effort label did not describe both cases,
while its actual per-case high/max correlations were correct.

| Source-status evidence directory | SHA256 |
| --- | --- |
| `20260913-source-status-startup` | `7624df35401f9bca8e7de597601c241528029813af6933b1cf10c254bc788274` |
| `20260913-source-status-gates/native-rank-0` | `ee7a4c715f24906823d966c48f23daa7eaeabba53954a40f19a370c1441ed2cb` |
| `20260913-source-status-gates/native-rank-1` | `183fd3e168809fc3c4db55ed79a0cb5e9572cf217273c68bfc65e35ab584d629` |
| `20260913-source-status-gates/native-rank-2` | `da1d24bc430728ef9e6979bddcf177e022684fbdbc3e6fbd265e759e18a3cc4e` |
| `20260913-source-status-gates/native-requirements-conflicts` | `39541ff295652f10832e66d00aa44b39ca2993567585ed84ab93ed81ae4f98d0` |

The following records retain the campaign chronology. Per V41E-D13, model-content
reviews are observations, not implementation completion gates. Subsequent work
uses bounded checks for changed behavior and the existing serving measurements.

### Source-status four-effort replay

All four public `primary` requests complete at the `e79dee15` source and
`41538267` served configuration recorded above. Each returns HTTP 200, SSE
`[DONE]`, `finish_reason=stop`, and passes the route/DAG and candidate-completion
contracts. Independent review reads the actual request, response, trace and
stage bodies, rather than treating the protocol verdict as a semantic verdict.

| API effort | Elapsed seconds | Final words, including head | Draft / answer 1 / answer 2 / critique tokens | Audit IDs / tokens |
| --- | ---: | ---: | --- | --- |
| low | 289.594 | 246 | 1391 / 2497 / 2388 / 1802 | R1–R11 / 5626 |
| omitted | 420.724 | 243 | 1391 / 2510 / 3767 / 2417 | R1–R13 / 6753 |
| high | 292.898 | 243 | 1391 / 2455 / 1866 / 1930 | R1–R14 / 5293 |
| max | 444.806 | 267 | 1391 / 2403 / 2388 / 7919 | R1–R10 / 4700 |

Draft cap is 2048; each policy answer cap is 4096. Critique caps are 8192 for
low, 32768 for omitted/high and 65536 for max. All four candidates contain a
complete memo below their token caps; omitted's answer 2 also contains the
private-counting spill described below. Each request has two distinct policy
framings, with one formal constraint-analysis policy and one adversarial or
stakeholder review policy. Each audit returns PASS on its first attempt:
**four audits, zero live repairs, no refinement exhaustion**.

All eight role hooks in each case uniquely correlate with their own closed
UTC trace windows. Requirement applies high to low/omitted/high and retains
max for API max, with 8192 total tokens and 4096 reserved thinking tokens.
Other DeepSeek roles inherit low/high/high/max respectively. Qwen draft,
answers and audit retain wire `high`, translated by the unchanged template
to its existing medium reasoning preamble. HIGH synthesis has 65518 total /
65262 thinking tokens after its 18-token head; MAX has 65512 / 65256 after its
24-token head. These observed hooks establish the effort/budget contract;
they do not prove reasoning quality or byte-for-byte reconstruction of the
expanded internal messages, which was not established by the bounded review.

All four final memos preserve A's 40 units/month and 80 ms, B's 70 units/month
and 30 ms, the current budget cap of 60 and latency cap of 50, and the explicit
conclusion that neither option is feasible. Each has a clean head/remainder
seam, remains under 350 whitespace-delimited words, proposes two distinct
conditional next steps, invents no benchmarks or citations, and ends exactly
`Decision: defer.`. The targeted final source-status defects from the preserved
`f737e0b0` runs are not observed: supplied hard limits remain requirements, and
missing evidence is not converted into a categorical absence claim. LOW's
redundant scope premises concern the current A/B decision and current limits;
they do not assert that alternatives or approved changes cannot exist.

Residual limitations remain and are not erased by these passes:

- Omitted `answer_2` spends part of its exposed body counting individual words,
  includes the literal `</think>`, then supplies a complete memo. It uses
  3767/4096 tokens, so the completion/cap gate passes despite contaminated peer
  content. The final memo omits this spill; V41E-D9 is not a universal guarantee.
- Omitted's final step 2 says the latency target is revised to "at most 80 ms or
  looser". A revised ceiling must be at least 80 ms to admit A, so this wording
  is ambiguous. The step requires re-evaluation and preserves the budget cap;
  it does not claim automatic feasibility. Audit misses this wording defect
  and overstates the separately developed comparison of a merely named
  Pareto-style lens.
- Peers still misclassify given constraints as assumptions or add unnecessary
  premises. LOW critique and MAX answer 2 also assert unsupported absence of
  alternatives; HIGH answer 2 overstates each option's failure as a gap in both
  dimensions. These statements are removed or narrowed in the final memos.
  Complete candidates do not imply semantically correct candidates.

These are bounded fixture results, not full quality assurance or evidence that
every audit detects defects. In particular, the first-pass results exercise no
live repair or exhaustion behavior. Quality, public performance and stability
gates require their own completed evidence; prior semantic failures remain
preserved above.

The following files are beside their cases under
`20260913-source-status-gates/`; SHA256 values bind the independent reviews.

| Case / review file | SHA256 |
| --- | --- |
| `l2-low/l2-route-primary/independent-source-status-low-review.json` | `be462dd2782b3552cd75b794741d45eb37635a9ebd6a42f8cfd06fda846da935` |
| `l2-omitted/l2-route-primary-omitted/independent-source-status-omitted-review.json` | `21fa7b1976117be9395c3b28fb653344587cc6614f97b6473ba16b2155cf1457` |
| `l2-omitted/l2-route-primary-omitted/independent-source-status-omitted-wording.json` | `cdce42f8a53116969b35c54e3582abd961ceb3b9380bbe4e2d7025779546d872` |
| `l2-high/l2-route-primary-high/independent-review-metadata-manifest.json` | `bc86eab3fcd22a5c8ad620e99a5523e5a1c2479bfb596f83e67bde86f93d9cdd` |
| `l2-max/l2-route-primary-max/independent-review-metadata-manifest.json` | `b52aba02af5b8ee5c68229412f0510f05769ad851e33d2b12b27fbd7944e865e` |

HIGH/MAX metadata manifests bind the original request/response/result files,
body review, allowlisted hook observations and hook review; all eight remote
file hashes match the local copies. Their derived `stage-bodies.json` files
remain **local-only** under `/private/tmp/kairyu-v41-source-status-review/high/`
and `/private/tmp/kairyu-v41-source-status-review/max/`; each is extracted from
the original `response.json` already preserved in the same remote case. The
manifests explicitly distinguish those local-derived hashes. For HIGH/MAX,
raw Docker logs remain on the GPU host; the transferred hook evidence contains
only timestamps, roles, efforts, numeric budgets and message hashes.

### Earlier fixed-high replay after transfer approval

The user explicitly approved the transfer on 2026-09-13. That campaign used the
clean GPU checkout `121bec4f`, with the V41E-D9 prompt correction and fail-closed
paired-baseline verifier. `20260913-wordcount-startup` completes normal restart
at approximately `2026-09-13T03:06Z`, verifies all pinned worker images and
preserves the private key fingerprint/mode. Its new configuration hash is
`9072c5dac44545f68926af9f5826cbeac8b3ed67d101b4cdd8c7ac34247de23f`.
Its serial replay is under `20260913-wordcount-gates`; the intentional stop is
recorded above. Previous effort failures below remain tied to `05042b9b`. The
transfer blocker is resolved; the later high-floor policy uses separate evidence.

Initial restarted-worker checks pass **12/12**: default, nonthinking, sampled
thinking-budget and structured sampled thinking-budget requests on each DP
rank 0/1/2. All 546 raw returned logprob values are finite. The six sampled
budget cases report exactly 16 reasoning tokens, return `437` and finish with
`stop`; no case is truncated. Independent source/runtime checks agree across
all ranks. The files below are `independent-review-manifest.json` relative to
the persistent evidence root:

| Restarted-worker evidence directory | SHA256 |
| --- | --- |
| `20260913-wordcount-startup` | `1b75282bcf5c233cbb3af47ef072db5a263abe1b298cc5a400a0565323b580ce` |
| `20260913-wordcount-gates/native-rank-0` | `24c182d4e1b95c6d6cb70d80e6a5208654fcc6b56ed7a4016ceb4089e6a47905` |
| `20260913-wordcount-gates/native-rank-1` | `99a2c412eab755f4b652b3c2ec772462c13bfc5a75462b2e8070631d49f6d5b8` |
| `20260913-wordcount-gates/native-rank-2` | `377de93a2efd1e568b6b58afc26586d9c14ae7e8c7d28246767a5cf99e6ece6e` |

The first corrected public replays pass high and max, clearing the previous
policy-answer cap failures in these observed runs. Counts below are generated
tokens except the final conservative whitespace word count:

| API effort | Seconds | Draft | Answer 1 | Answer 2 | Refined draft | Audit | Final words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 338.9 | 1626/2048 | 2470/4096 | 2609/4096 | 2720/32768 | 5812 | 235 |
| max | 384.1 | 1626/2048 | 2410/4096 | 2358/4096 | 4938/65536 | 6071 | 252 |

All 16 role executions uniquely match hook records in their trace windows.
Requirement stays high/8192-total/4096-thinking; other DeepSeek thinking roles
inherit high/max; Qwen keeps its medium alias and existing reservations.
Both final answers retain the paragraph separator and exact required ending.
Each first audit passes; no live repair is exercised. The internal draft still
contains a short deliberation continuation followed by a second closing-think
marker and a complete memo. Synthesis removes that spill from the public answer.
This is a recorded quality limitation, not proof that forced thinking boundaries
always produce clean candidate bodies. MAX also has awkward wording around a
conditionally raised latency ceiling; its audit paraphrases the intended bound.

Reports are saved add-only below `20260913-wordcount-gates/l2-high-max/`:

| Independent effort report | SHA256 |
| --- | --- |
| `l2-route-primary-high/independent-high-review.json` | `d4614a1aa7ccceb80bd7d63b1ebebdf728216681074962396794dc82d9ca543c` |
| `l2-route-primary-max/independent-max-review.json` | `8f0c8a7beefa1fbb695dad5cb66e8ed3162f5d6b2453887ca8c59d6fc4f7508b` |

### Preserved failures before the candidate prompt correction

The earlier public effort matrix passes low (301.968 seconds) and omitted (347.8
seconds), with exact per-stage hook correlation: Requirement stays high,
other DeepSeek thinking roles follow low/default-high, and Qwen remains at
its medium alias. The explicit-high case fails after 386.4 seconds because
`answer_2` uses exactly 4096 tokens. Its body continues word-by-word counting
after the forced 2048-token thinking boundary, emits another `</think>` and
starts a memo that stops mid-audit, missing its next steps and required ending.
The supplied policy never asks for word enumeration. `answer_1` completes at
2479/4096; final synthesis and its 5799-token audit pass, which does not repair
the incomplete peer. Preserve this failure under
`20260913-budget-gates/l2-efforts/l2-route-primary-high/`.

The Qwen hook inside the failed stage's closed UTC window records
high/4096-total/2048-thinking and messages SHA256
`5fa3d27184d666e21151d95b050a3d420ae4404c4e6d244067909b89c5a4542c`.
V41E-D9 adds focused role guidance against repeated individual word counting;
it preserves effort, sampling and caps and requires native/composed replay.
The failure check remains strict. This is not a verified universal remedy for
deliberation continuing after forced thinking termination.

The max case similarly fails after 425.7 seconds: `answer_1` reaches exactly
4096 tokens while repeatedly counting words. Its exposed body continues a
count, revises an embedded draft and counts again until cutoff, rather than
finishing a clean peer. `answer_2` completes at 2517 tokens. The supplied policy
does not demand word enumeration. The four-case matrix exits 1 at
`2026-09-12T19:53:50.345345+00:00`; the parent driver stops before either formal
Requirement quality or public performance. Both high/max failures remain at
the original source/configuration, not the new prompt correction.

Independent review matches all **32/32 role executions** to exactly one hook
inside that role's closed trace window. Requirement is high in all four calls;
other DeepSeek thinking roles inherit low/high/max (omitted becomes high),
while Qwen keeps the same medium alias and budgets. All four final syntheses
stop, retain the required ending and stay below 350 words; each receives its
first audit PASS, with zero live repairs. These final-answer facts do not
override the two peer failures. Source response SHA256 values are
`23922e06e0e73479da0032f9c7ea206636495c22b37615d24dc40da5c9b687f2`
(high) and `257c5e0bce993bfb63d3f81573a36b1aa0c1b011a3cc4ece7a982f52198201f0`
(max). The local independent report is
`/private/tmp/kairyu-v41-budget-review/independent-four-effort-review.json`,
SHA256 `d8d60c59b88af2eb43d7738bf07f8e4029c1e045e99bffe1f277bfce72e95c93`;
it has not been transferred to the GPU host. Raw responses/traces remain in
the remote evidence tree and support repeating the review on another machine.

At that failed matrix checkpoint, the GPU checkout was `05042b9b` and all six
services were healthy. Automatic approval review twice rejected transfer of
the private Git difference to the already-used GPU host, even after checking its clean checkout
and the narrow source/document payload. Explicit transfer approval was
requested and later granted as recorded above; no alternative transfer was
attempted. PR commit `b50e9863` contains the then-unverified prompt correction,
and `499f8728` contains the verifier fix below.
Selected CPU tests pass 526/526; the final prompt wording also passes 163 focused
DAG/hook/probe tests. CI results are separate from GPU readiness.

After approval, the full branch was synchronized into the existing GPU checkout
with bind-mounted inodes preserved until normal `run.sh up`. The restarted
configuration, unchanged pinned worker images, clean checkout and private-key
persistence were verified above. Fresh replay retains the original failed
artifacts. Candidate completion remains a strict gate even when synthesis and
audit succeed.

Before the fresh public performance matrix, the coding verifier was tightened
to reject a failed paired native benchmark or a missing/nonfinite/nonpositive
baseline TTFT, including rows whose public routes make the TTFT comparison N/A.
Eight CPU regressions cover failure cases and the valid N/A case. This changes
verification only; it does not change the serving configuration or any recorded
model measurement.

Native fixed-output capacity at c1/8/16/32, the five native retrieval sizes,
and native DP-rank cancellation cleanup now have the successful evidence above.
Those native measurements do not establish composed L2 behavior. The later
effort/headless/protocol and cancellation records above provide the applicable
composed evidence at their stated revisions. Public serving measurements still
need fresh same-topology baselines. Initial startup/NaN/candidate/measurement
failures remain preserved; model-content review is not a new completion gate.

The sibling TP8 result and PR #595's Qwen Requirement measurements are not
measurements of this example. Native DeepSeek retrieval has passed with
1,039,902 processed prompt tokens; that does not bypass the Qwen 256K limit in
the ensemble. Qwen's near-limit native retrieval passes as recorded above;
composed requests with similarly large inputs remain unmeasured.

## Completed diagnostic pass and verification scope correction

`20260913-source-status-quality` finishes at 2026-09-13 04:58:09 UTC on
`e79dee15`/configuration `41538267`. All three requests complete the protocol
checks. The image fixture's heuristic ending check fails because L3 inserts
`<image:0>` into the user's required literal; its native Requirement input,
checklist and final response contain that appended marker. V41E-D12 corrects
the reproducible text/image serialization defect. Related CPU request and DAG
tests pass 94/94 before deployment.

Content review also observes unsupported premises in the headed memo despite
the model's audit PASS. This is retained as an observation about model behavior,
not a required quality guarantee or a reason for more prompt-tuning campaigns.
The owner explicitly corrected that overreach. V41E-D13 defines the remaining
verification scope; no additional semantic reviewer approval is required.

The extra native audit-control run `20260913-source-status-audit-controls`
was stopped at the owner's instruction at 05:08:30 UTC, before either case
produced a completed summary. Its owned HTTP client and runner were stopped;
all five engine instances across the three worker services returned to stable
idle. `operator-user-scope-stop.json` and `user-scope-stop/` retain that record.
Partial artifacts remain; completing those controls is **not a remaining task**.
The previous campaign's performance steps had not started. Do not launch the
prepared continuation script that requires semantic audit-control approval.

## Next evidence record

Follow [README.md](README.md#gpu-validation-execution-order). Record
checkout SHA, exact runtime image IDs, checkpoint attestation, configuration
hash, request/response/trace artifacts, first failures and explicit skipped
checks. Record the behavior actually exercised and observed metrics. Preserve
content diagnostics separately, without presenting them as guarantees or making
them prerequisites for implementation completion.
