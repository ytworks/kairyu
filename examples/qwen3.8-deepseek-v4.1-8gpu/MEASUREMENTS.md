# V4.1 ensemble evidence

Status: **GPU validation in progress.**

The user released all eight GPUs on 2026-09-13 (JST). GPU trials are recorded
below. The original CPU-only implementation and the sibling TP8 results do not
establish GPU correctness for this example.

## What is established

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
- Requirement is fixed DeepSeek high, while Qwen retains its existing effort
  mapping. The removed image-description stage is not part of the new DAG.

## GPU startup investigation

Hardware: 8 × RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, PCIe;
1 TiB host RAM. Initial trials use parent `027bf47b2bd6`; the selected child
image is `8b8bcb200f3a` (full IDs in example.json).
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

| Literal diagnostic artifact | SHA256 |
| --- | --- |
| `results.json` | `0602b278644b957d5028d88543140a737b0b6531544828051dd23e538280778d` |
| `manifest.json` | `85054b81ed4d1086375a9fd5b6977d445e17aa06db5bbd1ac15508b3485c9cca` |

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

## Remaining gates

Native fixed-output capacity at c1/8/16/32, the five native retrieval sizes,
and native DP-rank cancellation cleanup now have the successful evidence above.
Final composed L2 validation remains open: revalidate candidate completion after
the Qwen reservation, primary image/headless and four-effort paths,
audit/refinement, public cancellation propagation, normal restart and public
concurrency with fresh same-topology baselines. Do not infer these gates from
native results or the initial L2 fixture's final-answer success. Initial
startup/NaN/candidate/measurement failures remain preserved.

The sibling TP8 result and PR #595's Qwen Requirement measurements are not
measurements of this example. Native DeepSeek retrieval has passed with
1,039,902 processed prompt tokens; that does not bypass the Qwen 256K limit in
the ensemble. Qwen's near-limit native retrieval passes as recorded above;
composed requests with similarly large inputs remain unmeasured.

## Next evidence record

Follow [README.md](README.md#gpu-validation-execution-order). Record
checkout SHA, exact runtime image IDs, checkpoint attestation, configuration
hash, request/response/trace artifacts, first failures and explicit skipped
gates. Preserve protocol results separately from model/task quality and human
semantic review. Mark only completed gates as passed; a completed implementation
or a healthy process alone is insufficient to close the remaining gates.
