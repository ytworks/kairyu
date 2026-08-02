# Benchmark ownership and entrypoints

Kairyu has two benchmark surfaces with different distribution contracts. The
installed `kairyu.bench` package owns reusable behavior and the public CLI. The
top-level `bench/` directory owns source-checkout-only executable wrappers and
reviewable result artifacts. Do not move code across that boundary without
preserving the affected command and evidence paths.

## Ownership boundary

| Surface | Owner | Distribution and compatibility contract |
|---|---|---|
| Reusable config, target types, credential resolution, statistics, atomic reporting, adapters, and runners | `kairyu/bench/` | Installed in the Kairyu wheel; may be imported by both the public CLI and checkout-only wrappers |
| Public benchmark CLI | `kairyu bench` | Installed console surface: `run`, `download`, `report`, `list`, and `entrypoints` |
| Synthetic offline fixtures | `kairyu/bench/fixtures/` | Installed package data; all eight JSONL files must be readable through `importlib.resources` |
| Entrypoint inventory | `kairyu/bench/entrypoints.toml` | Installed, machine-readable source of truth for every supported top-level wrapper |
| Gate, comparison, operator, and microbenchmark executables | `bench/*.py` | Repository-only; both `python bench/<name>.py` and `python -m bench.<name>` are supported from a checkout |
| Measurement and decision artifacts | `bench/results/` | Repository-only and never shipped in a wheel; routine output is ignored, while explicitly reviewed formal evidence may be retained by Git |
| Tests | `tests/` | Repository-only and never shipped in a wheel |

The default Fugu result location remains `bench/results/fugu/` for command
compatibility. For an installed CLI used outside this repository, that is a
path relative to the caller's working directory; it does not mean the
top-level `bench/` tree is installed.

Installed `kairyu` code must not import the repository-only `bench` namespace.
If two wrappers need the same config, type, statistics, result writer, or
provenance rule, put that reusable contract in `kairyu.bench` and keep each
top-level file as a thin composition layer.

## Inventory and checkout validation

The installed manifest can be inspected without a source checkout:

```bash
uv run --frozen kairyu bench entrypoints
uv run --frozen kairyu bench entrypoints --json
```

In a checkout, validate the manifest against executable files, documentation,
main guards, invocation forms, and the one-way package boundary:

```bash
uv run --frozen kairyu bench entrypoints --check-repo .
```

Every `bench/*.py` entrypoint must:

- have exactly one sorted `[[entrypoints]]` record in
  `kairyu/bench/entrypoints.toml`;
- preserve both its path and module invocation forms;
- expose non-executing `--help`; CI supplies the declared development import
  dependencies, but help must not contact a GPU, service, Docker, Kubernetes,
  vLLM runtime, or model;
- keep reusable semantics in `kairyu.bench`, not in a new cross-wrapper helper;
- name its runtime prerequisites and at least one document in the manifest;
- retain its historical flags and default result path, or provide an explicit
  compatibility wrapper and documented migration.

Formal G2/G4/G5/G6 artifacts bind source paths, hashes, commands, and result
locations. A refactor may delegate a stable wrapper to package-owned code, but
must not silently rename the wrapper, invocation form, or recorded evidence
path. Existing wrapper-to-wrapper imports are compatibility dependencies; new
shared behavior belongs in the installed package. The exact seventeen retained
composition edges are allowlisted in the manifest's
`[compatibility_imports]` table; checkout validation fails on any undeclared,
removed, or redirected edge.

### Target and credential migration

Every benchmark target now uses one package-owned grammar:
`name=base_url=model[=api_key_env]`. The optional fourth field is the **name of
an environment variable**, never a credential value. Older
`bench/frontier_compare.py` help text treated that field as a literal key; set
the key in the environment and replace the literal field with its variable
name:

```bash
export FRONTIER_API_KEY=...
python bench/frontier_compare.py \
  --target frontier=https://api.example/v1=model=FRONTIER_API_KEY
```

`bench/serving_bench.py --api-key` remains as a deprecated compatibility flag
for existing commands. New commands should use `--api-key-env` or the shared
`--target` form. An explicitly named but unset credential variable fails
closed across installed and repository-only runners. Resolved secret values
are never written to result config or validation errors.
Direct, YAML, split-flag, and combined-target construction all normalize API
roots to `/v1` and validate credential variable names before fingerprinting.
Historical run directories remain reportable, but a pre-migration run whose
stored YAML URL omitted `/v1` has a different fingerprint; use a new run ID
instead of silently resuming it.

Generic serving/frontier reports now record
`percentile_method: nearest-rank-v1`. Older reports from those two wrappers
that lack the field used frontier's floor-index calculation and serving's
median p50/floor-index p99; they must not be compared as though the percentile
definitions were identical. Formal gate wrappers retain their versioned,
source-bound artifact schemas; those embedded methods are gate contracts, not
alternative reusable reporting helpers.

The human-readable path index below mirrors the manifest. Kind, prerequisites,
and documentation metadata are authoritative in the TOML and in the
`entrypoints --json` output.

```text
bench/agentic_kv_tier_f4b_bench.py
bench/attention_backend_profile_bench.py
bench/audit_io_bench.py
bench/auto_params_bench.py
bench/batched_prefill_qwen.py
bench/batched_spec_verify_qwen.py
bench/decode_page_table_cache_qwen.py
bench/dp_scaling_g2_a8_bench.py
bench/draft_quant_qwen.py
bench/dram_kv_tier_qwen.py
bench/fleet_churn_bench.py
bench/fleet_gateway_bench.py
bench/fleet_rollout_bench.py
bench/fleet_usage_replay.py
bench/fp8_kv_g4_ekv_bench.py
bench/frontier_compare.py
bench/future_token_bench.py
bench/g2_a6_vllm_bench.py
bench/g2_a9_dp_tp_crossover_bench.py
bench/g4_ma1_qwen3_235b_nvfp4_bench.py
bench/g4_ma1_qwen3_235b_nvfp4_capture.py
bench/gate_a1.py
bench/gate_a2.py
bench/global_kv_pool_decision.py
bench/kv_aware_ttft_f2c_bench.py
bench/kv_event_f2b_bench.py
bench/kv_event_hash_bench.py
bench/kv_transfer_bench.py
bench/multiturn_prefix.py
bench/noisy_neighbor_bench.py
bench/noisy_neighbor_gpu_bench.py
bench/op_queue_bench.py
bench/orchestration_mock_bench.py
bench/orchestration_stream_bench.py
bench/parity_hf.py
bench/parity_tp.py
bench/pd_mixed.py
bench/pd_overlap_qwen.py
bench/prefix_routing_f2a_bench.py
bench/prefix_weight_f2d_bench.py
bench/priority_overload_bench.py
bench/priority_overload_gpu_bench.py
bench/proc_wire_bench.py
bench/quant_gemm_bench.py
bench/radix_eviction_bench.py
bench/reduce_scatter_bench.py
bench/router_latency.py
bench/run_g2_a6_formal.py
bench/sampler_penalty_state_bench.py
bench/scheduler_queue_bench.py
bench/serving_bench.py
bench/slo_admission_bench.py
bench/tiered_auto_bench.py
bench/tp_kv_hit_g2_a7_bench.py
bench/tp_sampling_owner_bench.py
bench/tp_sampling_owner_qwen.py
bench/typed_prompt_qwen.py
bench/usage_architecture_bench.py
bench/vllm_quant_kernel_bench.py
```

### G2 A8 DP scaling evidence

`bench/dp_scaling_g2_a8_bench.py` is the checkout-only formal operator for
G2 A8. It compares one Qwen3-32B TP4 replica with two independent TP4 replicas
behind the L2 gateway on the same eight-GPU host. The performance arm uses a
predeclared open-loop arrival-rate grid, at least three fixed-seed paired runs,
with explicit seed 0, and excludes warmup. It does not substitute A6's synchronized-concurrency
binding point for A8's saturation sweep.

The operator retains every request sample and correlates gateway responses with
the replica-placement JSONL log. Its independent verifier recomputes all three
binding verdicts:

- the median paired peak-goodput ratio is at least 1.9;
- nearest-rank ingress-to-replica-selection latency p99 is below 10 ms; and
- the engine-originated multi-turn KV hit rate through the two-replica
  session-affinity gateway is at least 90% of the single-replica value.

Gateway counters and placement reasons prove routing behavior but are never
used as cache-hit truth; cache hits come only from response
`prompt_tokens_details.cached_tokens`. A report cannot say PASS without
complete raw performance, placement, and cache-usage evidence from a real
eight-GPU run. Offline fixtures and unit tests validate the verifier only.
Before traffic, the live read-only model volume is full-hashed (all 17 weight
shards plus index, tokenizer, and model config) and compared with the pinned A7
checkpoint evidence.
The exact launch and replay procedure is in `docs/gpu-runbook.md` §6.

The retained 2026-07-31 eight-GPU artifact contains 2,992/2,992 successful,
retry-free requests and 1,496 correlated placement rows. Router p99 is
3.723 ms and DP retains 99.53% of the single-replica cache-hit rate. The three
paired peak-goodput ratios are 1.9988×, 1.7342×, and 1.7993×; the 1.7993×
median misses the original 1.9× threshold, so verify and replay intentionally
report `passed: false`. The product owner accepted this measured median as an
explicit closure deviation; neither the artifact nor the operator rewrites the
original threshold or claims a formal PASS. Evidence is retained under
`bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31/`.

### Quantized EAGLE-3 draft-head evidence

`bench/draft_quant_qwen.py` compares the pinned public
`thoughtworks/Qwen3-32B-Eagle3` dense checkpoint with four offline-packed
dynamic FP8 variants against the same real Qwen3-32B target traces. Each arm
receives identical target embeddings, auxiliary residuals, KV contents, and
verification positions. The report retains exact greedy proposals,
target-corrected committed-token counts, draft and verification timing,
acceptance, module/CUDA memory, generated-checkpoint hashes, and environment
provenance. Accepted prefixes must exactly match the independently generated
teacher trace. Target verification and sequential teacher shapes may choose
different tokens only when both selected tokens remain within the established
0.25-nat reciprocal selected-logprob bound; exact correction counts and the
two token IDs, four cross-distribution log-probabilities, individual deltas,
and maximum delta remain visible in the artifact. The bound is fixed by the
formal operator and is not a CLI-adjustable pass criterion. The operator selects the
highest-goodput quantized arm that retains
at least 95% of dense acceptance, then requires memory reduction and at least
95% of dense standalone-cycle committed-token goodput. Five prompts and three
repeats rotate all five arm orders after every measured context; every draft
and target-verify shape is warmed first. This is not production serving E2E: it
includes context construction through target correction but excludes scheduler
and serving overhead. It does not enable model-draft serving or turn a slower
quantized result into a default. The exact invocation and pinned public-weight
digest are in `docs/gpu-runbook.md` §3.1.

The retained RTX PRO 6000 Blackwell run on source `d8dbdba` passes every gate.
Dense and all four FP8 arms accepted exactly 90/270 proposals (33.33%); the
selected `fp8_dense_fc` arm retained 100% of acceptance, used 55.06% of dense
module memory (861,854,720 versus 1,565,296,640 bytes), and retained 98.74% of
dense standalone-cycle goodput (25.842 versus 26.172 committed token/s). Its
draft median was 5.218 ms versus dense 4.287 ms, so the measured result is a
memory win with a 21.71% draft-latency cost and a 1.26% cycle-goodput cost, not
a speedup claim. Every teacher prefix was exact. Corrections were exact in
87/90 repeated rows per arm; the one unique cross-shape divergence had a
0.13118-nat reciprocal delta, and the maximum over every compared correction
was 0.16395 nat, both below the fixed 0.25-nat bound. Evidence:
`bench/results/issue-234-draft-quant-qwen3-32b-rtxpro6000-2026-07-31.json`
(SHA-256 `850191a039edd6e3ff5ae4bf974eadeef3227b3700b1747d281c595daad63c59`).

No compatible trained public MTP target/checkpoint is present in this
environment. MTP therefore makes no trained acceptance or performance claim:
its evidence is canonical packed-checkpoint loading, numerical tolerance, and
the real fused CUDA path with dequantization made fatal. Native EAGLE/MTP
proposal-state integration remains the existing G4 runtime boundary.

### G2 A9 DP-versus-TP crossover evidence

`bench/g2_a9_dp_tp_crossover_bench.py` produces the report-only Qwen3-32B
DP=2×TP4 versus TP8 arrival-sweep artifact. It independently replays the
post-SSE-fix A8 DP evidence retained under
`bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`, then
measures only the missing TP8 arm through a one-replica production gateway.
That comparator has 2,992/2,992 retry-free successes and 1,496 exact
placements; its sole false check is the accepted 1.9× target at a measured
1.7711× median. Both arms use the same image, commit-`86d4922` materialized
read-only runtime source, checkpoint, workload,
per-engine 8,192 usable KV pages, scheduler limits, pipeline depth, and CUDA
Graph envelope. The report explicitly records that two DP replicas therefore
have twice TP8's aggregate configured KV, sequence, batch-token, and graph-batch
capacity; this is a deployment-topology comparison, not an equal-aggregate-
capacity microbenchmark. TP8 uses the exact A8 DP cache namespace so request
bytes and token IDs match the retained comparator. Every rate
retains three fixed-seed repeats, exact request/placement rows, goodput,
TTFT, and versioned terminal-stream TPOT. PASS means evidence completeness
only; no topology wins by threshold. The report explicitly records whether
any measured ordering transition exists and attaches both arms' observed
concurrency to every transition bracket. The exact launch and replay procedure is
in `docs/gpu-runbook.md`.

The retained 2026-07-31 A9 artifact passed all 14 checks with 984/984
retry-free TP8 successes and placements. Median DP/TP8 goodput at
4/8/16/32/64 offered req/s was 3.884/3.902, 7.383/7.313,
12.948/8.994, 16.042/11.707, and 19.612/12.440 req/s. The measured
ordering transition is between 4 and 8 req/s with no interpolation, and DP is
first noninferior at 8 req/s. TP8 retains lower terminal-stream TPOT at
16–64 req/s, while DP retains higher goodput and lower TTFT under load.
Evidence is retained under
`bench/results/g2-a9-dp-tp-qwen3-32b-rtxpro6000-2026-07-31-ssefix/`.

### G4 E-KV FP8 KV evidence

`bench/fp8_kv_g4_ekv_bench.py` is the formal G4 E-KV correctness operator.
It measures the pinned Qwen3-32B checkpoint on one visible SM120 GPU, writes
raw JSONL plus a derived manifest even on failure, and supports independent
`verify` and raw-only `replay` commands. BF16 KV remains the product default;
the E4M3 candidate arm is explicit and timing is non-binding. The retained
2026-07-31 bake failed output, common-prefix logprob, and cache-NRMSE gates, so
public `fp8_e4m3` startup remains disabled. The exact mounted source-JIT
procedure, thresholds, and retained evidence are in `docs/gpu-runbook.md`
§9.9.

### G5 F4a DRAM KV crossover evidence

`bench/dram_kv_tier_qwen.py` measures the production rank-local pinned-DRAM
tier on Qwen3-32B at TP4 and TP8. Each shard uses the real RadixKV and
all-rank Gloo control path, compares restore with uncached model recomputation
over the fixed 16–8,192-token grid, and retains nine alternating paired
measurements per length. The primary metric is the rank-0 controller wall from
empty destination pages through one next-token result; pure CUDA D2H/H2D
intervals remain diagnostic evidence rather than replacing the production
boundary.

Restore validates the logical page checksums, transfers KV, replays the final
prompt-token query, and samples. Cold recompute processes the complete prompt
through its natural production chunks, including the final prompt token, then
samples that hidden state without an extra one-token model invocation. Schema
v2 binds every raw shard and runtime profile to the exact versioned
fragment-major CUDA transfer backend. Schema-v1 raw cannot be mixed into,
relabelled as, or used to seed a v2 profile.

Run TP4 and TP8 in separate, non-overlapping containers from the same clean
commit, immutable image, and checkpoint, then seal and verify the artifact:

```bash
python bench/dram_kv_tier_qwen.py run --tp 4 ... --output tp4-raw.jsonl
python bench/dram_kv_tier_qwen.py run --tp 8 ... --output tp8-raw.jsonl
python bench/dram_kv_tier_qwen.py assemble \
  --tp4-raw tp4-raw.jsonl --tp8-raw tp8-raw.jsonl \
  --output-dir bench/results/<f4a-artifact> --assert-gate
python bench/dram_kv_tier_qwen.py verify \
  --artifact-dir bench/results/<f4a-artifact> --assert-gate
```

The generated TP-specific profiles are startup inputs, not editable tuning
files. Runtime binding replays every raw pair, requires a stable measured
suffix (median restore/recompute ratio below 1 with at least 8/9 restore wins),
and rejects any model, TP, KV-layout, attention implementation, source,
hardware-transport, or host-placement identity mismatch.

The retained 2026-08-01 exact-source run closes F4a. TP4's stable suffix starts
at 1,024 tokens: its 512-token cell failed at a 1.021531 median ratio and 2/9
wins, while 1,024 tokens passed at 0.975449 and 8/9 and every larger cell
passed. TP8 passed all cells from the 16-token measured lower bound, so the
manifest reports the crossover at or below 16 tokens and its deployable
profile conservatively sets `min_restore_tokens` to 16. Assembly, verification
from the retained copy, and independent raw replay all pass. The raw shards,
profiles, manifest, and full container provenance are retained at
`bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`.

### G5 F4b agentic DRAM-tier evidence

`bench/agentic_kv_tier_f4b_bench.py` is the checkout-only formal operator for
the Qwen3-32B TP4 agentic tier on/off gate. The predeclared transcript has 16
sessions and eight turns per session. Every prompt contains a 2,048-token
fleet-shared prefix plus 512 new session-history tokens per turn, and every
request generates exactly 32 output tokens. Cached-prompt truth comes from the
production engine's own cached-token accounting, not from an expected prefix
length or a benchmark-side cache model.

The experiment uses two disjoint four-GPU cohorts and four fresh, sequential,
non-overlapping arms. Round 0 runs tier off on GPUs 0--3 and tier on on GPUs
4--7; round 1 swaps the arm assignment, running tier on on GPUs 0--3 and tier
off on GPUs 4--7. This AB/BA cohort swap separates the tier effect from a
fixed cohort effect. Assembly requires a positive engine cached-token-rate
gain in the pooled requests and independently in each cohort, and reports both
absolute and relative gains.

TPOT is request-level `stream-terminal-token-v1`: each request retains its
stream token-boundary timestamps, including the terminal boundary, and p99 is
computed by nearest rank. Tier-on is noninferior only when both the pooled
tier-on/off p99 ratio and the geometric mean of the two cohort ratios are at
most 1.10. The 10% bound is predeclared to tolerate ordinary host and OS
timing variation; it does not permit missing requests, retries, overlapping
arms, or replacement of real stream timestamps with model-only CUDA timing.

Decode-path exclusion is measured at synchronous `EngineLoop` step
boundaries. The raw evidence records DRAM-tier counters and free-page counts
at every output boundary. Offload and restore counters must not advance after
the first output token. The expected free-page decrease from output 16 to 17
must also be observed, proving that the same boundary instrumentation sees a
real decode-page allocation rather than making the zero tier-delta check
vacuous. Raw request timing, cached-token usage, output tokens, and tier/cache
events are retained for independent replay.

Free-running greedy output equality across tier-off and tier-on is retained as
a diagnostic, not treated as a semantic invariant. A BF16 near-tie can choose
a different first token after the cache-prefill shape changes even when both
paths remain numerically equivalent; later tokens then have different
generated prefixes and cannot be compared as though they had the same input.
The performance arms still bind the prompt/order exactly and report same-arm
cross-cohort output stability.

Distribution quality is therefore measured in a separate timing-nonbinding
companion run so that requesting logprobs cannot alter the TPOT evidence. Two
fresh, sequential cohort-A containers repeat the exact tier-off and tier-on
trace with top-64 logprobs. Their checkpoint, trace, GPU cohort, calibrated
image, engine/runtime identity, output tokens, cached-token usage, and
tier-state behavior must match the corresponding performance arms. While the
generated prefix is still common, every selected-token logprob must agree
within 0.25 nat. At the first divergent token, each arm's selected token must
be present in the other arm's top 64 and both cross-arm differences must also
be at most 0.25 nat. Positions after that divergence are not compared. The
quality artifact retains and independently validates the two containers'
created/exited Docker records just as the performance artifact does.

After four raw arms have completed from the same clean source, checkpoint,
retained TP4 F4a runtime profile, and the exact immutable image in which that
profile was calibrated, assemble and replay the evidence without editing a
measured verdict. The calibrated execution image and the read-only
bind-mounted measured source are separate provenance authorities: the image
pins the compiled runtime dependencies, while the source record pins every
executed Kairyu Python file and the engine rollup. Assembly requires the image
inspect plus each arm's full container ID and created/exited inspect pair. It
validates both provenance authorities and those lifecycle records, embeds
their descriptors in the combined raw, and copies the exact input files into
the artifact's `container-metadata/` directory.

```bash
python bench/agentic_kv_tier_f4b_bench.py assemble \
  --raw round0-off-raw.jsonl \
  --raw round0-on-raw.jsonl \
  --raw round1-on-raw.jsonl \
  --raw round1-off-raw.jsonl \
  --metadata-dir metadata \
  --output-dir bench/results/<f4b-performance-artifact> --assert-gate
python bench/agentic_kv_tier_f4b_bench.py verify \
  --artifact bench/results/<f4b-performance-artifact> --assert-gate
python bench/agentic_kv_tier_f4b_bench.py replay \
  --raw \
    bench/results/<f4b-performance-artifact>/agentic-kv-tier-f4b-raw.jsonl \
  --assert-gate
python bench/agentic_kv_tier_f4b_bench.py seal-quality \
  --performance-artifact bench/results/<f4b-performance-artifact> \
  --quality-raw quality-off-raw.jsonl \
  --quality-raw quality-on-raw.jsonl \
  --quality-metadata-dir quality-metadata \
  --output-dir bench/results/<f4b-artifact> --assert-gate
python bench/agentic_kv_tier_f4b_bench.py verify-quality \
  --artifact bench/results/<f4b-artifact> --assert-gate
python bench/agentic_kv_tier_f4b_bench.py replay-quality \
  --performance-raw \
    bench/results/<f4b-artifact>/agentic-kv-tier-f4b-raw.jsonl \
  --quality-raw \
    bench/results/<f4b-artifact>/agentic-kv-tier-f4b-quality-raw.jsonl \
  --assert-gate
```

`replay` recomputes the verdict from the combined raw and validates its
embedded container lifecycle. `verify` does that same replay, checks the
retained manifest, and additionally rehashes every retained
`container-metadata/` byte against the embedded descriptors.
`replay-quality` independently recomputes the distribution verdict from the
unchanged performance raw plus the quality raw, while `verify-quality` also
rehashes both metadata trees and both retained manifests. A publishable
artifact therefore retains the performance and quality combined raw files,
both manifests, and the complete `container-metadata/` and
`quality-container-metadata/` trees together.

The complete clean-source container procedure is in
`docs/gpu-runbook.md` §9.8b. F4b is closed by the retained Qwen3-32B TP4
artifact at
`bench/results/g5-f4b-agentic-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`.
Tiering raised the pooled engine prefix-hit rate from 47.7941% to 60.2338%
(+12.4397 percentage points). The pooled TPOT p99 ratio was 1.03721 and the
cohort-ratio geometric mean was 1.04488, both within 1.10. No tier counter
advanced after first content, while the decode allocation control fired.
The separate quality arms reproduced their parent outputs, cache usage, and
per-request tier counters exactly. Across 3,968 comparable generated-prefix
positions the maximum selected-logprob difference was 0.195256 nat; the
maximum reciprocal difference at four first divergences was 0.213124 nat.
Every divergent tier-on request restored 160 pages with no fallback or
ownership failure. Sealing, retained-copy verification, and independent raw
replay all passed.

## Fixtures, results, and wheel verification

The eight installed fixtures are synthetic plumbing inputs, never substitutes
for publishable benchmark measurements:

```text
charxiv-reasoning.jsonl
gpqa-diamond.jsonl
hle.jsonl
livecodebench-pro.jsonl
livecodebench.jsonl
long-context-reasoning.jsonl
mrcr-v2.jsonl
scicode.jsonl
```

Routine measurements go under `bench/results/` and remain ignored. Retain an
artifact only when a goal or design decision explicitly requires reviewable
evidence, and retain its complete config/provenance rather than an isolated
summary number.

The packaging gate builds a real wheel, inspects its contents, and imports it
from an isolated temporary directory. It proves that the console dispatch,
entrypoint manifest, and all eight fixtures are present, while the top-level
`bench/`, `bench/results/`, and `tests/` trees are absent:

```bash
uv run --frozen python scripts/verify_bench_entrypoints.py
uv run --frozen python scripts/verify_bench_wheel.py
```

The first command separately exercises all 56 registered wrappers through
both their path and module `--help` forms. It runs once in CI, on Python 3.12,
after the declared development dependencies are synced, without duplicating
110 subprocesses in every portable test cell.
