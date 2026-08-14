# Quantization x task-accuracy sweep (issue #372)

Status: implemented

## Decision

Kairyu has a separate `quantization` benchmark suite with the fixed task order
`gsm8k`, `mmlu`, `ifeval`, `gpqa-diamond`. The existing three-row `core` suite
is unchanged: adding GPQA there would silently change its history and cheap-core
call-count contract. GPQA supplies one judge-free reasoning evaluation; all
four headline item outcomes are binary.

One complete indexed run contains exactly seven unique served targets in any
configuration order. The published artifact always renders them in this
versioned order:

1. dense weights, BF16 compute, BF16 KV (`bf16`, the reference);
2. FP8 weights, BF16 compute/KV;
3. INT8 weights, BF16 compute/KV;
4. AWQ weights, BF16 compute/KV;
5. GPTQ weights, BF16 compute/KV;
6. NVFP4 weights, BF16 compute/KV; and
7. dense weights, BF16 compute, FP8-E4M3 KV (`fp8-kv`).

BF16 is deliberately not encoded as a weight quantization method. A dense
checkpoint declares `weight_method: none` and separately declares
`compute_dtype: bfloat16`. The KV field is the effective storage dtype and
never accepts unresolved `auto`. Missing, duplicate, additional, or malformed
profiles and reused deployment digests are input errors; a sweep cannot shrink
itself to whichever arms happened to finish.

## Identity and support boundary

Every target requires a distinct `served_config` label and lowercase SHA-256.
Operators hash a canonical deployment manifest that binds at least the common
base-model and tokenizer revisions, checkpoint tensors, exact quantization
dialect/bits/group/scales/calibration/ignored layers, compute and effective KV
dtypes, Kairyu build or immutable image, parallel topology, and relevant
hardware. The compact `quantization` profile only classifies table rows; the
manifest digest carries exact configuration identity.

Both are operator declarations. The benchmark client does not remotely attest
the checkpoint, selected kernels, compute/KV dtype, or every replica behind a
pool. Reports say `operator_declared_not_remotely_attested` and never promote a
declared profile into a runtime support claim. Target `model`, sampling and
reasoning options, context/output limits, vision policy, and every other request
field must remain identical across arms. Only target label, endpoint,
credential-variable name, served manifest, and quantization classification may
differ.

The current source explicitly rejects public `fp8_e4m3` KV after the G4 E-KV
bake failed output/logprob/cache-quality gates. The required `fp8-kv` row can
therefore measure only an explicitly available external or experimental
deployment under this source; adding the row does not enable or claim native
Kairyu support. Weight formats likewise retain their actual loader, hardware,
model-family, and parallelism restrictions. A downstream task score is evidence
about the declared served arm, not proof that every topology supports that
format.

## Evidence and gate contract

`python -m evals quant-sweep --run RUN` first validates the entire suite-local
SHA-256-chained scoreboard index. It then reloads all 28 raw `PairResult` cells
through the authoritative no-symlink indexed loader. The task tolerance map
must name all four tasks exactly once. One margin per task applies to every
candidate; candidate-specific after-the-fact margins are not accepted.

For each of the six candidates, the public issue-#365
`build_config_comparison()` is the sole evidence and statistics core. This
requires a clean, full-data, non-smoke, non-fixture run; complete pairs and
items; exact unique item-ID sets; exact counts and recomputed means; allowed
cross-run provenance; matching harness/runtime/dataset/evaluator/request
methodology; and independent judging where relevant. MMLU requires its native
teacher-forced likelihood evidence and GPQA requires its gated pinned dataset;
an unavailable/skipped row is unevaluable, never a zero score.

All four rows use Newcombe paired-score method 10. Each cell retains absolute
candidate/reference scores, full-precision delta, two-sided 95% interval,
one-sided 95% lower confidence bound, tolerance, decision, item count and ID
digest, and both pair SHA-256s. A cell passes only when `LCB >= -tolerance`.
A candidate passes only when all four cells pass, and the sweep passes only
when all six candidates pass. Tasks and schemes are never averaged.

## Retained artifacts and compatibility

The source run owns dedicated atomic `quantization-sweep.json` and
`quantization-sweep.md` files. A valid quality failure is saved before exit 1;
invalid inputs/evidence return exit 2 without replacing a valid artifact. Exit
0 means all cells passed. The JSON embeds all six complete #365 comparison
artifacts plus a scheme-major summary, source run/fingerprint/index-record
bindings, fixed coverage contract, policy, operator-attestation boundary,
comparison/runtime identities, and a content hash over the sweep itself. Its
protocol content-binds the sweep, #365 statistics, history, schema/store, and
weight/KV classification sources.

The Markdown contains one absolute-accuracy row per profile, a second table of
all 24 candidate/task gates, evidence hashes, and the FP8-KV support warning.
The aggregate files and quantization scoreboard index are explicitly eligible
for Git retention; raw routine pair data stays ignored.

Core row order and recorded target shape remain unchanged by the
quantization sweep. `BenchTarget.quantization` is optional, and absent values are
omitted from durable configuration rather than adding `quantization: null` to
older run identities. Ordinary configuration A/B now treats a declared
quantization profile as an intended deployment-arm difference while preserving
all existing request-policy checks.
