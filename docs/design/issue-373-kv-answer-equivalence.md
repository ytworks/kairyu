# Issue #373 Design: B7 KV Answer-Equivalence Companion Gate

Status: **Implemented** (2026-08-05).

Related contracts: M2 radix-paged KV, M10 A32/A33 (F2c/F2d), G5 F4a
pinned-DRAM crossover, and G5 F4b agentic DRAM-tier evidence.

## 1. Goal and scope

B7 adds one narrow semantic prerequisite to every existing KV feature gate:
the native engine must emit exactly the same 32 greedy token IDs when an
identical prompt is evaluated first from a cold cache and then from a proven
warm cache in the same persistent runtime.  The companion is intentionally
cheap and timing-nonbinding.  It isolates reuse correctness from the routing,
placement, crossover, and agentic-performance questions answered by F2c,
F2d, F4a, and F4b.

This is an additive child artifact.  It does not change, upgrade, reseal, or
reinterpret any F2c, F2d, F4a, or F4b schema, raw file, threshold, or retained
verdict.  Each B7 cell instead carries SHA-256 identities for its reviewed
parent manifest and raw evidence.  The fixed production-topology matrix has
five cells in this exact order: `f2c-tp2`, `f2d-tp2`, `f4a-tp4`, `f4a-tp8`,
and `f4b-tp4`.  F4a therefore covers both topology rows retained by its one
parent manifest.  A publishable aggregate requires all four parent gates and
all five native cells to validate.  Changing a parent byte makes a new B7
assembly necessary; it never permits editing the parent into a different
result.

For F4b, “parent gate” means its sealed performance+distribution-quality
closure, not the performance manifest alone.  The sibling quality manifest
and raw must be present in the performance artifact directory, independently
replay to PASS through the owning quality verifier, and bind that exact
performance result.

The gate does not claim cross-model, cross-GPU, cross-process, or cross-kernel
bitwise reproducibility.  It also does not replace the existing unit coverage
for radix eviction, page-table reuse, or DRAM ownership.  Its one claim is
causal and end to end: after the evidence proves that the same runtime served
the same prompt through the intended KV reuse path, that reuse did not change
the generated answer.

## 2. Why existing output comparisons remain diagnostic

The F2c performance arms intentionally route a request to different replicas,
GPU pairs, and independently populated caches.  F4b's performance and quality
arms likewise compare separate tier-off and tier-on containers whose prefill
shapes can differ.  In BF16 a numerically harmless near tie can therefore move
the first greedy argmax.  Once one token differs, later positions have
different autoregressive inputs and exact full-sequence equality can no longer
diagnose cache corruption.  F2d is a deterministic policy-replay proof rather
than a native model-output experiment.

Those cross-arm output-match rates remain useful diagnostics, and F4b's
common-prefix top-64/logprob quality contract remains unchanged.  They do not
become B7 pass/fail inputs.  B7 removes the confounds by keeping one native
runtime, checkpoint, engine configuration, prompt token IDs, sampler settings,
and execution implementation fixed.  Only the prompt-KV state changes from a
proved miss to the feature's proved reuse path.

## 3. Native cold/warm contract

`bench/kv_answer_equivalence_bench.py run-native` executes one fixed cell in one
persistent native runtime.  The prompt and sampling contract are fixed before
the first request.  Both requests use `temperature=0`, `seed=0`,
`min_tokens=max_tokens=32`, and `ignore_eos=true`; early termination, a missing
token, text-only agreement, or prefix-only agreement fails closed.

The sequence is:

1. Start from an isolated cache namespace with no reusable prompt KV.  Run the
   complete tokenized prompt and retain the engine-reported cold cache usage,
   exact prompt token IDs plus their derived descriptor, and 32 output
   IDs/pieces.
2. Keep that runtime alive and submit the token-identical prompt
   again with the identical native sampling configuration.  Retain the warm
   cache usage, the same exact prompt token IDs and derived descriptor, and 32
   output IDs/pieces.
3. Prove the feature-specific reuse prerequisite below, then require the cold
   and warm output-ID arrays to be exactly equal.

All evidence comes from the native engine boundary.  A benchmark-side expected
prefix length, decoded text, router reason, synthetic cache label, or copied
answer cannot establish a hit.  Both native requests must complete without an
error and report `finish_reason=length`, 32 native token IDs, 32 token pieces,
and usage whose prompt count equals the descriptor and whose completion count
is 32.  Their
prompt records (`token_ids`, their recomputed `token_ids_sha256`, and their
recomputed `token_count`), output IDs, pieces, and text must be exact.  Both
requests must carry the cell's same per-run nonce and structured runtime
identity.  The five assembled cells must use five distinct nonces, preventing
one captured runtime from being relabelled as multiple topology cells.
The raw row is still retained when the equality or path proof fails, and
`--assert-gate` returns nonzero.

The retained IDs are tokenizer-domain evidence, not arbitrary model-logit
indices. Qwen3-32B's model configuration pads the LM-head/logits width to
151,936, but its tokenizer ID domain contains only 151,669 entries. B7 fixes
`TOKENIZER_VOCAB_SIZE = 151669` and validates every prompt and response ID
against `0 <= token_id < TOKENIZER_VOCAB_SIZE`; IDs 151,669 through 151,935
are logits padding and fail closed even though the model exposes those rows.
The 151,936 logits width and the 151,669 tokenizer boundary must never be
described or validated as one vocabulary size.

### 3.1 F2c and F2d radix prerequisite

The `f2c-tp2` and `f2d-tp2` cells exercise ordinary production RadixKV reuse.
The cold
request must report no cached prompt tokens.  The second request must report a
native cached-token count equal to the complete prompt length, with the same
prompt token IDs and live radix-cache/runtime identity.  Immediately before
that request, the cache's non-mutating native residency probe must independently
report the same complete prefix; the proof cannot copy the response usage
claim.  Its `radix_reuse` proof must set `same_runtime=true` and its
`cache_hit_tokens` must equal both the probe and warm response count.  This is the semantic
prerequisite for F2c's KV-aware routing benefit and F2d's learned prefix-weight
policy; B7 does not rerun their multi-replica routing or policy performance
experiments.

### 3.2 F4a and F4b DRAM prerequisite

The `f4a-tp4`, `f4a-tp8`, and `f4b-tp4` cells enable the exact reviewed DRAM
profile for their parent topology and use fixed device and DRAM capacities.
One or more pressure requests run only between the cold and warm comparison
requests and force the prompt's reusable pages through the native
offload/restore lifecycle; pressure outputs never participate in answer
equality.  A cache-hit count alone is insufficient.  From before/pre-warm/after
values and deltas over the six retained tier counters (`offload_pages`,
`offload_bypassed_pages`, `restore_pages`, `restore_attempts`,
`restore_fallbacks`, and `ownership_failures`), the raw evidence must prove
positive offload and restore deltas and zero restore fallback or ownership
failure.  The pressure interval must account for offload without restore; the
subsequent warm interval must independently account for restore, so unrelated
pressure traffic cannot satisfy the path proof.  The warm request must report
the complete prompt length as restored cached tokens before its exact 32-token
answer can satisfy equality.

F4a TP4, F4a TP8, and F4b TP4 retain separate cells because their parent
claims, topology, capacity, and profile bindings differ, even when they use the
same native DRAM mechanism.  B7 does not change F4a's crossover profile or
F4b's TPOT/cache-gain/distribution thresholds.

### 3.3 Frozen topology and runtime identity

Every geometry field is selected by `--cell`; callers cannot override TP,
page count, capacity, decode mode, batch-token bound, or model-length bound.
All cells use 16-token pages and a 1,024-token comparison prompt.  F4a TP8's
reviewed restore crossover is one 16-token page, but the comparison keeps the
1,024-token pressure floor so target eviction remains attributable.

| Cell | TP | Device pages | DRAM pages | Decode | Max batched tokens | Max model length | Reviewed minimum restore |
|---|---:|---:|---:|---|---:|---:|---:|
| `f2c-tp2` | 2 | 8,192 | 0 | CUDA graph | 1,024 | 8,192 | n/a |
| `f2d-tp2` | 2 | 8,192 | 0 | CUDA graph | 1,024 | 8,192 | n/a |
| `f4a-tp4` | 4 | 1,024 | 512 | eager | 2,048 | 8,193 | 1,024 tokens |
| `f4a-tp8` | 8 | 1,024 | 512 | eager | 2,048 | 8,193 | 16 tokens |
| `f4b-tp4` | 4 | 1,024 | 2,048 | eager | 2,048 | 8,192 | 1,024 tokens |

Before GPU construction, `run-native` independently replays the supplied
parent through that parent's owning verifier and requires its complete
retained verdict to pass.  It then full-hashes the pinned Qwen3-32B revision:
all 17 weight shards with names, byte sizes, and SHA-256s; both weight
rollups; architecture and canonical config identity; the exact six metadata
files (`config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, and `merges.txt`); the separate
safetensors-index hash; and the pinned parent revision identity.  The
five cells must retain exactly the same checkpoint identity.

The structured runtime descriptor also binds one Git commit, a clean tracked
tree, and SHA-256s for the wrapper, pure replay contract, engine/KV paths,
package metadata, and lockfile.  Clean-source preflight rejects untracked or
ignored import artifacts anywhere in the repository (source, direct or cached
bytecode, path hooks, native extensions, and symlinks), and every loaded repo
module must be tracked and byte-identical to `HEAD`.  Repository
`__pycache__` directories must therefore be removed before formal capture.
The repo-local `.venv` is the sole generated-code exclusion; its locked
interpreter, site bootstrap, and installed packages are part of the formal
environment TCB.  The same source identity is recomputed after
backend shutdown.  Model, parent, and profile paths are resolved once before
use; the complete checkpoint is rehashed after shutdown and must exactly match
its preflight identity.  All five cells must retain exactly the same clean
source and checkpoint identities.  Each cell additionally records and
validates its frozen engine geometry and DRAM-profile digest against the
constructed native backend, rather than trusting command-line labels.

Publishable B7 work therefore has a narrower startup contract than ordinary
checkout entrypoint discovery. Each `run-native`, `assemble`, `verify`, or
`replay` command must start through the direct checkout script path with an
already-isolated `python -I -B` interpreter; the wrapper refuses a plain
process instead of trying an unsafe late restart. Before exposing the repository on `sys.path`, the
wrapper internally creates an unpredictable, exclusive 0700 directory under
`/tmp` with `tempfile.TemporaryDirectory`, assigns that directory to
`sys.pycache_prefix`, proves `git diff --quiet HEAD --`, and rejects untracked
or ignored `.py`, `.pyc`, `.pth`, native `.so`, and symlink import artifacts
outside `.venv`. Existing repository `__pycache__` therefore fails before a
repository package can execute. An operator-side clean/artifact check may fail
earlier, but cannot replace this wrapper-owned pre-import preflight. `-I`
removes ambient `PYTHONPATH` and user-site influence, but does not attest the
chosen interpreter, `.venv` site bootstrap, or installed packages; those
remain TCB. Module invocation is excluded from the declared inventory; its
unsupported non-evidence `--help` convenience remains harmless, but no
`python -m` evidence command is supported or publishable.

## 4. Parent binding and aggregate verdict

Every `run-native` invocation requires `--parent-manifest` and `--parent-raw`.
The child records their content hashes, exact parent topology, and the
checkpoint constraint needed to keep the cell label honest.  F2c must prove
the reviewed four-service/eight-GPU TP2 topology and directly bind the parent
model, revision, and weight rollup.  F4a TP4/TP8 and F4b TP4 directly bind the
full parent checkpoint identity and exact TP.  The two F4a cells must share one
parent manifest while using its distinct TP4 and TP8 raw shards.

Parent manifest, raw, the F2d router sibling, F4a sibling-profile bytes, and
both F4b quality files are
snapshotted before the owning verifier runs.  Replay, digest binding, and
semantic extraction must all refer to that one snapshot, and each path must
remain byte-identical through the end of extraction.  A verifier-time or
extraction-time artifact swap fails closed instead of binding an unverified
file version.  The F4b child records separate quality-manifest and quality-raw
SHA-256s; all other cells retain `null` for those two fields.  Parent structures
and owner replay results compare as canonical, type-exact JSON rather than
Python equality, and F4b quality raw is parsed as strict canonical JSONL with
duplicate keys and non-finite values rejected. Every parent row index and
schema-fixed numeric/boolean field uses its exact JSON type; F2d's noncanonical
legacy router encoding is snapshot-parsed with duplicate/non-finite rejection
and an exact row schema.

F2d is a virtual policy replay and its retained parent contains no model or
checkpoint assertion.  B7 does not invent one.  Its
`aggregate-common-f2c` constraint instead requires the `f2d-tp2` runtime
checkpoint to be exactly the same checkpoint already bound by `f2c-tp2`, and
the aggregate also requires that identity across all five cells.

`assemble` accepts exactly one raw child for each fixed cell in the frozen
order.  It independently recomputes the cold/warm verdicts, validates parent
and topology bindings, enforces common checkpoint/source identities and five
unique runtime nonces, writes a combined raw file and derived manifest, and
reports PASS only when all five cells pass.  It does not import any parent's
performance result into the B7 math or pretend that a historical parent was
rerun; the owning verifier replay establishes only that the supplied retained
parent is internally replayable and passing.

`verify` rehashes the retained artifact, checks the derived manifest, and
replays every row.  `replay` derives the verdict from the artifact's retained
raw evidence rather than trusting the stored summary.  Omission, duplication,
unknown fields that weaken a binding check, non-finite or malformed evidence,
hash drift, or a derived verdict mismatch fails closed in all applicable
commands.

## 5. Reproducible command surface

Run every cell sequentially from the same clean commit and pinned checkpoint,
using a fresh isolated process, wrapper-owned temporary cache namespace,
evidence location, and native runtime each time. Execute from the reviewed
checkout root; the wrapper repeats its tracked-clean and import-artifact
preflight before making the repository importable on every command below.

`--cell` supplies the frozen geometry above.  Parent raws and DRAM profiles
must match that cell's reviewed deployment; the placeholders below are not
permission to mix parents, TP degrees, or runtime profiles.

```bash
python -I -B \
  bench/kv_answer_equivalence_bench.py run-native \
  --cell f2c-tp2 --model-path <model-path> \
  --parent-manifest <f2c-parent-manifest> \
  --parent-raw <f2c-parent-raw> \
  --output <run-root>/f2c-tp2.jsonl --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py run-native \
  --cell f2d-tp2 --model-path <model-path> \
  --parent-manifest <f2d-parent-manifest> \
  --parent-raw <f2d-parent-raw> \
  --output <run-root>/f2d-tp2.jsonl --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py run-native \
  --cell f4a-tp4 --model-path <model-path> \
  --parent-manifest <f4a-parent-manifest> \
  --parent-raw <f4a-tp4-parent-raw> \
  --dram-profile <f4a-tp4-dram-profile> \
  --output <run-root>/f4a-tp4.jsonl --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py run-native \
  --cell f4a-tp8 --model-path <model-path> \
  --parent-manifest <f4a-parent-manifest> \
  --parent-raw <f4a-tp8-parent-raw> \
  --dram-profile <f4a-tp8-dram-profile> \
  --output <run-root>/f4a-tp8.jsonl --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py run-native \
  --cell f4b-tp4 --model-path <model-path> \
  --parent-manifest <f4b-parent-manifest> \
  --parent-raw <f4b-parent-raw> \
  --dram-profile <f4b-bound-dram-profile> \
  --output <run-root>/f4b-tp4.jsonl --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py assemble \
  --f2c-tp2 <run-root>/f2c-tp2.jsonl \
  --f2d-tp2 <run-root>/f2d-tp2.jsonl \
  --f4a-tp4 <run-root>/f4a-tp4.jsonl \
  --f4a-tp8 <run-root>/f4a-tp8.jsonl \
  --f4b-tp4 <run-root>/f4b-tp4.jsonl \
  --output-dir bench/results/<b7-artifact> --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py verify \
  --artifact bench/results/<b7-artifact> --assert-gate

python -I -B \
  bench/kv_answer_equivalence_bench.py replay \
  --artifact bench/results/<b7-artifact> --assert-gate
```

Only the direct path commands above are supported for evidence capture,
assembly, verification, and replay. The module form is excluded from the
declared inventory and remains only as unsupported, non-evidence `--help`;
its output cannot be published as B7 evidence. The wrapper and artifacts
remain checkout-only and are never installed in the wheel.

## 6. Portable verification

CPU tests use deterministic fake native runs to cover exact 32-token equality,
cold/warm usage truth, radix miss/hit transitions, separated DRAM
pressure/offload and warm/restore evidence, owning-parent verification seams,
checkpoint/source/topology binding, five-cell order and nonce uniqueness,
independent radix-residency proof, start/end identity drift, immutable parent
and replay snapshots, import-shadow rejection, tamper rejection, manifest
verification, and raw-only replay.  They do not
claim that a GPU model ran, a GPU cache hit occurred, or a GPU DRAM transfer
occurred.  A publishable B7 artifact still requires the five native
real-checkpoint `run-native` cells described above.
