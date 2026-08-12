# Benchmark Suites (`kairyu bench`)

One command runs an installed answer-quality suite against a deployed Kairyu
gateway — single models and orchestrations side by side — then writes a dated,
footnoted scoreboard. The default `accuracy` suite runs every benchmark from
Sakana's Fugu release table
([sakana.ai/fugu-release](https://sakana.ai/fugu-release/)) against a deployed
gateway. The `core` suite provides judge-free, Docker-free GSM8K, MMLU, and
IFEval regression checks. The `structured` suite pairs enforced JSON-Schema
generation with an otherwise identical unconstrained control over a fixed
five-category corpus. This implements goal G6 gate P-C1 ("one command → dated
scoreboard") and the roadmap §6 evidence rules (per-item results, methodology,
config committed next to every number).

The perf and formal-gate harnesses in the top-level `bench/` directory are a
separate, checkout-only surface; this installed suite measures answer quality.
The complete ownership policy and wrapper inventory live in
[`bench/README.md`](../bench/README.md).

## Installed package vs source-checkout tools

`kairyu/bench/` is the single owner of reusable benchmark config, target types,
credential resolution, statistics, atomic reporting, adapters, the public
`kairyu bench` CLI, 17 synthetic benchmark stand-ins, the fixed structured
conformance corpus, and the published-gold judge calibration corpus. All 19
JSONL resources ship in the wheel.
`kairyu/bench/entrypoints.toml` is also packaged and records every supported
repository-only benchmark executable.

Top-level `bench/*.py` files are developer/formal wrappers, not installed
commands. From a source checkout each registered wrapper supports the path and
optional module forms declared in the packaged inventory; B7 and A12 evidence
are path-only. This keeps historical commands and artifact provenance replayable.
`bench/results/` is likewise
checkout-only: routine outputs are ignored, while explicitly reviewed formal
evidence can be retained. Neither wrappers, result artifacts, nor `tests/` are
included in the wheel.

Retained top-level evidence is discoverable through the checkout-only
`bench/results/index.json`. Its strict validator proves exact coverage of
Git-tracked result roots and points bundle records at their authoritative
summary, without treating ignored or untracked runtime output as published
evidence. Missing historical measurement metadata stays `null`, and the index
verdict never replaces the owning artifact's replay or integrity check.

Inspect the packaged inventory or validate a checkout with:

```bash
kairyu bench entrypoints
kairyu bench entrypoints --json
kairyu bench entrypoints --check-repo .
uv run --frozen python scripts/verify_bench_entrypoints.py
uv run --frozen python scripts/verify_bench_results_index.py
uv run --frozen python scripts/verify_bench_wheel.py
```

After the declared development dependencies are synced, the first verifier
exercises all 66 registered wrappers through their 130 declared `--help` forms
without executing workloads or contacting external runtimes.
The last command builds and imports a real wheel from an isolated temporary
directory. It verifies the public CLI dispatch, packaged manifest and all 19
JSONL fixtures, and rejects accidental inclusion of top-level benchmark
scripts, results, or tests. Gate-specific code stays in its stable wrapper;
semantics shared by the installed CLI and gate scripts belong in
`kairyu.bench`.

The package-owned `kairyu.bench.evidence` module provides the narrow common
artifact substrate used by repository-only formal gates: canonical JSON and
SHA-256, atomic indexed JSONL, strict fail-closed framing, artifact path
resolution, raw-only replay, and exact retained-manifest verification. It does
not define gate row schemas, acceptance thresholds, diagnostics, or verdicts;
those remain with each stable `bench/*.py` owner. The boundary and migration
are recorded in `docs/design/issue-382-evidence-library.md`.

The shared target form is `name=base_url=model[=api_key_env]`; the fourth
field names an environment variable and is never a literal secret. This
corrects the old `frontier_compare.py` interpretation. Export the credential
and pass its variable name when migrating an old command. The legacy
`serving_bench.py --api-key` flag remains executable but is deprecated in
favor of `--api-key-env`. An explicitly named but unset credential variable
fails closed, and no resolved key is recorded in artifacts or validation
errors.
All construction paths normalize the API root before run fingerprinting.
Historical runs remain reportable, but a run fingerprinted from a YAML URL
without `/v1` must use a new run ID rather than silently resume.
New serving/frontier artifacts explicitly record
`percentile_method: nearest-rank-v1`; older artifacts without that field used
frontier's floor-index calculation and serving's median p50/floor-index p99.
Versioned formal-gate artifact schemas remain source-bound compatibility
contracts rather than generic reporting implementations.

`bench/serving_bench.py --stage-trace` opts Kairyu requests into the structured
trace contract and reports native `tokenize`, `queue_wait`, `schedule`,
`prefill`, `decode_step`, `detokenize`, and `sse_write` p50/p99 durations beside
TTFT/TPOT. Artifacts distinguish valid, partial, missing, invalid, and
not-requested coverage, and every expected native stage records its observed
and missing request denominator. Unsupported targets are missing rather than
zero-valued. Unsafe free-form orchestration producer labels are omitted from
retained events without invalidating the rest of an otherwise valid envelope.
The flag is off by default so diagnostic instrumentation does not contaminate
ordinary Kairyu/vLLM/SGLang comparisons.

`bench/serving_bench.py --profile` is a separate, local diagnostic. It wraps
only the benchmark client's measured request wave in a CPU-only
`torch.profiler` scope; it cannot observe the target server, remote GPUs,
Kairyu engine subprocesses, vLLM, or SGLang. The option therefore requires
PyTorch from the `engine` extra or development dependencies and a non-empty
`--results-dir`. Server-process Python/CUDA diagnosis remains owned by
`scripts/profile_server.py`, while `--stage-trace` remains the target-reported
structured timing contract. The two serving flags may be combined, but their
evidence is never relabeled or merged.

Each profiled run creates one UTC-microsecond `*-serving.json` result and one
same-stem `*.client.pt.trace.json` sidecar. The result binds the trace basename,
format, byte length, SHA-256, CPU activity, local-client scope, and the facts
that the target process is excluded and timings are diagnostic-only. Trace
publication is private (`0600`), bounded to 64 MiB, strict-JSON validated, and
exclusive: an existing file, directory, symlink, unsafe path, failed export,
or concurrent publisher fails rather than replacing evidence. These checks run
before target traffic where possible, and an export failure cannot leave a
successful result JSON. Result publication is also exclusive; if it loses a
race, only the exact trace just published by that run is rolled back. Profiling
is off by default and imports no PyTorch on the normal or `--help` path. Raw
Chrome traces can contain operator names and local runtime metadata; inspect
them before sharing. Profiled TTFT, TPOT, throughput, and goodput must not be
compared with ordinary runs. The example Qwen multi-GPU serving reporter skips
profiled result files and reports them as rejected diagnostics.

`bench/proc_wire_bench.py` is a deterministic process-boundary complexity
gate. It encodes the same cumulative generation trace through the legacy and
versioned-delta `kairyu-proc` msgpack paths, retains every frame size, and
asserts linear v2 byte growth versus the legacy quadratic control. Serialized
bytes are binding; wall time is deliberately absent so OS jitter cannot change
the result. The retained issue #212 closure artifact is
`bench/results/proc-wire-delta-2026-07-29.json` (source `5c634ee`, artifact
SHA-256 `02054d9def30281f493c50ac6a774069b51f9eaca6178374609e3aa31021fb0f`).
At 1,024 output tokens it records 31,012,271 legacy bytes versus 356,199 v2
bytes; doubling output produces 3.91–3.98x legacy growth and 2.01–2.03x v2
growth.

`bench/frontier_compare.py` requests OpenAI-compatible streaming usage and defines
token TPOT as `(last content chunk time - first content chunk time) /
(completion_tokens - 1)`, using the final streamed `completion_tokens`. It never
uses SSE chunk count as a token count. If an endpoint omits usage (or reports fewer
than two completion tokens), TTFT and output characters remain available, TPOT is
`null`, and the scoreboard reports how many trials omitted usage.

The manual real-checkpoint gate in `scripts/parity_real_model.py` requires exact,
deterministic greedy token parity: Kairyu and the Transformers reference must emit
the same token IDs in the same order and with the same length. Prefix equality,
early EOS, and any other truncation fail with an explicit length diagnostic; there
is no tolerance or text-only equivalence.

## Quick start

```bash
# 1. deploy a gateway (mock engines shown; swap for real backends)
kairyu serve examples/deploy_multi_orchestrator.yaml &

# 2. one command: download missing datasets, run all 11 slots, print the table
kairyu bench run --base-url http://localhost:8000/v1 \
    --model m1 --model kairyu-auto --model kairyu-auto-max

# or config-driven (targets + judge in one file, CLI flags still override):
kairyu bench run --config bench/configs/accuracy.yaml

# deterministic core regression suite (60 requests with the smoke preset):
kairyu bench run --suite core --smoke \
    --base-url http://localhost:8000/v1 --model m1
# or: kairyu bench run --config bench/configs/core.yaml --smoke

# five fixed JSON-Schema cases, paired constrained/control (10 calls):
kairyu bench run --config bench/configs/structured.yaml
# or: kairyu bench run --suite structured \
#         --base-url http://localhost:8000/v1 --model m1

# full seven-arm quantization sweep (use the example's real manifest digests):
kairyu bench run --config bench/configs/quantization.yaml \
    --run-id qwen3-quant-accuracy
kairyu bench quant-sweep --run qwen3-quant-accuracy \
    --tolerance gsm8k=1.0 --tolerance mmlu=1.0 \
    --tolerance ifeval=1.0 --tolerance gpqa-diamond=2.0

# deterministic 4K -> 128K single-key retrieval curve (120 calls):
kairyu bench run --suite long-context \
    --base-url http://localhost:8000/v1 --model m1 \
    --max-context-tokens 131072
```

Results land in `bench/results/accuracy/<run_id>/`:

```
run.json                                      # fingerprint + identity + config + environment
<benchmark>--<sha16>/<target>--<sha16>.json   # one PairResult per scoreboard cell
scoreboard.json                               # machine-readable table
scoreboard.md                                 # Accuracy-suite table (also printed to stdout)
comparison.json                               # measured vs published, machine-readable
comparison.md                                 # accuracy report vs six frontier references
config-comparison.json                        # optional config A/B gate artifact
config-comparison.md                          # optional config A/B gate report
quantization-sweep.json                       # seven-arm task-accuracy artifact
quantization-sweep.md                         # scheme-major accuracy/gate report
```

Core results default to `bench/results/core/<run_id>/`; structured results
default to `bench/results/structured/<run_id>/`; quantization results default
to `bench/results/quantization/<run_id>/`; and long-context results default to
`bench/results/long-context/<run_id>/`. All four contain the same
run, pair, and scoreboard artifacts. They intentionally omit
`comparison.json` and `comparison.md`: the published reference catalog is an
Accuracy-suite contract, not a generic benchmark baseline. A completed quantization run adds
its dedicated `quantization-sweep.{json,md}` only after the strict sweep command
validates all raw evidence.

A completed real-data run from a clean, tracked checkout is also snapshotted in
the suite-local `bench/results/<suite>/scoreboards.jsonl`. The append-only
hash-chained index is keyed by the local benchmark-harness Git commit and the
run fingerprint; it stores the scoreboard itself rather than trusting a path
to a later-mutated run directory. Dirty/Git-less executions and synthetic
offline fixtures still write their ordinary run artifacts but are not history
baselines. The recorded commit identifies the Kairyu code that ran the local
benchmark harness. It does **not** attest the build deployed behind a target
URL, so operators must use distinct fingerprinted target declarations when a
served deployment changes.

Every indexed pair carries the same clean source attestation as its run. Source
drift observed after initialization permanently taints that run id, so restoring
the checkout cannot relabel cached evidence; start a new run id instead. The
fingerprint content-binds each adapter, shared scoring/aggregation code, judge
protocol, referenced cache assets, score-time Python distributions, and any
resolved third-party harness distribution and owned console script. Editable or
otherwise unreadable installed distributions and executables without verified
distribution ownership are never admitted to history. A detected evaluator or
dataset-identity drift converts affected cached evidence to failed, so a restored
resume must execute the pair again rather than relabel it.

History records a complete eligible run without pretending that every cell has
the same provenance strength. SWE-Bench Pro, Terminal-Bench, and τ³ retain
their normal cells and the run may be registered, but those cells carry the
structured `withheld_unresolved_runtime` policy while their harness-managed
remote data, images, or sandbox inputs cannot be resolved to immutable content.
SciCode and the LiveCodeBench slots similarly carry
`withheld_unpinned_execution` under the local runner. They become cross-run
comparable only when Docker is available and inspection resolves the configured
content-addressed image to an exact image ID plus OS, architecture, and optional
variant. A withheld cell never emits a cross-commit delta, even when both runs
have the same policy and reason; independently source-complete cells in the same
scoreboard remain comparable.

Every index record also retains a SHA-256 of each complete `PairResult` and an
explicit pair summary (status, score, denominators, reason, published-score
comparability, cross-run policy, and confidence interval). The summary must
match its scoreboard cell byte-for-byte; failed cells, missing pair evidence,
and offline-fixture configuration are rejected by both append and load
validation.

Opaque `extra_body_json` values are used in memory but retained in durable
metadata only as a semantic JSON SHA-256. Endpoint URLs with userinfo, query, or
fragment components are rejected. This keeps the Git-trackable history from
becoming a credential store while preserving exact configuration identity.

Benchmark and target components retain a readable sanitized prefix and append
the first 16 hexadecimal characters of the raw name's SHA-256. Thus names such
as `org/model` and `org__model`, which otherwise sanitize to the same path, do
not overwrite one another. A run id must be one non-dot path component;
absolute paths, separators, Windows drive paths, and symlink escapes outside
the results or run directory are refused. Result writes are atomic.

Every scored `scoreboard.md` cell shows its sample count; partial cells use
`n=scored/total`. When the retained per-item evidence is entirely binary and
the adapter explicitly declares Bernoulli outcomes, its successes, scored
denominator, total item count, and point estimate must all agree before the cell
shows a two-sided 95% Wilson score interval as `score [lower–upper]`.
Intervals are limited to completed cells whose scored and total denominators
match; partial/failed cells still expose both counts and their status caveat.
Continuous/reward-valued metrics such as MRCR, Terminal-Bench, and τ³ remain
interval-free even when one run happens to contain only zeroes and ones;
malformed counts and legacy artifacts without item evidence likewise fail
closed to point estimate plus `n`. The same interval is retained structurally
in `scoreboard.json`.

Useful subcommands:

```bash
kairyu bench list                      # slots, requirements, cache status
kairyu bench download [--only a,b]     # pre-fetch datasets (idempotent)
kairyu bench report <run_id>           # rebuild + print a stored scoreboard
kairyu bench compare-runs BASE CANDIDATE  # print CANDIDATE - BASE deltas
kairyu bench compare --baseline BASE --candidate CANDIDATE \
    --tolerance gpqa-diamond=2.0       # paired config A/B non-inferiority gate
kairyu bench quant-sweep --run RUN \
    --tolerance gsm8k=1 --tolerance mmlu=1 \
    --tolerance ifeval=1 --tolerance gpqa-diamond=2
kairyu bench entrypoints               # installed/repository ownership inventory
kairyu bench list --suite core
kairyu bench download --suite core
kairyu bench report --suite core <run_id>
kairyu bench compare-runs --suite core BASE CANDIDATE
kairyu bench list --suite structured
kairyu bench download --suite structured --strict
kairyu bench report --suite structured <run_id>
```

`compare-runs` reads only validated index snapshots and never modifies either
run. It requires matching suite, fingerprint, Python/execution runtime, target
layout, benchmark layout, methodology reasons, and scored denominators.
Completed finite cells with an explicit `allowed` policy on both sides show the
candidate-minus-baseline change in percentage points. Partial, failed,
denominator-mismatched, synthetic-fixture, runtime-withheld, or
differently-substituted cells fail closed to an explicit
unavailable/non-comparable marker. When both runs carry the exact same subset
or substitution boundary, a numeric diagnostic delta is shown together with
that boundary; it is not promoted to like-for-like published comparability. An
identical runtime-withholding policy never permits that diagnostic exception.
A negative delta is a report, not a policy gate; thresholding and
trailing-window alert policy belong to the nightly comparator.

### Configuration A/B quality gate

`compare-runs` answers whether the same declared target changed across local
harness commits. Configuration A/B intentionally answers a different question:
whether one immutable served configuration is non-inferior to another under
the same harness, samples, and request policy. First run both configurations
either as two targets in one run or as two separately indexed runs. Every arm
used by the gate must declare an operator-owned immutable identity:

```yaml
targets:
  - name: bf16-baseline
    base_url: http://baseline.example/v1
    model: m1
    served_config:
      label: qwen3-32b-bf16
      sha256: 1111111111111111111111111111111111111111111111111111111111111111
  - name: fp8-candidate
    base_url: http://candidate.example/v1
    model: m1
    served_config:
      label: qwen3-32b-fp8
      sha256: 2222222222222222222222222222222222222222222222222222222222222222
```

The SHA-256 should cover a canonical deployment manifest that binds the
checkpoint/tokenizer revisions, Kairyu build, and all relevant engine,
quantization, KV-cache, expert-parallel, and speculation settings. It is
fingerprinted and retained, but remains an operator declaration: the benchmark
client does not remotely attest which deployment answered a request. For a
single target per run the same identity can be supplied with
`--served-config-label` and `--served-config-sha256`; the flags must be used
together.

Gate two targets from the same immutable run:

```bash
kairyu bench run --config bench-ab.yaml --run-id qwen3-ab
kairyu bench compare --suite core \
    --baseline qwen3-ab --candidate qwen3-ab \
    --baseline-target bf16-baseline --candidate-target fp8-candidate \
    --tolerance gsm8k=1.0 --tolerance mmlu=0.5
```

Or use different run IDs when the deployments cannot be measured in one run:

```bash
kairyu bench compare --suite accuracy \
    --baseline qwen3-bf16 --candidate qwen3-fp8 \
    --tolerance gpqa-diamond=2.0 --tolerance scicode=1.0
```

Tolerance values are percentage points; every named row must pass and scores
are never averaged across benchmarks. The command validates the complete
hash-chained history, then reloads each raw `PairResult` and matches its
canonical SHA-256 to the index. Item IDs must be unique and identical across
arms, all items must be completed, denominators and recomputed means must
agree, and subset/smoke/offline or provenance-withheld evidence is rejected.
Different full run fingerprints are expected because targets differ; after
removing only `identity.config.targets`, the complete methodology identity,
local commit/source tree, Python/execution runtime, judge, dataset/evaluator
identity, and sample-selection configuration must match. Target model,
sampling/reasoning options, context/output limits, and vision policy must also
match; only the arm label, endpoint, credential-variable name, declared served
configuration, and declared quantization classification may differ.

Independent binary items use a paired 2x2 table. Their artifact records a
two-sided 95% Newcombe paired-score method-10 interval for candidate minus
baseline and a one-sided 95% lower confidence bound. MRCR and other bounded
non-binomial item scores use a deterministic paired percentile bootstrap.
SciCode's sequential sub-steps are dependent within a problem, so they use a
problem-clustered paired bootstrap that resamples whole problems while retaining
sub-step weighting. Both bootstrap variants fix 20,000 resamples, symmetric
nearest-rank quantiles, and the repository-owned
`splitmix64_rejection_v1` sampler. The run fingerprint records each adapter's
binary and cluster declarations; comparison never reclassifies old evidence
from a later adapter registry. A row passes only when its unrounded one-sided
lower bound is at least the negative tolerance; a point estimate inside the
margin with insufficient evidence still fails. Exit 0 means all rows passed,
1 means a valid gate did not demonstrate non-inferiority, and 2 means the
inputs/provenance/evidence were invalid.
`config-comparison.json` and `.md` are saved atomically under the candidate run
before exit 1, without changing `comparison.*` or `scoreboards.jsonl`.

## Single model vs orchestration

Orchestration is benchmarked as **just another model name** on the same
endpoint. `DeploymentSpec.orchestrators` serves any number of named
orchestrations (arbitrary worker/role DAGs via the kairyu DSL):

```yaml
engines:
  m1: { backend: mock }
orchestrators:
  kairyu-auto: { spec: agent_pool.yaml }
  kairyu-auto-max: { spec: agent_pool_max.yaml }
```

Every `--model` flag adds a scoreboard column; compare `m1` vs `kairyu-auto`
vs `kairyu-auto-max` in one run.

## Core suite

The core suite uses only deterministic local scorers. Install the benchmark
extra before downloading real datasets:

```bash
uv sync --extra bench
kairyu bench download --suite core --strict
```

| Slot | Pinned source | Headline score | Methodology boundary |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, `main/test`, 1,319 rows | exact final numeric string after `####` (commas removed) | zero-shot Kairyu chat prompt; upstream answer extractor |
| MMLU | `cais/mmlu`, `all/test`, 14,042 rows / 57 subjects | exact teacher-forced A-D continuation-likelihood argmax, item-micro accuracy | zero-shot raw-completion variant, not canonical five-shot MMLU |
| IFEval | `google/IFEval`, `default/train`, 541 prompts / 834 instructions | strict prompt-level accuracy | all four strict/loose × prompt/instruction metrics retained; pinned Google 25-checker plus the documented two-row exact-character amendment |

GSM8K caps generated output at 1,024 tokens. MMLU does not free-generate an
answer: it teacher-forces the ordered continuations `" A"` through `" D"` over
the native `/v1/completions` extension and ranks their raw, pre-processor
natural-log probabilities. Every candidate must resolve to exactly one target
token. A missing capability, token-boundary mismatch, malformed/non-finite
evidence, or partial candidate set is skipped or failed and unmeasured rather
than converted into a wrong answer. Direct native targets support this exact
path; orchestration and remote/chat-only targets are visibly skipped for the
MMLU row. IFEval keeps the target's configured output allowance because valid
prompts require as many as 1,200 words or 100 sentences. The IFEval dataset
prompt is the sole user message; adding boilerplate would corrupt repeat,
start, end, and formatting checks.

IFEval's Google checker source and English NLTK Punkt parameters are immutable
score-bearing inputs. The dataset download path fetches the pinned Punkt
archive, verifies its SHA-256, and stores only the expected English parameters
in the adapter cache. Evaluation never invokes `nltk.download()` or reads a
mutable global tokenizer cache. `langdetect` is seeded before its first use.
Pinned keys 1122 and 1129 request `#` and `!`; upstream
`LetterFrequencyChecker` replaces those non-ASCII-letter arguments with a
random ASCII letter. Kairyu's documented dataset-consistency amendment keeps
the exact single non-whitespace character, so those rows are deterministic.
The scoreboard uses strict prompt-level accuracy as Kairyu's headline; Google
reports all four metrics without designating one official headline.

A full core run sends 58,028 target calls: one each for GSM8K and IFEval and
four exact continuation calls per MMLU item. `--smoke` or `--limit` is the intended
fast development loop, but the resulting artifact remains visibly a subset and
is never promoted to a full score. Dataset counts, MMLU subject coverage,
IFEval keys/checker IDs/kwargs, and fixed scorer resources all fail closed on
drift. Full design and source identities are recorded in
[`docs/design/issue-367-core-evals.md`](design/issue-367-core-evals.md) and the
exact likelihood transport/scoring contract is recorded in
[`docs/design/issue-368-loglikelihood.md`](design/issue-368-loglikelihood.md).

## Structured-output conformance suite

The dedicated `structured` suite tests the OpenAI-compatible `response_format`
contract independently of the general answer-quality rows. Its package-owned
corpus contains exactly one Draft 2020-12 schema in each of five categories:

| Category | Contract exercised |
|---|---|
| Nested | required nested objects and a typed array |
| Recursive | a tree expressed through local `$defs` and `$ref` only |
| Enum | selection from a closed string set |
| Pattern | a string constrained by a regular expression |
| Union | an `anyOf` value with type-sensitive expected output |

Every source item produces a matched pair. Both arms receive the same prompt,
including the same schema text, and use the same target sampling policy and
seed. Only the constrained arm receives `response_format: {type:
"json_schema", ...}`; the control arm omits that field. The arm sent first is
alternated over the item-major/seed-minor matrix to reduce ordering bias. An
any supplied `extra_body_json` is rejected for this suite because it could constrain
the nominal control. With `attempts > 1`, each source item retains one paired
observation under every scheduled seed rather than counting repeated attempts
as new independent tasks.

Scoring keeps syntax, conformance, and usefulness separate. The strict JSON
parser accepts exactly one value and rejects duplicate object keys and
`NaN`/`Infinity`. A pinned `jsonschema` Draft 2020-12 validator then checks the
parsed value, and task correctness requires type-sensitive canonical JSON
equality with the corpus answer. Both arms report the following rates:

| Metric | Numerator | Denominator |
|---|---|---|
| Request acceptance | completed HTTP 200 responses | all scheduled paired observations |
| JSON validity | strictly parseable responses | accepted HTTP 200 completions |
| Schema conformance | independently schema-valid responses | all scheduled paired observations |
| Exact-task accuracy | schema-valid exact expected values | all scheduled paired observations |
| Malformed JSON | accepted responses that fail strict parsing | accepted HTTP 200 completions |

A recognized constrained-schema rejection at HTTP 400 or 422 is a measured
zero for acceptance, schema conformance, and task accuracy; it is not called
malformed model output. Transport failures, retry-exhausted server failures,
and malformed API envelopes are execution failures, and the complete derived
conformance summary is withheld rather than changing its population.
An OpenAI-compatible HTTP 200 safety refusal (`content: null` plus non-empty
`refusal`), content-filtered empty message, or unexpected call payload is
retained as raw accepted evidence and counted as non-JSON and task-incorrect,
so over-refusal cannot disappear into an environment failure.
Constrained-minus-control deltas are shown separately from each arm's absolute
rates. The headline cell score is constrained exact-task accuracy.

Endpoint-reported prompt, completion, and total token counts are retained when
present. The report states usage coverage over accepted completions and computes
paired token deltas only where both arms supplied usage; these fields show what
the endpoint reported and do not attest its tokenizer or accounting. Client
latency is retry-inclusive, counterbalanced, and diagnostic. No per-token price
is configured, so the suite deliberately reports no currency cost.

The corpus is not fetched from Hugging Face. Its installed bytes are bound by
the package revision
`sha256:a40b41e1f91f6f33803a55ab4967c75d610dc82d2c14fc11d03c19ace45b05be`,
and loading fails if that digest changes. This content SHA-256 is deliberately a
different provenance mechanism from the immutable HF Git commit pins used by
downloaded Accuracy/Core datasets: a Git revision identifies a repository snapshot,
whereas this digest identifies the exact JSONL bytes shipped in the wheel. Both
the corpus and the `jsonschema` evaluator distribution are included in the run
identity and fresh-history validation.

```bash
uv sync --extra bench
kairyu bench download --suite structured --strict
kairyu bench run --config bench/configs/structured.yaml \
    --run-id structured-conformance
kairyu bench report --suite structured structured-conformance
```

A full one-attempt run schedules five source items, two endpoint calls per
item, and ten calls in total. `--limit` remains useful for plumbing but creates
an explicit subset. No GPU, judge, Docker runner, or remote dataset is required;
the deployed target must implement the chat `response_format` contract.

## Long-context length curve

The dedicated `long-context` suite turns sequence length into the scoreboard's
row dimension. It runs the same deterministic, single-key needle retrieval task
at 4K, 8K, 16K, 32K, 64K, and 128K user-message content tokens. Each row has 20
items with the needle placed at evenly spaced target depths from 2.5% through
97.5%, so a normal scoreboard is the accuracy-vs-length curve rather than one
average that hides the failure point. Each answer is a unique deterministic
value and scores one only when the stripped response is exactly that value.

The rows are generated locally during `download`; no remote dataset is needed.
Generation uses the pinned `o200k_base` vocabulary and recounts the complete
message content to require its exact row length. Endpoint-specific chat-template
tokens are outside that count and are disclosed in every cell. This is a
RULER-style single-key NIAH probe, not the official 13-task NVIDIA RULER suite,
and its scores must not be reported as official RULER numbers. The design uses
RULER's configurable-length retrieval principle and standard 4K-to-128K curve,
while deliberately avoiding a second external harness. See the
[official RULER repository](https://github.com/NVIDIA/RULER) for the full task
family and [`docs/design/issue-374-long-context-sweep.md`](design/issue-374-long-context-sweep.md)
for Kairyu's exact boundary.

`--max-context-tokens` declares the input limit of every CLI target. A curve
point above that limit is skipped in full and is never truncated into a shorter
test. Omitting the limit attempts every point; a standard context-length HTTP
400 is retained as skipped evidence with its raw error, while transport or
retry-exhaustion errors remain failed evidence. A full run makes 120 calls per
target. `--limit` and offline fixtures remain plumbing aids with the existing
subset/fixture incomparability markers. Clean full runs can use the ordinary
`compare-runs` history report or the paired `compare` command with one tolerance
per length row, which makes the curve a direct gate for RoPE or KV-cache changes.

```bash
uv sync --extra bench
kairyu bench download --suite long-context --strict
kairyu bench run --suite long-context \
    --base-url http://localhost:8000/v1 --model qwen3-32b \
    --max-context-tokens 131072 --run-id qwen3-long-context
kairyu bench report --suite long-context qwen3-long-context
```

## Quantization x task-accuracy suite

Kernel throughput and short teacher-forced parity do not answer whether a
quantized model still solves downstream tasks. The `quantization` suite keeps
the complete deterministic Core task order and adds judge-free GPQA Diamond as
one reasoning row:

| Task | Role in the sweep |
|---|---|
| GSM8K | generative mathematical exact match |
| MMLU | exact teacher-forced continuation-likelihood accuracy |
| IFEval | strict prompt-level instruction-following accuracy |
| GPQA Diamond | graduate-level reasoning MCQ exact match |

A complete run has exactly seven declared profiles: dense BF16 reference, FP8,
INT8, AWQ, GPTQ, NVFP4, and dense BF16 with FP8-E4M3 KV. Dense BF16 is encoded
as `weight_method: none` plus `compute_dtype: bfloat16`; BF16 is not a
checkpoint quantization method. All weight-quantized arms use effective BF16
KV, while the final arm isolates the KV dtype change. See
[`bench/configs/quantization.yaml`](../bench/configs/quantization.yaml) for
the complete configuration.

Each target also requires a distinct `served_config` SHA-256. Hash a canonical
deployment manifest covering the common model/tokenizer lineage, exact
checkpoint tensors, quantization dialect/bits/group/scales/calibration/ignored
layers, compute and effective KV dtypes, Kairyu build/image, topology, and
hardware. The profile and manifest are operator declarations: `/backends` does
not remotely attest all of those fields or every replica, and the report never
claims otherwise. API model and every sampling/reasoning/context/output request
field must match across arms.

Current native Kairyu deliberately rejects `fp8_e4m3` KV because its retained
G4 E-KV bake failed the output/logprob/cache-quality gates. The required FP8-KV
row can classify an explicitly served external or experimental endpoint, but it
does not enable or demonstrate native support. The other formats retain their
real hardware, dialect, model-family, and parallel-topology restrictions; a
sweep score is not a universal loader-support claim.

A full run sends 58,226 target calls per arm (58,028 Core plus 198 GPQA), or
407,582 across all seven arms. `--smoke` and `--limit` are useful for plumbing
only: `quant-sweep` rejects subsets, offline fixtures, skipped/partial/failed
cells, source/runtime withholding, mismatched request policy, and any missing,
duplicate, additional, or anonymously configured arm.

```bash
kairyu bench download --suite quantization --strict
kairyu bench run --config bench/configs/quantization.yaml \
    --run-id qwen3-quant-accuracy
kairyu bench quant-sweep --run qwen3-quant-accuracy \
    --tolerance gsm8k=1.0 --tolerance mmlu=1.0 \
    --tolerance ifeval=1.0 --tolerance gpqa-diamond=2.0
```

The four margins are percentage points and must be supplied exactly once. For
each candidate/task cell the command reuses the source-, item-, and
pair-SHA-bound configuration A/B comparator. All four headline outcomes are
binary, so it retains Newcombe paired method-10 two-sided 95% intervals and a
one-sided 95% non-inferiority lower bound. A cell passes only when the unrounded
lower bound is at least the negative task margin. There is no averaging across
tasks or schemes: all 24 cells must pass.

The source run receives atomic `quantization-sweep.json` and `.md` files. JSON
embeds all six complete A/B comparisons, the fixed profile/task contract,
source/index/raw-pair bindings, runtime and protocol identities, evidence
hashes, policy, support/attestation boundaries, and its own canonical SHA-256.
Markdown transposes the ordinary benchmark-major scoreboard into one absolute
accuracy row per scheme and lists every paired gate separately. Exit 0 means
all gates passed, exit 1 is a valid retained quality failure, and exit 2 means
the input or evidence was invalid. The complete design is in
[`docs/design/issue-372-quantization-sweep.md`](design/issue-372-quantization-sweep.md).

## The 11 Accuracy slots

| Slot | Source | Scoring | Requires |
|---|---|---|---|
| SWE-Bench Pro | `ScaleAI/SWE-bench_Pro` at a pinned commit (731 tasks) | mini-swe-agent text actions (1,000 steps) + pinned ScaleAI local-Docker evaluator, resolved rate | docker, `[bench-agentic]` |
| Terminal-Bench 2.1 | `terminal-bench/terminal-bench-2-1` (Harbor Hub) | `harbor run` (terminus-2, 500 turns), Harbor Mean | docker, `[bench-agentic]` |
| LiveCodeBench | `livecodebench/code_generation_lite` `release_v6` (1,055 problems, pinned commit) | sandboxed pass@1 (public+private tests) | — |
| LiveCodeBench Pro | `QAQAQAQAQ/LiveCodeBench-Pro` split `quater_2025_4_6` + `-Testcase` ZIPs | sandboxed pass@1 (lower bound: no testlib checker) | HF token |
| Humanity's Last Exam | `cais/hle` (gated) | MCQ exact match + judge for free-form | HF token; judge for free-form |
| CharXiv Reasoning | `princeton-nlp/CharXiv` | judge-graded, vision content-parts | vision target + judge |
| GPQA Diamond | `Idavidrein/gpqa` (gated) | MCQ exact match, seed-shuffled choices | HF token |
| SciCode | `SciCode1/SciCode` | sequential sub-step tests (+`test_data.h5` golden data) | numpy in venv |
| τ³-Bench Banking | official `tau2` v1.x package, `banking_knowledge` + `alltools` | official reward (agent = target, user-sim = judge) | tau2-bench `[knowledge]` + judge |
| Long Context Reasoning | `THUDM/LongBench-v2` **substitute** | MCQ exact match | — |
| MRCRv2 | `openai/mrcr` (8-needle, ≤128K) | official prepend + SequenceMatcher ratio | long-context target |

Annotated caveats appear as scoreboard footnotes automatically, notably:
the Long Context Reasoning slot is a **LongBench v2 substitute** (Fugu's own
suite is unpublished; numbers are not directly comparable), and LiveCodeBench
Pro is scored by the local sandbox, not the official judge.

### Dataset acquisition notes

- **LiveCodeBench** reads the repo's `test.jsonl`…`test6.jsonl` shards directly
  at a pinned commit. `release_vN` is a *config name*, not a git ref, and the
  loading-script path needs `trust_remote_code` (gone in `datasets` 4.x), so
  going through the files is what keeps the slot working. `release_v6` must
  yield exactly 1,055 problems; any other count fails closed as `unavailable`
  rather than scoring a silent subset.
- **LiveCodeBench Pro** pins Fugu's 2025 Q2 slice (`quater_2025_4_6`, 167
  problems) and joins each `problem_id` to a `<problem_id>.zip` in the testcase
  repo (`testdata/<n>.in` / `.ans`). Acquisition **fails closed**: the split must
  yield exactly 167 problems, every archive must download, and each archive's
  usable cases must match the `sum(subtasks[].n_cases)` it declares, with no
  unpaired half in either direction. An archive that declares **no** count is not
  "as complete as whatever arrived" — that declaration is the only denominator
  evidence there is, so a missing or malformed `config.yaml` fails closed too. `download_file()` turns a timeout, a 401 and a 404 alike into
  `None`, so excluding a problem would cache a smaller denominator permanently —
  and a rate over a shrunken set is not even a lower bound on the full 167. The
  testcase repo's pin is part of the cache identity (`AdapterInfo.extra_sources`)
  so repinning it rebuilds rather than leaving stale bytes "ready" under a new
  methodology. The archives also ship a per-problem testlib `checker.cpp` that
  kairyu does **not** compile: grading is per-line whitespace-normalized
  comparison, so multi-answer problems can only lose points and the cell is a
  **lower bound**.

**MRCRv2 population.** The published `openai/mrcr` split mixes 2-, 4- and
8-needle items across eight length bins up to 1M tokens, with **100 samples per
(needle count, bin)**. The card defines those bins by the tokens used by
**prompt + answer** under `o200k_base`, with boundaries `[4096, 8192]`,
`(8192, 16384]`, … `(524288, 1048576]`.

Fugu reports the **8-needle** subset at up to **128K**, which is the five bins at
or below 131,072 — exactly **500 rows**. The adapter counts tokens with the
official encoder (so `tiktoken` is required; without it the cell is skipped
rather than approximated), assigns each row to its official bin, keeps the
selected bins, prints the per-bin counts, and **fails closed** unless there are
exactly 100 rows in *each* of them — 500 in total weighted 99/101/100/100/100
would be a different population reported as the official slice. An approximation such as chars/4 over the prompt alone cannot reproduce
those boundaries, and averaging the whole 2,400-row split would score an easier,
shorter population against Fugu's number.

The target's own `max_context_tokens` gate is separate: it uses the exact
prompt-only token count, matching the official runner's
`n_tokens(messages) > MAX_CONTEXT_WINDOW` check. (The chars/4 heuristic survives
only as a fallback for rows normalized before that field existed; near a target's
limit the two disagree and would skip a fitting row or send an oversized one.)

### SciCode: sequential sub-steps and golden data

The published `SciCode1/SciCode` export ships **no reference code** — every
sub-step's `ground_truth_code` and every problem's `general_solution` is null.
There is therefore no "gold previous steps" setting to run, so sub-steps execute
**sequentially per problem** and each step sees the model's *own* earlier code in
both its prompt and its executed program (SciCode's main setting, which is what
makes the cascade visible). Grading a later step in isolation could only raise
`NameError` on the helper an earlier step was meant to define.

Two consequences:

- `--limit` / `--smoke` select **whole problems**, never a truncated chain.
- The scored population is **288 of the 291** test-split sub-steps. The official
  evaluator `continue`s past three of them (problem 13 step 6, 62 step 1, 76
  step 3) and instead supplies their implementation as a text file, because later
  steps of those problems call the helpers they define. kairyu does the same: those
  three are excluded from scoring and their pinned-by-hash implementation is
  carried into the context. 288 is also the denominator Fugu reports, and
  acquisition fails closed unless it lands on 291 sub-steps / 288 scoreable — and
  also if any of those three implementations cannot be fetched at its pinned hash,
  because scoring their dependents without them would charge the model for a
  missing harness file.
- Nearly all of those compare against golden data (`target`) from `test_data.h5`,
  which the HF export does not contain. It is fetched from the upstream repo first
  and otherwise from a public mirror (`Srimadh/Scicode-test-data-h5`), and is
  accepted only when its size and **SHA-256 content hash** match the pin: magic
  bytes alone prove the file format, so a different-but-valid HDF5 would otherwise
  be trusted as every expected value in the benchmark. The check runs again when a
  cached asset is reused (once per pair, since the file is ~1 GB), so a replaced or
  truncated file cannot become the expected-answer source under a manifest that
  still advertises the pin. The pin says *which* bytes
  were scored against — it has **not** been cross-checked against the official
  Google Drive artifact, and the methodology says so. Sub-steps left without the
  file are `unjudged`, never guessed.

Prompts include the problem-level and step-level background, matching Fugu's
with-background condition, and each prior step is rendered the way the official
`process_problem_steps()` does: its description, its background, then its code,
with steps separated by `------`. Passing only the concatenated code would lose
the statement of what each helper was for.

## Live progress

A full Accuracy run is thousands of judged items across eleven slots and can take
hours; a full Core run is 15,902 target calls. The runner therefore reports
what it is doing for either suite:

- **On a TTY** — a `tqdm` bar for the suite (pairs) plus one for the current
  benchmark×target (items). `tqdm` comes with `kairyu[bench]`; without it the
  run falls back to log lines rather than failing to import.
- **In a log** (CI, `docker compose logs`, nohup) — one self-contained line per
  event plus a throttled item counter, so a 2,500-item slot emits a handful of
  lines instead of 2,500 and no line depends on the previous one being visible:

  ```
  [bench] 22 benchmark×target pairs to run
  [bench 7/22] hle × qwen3-32b
  [bench 7/22] hle × qwen3-32b: 2500 items
  [bench 7/22] hle × qwen3-32b: 412/2500 items (15s)
  [bench 7/22] hle × qwen3-32b: done — partial (score=8.4)
  ```

- `--no-progress` disables it. The reporter is a pure observer: `progress` is
  excluded from the run fingerprint, and scoreboard/pair evidence is identical
  either way. Every callback is wrapped so a closed stream, a broken pipe or a
  bar bug cannot end a run that is producing evidence, and the reporter is closed
  in a `finally` so cancellation does not leak it.
- Agentic slots have no item count until their harness returns, so they are
  labelled `agentic harness` and emit a **heartbeat** every 15s. Without it an
  8-hour SWE-Bench Pro or Terminal-Bench run would print one line and go silent —
  the exact case where "working" and "hung" must stay distinguishable.

The play-by-play goes to **stderr** and the artifacts (download notes, the
scoreboard, and the accuracy report when the suite has one) to **stdout**, so
`kairyu bench run … > scoreboard.txt` keeps the two apart.

## Accuracy report vs published frontier scores

Every Accuracy run also writes `comparison.md` / `comparison.json` (and prints the
report), placing each measured cell next to the available published values for
**Fable 5, GPT-5.6 Sol, DeepSeek-V4-Flash-0731, Qwen3.8 MAX, Kimi K3, and
Fugu**. A second table reports measured-minus-reference gaps for every available
model value; the legacy `Δ target` column remains measured minus Fugu.
`kairyu bench report <run_id>` rebuilds it (`--no-comparison` to skip).

The published values are a **committed, SHA-256 identified catalog** in
`kairyu/bench/reference.py`, sourced from provider launch pages, the Kimi K3
technical report, and the Fugu release figures. Reports embed source URL, tier,
retrieval date, exact condition text, and alternate-condition records. Missing
values remain `—`; an older model, a preview, a tool-enabled score, or a different
context bucket is never substituted.

What the report refuses to do:

- **Invent a number.** A skipped cell is `—`, never 0.
- **Hide a denominator.** Every score carries its item count, `partial` carries
  `*`, `failed` carries `!` (**even without a score**, so a failed cell never
  reads as merely absent), and the reason is reprinted.
- **Print a delta for anything that is not a full-suite measurement of the same
  thing.** Comparability is carried per cell, so all of these render `n/c`:
  a substituted dataset (Long Context Reasoning → LongBench v2), a *run-time*
  substitution (the τ2 harness standing in for τ³), a partial or failed cell, and
  a **subset or fixture run** — `--limit`/`--smoke` cells are legitimately
  `completed`, so without this a 20-item run would print an unmarked delta
  against a full-suite published score.
- **Bury the caveat.** When a reason applies to every cell, both `scoreboard.md`
  and `comparison.md` open with a banner saying so, because a shell warning does
  not survive into the file an operator opens hours later.
- **Let a resumed pair keep someone else's comparability.** Run-level reasons
  belong to the run doing the reporting, so a reused pair is re-stamped (and
  re-saved) with them. A pair written before these fields existed validates as
  `comparable=True` by model default under an unchanged fingerprint, and would
  otherwise resume into a subset run with a numeric delta and no banner.
- **Imply the baselines are comparable.** The page states that every non-Fugu
  score is *provider-reported*; the report repeats that, so those columns read
  as orientation rather than as measurements made under this harness.

It also reprints the run's own methodology footnotes (substituted datasets,
uncompiled checkers, self-judging, degraded cells) and the release's HLE
**text-only** variant, which the figure reports separately from the headline
table's full set.

## Target generation TTFT and TPS

Every scoreboard includes a target-only generation-performance table. Direct
chat adapters request SSE with endpoint usage and retain one timing record per
successful target call:

- **TTFT** (the requested “TFTT” metric) is request start to the first non-empty
  content, reasoning, refusal, or tool-call delta. Role-only and usage-only
  chunks do not count.
- **TPS** is `(completion_tokens - 1) / (last semantic delta - first semantic
  delta)`. It is `—` when usage is absent, fewer than two completion tokens are
  reported, or the measured span is zero.
- The report shows TTFT p50/p95, TPS p50, valid/total coverage, missing usage,
  request errors, and retry attempts. It never compares these values with the
  six published models.

MMLU's teacher-forced log-likelihood row is `not applicable`, because it does
not generate an output stream. SWE-Bench Pro, Terminal-Bench, and τ³-Bench run
inside external harnesses that do not currently return per-target SSE timing;
their performance cells are explicitly `unavailable` rather than mixing in
harness wall time, judge traffic, or user-simulator traffic.

## Degradation model (why one command always completes)

Every unmet precondition becomes data, never a crash. Per (benchmark, target)
pair the status is one of:

- `completed` — every item resolved.
- `partial` — a score exists but some items were unjudged/skipped/failed
  (reason recorded, e.g. `312/2500 items unjudgeable`).
- `skipped` — a precondition failed, zero items ran: `docker unavailable`,
  `dataset not in cache (gated…)`, `requires a judge endpoint`, non-vision
  target, harness not installed.
- `failed` — the adapter crashed or most items hard-errored. **Only this
  affects the exit code.**

### Resume identity

`--run-id` names immutable evidence; it is not a mutable output slot. Before
the first backend request or pair write, the runner downloads or preflights the
selected adapters, constructs a canonical JSON identity, and stores its
SHA-256 fingerprint in `run.json`. The identity contains:

- the selected adapter names and each adapter's pinned dataset id, revision,
  and validated `data.jsonl` SHA-256 (or an explicit unavailable marker).
  Each adapter also records its binary-outcome declaration and optional
  versioned paired-cluster key, which determine config-A/B uncertainty without
  consulting a later checkout's registry.
  HLE and CharXiv additionally carry the logical judge-template name, variant,
  and SHA-256 of the exact UTF-8 template used for scoring. Each core adapter
  carries package/resource/SHA-256 identities for its complete score-bearing
  source; IFEval binds both the adapter and every vendored checker module. The
  structured adapter binds its exact package-corpus SHA-256, paired protocol,
  evaluator resources, and score-time `jsonschema` dependency set; and
- the output-affecting `BenchConfig` fields `suite`, `targets`, `judge`,
  `execution`, `limit`, `smoke`, `offline_fixtures`, `only`, `exclude`, `seed`,
  `attempts`, `concurrency`, `request_timeout_s`, and `retries`. `targets`
  includes every target's name,
  base URL, model, API-key environment-variable name, context/output limits,
  vision capability, optional operator-declared `served_config` label/SHA-256,
  and sampling policy (`sampling_mode`, `temperature`, `reasoning_effort`,
  `top_p`, `seed`, `extra_body_json`); `judge` likewise includes every ordered grading-panel
  endpoint/model, API-key environment-variable name, concurrency, retry limit,
  and sampling policy. Changing a judge template, panel member, vote policy, or
  reasoning effort is therefore a different experiment, not a resumable run.

Exactly six execution, location or display controls are excluded: `run_id`,
`results_dir`, `cache_dir`, `rerun`, `download`, and `progress`. API-key *environment
variable names* remain part of the endpoint identity, but resolved secret
values are never read into or hashed by the fingerprint. Environment metadata
such as the timestamp, git commit, Python version, and kairyu version remains
in `run.json` as provenance and does not affect identity equality. Canonical
JSON uses sorted keys and compact separators before hashing.

Re-running with the same `--run-id` resumes only when `run.json` has the exact
fingerprint. A missing or different fingerprint—including a legacy run
directory—or a changed target, dataset bytes/revision, limit, seed, judge, or
methodology-affecting configuration is refused without overwriting `run.json`
or pair evidence and before backend HTTP calls. Under a matching run, only a
non-failed pair carrying the same `run_fingerprint` is reused; failed pairs and
legacy/mismatched pair files run again.

`--rerun` bypasses matching pair reuse, but it does **not** bypass the
run-directory fingerprint check. To intentionally change immutable inputs,
choose a new `--run-id`; `--rerun` cannot repurpose existing evidence.

## Datasets, cache, tokens

- Cache dir: `--cache-dir` > `$KAIRYU_BENCH_CACHE` > `~/.cache/kairyu/benchmarks`.
  Datasets are normalized to JSONL once at download; nothing is committed to
  the repo (`bench/results/` and `bench/data/` are git-ignored; the committed
  fixture set contains 17 tiny synthetic stand-ins for offline testing, one
  fixed five-row structured conformance corpus, and one judge-calibration
  corpus).
- A cache entry is ready only when `manifest.json` and `data.jsonl` exist, the
  manifest contains a well-formed lowercase SHA-256, a streaming hash of the
  current JSONL bytes matches it, and any requested dataset id/revision pins
  match. Every `assets/...` reference in normalized rows is also recorded with
  its content SHA-256 and re-read without following symlinks. Missing, malformed,
  unreadable, stale, or modified entries fail closed
  as not ready; a readiness check never rewrites or deletes them. The same
  identity is checked again immediately before each pair, so bytes that change
  after run initialization are skipped rather than scored as valid input.
- Download deps are an extra: `uv sync --extra bench` (or
  `pip install 'kairyu[bench]'`).
- **Pinned revisions.** Every slot whose data kairyu downloads is pinned to a
  commit in `kairyu/bench/pins.py`, and that commit is passed to the fetch — a pin
  recorded in the manifest while the bytes came from a moving `main` would make
  the cache and run fingerprint attest something false. `revision` is a git ref,
  so a declared value that is not a commit sha (a config name such as
  `release_v6`) is replaced by the registry pin; the config name goes to `name=`.
  Immutable secondary source revisions that decide a slot's tests or expected
  answers — the LiveCodeBench Pro testcase archive, vendored IFEval checker,
  and IFEval Punkt source — are registered in `SECONDARY_PINS` and carried in
  the adapter's `extra_sources`. Adapter-owned raw files such as SciCode's
  `test_data.h5` and IFEval's Punkt archive additionally bind verified content
  hashes, so cache invalidation and provenance cover the actual score-bearing
  bytes. This matters: `openai/mrcr` was corrected in December 2025 and HLE's
  item count has shifted since release, so a score taken
  against "whatever `main` was that day" is comparable to neither Fugu's number
  nor an earlier kairyu run. A pin only applies when the recorded dataset id
  still matches, and an adapter that declares its own revision keeps it.
  Refreshing a pin changes the run fingerprint, so stored runs are refused for
  resume rather than silently reinterpreted — the procedure is in that module's
  docstring.
  The structured corpus is package-owned instead: its `sha256:...` revision is
  recomputed directly from the installed JSONL before parsing, so it neither
  claims nor needs an HF Git pin. A package content digest and an HF repository
  commit are intentionally not presented as interchangeable provenance.
  Terminal-Bench and τ remain exceptions because their harnesses own data
  acquisition without an equivalent immutable revision input. SWE-Bench Pro is
  pinned separately: Kairyu exports exactly 731 rows from
  `ScaleAI/SWE-bench_Pro` at the adapter revision and runs the official
  `SWE-bench_Pro-os` evaluator at its recorded commit. Its Docker Hub task image
  tags are still mutable, so the adapter discloses incomplete execution-image
  provenance instead of claiming a content-addressed historical result.

- **Gated datasets** (GPQA Diamond, HLE, LiveCodeBench Pro): accept the license on the dataset
  page (e.g. <https://huggingface.co/datasets/Idavidrein/gpqa>) and set
  `HF_TOKEN`. Without it those cells report `skipped (gated)` and the run
  continues.

## Sampling policy and sensitivity

Fugu reports every model at its **maximum reasoning effort**, and ran the τ³
user simulator at **low**. Sampling belongs to the endpoint, not to a
benchmark, so it is configured per target (and per judge) and applies to every
slot:

Target chat requests have two mutually exclusive generation-default modes:

- `sampling_mode: adapter` is the default. The adapter-authored temperature
  (currently `0.0`) reaches the wire unless the target supplies an explicit
  `temperature`; `--temperature` is the CLI form. The default mode and an
  absent target temperature are omitted from serialized configuration, so an
  ordinary one-attempt run retains its pre-sensitivity configuration shape and
  canonical configuration-fingerprint input.
- `sampling_mode: recommended` (CLI: `--recommended-sampling`) omits
  `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty` from the
  request so the endpoint may apply its model defaults. It cannot be combined
  with an explicit `temperature`, `top_p`, or those other generation fields.
  For a native Kairyu deployment, checkpoint values can be selected only when
  the server itself uses `generation_config: auto`; `vllm` and `none` retain
  their documented neutral policies. The artifact proves the requested wire
  omission, **not** that a remote endpoint loaded or applied any particular
  `generation_config.json`.

For the native Kairyu Qwen3 endpoint, use the model's chat-template control;
`reasoning_effort` is a provider-specific OpenAI field that this endpoint does
not implement:

```bash
kairyu bench run --base-url http://localhost:8000/v1 --model qwen3-32b \
    --temperature 0.6 --top-p 0.95 --sampling-seed 100 --attempts 4 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": true}}' \
    --judge-model qwen3-32b \
    --judge-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```

```yaml
targets:
  - name: qwen3-32b
    base_url: http://localhost:8001/v1
    model: qwen3-32b
    temperature: 0.6
    top_p: 0.95
    seed: 100
    extra_body_json: '{"chat_template_kwargs": {"enable_thinking": true}}'
attempts: 4
judge:
  base_url: http://localhost:8001/v1
  model: qwen3-32b
  extra_body_json: '{"chat_template_kwargs": {"enable_thinking": false}}'
```

To request endpoint defaults instead, replace the explicit `temperature` and
`top_p` above with `sampling_mode: recommended`, or use
`--recommended-sampling`. Keep `served_config` as the operator-owned binding of
the deployment that is expected to interpret those omissions.

`--sampling-seed` is the target request `seed`; `--seed` remains the *item
selection* seed. With one attempt, an unset target seed remains absent from the
wire. With `--attempts N` greater than one, chat adapters send one request per
source item and seed: the ordered seeds are `target.seed + 0 .. N-1`, or
`0 .. N-1` when the target seed is unset. Thus the example above uses
`100, 101, 102, 103`. Other unset endpoint knobs remain absent. Use
`reasoning_effort` only with an endpoint that documents support for it.

`extra_body_json` is validated at load time and applied only to unreserved
endpoint extension fields: it must be a JSON object, and it may not override
`model`, `messages`, `stream`,
`temperature`, `max_tokens`, `reasoning_effort`, `top_p`, `seed`, or
the other typed request fields. Those come from the adapter's request and this endpoint's typed policy — the values the run
fingerprint and methodology record — so letting them through would make the
effective request disagree with the recorded configuration.
General chat suites may use `response_format` as an endpoint extension; the
paired structured suite rejects the entire `extra_body_json` escape hatch so
the control arm cannot inherit a hidden constraint.

This policy reaches every slot that issues its own chat requests. The three
external-harness slots (SWE-Bench Pro, Terminal-Bench, τ³) drive a separate CLI,
so each maps what its harness exposes and annotates what it cannot forward.
`temperature` and `sampling_mode: recommended` are chat-only: all three
external-harness rows fail closed as skipped when either is selected, because
their pinned wrappers expose no verified equivalent. Run those rows separately
with the default target policy. A full Accuracy chat sensitivity run can use
`--exclude swe-bench-pro,terminal-bench,tau-bench-banking`; SWE-Bench Pro also
requires `attempts: 1`, while Harbor and τ interpret a larger attempt budget as
their own harness trial count rather than as the grouped seed protocol below.

### Sampling-sensitivity evidence and statistics

A multi-attempt chat result remains one `ItemResult` per source dataset item;
its ordered `sampling_attempts` children retain the 1-based attempt number,
seed, and complete seed-specific item result. Attempts are deliberately not
flattened into `PairResult.items`: repeated samples of one question are
correlated and are not additional independent dataset items. A source item's
score is the arithmetic mean of its attempts. If any child is incomplete, the
ordinary pair remains visibly partial/failed as appropriate and the sampling
summary is withheld rather than calculated from a changing denominator.
The runner, standalone report path, and fresh history append all rebind the
retained methodology to the authoritative run configuration: attempt budget,
ordered schedule derived from the target seed, mode, and explicit temperature
must agree. Parent/item status, scored/total denominators, and pair score are
recomputed from the children. History schema 1 retains a protocol marker for
new multi-attempt runs, so these checks do not reinterpret or reject older
agentic harness records that used `attempts` only as a native trial count.

For complete evidence, let `s_j` be the mean item score at seed `j`, across the
same `N` source items. The scoreboard reports the mean of the `s_j` values, their
minimum and maximum, and the **sample** standard deviation

```text
SD = sqrt(sum_j (s_j - mean(s))^2 / (n - 1))
```

where `n` is the number of seeds, not the number of dataset items. This is why
the sensitivity summary requires at least two distinct seeds. It is a spread
over the fixed seed sweep, not a confidence interval. Ordinary Wilson intervals
are withheld for these cells because attempts within an item are repeated
measures.

For an adapter that declares binary outcomes, let item `i` have `c_i` successes
among `n` attempts. Its unbiased estimator is

```text
pass_i@k = 1 - C(n - c_i, k) / C(n, k)
pass@k   = (1 / N) * sum_i pass_i@k
```

The report includes `k = 1, 2, 4, ...` up to the attempt budget and also the
actual budget when it is not a power of two. Non-binary rows still report the
seed mean/SD/range but do not manufacture pass@k. Configuration A/B requires
`attempts: 1` and rejects any run configured otherwise: its paired Newcombe and
bootstrap procedures assume one scored outcome per source item and must not
reinterpret correlated attempt means as independent evidence. Run a separate
one-attempt experiment for that gate. Stored rendering accepts pass@k only for
an adapter that declares binary outcomes and only when its item/seed margins
form a realizable binary matrix; otherwise the label is withheld.

## Judge configuration

Free-form grading (HLE, CharXiv) and the τ-bench user simulator use a
configurable OpenAI-compatible primary judge endpoint:

```bash
kairyu bench run ... --judge-base-url http://localhost:8000/v1 --judge-model kairyu-auto
```

Headline runs may add one or more additional pointwise graders. The compact
CLI form inherits default sampling for each additional member:

```bash
kairyu bench run ... \
  --judge-base-url http://judge-a:8000/v1 --judge-model judge-a \
  --judge-secondary http://judge-b:8000/v1=judge-b=JUDGE_B_API_KEY
```

Use YAML when members require distinct sampling controls:

```yaml
judge:
  base_url: http://judge-a:8000/v1
  model: judge-a
  reasoning_effort: high
  additional_judges:
    - base_url: http://judge-b:8000/v1
      model: judge-b
      api_key_env: JUDGE_B_API_KEY
      reasoning_effort: high
```

Panel grading is strict-majority and fail-closed. Every configured member must
return a parseable `correct: yes|no` vote. A failed/unparseable member or a tie
makes the item `unjudged`; the primary never breaks a tie. Consequently two
total judges are deliberately unanimous consensus, while three permit a real
2-of-3 majority. Ordered per-member votes are retained in item evidence. Exact
duplicate resolved endpoint/model members are rejected. The τ-bench user
simulator remains the primary endpoint only—additional graders never become
simulated users and do not imply that τ rewards were judge-calibrated.

The judge configuration is disclosed in `run.json`, and each judged item keeps
its verdict evidence. Self-judging is
detected from the resolved endpoint/model identity used for requests: trailing
slashes are removed and the standard OpenAI `/v1` path is appended when absent,
while scheme, host, port, any other path, and the exact model remain significant.
Display aliases therefore cannot hide the bias, and matching any panel member
is self-judging. Legacy reports that indicate a judge but lack any required
resolved identity are annotated `judge independence unknown` rather than being
assumed independent; an explicitly disabled judge is not. These annotations
appear only on HLE/CharXiv cells that actually use a judge template.
Without a judge, MCQ items still score exact-match; free-form items are recorded
`unjudged`. Judge verdicts that fail to parse degrade the item, never the run.

### Judge calibration

Before promoting a headline run, bind calibration directly to its immutable
`run.json`:

```bash
kairyu bench calibrate-judge \
  --run 20260804-headline --results-dir bench/results/accuracy \
  --output bench/results/accuracy/20260804-headline/judge-calibration.json
```

The command recomputes the run fingerprint from its recorded identity, verifies
that the disclosed target/judge config agrees with the fingerprinted config,
and requires every selected judge-backed adapter's recorded production-template
and judge-protocol identities to match the code doing calibration. Missing,
duplicate, or stale identities and all judge/config overrides are rejected.
Thus it cannot accidentally calibrate a different judge or prompt from the one
used by the run. Passing a run directory instead of an id is also supported.

The packaged set contains 12 clear correct/incorrect pairs, split evenly across
the HLE and CharXiv rubrics (24 responses: six correct and six incorrect per
rubric). They are selected from the published gold labels in
[Princeton LLMBar at commit `900616b`](https://github.com/princeton-nlp/LLMBar/tree/900616bff90b6c6c8e1681f7d079250637c55992),
whose Natural set defines one output as faithfully/correctly following the
instruction and the other as deviating. Every row records the source path,
commit, record/output locator, and MIT license; the upstream notice is packaged
as `kairyu/bench/fixtures/LLMBAR_LICENSE`.

Every response is graded twice (48 aggregate panel decisions): once with the
production template and once with only the reference/response block order
counterbalanced. The JSON artifact binds the raw set, both template variants,
the parser/request protocol, normalized judge config, thresholds, benchmark run
fingerprint, and target identities into its own SHA-256 fingerprint. It reports
coverage, confusion matrices, overall/per-template gold-label agreement,
position flips, ordered member votes, and the complete source provenance.

The default fail-closed gate requires 100% parseable coverage, at least 11/12
gold-label agreement for each template in each order (and therefore at least
22/24 overall), and zero order-dependent verdict flips. Headline eligibility
also requires at least six paired prompts/12 responses per rubric, attested
label provenance, standard-or-stricter thresholds, and a verified run binding.

The artifact deliberately separates `passed` (the caller-selected exploratory
thresholds) from `headline_eligible` (the fixed promotion floor). A config-only
invocation remains useful for exploration but can never be headline-eligible;
weaker flags can never promote a run. With `--run`, the process exits nonzero
unless `headline_eligible` is true. Without it, exit status follows `passed`.
The complete artifact is printed and is written atomically when `--output` is
supplied.

The built-in rows attest the correctness label but do not claim which endpoint
generated either response, so they cannot invent self-preference evidence. A
custom `--calibration-set` may add both `response_base_url` and
`response_model`. Measurement then requires known producer provenance for every
row and at least two responses in every rubric × correctness label × self/nonself
stratum. (`nonself` means a different resolved endpoint/model identity; it does
not prove different weights.) True- and false-positive gaps are computed in
both prompt orders, and the worse self-favouring gap is gated at 0.20 by default.

If any bound target matches a panel member, this evidence is required
automatically and missing/thin provenance makes `headline_eligible` false.
`--require-self-preference` additionally forces the measurement for members
that do not match a target. Custom rows with attested labels must set
`label_provenance` (`published-gold` or `human-reviewed`) plus
`label_source`, `label_source_revision`, `label_source_record`, and
`label_source_license`. This is a promotion smoke gate, not a statistical proof
of general judge reliability.

## Agentic benchmarks (docker)

```bash
uv sync --extra bench-agentic          # mini-swe-agent, ScaleAI eval deps, harbor
# Official tau-three is not on PyPI. Its v1.x package and CLI remain named
# tau2; pin the v1.0.1 release commit and include the banking knowledge extra:
uv pip install 'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench.git@fc0055dc4e0a316c3f83133267fbd6faaa770992'
# A non-editable install omits task data; retain the same pinned checkout and:
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

SWE-Bench Pro and Terminal-Bench evaluate inside per-task docker containers.
`kairyu bench run` probes `docker info` once; without a working daemon those
two rows report `skipped: docker unavailable` and everything else completes.
The τ-bench harness needs the user simulator (judge) served by the **same
gateway** as the target (single `OPENAI_BASE_URL`).

Fugu's published turn and trial conditions are pinned in the invocations:

| Slot | Condition | How it is passed |
|---|---|---|
| SWE-Bench Pro | 1,000 agent steps (harness default is 250) | `-c swebench_backticks.yaml -c agent.step_limit=1000`; `OpenAICompatTextbasedModel` keeps only standard OpenAI chat-message fields between turns and forwards the target output limit |
| Terminal-Bench 2.1 | terminus-2, 500 turns | `-a terminus-2 --ak max_turns=500`, dataset `-d terminal-bench/terminal-bench-2-1`, results in `--jobs-dir` |
| τ³ Banking | `banking_knowledge`, all retrieval tools, low-effort user simulator | `--domain banking_knowledge --retrieval-config alltools --user-llm-args '{"reasoning_effort":"low"}'` (from the judge's sampling policy), results addressed by `--save-to <name>` under the harness data dir |

Harness output and sampling, verified against the pinned harnesses:

- **Harbor** writes a job-level `result.json` holding `trial_results`, each trial
  carrying its verdict under `verifier_result.rewards` — a *task-defined* dict.
  The adapter prefers the conventional keys (`reward`, `resolved`, `accuracy`,
  `score`, `passed`), accepts a single-key dict whatever it is called, and
  records an ambiguous dict as a **failed** item listing the keys rather than
  guessing. `trial_name` is the item id so `-k > 1` keeps attempts distinct. The
  score is Harbor's own `Mean` — **every** trial counts, an errored one as zero,
  because `aggregate_reward_dicts()` maps a missing reward to zero before
  averaging; excluding errors would report a crashed run as a better score.
- **τ** resolves its data directory itself (`TAU2_DATA_DIR`, else a path *beside*
  `site-packages`), so the adapter imports the harness's own `DATA_DIR` instead
  of reconstructing that layout. `--save-to` is unique per invocation and carries
  the kairyu run id: the harness prompts before resuming an existing results
  file, so a fixed name would make a second run interactive or resume
  simulations from another configuration.
- **Sampling**: τ takes `--agent-llm-args` / `--user-llm-args`, and mini-swe-agent
  takes `model.model_kwargs.*`, so the target output limit and verified
  `reasoning_effort`, `top_p`, and `seed` fields reach the harness. Vendor
  `extra_body` has no equivalent in
  mini-swe-agent, and Harbor exposes no documented sampling passthrough for
  terminus-2; those omissions are annotated on the cell. Explicit target
  `temperature` and recommended-default omission are not claimed through any
  external harness: those rows skip as described above.

`--attempts N` also controls the grouped multi-seed chat runs described above.
For the external harnesses it continues to set trials per task where exposed
(`-k` for Harbor and `--num-trials` for τ). It defaults to **1** because each
attempt is another model request or full container run; Fugu reports τ³ Banking
as **pass@4** and the Terminal-Bench leaderboard requires at least five, and
both facts are annotated on the cell so a single-attempt number is never
mistaken for either. SWE-Bench Pro has no verified repeated-trial flag and
therefore skips unless `attempts: 1`.

## Scale and cost

The full Accuracy suite is expensive by design (HLE alone is ~2500 judged items per
target). Core avoids judges, Docker, and vision, but its complete MMLU population
still makes 14,042 requests. Published request counts assume `attempts: 1`;
increasing it multiplies generated-chat requests while teacher-forced MMLU keeps
its fixed continuation-scoring calls and external harnesses apply their own
trial semantics. For quick runs:

- `--smoke` — deterministic ≤20-item subset per benchmark (CI uses this).
- `--limit N` — cap items per benchmark (seeded, comparable across runs).
- `--only`/`--exclude` — comma-separated slot names.
- `--offline-fixtures` — committed package fixtures/data, with no dataset
  network access (a diagnostic mode without cache-bound dataset identity, used
  to verify plumbing end-to-end).

## Execution runners and threat model

LiveCodeBench and SciCode execute model-generated Python. The execution runner
is an explicit, fingerprinted part of every run:

```yaml
execution:
  runner: docker
  image: sha256:<immutable-local-image-id>
  cpus: 1.0
  pids_limit: 64
  disk_mb: 256
```

The equivalent CLI starts with `--exec-runner docker --exec-image
sha256:<id>`. `local` remains the default for trusted development. It uses a
fresh cwd, a scrubbed environment, `python -I`, Linux rlimits, a new process
group, and a wall-clock kill, but it is **not** a security boundary against
hostile code.

The Docker runner is the unattended/untrusted option. It accepts only
`repository@sha256:<64 hex>` or a local `sha256:<64 hex>` image ID; mutable tags
are rejected instead of being silently resolved. It starts each test with:

- no network and no inherited host secrets;
- a read-only root filesystem, all Linux capabilities dropped,
  `no-new-privileges`, and an unprivileged UID;
- bounded CPU, memory/swap, PIDs, output, writable `/work`, and wall time;
- daemon-side container logging disabled so output cannot bypass the in-process
  output cap and consume host disk;
- no Docker socket, repository, cache, home directory, or arbitrary host mount;
- one private read-only staging mount containing only `main.py`, stdin, and the
  explicitly supplied test artifacts; completed creation gives cleanup an
  immutable container ID before execution starts, and every normal, failed,
  cancelled, or timed-out return removes that exact ID.

The user-code wall timer starts with `docker start --attach`. The preceding
`docker create` cannot execute code. A signal-isolated helper has ten seconds to
transfer the exact ID into an already-armed cleanup lease; after a caller
timeout or interruption it retains cleanup ownership and removes any ID the
trusted daemon returns later. Cleanup therefore never guesses whether a delayed
create request acquired an ID, while a wedged control plane cannot hold the
benchmark caller indefinitely.

Numerical-library worker pools are fixed to one thread so they fit inside the
PID boundary and produce stable scoring behavior across hosts. Large read-only
assets such as SciCode's approximately 1 GB `test_data.h5`
remain in that staging mount and are symlinked into `/work`; they are not
duplicated into the memory-backed work filesystem. Generated files remain
bounded by the tmpfs limits.

Build the supplied runtime (Python plus the hash-pinned NumPy/HDF5 wheels), then
record its immutable ID:

```bash
docker build -f deploy/bench/Dockerfile.exec -t kairyu-bench-exec:local deploy/bench
docker image inspect kairyu-bench-exec:local --format '{{.Id}}'
```

Use the returned `sha256:...` value in the benchmark config. A different
runtime may be used, but it must contain `/usr/bin/env`, a `python` executable
on the runner's scrubbed system `PATH`, and every module the selected suite
requires.
`--exec-runner docker` never falls back to local execution if Docker or the
pinned image is unavailable.

The remaining trusted computing base is the Docker daemon/runtime, host kernel,
the Kairyu benchmark supervisor, host users allowed to inspect its temporary
directory, and the contents of the explicitly trusted digest-pinned image.
Containers use the `io.kairyu.bench-exec=true` label and restart policy `no` so
an operator can remove them after an uncatchable supervisor/host failure.
Container isolation does not defend against a killed supervisor, malicious
image, Docker/kernel escape, hostile privileged host user, or an operator who
mounts additional host resources outside Kairyu.
