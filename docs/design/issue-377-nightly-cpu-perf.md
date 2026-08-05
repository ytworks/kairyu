# Nightly CPU performance series (issue #377)

Status: implemented

## Decision

Kairyu runs one main-branch CPU performance series nightly and on explicit
manual dispatch.  The workflow reuses the six source-checkout measurements in
`scripts/cpu_microbench_gate.py` from issue #378, then appends the validated
report to a tamper-evident JSONL series and compares its hard-alert metrics
with trailing medians.  This is a shared-runner smoke alarm.  It does not
upgrade absolute CPU timings or ratios into formal benchmark evidence, and it
does not replace reproduction on an owning benchmark.

Pull requests retain the existing same-process, absolute-bound smoke gate in
`ci.yml`.  They do not append to the nightly series.  Existing seven-day
issue-#378 artifacts are not imported as history because they predate this
series schema, runtime identity, and hash chain.  A new nightly segment must
therefore complete its own five-record warmup.

## Measurement and alert policy

All binding series metrics are higher-is-better ratios measured within one
child process.  The fixed allowlist is:

| Metric | Report field |
|---|---|
| Scheduler FIFO drain | `benchmarks.scheduler_queue.fifo_full_drain.median_speedup` |
| Scheduler indexed removal | `benchmarks.scheduler_queue.indexed_removal.median_speedup` |
| Scheduler priority drain | `benchmarks.scheduler_queue.priority_full_drain.median_speedup` |
| Radix eviction | `benchmarks.radix_eviction.median_speedup` |
| Operation-queue add | `benchmarks.op_queue.add_burst.elapsed_speedup` |
| Operation-queue abort | `benchmarks.op_queue.duplicate_abort_burst.elapsed_speedup` |
| Sampler legacy path | `benchmarks.sampler_penalty_state.legacy_speedup` |
| Sampler append-legacy path | `benchmarks.sampler_penalty_state.append_legacy_speedup` |

For each current metric, the comparator selects at most the seven most recent
preceding records anywhere in the chain with the exact same compatibility
fingerprint and computes their ordinary median.  Intervening records from a
different fingerprint are ignored, and the current record is excluded.  At
least five preceding compatible observations are required.  With zero through
four, the metric is `warmup` and cannot make the cross-run verdict fail.

Let `current` be the new finite positive ratio and `baseline` the finite
positive trailing median.  A regression is binding only when
`current <= baseline * 0.85`; the exact 15% boundary is binding.
Any binding regression makes the comparison command return nonzero after it
has durably written the run artifacts.  The 15% threshold is deliberately not
relaxed to hide shared-runner variance.  In particular, these alerts can be
noisy; they are a prompt to reproduce the affected owning benchmark, not a
performance conclusion or authorization to change production code.

Absolute optimized and legacy timings do not enter the trailing-median
regression comparison.  Router p50/p99, process-wire frame sizes and growth,
and the remaining raw benchmark measurements/checks are also report-only for
that cross-run comparison.  The unchanged issue-#378 same-run source gate is
still binding: one of its performance failures keeps the workflow red even
when it is not promoted to a ninth series metric.  The series does not claim a
serving TTFT measurement because the reused CPU gate does not exercise real
HTTP/model serving.

## Compatibility segmentation and provenance

A comparison is permitted only inside an exact compatibility segment.  The
segment fingerprint binds:

- the nightly method and input report schema versions;
- the ordered benchmark inventory and every fixed workload argument;
- the CPU-only and single-thread environment controls;
- Python implementation and major/minor version;
- runner operating-system, architecture, and hosted-image class;
- every hard metric path and its higher-is-better direction; and
- the 15% threshold, seven-record window, and five-record minimum.

A previously unseen fingerprint starts a new warmup segment while older
records remain visible and validated.  If the environment later returns to an
exact prior fingerprint, that prior compatible history resumes; it is not
discarded merely because incompatible records intervened.  The Git source
commit and workflow run identity are recorded as provenance but are
deliberately excluded from compatibility;
making source revision part of the key would create one segment per main
commit and defeat cross-commit regression detection.  A methodology change
must explicitly bump its stable method/schema identity rather than relying on
an incidental source hash.

Reports and history fail closed on non-finite or non-positive hard metrics,
unexpected metric sets/directions, malformed schemas, duplicate JSON keys or
run identities, and inconsistent runtime/workload/policy identities.  A
failed input is not coerced, omitted, or treated as warmup evidence.

## Series and artifact contract

`series.jsonl` is canonical, append-only JSONL.  Each record binds its prior
record hash and its own SHA-256 over the canonical payload, producing one
ordered chain.  Loading validates the complete file before any record is
eligible.  Truncated, reordered, duplicated, non-canonical, non-finite, or
hash-mismatched input invalidates the whole candidate; recovery never salvages
a prefix.  The current run identity is unique, so a retry is a new attempt and
an already-recorded identity cannot be silently replaced.

Each successfully recorded observation publishes exactly these files from a
dedicated directory beneath `RUNNER_TEMP`:

- `report.json`: the unmodified issue-#378 CPU measurement report;
- `series.jsonl`: the validated prior chain plus the current canonical record;
- `comparison.json`: the current per-metric baseline, delta, status, and
  overall diagnostic/regression verdict.

Each evaluated per-metric `decline_fraction` is a finite decimal string (or
`null` during warmup), so even the full range of valid finite IEEE-754 inputs
cannot turn the JSON artifact into `Infinity` or `NaN`.

A setup, download, or structurally invalid input failure can publish only the
safe subset it produced for diagnosis; such an incomplete artifact is not
eligible history.  Conversely, a structurally valid issue-#378 report remains
recordable when its same-run absolute gate has `all_passed: false`.  The record
retains that source-gate outcome and the workflow stays red, but the real
observation is not erased from later cross-run history.  This exception is
limited to an explicit allowlist of performance checks.  A mismatch in fixed
requests, repeats, benchmark configuration, protocols, shapes, or structural
source checks is invalid input and is never appended.

Artifacts use the immutable per-attempt name
`nightly-cpu-perf-<run-id>-<run-attempt>` and are retained for 90 days.  The
attempt suffix prevents an Actions rerun from colliding with or overwriting
its previous attempt, whose artifact remains eligible history.  Retention
bounds the recoverable series; the comparator's policy needs at most seven
compatible predecessors, while the chain still preserves all records carried
by the newest retained artifact.  No file under `bench/results/` is mutated by
CI.

The intended checkout CLI boundary is:

```bash
uv run --frozen python scripts/cpu_perf_series.py select-history \
  --candidates "$CANDIDATE_ROOT" \
  --output "$HISTORY"

uv run --frozen python scripts/cpu_perf_series.py record \
  --report "$RESULT_DIR/report.json" \
  --history "$HISTORY" \
  --series "$RESULT_DIR/series.jsonl" \
  --comparison "$RESULT_DIR/comparison.json" \
  --repository "$GITHUB_REPOSITORY" \
  --run-id "$GITHUB_RUN_ID" \
  --run-attempt "$GITHUB_RUN_ATTEMPT" \
  --sha "$GITHUB_SHA"
```

`select-history` treats an authoritatively empty candidate set as first-run
warmup and writes the canonical genesis history, validates candidate chains
without modifying them, and copies only the newest valid candidate to the
fixed history path.  A non-empty candidate set with no valid chain returns an
error and does not create an output.  `record` consumes that history and the
already-produced report, writes the updated series and comparison, and then
returns zero for pass/warmup or nonzero for a binding regression.  The
workflow uploads those two outputs together with the unchanged measurement
report.  Artifact, network, and input failures remain distinct hard errors.

## Workflow and recovery boundary

`.github/workflows/nightly-cpu-perf.yml` has only `schedule` and
`workflow_dispatch` triggers, plus a `refs/heads/main` job guard.  One fixed
concurrency group serializes the chain and does not cancel an in-flight run.
The job uses Python 3.12, `uv sync --frozen --dev`, SHA-pinned checkout/setup
and upload actions, and `persist-credentials: false`.

The workflow grants only `contents: read` and `actions: read`; the Actions
token is exposed only to the recovery step.  It has no issue, pull-request,
contents-write, or repository mutation permission and never opens or updates
a GitHub issue.

Recovery enumerates this exact workflow's main-branch artifacts and explicitly
sorts them by artifact creation time and ID in newest-first order without
filtering on run conclusion.  This is required
because a real regression writes and uploads a valid new chain before its run
is marked failed; excluding failed runs would repeatedly compare against a
stale baseline.  Expired entries, archives with anything other than the exact
three regular root files, and histories rejected by the core validator are
skipped while searching for the newest valid candidate.  A successful empty
API listing is the only no-history/warmup case.  Listing, attempt-metadata, or
download transport failure is a hard error; one or more eligible candidates
with no valid history also fails closed rather than silently resetting the
comparison baseline.  Archive members are copied by
exact name into a fresh fixed candidate directory rather than extracted using
archive-provided paths, preventing path traversal.

Before a candidate is selected, its attempt-specific Actions API record must
be completed and must bind the repository, workflow path, main branch,
schedule/manual event, run ID, attempt, and head SHA.  The validated series
tip must carry the same repository, run ID, attempt, and SHA, and the artifact
name must be `nightly-cpu-perf-<run-id>-<attempt>`.  The attempt-specific API
is necessary on rerun: GitHub reuses the run ID and reports the new attempt as
in progress, while the prior completed attempt must remain eligible rather
than being bypassed or overwritten.

The benchmark, recovery, and record steps retain their exit outcomes without
ending the job.  An `always()` upload step runs first.  A final enforcement
step then fails the job if measurement, recovery, or comparison failed.
Consequently both green and regression-red completed runs remain eligible
history, while setup, download, or malformed-artifact failures cannot be
mistaken for a valid data point.
