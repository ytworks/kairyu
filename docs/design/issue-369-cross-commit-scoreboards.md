# Cross-commit benchmark scoreboards (issue #369)

Status: implemented

## Decision

Completed, source-attested `python -m evals run` executions are recorded in a
suite-local `scoreboards.jsonl`.  The stable key is
`(git_commit, fingerprint)`, where `git_commit` identifies the **local Kairyu
benchmark harness** and `fingerprint` binds the immutable run configuration,
target endpoint/model declarations, dataset revisions, scorer protocols, and
judge prompts.  It does not identify or attest the software deployed behind a
target URL.

The index owns immutable scoreboard snapshots rather than paths to mutable run
directories.  Each canonical JSONL record contains the complete run metadata
and scoreboard, its own SHA-256, and the preceding record's SHA-256.  Readers
validate the complete file and chain before returning any entry.  The same key
and same canonical payload is idempotent; the same key with different evidence
is a conflict and is never overwritten.

The index is updated under a fixed sibling lock and published by atomic replace
plus directory fsync.  The index, lock, temporary file, results root, and run
artifacts are containment-checked and may not be symlinks.  A malformed,
non-canonical, non-finite, duplicate-key, duplicate-run, truncated, or
hash-chain-invalid index fails closed.

## Source and resume boundary

Source provenance is derived from the loaded `kairyu/bench/runner.py`, not the
process working directory.  A history entry requires all of the following:

- a tracked module in its own Git checkout;
- a full lowercase commit ID and a clean benchmark source tree;
- `git_commit_role: local_benchmark_harness`;
- a SHA-256 over the source-tree identity;
- a completed run with internally consistent run metadata, pair fingerprints,
  per-pair source identities, complete PairResult digests and summaries,
  target/benchmark matrix, and scoreboard snapshot;
- content identities for adapter/aggregation code and every resolved external
  harness distribution, score-time Python dependency, referenced cache asset,
  and distribution-owned executable.

Dirty or Git-less package executions still retain normal per-run artifacts but
are not cross-commit baselines.  Synthetic offline fixtures are diagnostics,
not measurements, and are not registered as quality history.

Resuming a run requires both the fingerprint and stable environment/source
identity to match the first `run.json`.  `created_at` and transient execution
availability text are not identity. Any observed source drift writes a
permanent taint marker before the run may resume; restoring the checkout does
not make that run id eligible again. The stored metadata remains authoritative
for a rebuilt scoreboard, preventing old pairs from being attributed to a new
commit. Failed runs without a source taint may be resumed. `python -m evals report` only
rebuilds per-run display artifacts; it never backfills history using later
renderer code.

Evaluator or dataset identity is checked before and after every pair and again
after rendering, immediately before append. Any observed drift converts affected
successful evidence to a persisted failed pair. A later resume therefore reruns
it; restoring bytes cannot make the old result indexable.

Opaque request-body extensions are represented in persisted config by their
canonical JSON SHA-256, never the literal body. Credential-bearing URL
components and credential-capable config keys fail closed before a tracked
snapshot is written.

## Run comparison

`python -m evals compare-runs BASE CANDIDATE` resolves both run IDs from the
selected suite's validated index and renders `CANDIDATE - BASE` in percentage
points.  It writes no run artifact and a negative delta alone does not change
the exit status.

Suite, fingerprint, Python/execution runtime, ordered targets, benchmark
structure, and display names must agree. A cell delta is emitted only when both
sides completed with finite scores, matching methodology/incomparability
reasons, matching positive `n`/`n_scored` denominators, and an explicit
`allowed` cross-run policy on both sides. Missing, failed, partial, withheld,
differently substituted, or denominator-mismatched cells are visibly
unavailable or non-comparable rather than silently coerced into a regression
number. An identical subset/substitution boundary permits a labelled diagnostic
delta only; a structured runtime/execution withholding policy never does.
Offline-fixture evidence is labelled non-measurement and never becomes a
quality delta.

This command compares observations associated with target declarations.  A
target redeployed at the same URL and model name is not distinguishable unless
the operator changes a fingerprinted target declaration; no served-engine
commit claim is inferred.
