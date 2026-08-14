# Configuration A/B quality gates (issue #365)

Status: implemented

## Decision

`python -m evals compare --baseline BASE --candidate CANDIDATE` compares one
explicit target from each completed, source-attested scoreboard-history run.
The two runs may have different full run fingerprints because target identity
is the intended experimental variable.  The comparison instead requires an
exact methodology fingerprint formed from the validated run identity with only
`config.targets` removed.  The local harness commit, clean source-tree digest,
Python and execution runtime, ordered adapter identities, datasets, evaluator
protocols, evaluator policy, sample selection, and every other score-bearing run option
must still agree.

Each selected target must carry an operator-declared `served_config` label and
lowercase SHA-256.  A configuration gate refuses anonymous or identical
digests.  The digest is part of the ordinary run fingerprint and makes a
redeployment at the same URL/model distinguishable.  It is not a remote
attestation: operators must hash a canonical deployment manifest that binds the
checkpoint, tokenizer, engine build, and relevant quantization/cache/speculate
settings.  The report states this boundary explicitly.  Apart from target
label, endpoint, API-key environment-variable name, and `served_config`, the
two selected target request policies must match exactly; changing model,
reasoning/sampling options, context/output limits, or vision support is not an
isolated engine-configuration A/B test.

The policy is supplied as one or more `--tolerance BENCHMARK=PP` options.
Values are non-negative percentage points and every named benchmark is a gate;
there is no averaging across rows.  The candidate run owns the derived
`config-comparison.json` and `config-comparison.md` artifacts.  They are saved
atomically before a quality failure is returned and never overwrite the
published-score `comparison.*` artifacts or the immutable history index.

## Evidence boundary

The suite-local `scoreboards.jsonl` is validated end to end before either run
is selected.  For every configured benchmark, the complete `PairResult` is
then reloaded from its contained run directory and its canonical binding,
including `pair_sha256`, is checked against the validated index entry.  A
missing or changed pair file is an error, not a reason to trust the aggregate
scoreboard cell.

Items are joined by unique, non-empty `item_id`; stored order is irrelevant.
The two sets must be identical.  Both pairs and every item must be completed,
all scores must be finite, `n_total == n_scored == len(items)`, and the stored
pair score must equal the recomputed item mean.  No intersection, imputation,
or denominator-only comparison is allowed.  Pair methodology, status reasons,
comparability boundaries, and structured cross-run policy must also agree.
Full-dataset evidence is required: smoke, explicit item limits, offline
fixtures, failed/partial cells, unresolved external-harness provenance, and
unresolved generated-code execution are not configuration gates.

## Statistical contract

The direction is always `candidate - baseline`, evaluated at full precision.
Rendering converts scores and deltas to percentage points only at the display
boundary.

Every run identity records the adapter's `binary_outcomes` declaration and an
optional versioned `paired_cluster_key`; both are fingerprint-bearing
methodology, not declarations re-read from a later checkout.  Independent
Bernoulli items use the paired 2x2 table and Newcombe's paired score method 10.
The artifact records both a two-sided 95% confidence interval for the score
difference and a one-sided 95% lower confidence bound for non-inferiority.
Fractional evidence from an adapter declared binary is an error; it is never
silently reclassified.

Bounded non-binomial scores use a paired percentile bootstrap over item
differences. Binary evidence with dependent sub-items uses a versioned cluster
key that groups item IDs and resamples whole clusters. Each replicate preserves
the declared item-weighted estimand rather than silently changing to an
equal-cluster mean. A single cluster cannot support an interval and is rejected.

Both bootstrap variants fix 20,000 resamples, symmetric nearest-rank quantiles,
the repository-owned `splitmix64_rejection_v1` sampler, and a
direction-independent seed derived from both pair hashes plus the item or
item-to-cluster identity digest.  They do not depend on Python's evolving
`randrange()` algorithm.  The method, PRNG, seed digest, resample count,
two-sided interval, and one-sided lower bound are retained in the artifact;
clustered rows also retain the cluster count and mapping digest.  Each arm is
never resampled independently.  The artifact separately records the Python
runtime used for measurement and the actual implementation/version executing
the comparison, while its protocol hash content-binds the statistics, history,
schema, and aggregate-validation sources.

For tolerance `m`, a row passes only when its one-sided 95% lower confidence
bound is at least `-m`.  Equality passes.  A point estimate inside the margin
with insufficient evidence still fails; missing or invalid evidence is an
error.  Exit status 0 means every configured row passed, 1 means a valid
comparison failed to demonstrate non-inferiority, and 2 means the input,
provenance, or evidence was invalid.
