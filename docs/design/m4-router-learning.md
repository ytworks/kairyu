# M4 Design: Router Learning Pipeline (serving logs → distillation → online updates)

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (agent design-review panel, 2026-07-02;
see §5). CPU-only, no GPU dependency. Human sign-off pending.
Milestone: M4 (pulled forward — it is GPU-independent, unlike M2's remaining half)
Date: 2026-07-02

## 1. Goal

Replace `RuleRouter` with a learned router on the same pluggable `Router` protocol (design
doc M1/D3), improving toward the acceptance criterion: ≥97% of tier2-only quality at ≥40%
lower inference cost. The pipeline: serving logs → labeled dataset → distilled classifier →
contextual-bandit online refinement.

## 2. Pipeline stages

### 2.1 Outcome logging (extends M1's JSONL decision log)

`JsonlRouterLog` already records `{query_sha256, target, features, confidence, reason}` per
decision. M4 adds outcome records `{query_sha256, target, quality, cost_usd, latency_s}`
(quality ∈ [0,1] from an LLM-as-judge or user feedback). Decisions and outcomes join on
`query_sha256` — raw text is never stored (privacy invariant from M1).

### 2.2 Dataset building (distillation labels)

`build_dataset(records, cost_weight)` groups outcomes by query hash and labels each query
with the **highest mean-utility observed target**, where `utility = quality - cost_weight
* cost_usd` (mean, not max — max is winner's-curse optimistic under noisy judge scores).
Queries with a single observed target still contribute (their label is that target if
utility exceeds a floor). Output: `(features_vector, label)` pairs.

**Known bias (amended per review):** counterfactual arms are only observed when the exact
same query hash recurs under exploration, which is rare in realistic traffic; with mostly
single-arm observations the distilled classifier learns "the chosen arm cleared the
utility floor", i.e., approximately the logging policy plus a positivity filter, NOT the
utility-optimal policy. Mitigations before trusting the distilled model: log propensity
(router name + confidence + ε are in decision records) and weight by inverse propensity,
or pool counterfactuals over feature-space neighborhoods instead of exact hashes. The
online bandit (§2.4) is the component that genuinely optimizes; distillation warm-starts
it.

### 2.3 Distilled classifier

Multinomial logistic regression over the 7 M1 query features, trained with plain SGD in
pure Python — at this dimensionality (7×3 weights) numpy/torch would be a dependency for
nothing, and inference is microseconds, trivially inside the <10ms routing budget.
Standardization statistics are learned with the weights and serialized together as JSON
(`RouterModel.save/load`). `LearnedRouter(model, fallback, min_confidence)` implements the
`Router` protocol; below `min_confidence` it defers to the fallback (`RuleRouter`).
Amended per review: this is an uncertainty deferral, not a "never worse than rules"
guarantee — softmax models are typically *confidently* wrong under distribution shift.
The default threshold is 0.55 (a 3-class softmax max-prob is always ≥ 1/3, so a
near-uniform threshold would never fire); calibrate on held-out data for a target
selective accuracy before serving.

### 2.4 Online refinement (contextual bandit)

`GreedyLinearBandit`: one linear reward model per arm (tier1/tier2/multi_agent) over the
same features, SGD-updated from observed rewards, ε-greedy exploration; scaled features
are clipped so one outlier query cannot destabilize the hot-path SGD, and the feature
space is derived from the same `FEATURE_ORDER` as the classifier (import-time assert).
`BanditRouter(bandit, base, min_updates_per_arm)` adapts it to the `Router` protocol:
until every arm has received `min_updates_per_arm` rewards it defers to the base router
(cold-start safety), after which it selects by predicted reward with ε exploration.
`update()` is O(features) — safe on the serving hot path. This completes the goal's
"ルール→分類器→contextual bandit" progression on the single Router seam.

## 3. Deliberately deferred

- LLM-as-judge harness itself (needs live traffic + a judge model budget) — the pipeline
  consumes its scores via `record_outcome`, no interface change later.
- LinUCB/Thompson posteriors: ε-greedy first (KISS); the bandit interface (`select`,
  `update`) doesn't change.
- The 40%-cost/97%-quality acceptance measurement — requires real traffic on real engines
  (GPU phase); the mechanics are testable now on synthetic logs.

## 4. Tests

- Dataset: max-utility labeling on synthetic multi-target logs.
- Classifier: ≥90% held-out accuracy on separable synthetic data; JSON round-trip.
- LearnedRouter: protocol contract, confidence fallback to RuleRouter, p99 < 10ms.
- Bandit: converges to the reward-optimal arm per context (≥80% correct after training);
  exploration rate honored.

## 5. Review record

Agent design-review panel, 2026-07-02. Verdict: **APPROVE-WITH-AMENDMENTS**. Disposition:

- Fixed in code same day: mean (not max) utility aggregation; `BanditRouter` adapter
  implementing the Router protocol with cold-start deferral; feature clipping + single
  feature-order source with import-time assert; saved-weights validation; outcome logging
  now validates quality ∈ [0,1] and cost ≥ 0; exploration-rate test added. All covered by
  tests.
- Fixed in doc same day: counterfactual/propensity bias disclosed in §2.2 (with inverse-
  propensity mitigation path); "never worse than rules" claim retracted and threshold
  calibration requirement stated in §2.3.
- Deferred: inverse-propensity weighting implementation (needs real traffic where the
  bias is measurable); bandit save/load (arrives with the serving integration); p99
  latency test uses wall-clock and may need a generous bound on loaded CI.
- Human sign-off: pending.
