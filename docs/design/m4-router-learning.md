# M4 Design: Router Learning Pipeline (serving logs → distillation → online updates)

Status: Draft for review; CPU-only, no GPU dependency
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
with the **highest-utility observed target**, where `utility = quality - cost_weight *
cost_usd`. Queries with a single observed target still contribute (their label is that
target if utility exceeds a floor) — exploration traffic (bandit ε) supplies the
counterfactuals over time. Output: `(features_vector, label)` pairs.

### 2.3 Distilled classifier

Multinomial logistic regression over the 7 M1 query features, trained with plain SGD in
pure Python — at this dimensionality (7×3 weights) numpy/torch would be a dependency for
nothing, and inference is microseconds, trivially inside the <10ms routing budget.
Standardization statistics are learned with the weights and serialized together as JSON
(`RouterModel.save/load`). `LearnedRouter(model, fallback, min_confidence)` implements the
`Router` protocol; below `min_confidence` it defers to the fallback (`RuleRouter`) so a
weak model can never make routing worse than M1's baseline.

### 2.4 Online refinement (contextual bandit)

`GreedyLinearBandit`: one linear reward model per arm (tier1/tier2/multi_agent) over the
same features, SGD-updated from observed rewards, ε-greedy exploration. It wraps any base
router: with probability 1-ε it takes `argmax` of predicted reward, ε explores uniformly.
`update(features, target, reward)` is O(features) — safe to call on the serving hot path.
This is the "ルール→分類器→contextual bandit" progression from the goal, all on one seam.

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
