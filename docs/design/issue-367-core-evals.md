# Issue #367 Design: Cheap Deterministic Core Evals

Status: **Implemented and merged** (2026-08-05); MMLU scoring amended by
issue #368 as described below.

Related contracts: M7 benchmark adapter/cache/result contracts and the Accuracy
suite's existing reproducibility, degradation, resume, and reporting rules.

## 1. Goal and non-goals

The existing `accuracy` suite deliberately follows Sakana's 11-row Fugu release table.
Most of those rows are expensive frontier, judge, vision, generated-code, or
agentic evaluations. Issue #367 adds a separate `core` suite for frequent,
deterministic quality regression checks:

1. GSM8K numerical exact match;
2. MMLU ordered continuation-likelihood choice ranking;
3. IFEval rule-based instruction following.

`accuracy` remains the default and keeps its row order, result path, published-score
comparison, and all 11 adapters. `core` is not inserted into the Accuracy table and
is never compared with Fugu's published values. It reuses the same target,
download/cache, request, retry, resume, pair evidence, scoreboard, fixture, and
Wilson-confidence-interval contracts.

“Cheap” describes the absence of an LLM judge, Docker, vision, and agent loops.
A full run sends 58,028 target calls: one per GSM8K/IFEval item and four exact
candidate-continuation calls per MMLU item.
Fast development runs use the existing deterministic `--smoke` or `--limit`
paths and are marked non-comparable by the existing run contract; a subset is
never presented as a full score.

## 2. Suite and artifact contract

The canonical core row order is `gsm8k`, `mmlu`, `ifeval`. The supported suite
names, display names, row order, and whether a published reference exists are
defined once and consumed by registry selection, aggregation, rendering, CLI
listing, run progress, and reporting.

```bash
kairyu bench run --suite core --base-url http://localhost:8000/v1 --model model
kairyu bench run --suite core --base-url http://localhost:8000/v1 --model model --smoke
kairyu bench download --suite core
kairyu bench list --suite core
```

When no result path is explicitly configured, a suite writes below
`bench/results/<suite>`. Core runs write `run.json`, pair JSON, and
`scoreboard.{json,md}` but no `comparison.{json,md}`. Accuracy runs retain their
existing comparison files. A comparison builder rejects a non-Accuracy scoreboard
instead of silently emitting a table full of missing reference values.

All three headline item outcomes are Bernoulli values, so a complete pair may
use the existing 95% Wilson interval. For the generative rows, a non-empty
completion with no required answer marker is an ordinary wrong answer (`0`),
while empty content is failed/unmeasured. MMLU has no generated-text or empty-
completion fallback: all four exact likelihood records must validate before an
item can be right or wrong. Unavailable data, scorer dependencies, fixed
resources, unknown checker IDs, or schema/count drift fail the adapter or pair
closed rather than manufacturing a zero.

## 3. Pinned data and adapter semantics

### 3.1 GSM8K

- Data: `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`,
  config `main`, split `test`, exactly 1,319 rows.
- Prompt: zero-shot Kairyu chat variant. The model may show work but must finish
  with `#### <number>`.
- Score: the upstream GSM8K answer pattern
  `#### (-?[0-9.,]+)`; commas are removed and the resulting strings are compared
  exactly. There is no float coercion, tolerance, or judge.
- Headline metric: micro accuracy over all items.

The zero-shot chat prompt is recorded as Kairyu methodology; only the answer
extraction is claimed to match the original scorer.

### 3.2 MMLU

- Data: `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`,
  config `all`, split `test`, exactly 14,042 rows and 57 subjects.
- Prompt: zero-shot multiple choice with the upstream A-D choice order
  preserved, ending at `Answer:` on the raw completions surface.
- Score: exact teacher-forced raw log-likelihood for the ordered continuations
  `" A"`, `" B"`, `" C"`, and `" D"`; each must resolve to exactly one target
  token. Stable candidate order resolves an exact tie, which remains recorded
  in item evidence.
- Headline metric: item-micro accuracy over the full test set.

Issue #368 supersedes the initial generated-letter transport while preserving
the predeclared zero-shot population and prompt boundary. Canonical MMLU uses
five subject-specific development examples as well as next-token A-D logprob
argmax, so Kairyu still neither labels this zero-shot variant canonical nor
compares it with published MMLU numbers. The zero-shot difference remains
permanent methodology and annotation data, not a footnote added after seeing a
result. Exact likelihood semantics and failure handling are defined in
`docs/design/issue-368-loglikelihood.md`.

### 3.3 IFEval

- Data: `google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84`,
  config `default`, split `train`, exactly 541 prompts, 834 instruction
  instances, and the complete set of 25 registered instruction IDs.
- Checker: the Apache-2.0 Google Research implementation at commit
  `066e1eda43f4785922e3994e95429e496080231f`, packaged with explicit
  provenance and deterministic dependency loading. There is one documented
  dataset-consistency amendment: pinned rows 1122 and 1129 require `#` and `!`,
  while upstream `LetterFrequencyChecker` silently replaces either character
  with a random ASCII letter. Kairyu accepts the dataset's exact single
  non-whitespace character instead, eliminating that random fallback without
  changing the other checker semantics.
- Prompt: the dataset prompt, byte-preserved as the sole user message. No
  prefix, suffix, few-shot example, or system message may change repeat/start/
  end constraints.
- Output budget: the target's configured maximum. The benchmark contains up to
  1,200-word and 100-sentence requirements, so a small adapter cap would make
  valid answers impossible.

The evaluator retains all four official aggregates:

- strict prompt-level accuracy;
- strict instruction-level micro accuracy;
- loose prompt-level accuracy;
- loose instruction-level micro accuracy.

The single scoreboard headline is strict prompt-level accuracy: an item is one
only when every instruction on that prompt passes. Google reports the four
metrics without declaring one official headline, so this selection is explicitly
Kairyu's display policy.

Loose evaluation tests the original response plus the official seven variants
formed by removing asterisks and/or the first and last lines. Every item retains
its strict and loose instruction outcomes as structured evidence so pair-level
metrics can be recomputed.

The reference checker uses `langdetect`; Kairyu fixes its global detector seed
to zero before any detection. The English NLTK Punkt table is fetched only by
the dataset download path from `nltk/nltk_data` commit
`550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`, verified against SHA-256
`e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`, and
loaded from that adapter cache. Scoring never calls `nltk.download()` and never
uses an unpinned global cache.

## 4. Fail-closed normalization and identity

Every dataset revision participates in the existing cache and run fingerprint.
IFEval additionally binds the checker source commit and Punkt source commit;
the archive content digest is verified before extraction and recorded in
methodology. Archive extraction materializes only the four expected English
parameter files and rejects missing, duplicate, or digest-mismatched members;
all other archive members are ignored rather than extracted.

Normalization enforces:

- GSM8K: 1,319 unique index-derived IDs, non-empty questions, and a parseable
  gold marker on every row;
- MMLU: 14,042 unique index-derived IDs, exactly 57 non-empty subjects, exactly
  four string choices, and integer answer indices in `[0, 3]`;
- IFEval: 541 unique keys and prompts, 834 total instructions, exactly the 25
  registered IDs, equal instruction/kwargs list lengths, and a concrete value
  for every checker argument required by the pinned data.

Changing any primary or secondary pin changes the run fingerprint and makes an
old cache/run ineligible for resume. Offline fixtures are synthetic contract
tests, not benchmark evidence; the existing fixture run reason prevents their
scores from being treated as full measurements.

The packaged source bytes implementing each core prompt, parser, and scorer
also enter that adapter's run identity. MMLU additionally binds the shared exact
likelihood client/parser/scorer implementation delivered by issue #368. IFEval
binds both its adapter and every vendored checker module. Consequently, changing
candidate reduction/boundary validation, the exact-character amendment, loose
variants, Punkt/dependency digest constants, or any other score-bearing
implementation changes the run fingerprint before stored pair evidence can be
reused.

## 5. Verification required for closure

Portable tests cover parser boundaries, wrong/missing markers, prompt bytes,
normalization counts and schema drift, all 25 checker registrations, strict and
loose behavior, four-metric aggregation, suite-local only/exclude validation,
core row order and heading, Wilson eligibility, comparison suppression, pinned
fetch arguments, offline end-to-end execution, and the exact packaged fixture
inventory. The wheel verifier must contain 11 benchmark fixtures plus the judge
calibration fixture while retaining the existing 64 checkout-only benchmark
entrypoints.

The full local CPU suite, coverage gate, lint, entrypoint/wheel boundary, and
GitHub CI remain mandatory before merge. Release validation downloads all three
pinned datasets into a temporary cache and replays all 541 IFEval prompts/834
instructions twice. Portable tests monkeypatch the pinned schemas, so normal CI
never relies on network state.
