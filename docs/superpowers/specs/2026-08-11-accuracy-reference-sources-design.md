# Accuracy Reference Sources Design

## Goal

Update the Accuracy benchmark comparison so a completed local run is compared
against the eight-model reference matrix supplied on 2026-08-11. The generated
report must make every published score traceable to its source and measurement
condition.

## Reference Matrix

The stable published-column order is:

1. Fugu
2. Fugu Ultra
3. Fable 5
4. GPT-5.6 Sol
5. DeepSeek-V4-Flash-0731
6. Qwen3.8 MAX
7. GLM-5.2
8. Kimi K3

All eleven Accuracy-suite rows remain in their current order. The committed
catalog uses the values and absences in the supplied final matrix. In
particular, DeepSeek-V4-Flash-0731 has a value only for Terminal-Bench 2.1;
values from earlier DeepSeek V4 Flash snapshots must not be promoted into this
model's column. Missing or inapplicable values remain missing and render as an
em dash rather than zero.

Each committed score record contains its model, score, source identifier,
measurement condition, source class, comparability declaration, and optional
notes. Alternative measurements, tool-enabled variants, different context
buckets, and newer third-party reruns are retained as variants rather than
silently replacing the selected matrix value.

## Report Shape

The existing local measured-score columns remain first. The eight reference
columns follow, then the existing local-minus-Fugu delta columns. The detailed
gap table continues to report the local score minus each available reference
score and withholds a numeric gap whenever the local cell is not comparable.

Every rendered reference score carries a compact source marker such as
`80.3 [S2]`. A new reference-details section contains one row per published
benchmark/model score with:

- benchmark and model;
- selected score or alternate score;
- linked source marker;
- official/provider or third-party classification;
- measurement condition;
- notes, including known harness or version differences.

The source catalog maps every marker to a source title, publisher, URL,
publication date, retrieval date, and source tier. The JSON comparison artifact
keeps the complete score records and source metadata so consumers do not need
to parse Markdown.

## Data and Rendering Boundaries

`kairyu.bench.reference` owns immutable source metadata, selected score records,
variants, column order, and the catalog digest. `kairyu.bench.compare` consumes
those records and owns source-marker assignment and Markdown presentation. No
network request occurs during a benchmark run; committed metadata keeps old
reports reproducible if a provider page changes.

The legacy Sakana headline table remains available for provenance and existing
callers. It is not used to fill missing values in the new eight-column matrix.

## Error Handling

Catalog invariants are enforced by tests: every selected and variant record
must reference a declared source, scores must be finite percentages, selected
records must be unique per benchmark/model, source URLs must be non-empty, and
all displayed reference columns must appear even when a row has no score.
Missing metadata is a development-time failure, not a report-time guess.

## Testing

Use test-driven development:

1. Add failing catalog tests for the eight-column order, the supplied selected
   values, deliberate absences, source classifications, and source binding.
2. Add failing renderer tests showing source markers on score cells, linked
   source catalog entries, per-score conditions, and retained variants.
3. Implement the minimal catalog and renderer changes needed to pass.
4. Run the focused comparison tests, the complete test suite, Ruff, and
   `git diff --check` before committing and opening the pull request.

## Non-goals

- Re-running external vendors' benchmarks.
- Scraping or refreshing source pages at report time.
- Treating provider and third-party scores as like-for-like measurements.
- Changing benchmark adapters, datasets, scoring, sampling, or run-to-run
  comparison semantics.
