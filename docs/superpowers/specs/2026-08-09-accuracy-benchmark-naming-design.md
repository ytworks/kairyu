# Accuracy Benchmark Naming Design

## Goal

Rename Kairyu's internally named `fugu` benchmark suite to `accuracy` so the
name describes its purpose: validating model and orchestration answer quality.
The benchmark's behavior, methodology, data, scoring, ordering, thresholds,
and comparison calculations must remain unchanged.

## Naming boundary

The canonical internal name becomes `accuracy`, with the display name
`Accuracy`. This applies consistently to:

- suite registry keys, type literals, defaults, constants, and help text;
- example configuration and shell-script filenames;
- result-directory defaults and `.gitignore` rules;
- tests, fixtures, function names, and assertions that describe the suite;
- operator documentation and current-status descriptions of Kairyu's suite.

No `fugu` compatibility alias is retained because that would preserve the
inappropriate internal name. Existing untracked result directories are not
moved or rewritten; new runs use `bench/results/accuracy` (or
`results/accuracy` in the Qwen example).

## References that remain unchanged

`Fugu` and `Fugu Ultra` remain where they identify Sakana's actual products or
published score columns. The Sakana release URL and asset paths also remain
unchanged. These are provenance data rather than names for Kairyu's benchmark.

Append-only progress history is not rewritten. Historical entries may retain
the old suite name exactly as originally recorded, as required by
`.claude/rules/progress-log.md`.

## Implementation

The suite registry changes from `fugu`/`FUGU_ROW_ORDER` to
`accuracy`/`ACCURACY_ROW_ORDER`, while preserving the exact eleven-element row
tuple. All consumers receive the new suite identity through their existing
interfaces; no benchmark adapter or scoring implementation changes.

The Qwen3-32B example is renamed to `accuracy-benchmark.sh` and
`run-accuracy-benchmark.sh`, and its static-contract test is renamed in lockstep.
Configuration, documentation, help output, default result paths, report titles,
and test expectations follow the same canonical name.

Text that currently conflates the Kairyu suite with its source comparison table
is rewritten to distinguish the `Accuracy` suite from the published Sakana
reference scores. Methodology notes that factually describe a published
product's conditions retain the product name.

## Functional invariants

The following must be byte-for-byte or value-for-value unchanged except for
suite labels, paths, filenames, headings, and prose naming:

- the eleven benchmark adapter identifiers and their order;
- dataset revisions, item populations, prompts, and sampling settings;
- execution limits, attempt counts, and external-harness arguments;
- aggregation, withholding, comparison, and calibration behavior;
- published reference models and numeric scores;
- JSON schema version and all non-suite fields;
- CLI command structure and option behavior.

## Verification

Run focused benchmark, CLI, wheel, and example-contract tests, followed by the
full test suite and Ruff. Compare the old and new eleven-row constants directly
during review to confirm that only the identifier changed.

Finally, scan filenames and live source/documentation for case-insensitive
`fugu`. Every remaining match must be either:

1. an immutable historical progress entry, or
2. a factual Sakana product, score-column, URL, asset-path, or methodology
   attribution.

Any remaining use of `fugu` as a Kairyu suite, file, test, result directory,
log prefix, or display label fails the migration.
