# SWE-bench Verified benchmark integration

Date: 2026-08-12
Issue: [#472](https://github.com/ytworks/kairyu/issues/472)

## Goal

Add `swe-bench-verified` to Kairyu's Accuracy suite as a first-class agentic
benchmark. It must be launchable through the same `kairyu bench run` surface as
the other benchmarks, use the upstream mini-SWE-agent generation flow and the
official SWE-bench evaluation harness, preserve auditable artifacts, and appear
in the existing score and frontier-comparison reports.

The supported suite command is:

```bash
kairyu bench run \
  --config bench/configs/accuracy.yaml \
  --only swe-bench-verified
```

The adapter also remains available through the ordinary direct benchmark
flags (`--base-url`, `--model`, `--results-dir`, `--concurrency`, `--limit`,
and `--seed`) accepted by `kairyu bench run`.

## Non-goals

- Kairyu will not replace or reimplement the official repository test harness.
- This change will not claim that a single Kairyu run is directly comparable
  with a vendor's multi-trial or differently scaffolded result.
- This change will not restore SWE-bench Verified as a recommended frontier
  benchmark. OpenAI retired it from model reporting because of contamination
  and test-quality concerns; Kairyu will surface that caveat while supporting
  the issue-requested run.
- A full 500-instance Docker run is not part of CPU CI. CI will validate command
  construction, artifact parsing, scoring, reporting, and failure behavior.

## Upstream contracts

### Generation

Use mini-SWE-agent's official SWE-bench batch entry point:

```bash
mini-extra swebench \
  --model <model> \
  --subset verified \
  --split test \
  --workers <concurrency>
```

The command must explicitly retain mini-SWE-agent's `swebench.yaml` base
configuration when Kairyu supplies overrides. mini-SWE-agent drops the default
config after the first `-c`, so Kairyu will put `-c swebench.yaml` before its
endpoint, output, seed, slice, and other overrides.

SWE-bench Verified uses the standard mini-SWE-agent step limit of 250. The
existing `swe-bench-pro` adapter keeps its deliberate 1,000-step override; the
shared implementation must not silently change that benchmark's method.

The upstream dataset selected by `--subset verified` is
`princeton-nlp/SWE-Bench_Verified`, split `test`, containing 500 instances.
mini-SWE-agent writes a prediction mapping to `mini-output/preds.json`; Kairyu
will preserve the complete mini-SWE-agent output directory as an artifact.

Primary references:

- [mini-SWE-agent SWE-bench usage](https://mini-swe-agent.com/latest/usage/swebench/)
- [mini-SWE-agent SWE-bench runner](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/benchmarks/swebench.py)
- [mini-SWE-agent standard SWE-bench configuration](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/benchmarks/swebench.yaml)
- [SWE-bench Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

### Evaluation

Evaluate the generated predictions with the official harness:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path <preds.json> \
  --instance_ids <exact prediction IDs...> \
  --max_workers <concurrency> \
  --run_id <stable run identifier> \
  --report_dir <adapter work directory>
```

Passing the exact prediction IDs is required. For `--limit N`, the official
report denominator must be the selected N instances, not all 500 dataset
instances. The adapter will read and validate the prediction mapping before
evaluation, reject an empty or malformed mapping, and pass its keys to
`--instance_ids`.

The official harness runs tests inside architecture-specific Docker images.
The adapter must therefore fail with a clear precondition message when Docker,
`mini-extra`, or the endpoint/model inputs are unavailable. Kairyu will not
pretend that an unsupported host architecture produced a benchmark score.

Primary references:

- [SWE-bench evaluation harness reference](https://www.swebench.com/SWE-bench/reference/harness/)
- [Official evaluation runner](https://github.com/SWE-bench/SWE-bench/blob/master/swebench/harness/run_evaluation.py)
- [Official report construction](https://github.com/SWE-bench/SWE-bench/blob/master/swebench/harness/reporting.py)

## Architecture

Extract the two-stage mechanics now embedded in `SweBenchProAdapter` into a
shared SWE-bench adapter implementation. A per-benchmark specification supplies
the public name, display name, mini-SWE subset, official evaluation dataset,
step limit, and comparison policy.

Two thin adapters use it:

| Adapter | mini-SWE subset | Evaluation dataset | Step limit |
|---|---|---|---:|
| `swe-bench-pro` | `ScaleAI/SWE-bench_Pro` | `ScaleAI/SWE-bench_Pro` | 1,000 |
| `swe-bench-verified` | `verified` | `princeton-nlp/SWE-bench_Verified` | 250 |

The refactor must preserve the Pro adapter's public behavior and its existing
tests. Shared code owns preconditions, environment construction, subprocess
execution, prediction validation, report discovery, parsing, scoring, artifact
metadata, and failure conversion. Dataset-specific adapters must not duplicate
those paths.

`swe-bench-verified` is registered in the Accuracy suite immediately after
`swe-bench-pro`. It is agentic and externally dependent, so ordinary inventory,
suite selection, availability, history, and CLI rendering work without a
parallel special-purpose entry point.

## Work directory and stored provenance

During execution, the isolated adapter work directory contains:

- the `mini-output/` generation directory, trajectories, logs, and `preds.json`;
- the official SWE-bench report JSON;
- the official harness's separately managed per-instance evaluation logs.

Kairyu's persisted pair result embeds the official report's validated category
counts and ID lists, plus the generated and evaluated commands, dataset names,
split, selected instance IDs, scaffold/config name, step limit, concurrency,
seed, and pinned evaluator distribution identities already captured by the
suite fingerprint. The potentially very large mini-SWE-agent trajectories,
container images, and upstream harness log tree are not copied into the normal
scoreboard result directory.

Secrets from endpoint credentials or environment variables must not be copied
into commands or report artifacts.

The normal suite reporter continues to produce `scoreboard.json`,
`scoreboard.md`, `comparison.json`, and `comparison.md`; no separate score
report format will be introduced.

## Official report parsing and score

Consume SWE-bench report schema version 2 and validate it fail-closed. The
parser must verify that count fields match their corresponding ID arrays, IDs
are unique and mutually exclusive where required, all submitted IDs belong to
the exact selected set, and category totals cover the selected denominator.
Malformed, ambiguous, missing, or multiple candidate reports are benchmark
failures, not partial scores.

Map official outcomes to Kairyu items as follows:

| Official category | Kairyu status | Item score |
|---|---|---:|
| resolved | completed | 1.0 |
| unresolved | completed | 0.0 |
| empty patch | completed | 0.0 |
| harness error | failed | none |
| incomplete | failed | none |

The benchmark score is `resolved / selected instances`. Empty patches, harness
errors, and incomplete runs remain in that denominator, matching the official
task-level resolution rate. The summary must expose category counts so a zero
score cannot hide infrastructure failures.

## Frontier comparison report

Add a `swe-bench-verified` row to the existing sourced Accuracy comparison.
Use only scores for the exact model represented by each existing comparison
column; do not substitute an older similarly named model.

The current eight-model row will contain:

| Model column | Published value | Policy |
|---|---:|---|
| Fugu | — | Fugu's published comparison uses SWE-bench Pro, not Verified |
| Fugu Ultra | — | Fugu Ultra's published comparison uses SWE-bench Pro, not Verified |
| Fable 5 | 95.0 | Anthropic system card, standard configuration, mean of five trials, thinking blocks included |
| GPT-5.6 Sol | — | No exact-model Verified score; OpenAI no longer reports this benchmark |
| DeepSeek-V4-Flash-0731 | — | No confirmed exact-model Verified score |
| Qwen3.8 MAX | — | No confirmed exact-model Verified score |
| GLM-5.2 | — | No confirmed exact-model Verified score |
| Kimi K3 | — | The model report does not report Verified |

The local comparison cell is marked not comparable to the 95.0 reference:
Kairyu runs one mini-SWE-agent trial, while Anthropic reports the mean of five
trials under its stated standard configuration. The report renders the local
score but withholds a misleading delta and explains the method difference.

Primary references:

- [Claude Fable 5 & Claude Mythos 5 System Card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf)
- [OpenAI: Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [OpenAI: Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)

The retirement warning belongs in the benchmark documentation and the sourced
comparison note so users do not mistake continued execution support for a
recommendation to use Verified for current frontier ranking.

## Error handling

Every external-stage failure returns a failed benchmark result with its stage
and a concise diagnostic. Expected cases include missing executables, Docker
unavailability, generation timeout/non-zero exit, missing or malformed
predictions, evaluation timeout/non-zero exit, absent/ambiguous report, and
invalid report schema. No such path may emit a numeric aggregate score.

Per-instance official harness errors remain scored failures in an otherwise
valid report, as described above; they do not invalidate other completed
instances.

## Test strategy

Development follows test-driven development. Focused tests will establish:

1. Registry, Accuracy inventory, config selection, history policy, and CLI
   discovery for `swe-bench-verified`.
2. The exact official generation/evaluation arguments, including the retained
   base config, `verified`/`test`, 250 steps, concurrency, and selected IDs.
3. `--limit` evaluation against exactly the generated IDs.
4. Strict schema-v2 parsing for resolved, unresolved, empty-patch, error, and
   incomplete outcomes plus rejected malformed/ambiguous artifacts.
5. Shared-code regression coverage proving Pro still uses its dataset and
   1,000-step method.
6. Scoreboard/history serialization and sourced comparison rendering, including
   Fable 5 at 95.0, missing exact-model values, the non-comparable local cell,
   and retirement/method caveats.
7. Precondition and subprocess failures that never produce a numeric score.

Completion requires focused benchmark tests, the full test suite, Ruff, and a
CLI smoke check. A real full-dataset run remains an operator GPU/CPU/Docker
exercise documented with the exact command above.

## Acceptance criteria

- `kairyu bench run --config bench/configs/accuracy.yaml --only
  swe-bench-verified` selects and launches the new adapter.
- Generation and evaluation match the official upstream interfaces documented
  above, with the 250-step standard Verified configuration.
- Full and limited runs use the correct official denominator and preserve all
  failure categories.
- Existing SWE-bench Pro behavior remains intact at 1,000 steps.
- Standard scoreboard, comparison, history, and artifact outputs include the
  new row with auditable sources and no fabricated scores.
- Documentation states Docker/architecture requirements and OpenAI's retirement
  caveat.
- Tests and static checks pass, and the pull request closes issue #472.
