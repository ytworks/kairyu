# Fugu Benchmark Suite (`kairyu bench`)

One command runs every benchmark from Sakana's Fugu release table
([sakana.ai/fugu-release](https://sakana.ai/fugu-release/)) against a deployed
kairyu gateway — single models and orchestrations side by side — then writes a
dated, footnoted scoreboard. This implements goal G6 gate P-C1 ("one command →
dated scoreboard") and the roadmap §6 evidence rules (per-item results,
methodology, config committed next to every number).

The perf harnesses in the top-level `bench/` directory (TTFT/TPOT/goodput)
are separate; this suite measures answer quality.

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
kairyu bench run --config examples/bench_fugu.yaml
```

Results land in `bench/results/fugu/<run_id>/`:

```
run.json                                      # fingerprint + identity + config + environment
<benchmark>--<sha16>/<target>--<sha16>.json   # one PairResult per scoreboard cell
scoreboard.json                               # machine-readable table
scoreboard.md                                 # Fugu-layout table (also printed to stdout)
```

Benchmark and target components retain a readable sanitized prefix and append
the first 16 hexadecimal characters of the raw name's SHA-256. Thus names such
as `org/model` and `org__model`, which otherwise sanitize to the same path, do
not overwrite one another. A run id must be one non-dot path component;
absolute paths, separators, Windows drive paths, and symlink escapes outside
the results or run directory are refused. Result writes are atomic.

Useful subcommands:

```bash
kairyu bench list                      # slots, requirements, cache status
kairyu bench download [--only a,b]     # pre-fetch datasets (idempotent)
kairyu bench report <run_id>           # rebuild + print a stored scoreboard
```

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

## The 11 slots

| Slot | Source | Scoring | Requires |
|---|---|---|---|
| SWE-Bench Pro | `ScaleAI/SWE-bench_Pro` | mini-swe-agent scaffold + swebench docker eval, resolved rate | docker, `[bench-agentic]` |
| Terminal-Bench 2.1 | Harbor registry | `harbor run` (terminus-2), accuracy | docker, `[bench-agentic]` |
| LiveCodeBench | `livecodebench/code_generation_lite` | sandboxed pass@1 (public+private tests) | — |
| LiveCodeBench Pro | `QAQAQAQAQ/LiveCodeBench-Pro(+-Testcase)` | sandboxed pass@1 (community mirror, not the official OJ) | — |
| Humanity's Last Exam | `cais/hle` (gated) | MCQ exact match + judge for free-form | HF token; judge for free-form |
| CharXiv Reasoning | `princeton-nlp/CharXiv` | judge-graded, vision content-parts | vision target + judge |
| GPQA Diamond | `Idavidrein/gpqa` (gated) | MCQ exact match, seed-shuffled choices | HF token |
| SciCode | `SciCode1/SciCode` | sandboxed sub-step tests (+`test_data.h5` golden data) | numpy in venv |
| τ³-Bench Banking | tau3 harness package | official reward (agent = target, user-sim = judge) | tau3/tau2 harness + judge |
| Long Context Reasoning | `THUDM/LongBench-v2` **substitute** | MCQ exact match | — |
| MRCRv2 | `openai/mrcr` | official prepend + SequenceMatcher ratio | long-context target |

Annotated caveats appear as scoreboard footnotes automatically, notably:
the Long Context Reasoning slot is a **LongBench v2 substitute** (Fugu's own
suite is unpublished; numbers are not directly comparable), and LiveCodeBench
Pro is scored by the local sandbox, not the official judge.

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
  and validated `data.jsonl` SHA-256 (or an explicit unavailable marker); and
- the output-affecting `BenchConfig` fields `suite`, `targets`, `judge`, `limit`,
  `smoke`, `offline_fixtures`, `only`, `exclude`, `seed`, `concurrency`,
  `request_timeout_s`, and `retries`. `targets` includes every target's name,
  base URL, model, API-key environment-variable name, context/output limits,
  and vision capability; `judge` likewise includes its endpoint/model,
  API-key environment-variable name, concurrency, and retry limit.

Exactly five execution or location controls are excluded: `run_id`,
`results_dir`, `cache_dir`, `rerun`, and `download`. API-key *environment
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
  fixtures are tiny synthetic stand-ins for offline testing).
- A cache entry is ready only when `manifest.json` and `data.jsonl` exist, the
  manifest contains a well-formed lowercase SHA-256, a streaming hash of the
  current JSONL bytes matches it, and any requested dataset id/revision pins
  match. Missing, malformed, unreadable, stale, or modified entries fail closed
  as not ready; a readiness check never rewrites or deletes them. The same
  identity is checked again immediately before each pair, so bytes that change
  after run initialization are skipped rather than scored as valid input.
- Download deps are an extra: `uv sync --extra bench` (or
  `pip install 'kairyu[bench]'`).
- **Gated datasets** (GPQA Diamond, HLE): accept the license on the dataset
  page (e.g. <https://huggingface.co/datasets/Idavidrein/gpqa>) and set
  `HF_TOKEN`. Without it those cells report `skipped (gated)` and the run
  continues.

## Judge configuration

Free-form grading (HLE, CharXiv) and the τ-bench user simulator use a
configurable OpenAI-compatible judge endpoint:

```bash
kairyu bench run ... --judge-base-url http://localhost:8000/v1 --judge-model kairyu-auto
```

The judge model is disclosed in every pair's methodology. Self-judging is
detected from the resolved endpoint/model identity used for requests: trailing
slashes are removed and the standard OpenAI `/v1` path is appended when absent,
while scheme, host, port, any other path, and the exact model remain significant.
Display aliases therefore cannot hide the bias. Legacy reports that indicate a
judge but lack either resolved identity are annotated `judge independence unknown`
instead of being declared independent; an explicitly disabled judge is not.
Without a judge, MCQ items still score exact-match; free-form items are recorded
`unjudged`. Judge verdicts that fail to parse degrade the item, never the run.

## Agentic benchmarks (docker)

```bash
uv sync --extra bench-agentic          # mini-swe-agent, swebench, harbor
# tau3 is not on PyPI: pip install git+https://github.com/sierra-research/tau3-bench
```

SWE-Bench Pro and Terminal-Bench evaluate inside per-task docker containers.
`kairyu bench run` probes `docker info` once; without a working daemon those
two rows report `skipped: docker unavailable` and everything else completes.
The τ-bench harness needs the user simulator (judge) served by the **same
gateway** as the target (single `OPENAI_BASE_URL`).

## Scale and cost

The full suite is expensive by design (HLE alone is ~2500 judged items per
target). For quick runs:

- `--smoke` — deterministic ≤20-item subset per benchmark (CI uses this).
- `--limit N` — cap items per benchmark (seeded, comparable across runs).
- `--only`/`--exclude` — comma-separated slot names.
- `--offline-fixtures` — committed synthetic fixtures, no network at all
  (used to verify the plumbing end-to-end).

## Execution sandbox caveat

LiveCodeBench/SciCode run model-generated code in a subprocess with a fresh
temp cwd, scrubbed env, `python -I`, rlimits (memory/CPU/procs/file size) and
a wall-clock kill. This contains runaway code but is **not a security
boundary against a hostile model** — run untrusted evaluations inside a
container (a `--exec-runner docker` hook is future work).
