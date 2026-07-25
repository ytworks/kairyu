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
| LiveCodeBench | `livecodebench/code_generation_lite` `release_v6` (1,055 problems, pinned commit) | sandboxed pass@1 (public+private tests) | — |
| LiveCodeBench Pro | `QAQAQAQAQ/LiveCodeBench-Pro` split `quater_2025_4_6` + `-Testcase` ZIPs | sandboxed pass@1 (lower bound: no testlib checker) | HF token |
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

### Dataset acquisition notes

- **LiveCodeBench** reads the repo's `test.jsonl`…`test6.jsonl` shards directly
  at a pinned commit. `release_vN` is a *config name*, not a git ref, and the
  loading-script path needs `trust_remote_code` (gone in `datasets` 4.x), so
  going through the files is what keeps the slot working. `release_v6` must
  yield exactly 1,055 problems; any other count fails closed as `unavailable`
  rather than scoring a silent subset.
- **LiveCodeBench Pro** pins Fugu's 2025 Q2 slice (`quater_2025_4_6`, 167
  problems) and joins each `problem_id` to a `<problem_id>.zip` in the testcase
  repo (`testdata/<n>.in` / `.ans`). Acquisition **fails closed**: the split must
  yield exactly 167 problems, every archive must download, and each archive's
  usable cases must match the `sum(subtasks[].n_cases)` it declares, with no
  unpaired half in either direction. An archive that declares **no** count is not
  "as complete as whatever arrived" — that declaration is the only denominator
  evidence there is, so a missing or malformed `config.yaml` fails closed too. `download_file()` turns a timeout, a 401 and a 404 alike into
  `None`, so excluding a problem would cache a smaller denominator permanently —
  and a rate over a shrunken set is not even a lower bound on the full 167. The
  testcase repo's pin is part of the cache identity (`AdapterInfo.extra_sources`)
  so repinning it rebuilds rather than leaving stale bytes "ready" under a new
  methodology. The archives also ship a per-problem testlib `checker.cpp` that
  kairyu does **not** compile: grading is per-line whitespace-normalized
  comparison, so multi-answer problems can only lose points and the cell is a
  **lower bound**.

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
  vision capability, and sampling policy (`reasoning_effort`, `top_p`, `seed`,
  `extra_body_json`); `judge` likewise includes its endpoint/model, API-key
  environment-variable name, concurrency, retry limit, and the same sampling
  policy. Changing the reasoning effort is therefore a different experiment,
  not a resumable run.

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
- **Pinned revisions.** Every slot whose data kairyu downloads is pinned to a
  commit in `kairyu/bench/pins.py`, and that commit is passed to the fetch — a pin
  recorded in the manifest while the bytes came from a moving `main` would make
  the cache and run fingerprint attest something false. `revision` is a git ref,
  so a declared value that is not a commit sha (a config name such as
  `release_v6`) is replaced by the registry pin; the config name goes to `name=`.
  Secondary artifacts that decide a slot's tests or expected answers — the
  LiveCodeBench Pro testcase archives, SciCode's `test_data.h5` — are registered
  in `SECONDARY_PINS` and carried in the adapter's `extra_sources`, so cache
  invalidation and provenance cover them too. This matters: `openai/mrcr` was corrected in
  December 2025 and HLE's item count has shifted since release, so a score taken
  against "whatever `main` was that day" is comparable to neither Fugu's number
  nor an earlier kairyu run. A pin only applies when the recorded dataset id
  still matches, and an adapter that declares its own revision keeps it.
  Refreshing a pin changes the run fingerprint, so stored runs are refused for
  resume rather than silently reinterpreted — the procedure is in that module's
  docstring.
  The **agentic** slots are the exception: mini-swe-agent, Harbor and the τ
  harness fetch their own datasets and expose no revision knob, so SWE-Bench Pro
  in particular tracks upstream (which has had post-release test fixes). That is
  a real limitation of those harnesses, not something this suite can pin.


- **Gated datasets** (GPQA Diamond, HLE, LiveCodeBench Pro): accept the license on the dataset
  page (e.g. <https://huggingface.co/datasets/Idavidrein/gpqa>) and set
  `HF_TOKEN`. Without it those cells report `skipped (gated)` and the run
  continues.

## Sampling policy (reasoning effort)

Fugu reports every model at its **maximum reasoning effort**, and ran the τ³
user simulator at **low**. Sampling belongs to the endpoint, not to a
benchmark, so it is configured per target (and per judge) and applies to every
slot:

```bash
kairyu bench run --base-url http://localhost:8000/v1 --model qwen3-32b \
    --reasoning-effort high --top-p 0.95 --sampling-seed 0 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": true}}' \
    --judge-model qwen3-32b --judge-reasoning-effort low
```

```yaml
targets:
  - name: qwen3-32b
    base_url: http://localhost:8001/v1
    model: qwen3-32b
    reasoning_effort: high
    extra_body_json: '{"chat_template_kwargs": {"enable_thinking": true}}'
judge:
  base_url: http://localhost:8001/v1
  model: qwen3-32b
  reasoning_effort: low
```

`--sampling-seed` is the request `seed`; `--seed` remains the *item sampling*
seed. Unset knobs are simply absent from the request body, so endpoints that
reject them are unaffected.

`extra_body_json` is merged **last**, so it is validated at load time: it must be
a JSON object, and it may not override `model`, `messages`, `stream`,
`temperature`, `max_tokens`, `reasoning_effort`, `top_p`, or `seed`. Those come
from the adapter's request and this endpoint's typed policy — the values the run
fingerprint and methodology record — so letting them through would make the
effective request disagree with the recorded configuration.

This policy reaches every slot that issues its own chat requests. The three
external-harness slots (SWE-Bench Pro, Terminal-Bench, τ³) drive a separate CLI,
so each maps what its harness exposes and annotates what it cannot forward.

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
