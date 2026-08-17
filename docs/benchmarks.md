# Checkout-only model evaluations

Kairyu keeps model evaluation separate from product verification.

- evals/ owns four explicit model-evaluation suites: Core, Quantization,
  Structured Output, and Long Context.
- verification/ owns Kairyu correctness, performance, resilience, and
  diagnostic gates, including HF/TP parity, TTFT, TPOT/TPS, throughput,
  goodput, and vLLM comparisons.
- evidence/ provides neutral strict JSON and SHA-256 contracts.
- bench/results/ is the immutable location of legacy tracked artifacts. Its
  existing paths and bytes are preserved.

These tools are source-checkout only. They are not included in the wheel and
there is no installed benchmark command.

## Install checkout dependencies

~~~bash
uv sync --extra evals --group dev
~~~

Datasets are normalized into ~/.cache/kairyu/benchmarks by default. Dataset
and secondary-source revisions are pinned and participate in run identity.
Credentials are read from named environment variables and are never recorded
as literal values.

## Suites

| Suite | Rows | Purpose |
|---|---|---|
| core | GSM8K, MMLU, IFEval | Judge-free deterministic regression checks |
| quantization | Core plus GPQA Diamond | Seven-arm served-configuration quality sweep |
| structured | fixed five-case paired corpus | JSON Schema acceptance, conformance, and exact task result |
| long-context | RULER NIAH at 4K through 1M | Retrieval quality as context length increases |

A suite or config is always required. There is no implicit aggregate suite.

~~~bash
uv run --frozen python -m evals list --suite core
uv run --frozen python -m evals list --suite quantization
uv run --frozen python -m evals list --suite structured
uv run --frozen python -m evals list --suite long-context
~~~

## Run and download

Run a deployed target directly:

~~~bash
uv run --frozen python -m evals run --suite core --smoke \
  --base-url http://localhost:8000/v1 --model m1
~~~

Or use a checked-in configuration:

~~~bash
uv run --frozen python -m evals run --config evals/configs/core.yaml
uv run --frozen python -m evals run --config evals/configs/structured.yaml
uv run --frozen python -m evals run --config evals/configs/quantization.yaml
~~~

Pre-fetch pinned data without running targets:

~~~bash
uv run --frozen python -m evals download --suite core --strict
uv run --frozen python -m evals download --suite long-context --strict
~~~

--smoke and --limit select deterministic development subsets. Their artifacts
remain explicitly non-comparable and must not be presented as full suite
evidence.

## Result contract

Unless overridden, results are written under
bench/results/<suite>/<run-id>/ for compatibility with retained history.

~~~text
run.json
pairs/<benchmark>--<target>.json
scoreboard.json
scoreboard.md
~~~

A clean full run can append an immutable snapshot to the suite-local
scoreboards.jsonl SHA-256 chain. Dirty checkouts, offline fixtures, subsets,
incomplete cells, and unresolved provenance are ineligible for history.
Existing historical artifacts are not rewritten when code ownership changes.

~~~bash
uv run --frozen python -m evals report --suite core RUN_ID
uv run --frozen python -m evals compare-runs --suite core BASE CANDIDATE
~~~

## Core methodology

Core has a stable row order of gsm8k, mmlu, and ifeval.

- GSM8K uses zero-shot numeric exact match with the pinned upstream marker.
- MMLU uses zero-shot teacher-forced raw continuation likelihood for the exact
  candidates A through D. A target that cannot provide the required evidence
  is skipped instead of being scored from generated top-k output.
- IFEval uses the pinned Google checker and reports strict and loose prompt and
  instruction rates. The headline is strict prompt-level accuracy.

All downloaded sources and evaluator bytes are content-bound. Missing scorer
assets or schema/count drift fail closed.

## Structured Output

The structured suite sends paired constrained and unconstrained requests with
identical prompt, sampling policy, and seed. It reports HTTP acceptance, JSON
validity, Draft 2020-12 conformance, exact expected value, and malformed output
separately.

~~~bash
uv run --frozen python -m evals run \
  --config evals/configs/structured.yaml --run-id structured-conformance
~~~

## Long Context

The long-context suite measures exact retrieval across fixed RULER NIAH rows
from 4K through 1M tokens. Target context limits are explicit, and an
unsupported length is skipped rather than silently truncated.

~~~bash
uv run --frozen python -m evals run --suite long-context \
  --base-url http://localhost:8000/v1 --model m1 \
  --max-context-tokens 131072 --run-id m1-long-context
~~~

## Quantization sweep

The quantization suite compares BF16, FP8, INT8, AWQ, GPTQ, NVFP4, and FP8-KV
served configurations across GSM8K, MMLU, IFEval, and GPQA Diamond. Each arm
requires a unique operator-declared served_config label and SHA-256 that binds
the deployment manifest. The declaration is recorded but is not remote
attestation of kernels or checkpoint contents.

~~~bash
uv run --frozen python -m evals quant-sweep --run RUN_ID \
  --tolerance gsm8k=1.0 --tolerance mmlu=1.0 \
  --tolerance ifeval=1.0 --tolerance gpqa-diamond=1.0
~~~

The artifact keeps every task gate separately. It never averages tasks or
shrinks the arm matrix to completed rows.

## Configuration A/B

The compare command gates two explicitly identified served configurations using
paired, item-bound evidence. Apart from target identity and served_config,
request policy and evaluator methodology must match.

~~~bash
uv run --frozen python -m evals compare --suite core \
  --baseline BASE_RUN --candidate CANDIDATE_RUN \
  --baseline-target baseline --candidate-target candidate \
  --tolerance gsm8k=1.0 --tolerance mmlu=1.0 --tolerance ifeval=1.0
~~~

A valid non-inferiority failure is saved before exit status 1. Invalid input,
provenance, or evidence exits 2 without replacing a valid artifact.

## Verification is a separate surface

Use the registry for Kairyu engine and system claims:

~~~bash
uv run --frozen python -m verification list
uv run --frozen python -m verification check
uv run --frozen python -m verification run GATE_ID -- --help
~~~

Formal performance execution requires an accepted correctness artifact and
records its exact SHA-256. See verification/README.md for the registry, scopes,
evidence envelope, and gate invocation contract.

## Repository checks

~~~bash
uv run --frozen pytest tests/evals
uv run --frozen python scripts/verify_verification_registry.py
uv run --frozen python scripts/verify_bench_results_index.py
uv run --frozen python scripts/verify_bench_wheel.py
~~~

The wheel check rejects accidental inclusion of evals/, verification/,
evidence/, bench/, and tests/.
