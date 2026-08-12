# Accuracy 12x3 L3 Readiness Execution Plan

> **Status:** In progress. Do not report completion or mark the PR Ready until one
> fresh real-data run satisfies every denominator, routing, scoring, vision, and
> evidence condition below.

**Goal:** Complete all twelve Accuracy slots with exactly three selected real-data
problems per slot through the public `kairyu-auto-max` L3 model, including official
or default scoring and complete fail-closed denominators.

**Architecture:** Reuse the installed `kairyu bench` Accuracy runner and the
existing eight-GPU tiered example. Add an example-owned entrypoint only where the
current runner cannot bind the required target, judge, executor, storage, and
fingerprint settings reproducibly. Preserve the merged single-ingress
L3 -> L2 -> deployment-owned L1 role DAG and require the publisher output to be
the final public `content`. Diagnose each observed failure from retained raw
artifacts before making the smallest configuration or code correction.

**Fixed run contract:**

- suite `accuracy`; all twelve registered slots; no `--only` or `--exclude` in
  the final run;
- one target named/modelled `kairyu-auto-max` at
  `http://127.0.0.1:${API_PORT:-8003}/v1`;
- `--limit 3 --attempts 1 --seed 0 --concurrency 1` initially;
- real datasets only; no `--smoke` or `--offline-fixtures` evidence;
- a declared context/output limit supported across the complete L3/L2/L1 path;
- immutable served-config SHA-256 and generated-code executor image identity;
- NVMe-backed cache, results, temporary harness work, and persistent raw
  external-harness artifacts;
- new run ID whenever any fingerprinted input changes.

## Phase 1: Baseline and prerequisite audit

- [ ] Confirm the clean dedicated branch and initial plan commit are pushed and
  the required Draft PR exists before benchmark execution or implementation.
- [ ] Record the exact eight-GPU names, UUIDs, compute capability, VRAM, PCI/NUMA
  placement, active processes, and NVMe path/free space without recording
  credentials.
- [ ] Verify Docker daemon operation, `.venv`, benchmark and `bench-agentic`
  dependencies, mini-swe-agent, the official SWE-bench evaluator, Harbor,
  pinned `tau2` v1.0.1, `TAU2_DATA_DIR`, and the immutable execution image.
- [ ] Verify the pinned Qwen3.6, DeepSeek V4, and vLLM identities against the
  example manifest and live model volumes.
- [ ] Check only the presence of `HF_TOKEN` and the readiness/licensing of GPQA,
  HLE, and LiveCodeBench Pro; never print a secret value.
- [ ] Validate every dataset manifest/hash, including the SciCode HDF5 asset,
  before traffic. Download only into the configured NVMe cache.
- [ ] Start or inspect the full stack; validate readiness, exact public model
  inventory, `/routing`, and trace-v2 evidence reaching the publisher.
- [ ] Run a small judge-format probe requiring machine-readable
  `correct: yes|no`.
- [ ] Prove a real CharXiv image can traverse L3 -> L2 -> a vision-capable L1 ->
  verifier/publisher. Treat a dropped/rejected image or a false capability flag
  as a failure requiring diagnosis.

## Phase 2: Reproducible example-owned entrypoint

- [ ] Confirm that no existing command already binds the complete fixed run
  contract. If none does, add the thinnest `accuracy-pilot` command to
  `examples/qwen3.6-deepseek-v4-8gpu/benchmark.py` and expose it through
  `bench.sh` without changing general benchmark semantics.
- [ ] Bind the single L3 target, same-gateway judge/user simulator, limit,
  attempts, item-selection seed, initial concurrency, context/output limits,
  served-config label/SHA-256, immutable Docker executor, NVMe cache/results,
  explicit run ID, and resumable logs/artifacts.
- [ ] Add focused example-control tests that assert all required flags and
  reject accidental L1 targeting, missing digest identities, unsupported
  sampling overrides, or non-NVMe evidence paths.
- [ ] Add a fail-closed final validator for twelve exact slots, one target,
  completed status, zero skipped/partial/failed/unjudged counts, complete
  per-slot denominators, SciCode problem-chain accounting, and retained run/
  dataset/executor/served-config/source identities.

## Phase 3: Targeted real-data diagnosis

- [ ] Start with one immutable diagnostic run identity and all twelve slots so
  independent prerequisite failures are visible together.
- [ ] For each incomplete slot, inspect the PairResult, scoreboard, gateway and
  L1 logs, external-harness logs, trajectories/predictions/reports, and runner
  artifacts to identify the first causal failure.
- [ ] Prefer an existing configuration, cache, dependency, or resume mechanism.
  Change code only for a reproduced product/adapter/runner defect.
- [ ] For a code defect, add the smallest regression test that fails before and
  passes after the correction. Do not weaken scorers, pins, denominators,
  official turn/step limits, or capability checks.
- [ ] Commit each logical correction separately after its focused tests, Ruff,
  and `git diff --check`; push it immediately and update the same Draft PR with
  cause, minimality, tests, run ID, and artifact path.
- [ ] Re-run only the affected slot with
  `--only <slot> --limit 3 --attempts 1 --seed 0`, using a new run ID whenever
  the fingerprint changes. Keep all failed artifacts.

## Phase 4: Final immutable twelve-slot run

- [ ] From the final clean commit and served configuration, create one new run
  ID and execute all twelve Accuracy slots together with no slot filter.
- [ ] Require a scoreboard with exactly one target column and all twelve cells
  `completed`; require zero skips, partials, failures, and unjudged items.
- [ ] Require all three selected ordinary items to be answered and scored;
  require all three agentic tasks to remain in the official denominator even
  when wrong; require every scoreable sub-step from the three SciCode problem
  chains to be accounted for.
- [ ] Verify CharXiv retained three image-bearing requests and evidence that the
  images affected a vision-capable L1 path before publisher completion.
- [ ] Retain run fingerprint, clean Git commit, served-config SHA-256, dataset
  identities/hashes, immutable executor identity, L3 trace, `/routing`, final
  publisher content evidence, and proof that the benchmark never targeted L1.
- [ ] Run focused and related benchmark/example/routing tests, Ruff, and
  `git diff --check`; run broader CPU tests if the corrected scope requires it.

## Phase 5: PR and handoff

- [ ] Keep the Draft PR body current with a twelve-row table containing status,
  `n`, `n_scored`, score, cause/fix notes, tests, run IDs, and artifact paths.
- [ ] Document methodology caveats without converting them into exclusions or
  success claims.
- [ ] Review the final diff for only changes necessary to complete the 12x3 L3
  run. Stop implementation after the completion conditions are met.
- [ ] Mark the PR Ready only after the final run and all validation gates pass.
- [ ] Report the PR, branch and commits, minimality rationale, test results,
  final run and artifact identities, twelve cell results, L3/L2/L1 and vision
  evidence, caveats, and remaining risks.

## Explicit non-goals

- Do not optimize benchmark scores, explore more than the selected three
  problems, add problem-specific prompts, or tune unrelated performance.
- Do not publish L1 pools, target them directly, expose intermediate/private
  reasoning as final content, or fabricate L3/L2/L1 evidence.
- Do not shrink denominators, drop missing predictions/sub-steps, relax official
  harness limits, disable hash/pin checks, or turn incomplete statuses into
  completed results.
- Do not commit datasets, model weights, credentials, caches, raw large
  artifacts, or unrelated refactors/dependency updates.
