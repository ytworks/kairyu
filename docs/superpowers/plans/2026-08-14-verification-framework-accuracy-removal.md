# Verification framework and Accuracy removal implementation plan

## Goal

Remove the externally migrated named Accuracy suite while preserving the Core,
Quantization, Structured Output, and Long Context evaluations. Reclassify
Kairyu engine/system checks as verification rather than benchmarks, without
rewriting historical evidence.

## Architecture

```text
kairyu/          installable product; no eval/verification imports
evals/           checkout-only retained model evaluations
verification/    checkout-only Kairyu correctness/performance gates
evidence/        neutral schemas, catalog, hashing, result primitives
bench/results/   immutable legacy artifacts and legacy index
```

## Hard constraints

- Delete only the named Accuracy suite and its dedicated adapters, fixtures,
  docs, tests, dependencies, and example invocations.
- Keep Core, Quantization, Structured Output, Long Context, and their tests.
- Keep all L1 correctness and performance coverage, but move it under
  `verification/` with stable gate IDs.
- Verification gates are themselves checkout-only test harnesses. Enforce their
  static invariants in their validation paths and registry checks; do not add a
  second pytest layer that tests fixed gate geometry, schemas, or CLI structure.
- Preserve every tracked file under `bench/results/` byte-for-byte.
- Do not ship `evals/`, `verification/`, or `evidence/` in the wheel.
- Do not leave compatibility imports or an installed `kairyu bench` command.

## Work packages

1. Add this decision, migration plan, and progress entry; open a draft PR.
2. Introduce `evidence/` contracts, SHA-256 references, and a versioned catalog
   that can also point to the legacy `bench/results/index.json`.
3. Add the verification registry and runner, then move L1 correctness gates.
4. Move L1 performance gates and classify orchestration, fleet, product, and
   diagnostic checks without changing their measurement semantics.
5. Move retained evaluation infrastructure to `evals/`; require an explicit
   suite/config and retain the four non-Accuracy suite definitions.
6. Remove Accuracy-only adapters, fixtures, agentic harnesses, comparison and
   calibration tools, dependencies, tests, and documentation.
7. Extract any workload needed by product verification (for example the fixed
   tiered coding workload) so verification does not import `evals/`.
8. Remove `kairyu bench`, update packaging, CLI, CI, scripts, examples, ignore
   rules, and documentation for the new command surfaces.
9. Retain only product and evaluation behavior tests under their owning test
   directories. Run dependency-boundary, registry, and evidence inventory checks
   directly, and verify legacy artifact hashes did not change.
10. Run focused tests after each move, then the full CPU suite, Ruff, wheel
    contents check, registry inventory check, and documentation link checks.

## Acceptance

- Searching tracked source/docs finds no active named Accuracy suite or
  Accuracy command; historical evidence may still contain the term.
- `evals` exposes only Core, Quantization, Structured Output, Long Context.
- Verification inventory contains the former correctness/performance gates,
  and formal performance results fail closed without a correctness digest.
- Import-boundary, registry, evidence inventory, and wheel-content checks pass.
- `bench/results/` has the same tracked path-to-SHA-256 map as before the move.
