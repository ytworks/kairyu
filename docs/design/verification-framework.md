# Verification framework and Accuracy removal

Status: accepted; implemented (2026-08-14)

## Decision

Kairyu separates product code, model evaluation, and system verification:

- `kairyu/` contains only the installable inference product.
- `evals/` is checkout-only evaluation tooling for Core, Quantization,
  Structured Output, and Long Context. The named Accuracy suite is removed.
- `verification/` is checkout-only verification for L1 correctness and
  performance, orchestration, fleet behavior, and product-level gates.
- `evidence/` contains neutral schemas, catalog helpers, and result primitives
  shared by evaluation and verification.
- `bench/results/` remains immutable legacy evidence. Existing bytes and paths
  are not rewritten merely to match the new source layout.

This replaces the overloaded meaning of `bench`: HF/TP parity, logprob
agreement, batch invariance, quantization/KV equivalence, TTFT, TPOT/TPS,
throughput, goodput, and vLLM comparisons are verification of Kairyu itself,
not model Accuracy evaluation.

## Dependency boundary

`evals/`, `verification/`, and `evidence/` may import `kairyu`; `kairyu` must
not import any of them. Evaluation and verification must not import each
other. Neutral result contracts belong in `evidence/`. All three top-level
trees are excluded from the wheel.

## Verification registry

Every active gate has a stable ID and declares:

- scope: `l1`, `orchestration`, `fleet`, or `product`;
- kind: `correctness`, `performance`, `resilience`, or `diagnostic`;
- entry point, requirements, documentation, and output schema.

Performance gates that make a formal claim record the SHA-256 digest of the
accepted correctness artifact for the same relevant build/configuration.
Schema revisions are versioned. Historical artifacts retain their original
schema and source paths.

## Command surfaces

The installed `kairyu bench` command is removed. Repository users run explicit
checkout-only commands instead:

```text
uv run python -m evals run --suite <suite> ...
uv run python -m verification run <gate-id> ...
```

There is no implicit Accuracy default and no replacement aggregate Accuracy
suite in this repository.
