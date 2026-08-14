# Issue #382 — Shared evidence-artifact mechanics

Status: accepted (2026-08-08); ownership amended (2026-08-14)

## Decision

Reusable evidence mechanics belong in the checkout-only, package-neutral
`evidence` namespace. Formal-gate meaning remains in repository-only
`verification/` entrypoints. `evidence` owns only:

- canonical UTF-8 JSON encoding and SHA-256 helpers;
- atomic canonical JSONL publication with contiguous `row_index` values;
- strict JSON/JSONL parsing that rejects duplicate keys, non-finite numbers,
  non-object rows, blank/truncated framing, empty evidence, and index gaps;
- stable resolution of a directory, raw path, or manifest path to one artifact
  pair; and
- raw-only manifest replay plus exact value comparison with the retained
  manifest.

Replay parses and hashes the same byte stream through one open file descriptor,
so the digest handed to the gate's manifest builder identifies exactly the rows
it evaluated. Missing raw or manifest files and malformed retained manifests
fail closed through the caller-selected gate error type.

## Ownership boundary

The shared module does not know a gate's schema version, row types, provenance
fields, thresholds, diagnostics, or pass/fail rule. Each top-level wrapper keeps
those definitions and supplies its existing `recompute_manifest(rows,
raw_sha256=...)` function as the replay callback. This preserves stable commands,
artifact names, manifest shapes, and historical replay while respecting the
one-way dependency boundary: `evidence` and verification may import `kairyu`,
while installed `kairyu` must not import either checkout-only namespace.
The 2026-08-14 amendment supersedes the earlier installed-package ownership.

The first migration covers the complete common artifact path in
`g4_ma3_sglang_bench.py` and `agentic_kv_tier_f4b_bench.py`. The highlighted
`fleet_churn_bench.py` adopts the shared file digest only: its sidecars are
heterogeneous JSONL without the combined-artifact `row_index` contract, so
forcing them into that schema would change evidence semantics. Further gates
may migrate only when their existing artifact contract matches these primitives.

## Compatibility and validation

Migrated verification entrypoints retain their public helper names as small adapters so
existing tests and imports do not move. Gate-specific error classes and
`--assert-gate` behavior are unchanged. Shared tests exercise canonical
round-trips, malformed and ambiguous JSON rejection, exact replay, retained
manifest tampering, raw tampering, path resolution, and custom error types;
the full owning tests for all three migrated wrappers remain the regression
authority.
