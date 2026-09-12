# V4.1 ensemble example implementation

User amendment: implement locally now; GPU validation is deferred. Create a PR
with a complete cross-machine handoff. Do not interrupt existing GPU workloads.

## Contract

- Separate `qwen3.8-deepseek-v4.1-8gpu` example; DeepSeek uses GPUs 0–5 and
  Qwen3.8 27B FP8 has two TP1 replicas on GPUs 6 and 7.
- Keep the existing five-route judge and dual-track DAG. Reduce four policies
  to two, and synthesize two policy answers plus the independently refined draft.
- DeepSeek defaults to official high; normal thinking roles inherit caller effort.
  Requirements always uses DeepSeek high. Qwen's non-thinking and medium roles
  retain their existing settings. Audit remains Qwen with two refinement rounds.
- Port PR #595 (`31f1adc`) extraction, propagation and audit contracts, not its
  historical measurements or changes to the original example.
- Both models receive images directly. Remove the image-description stage and
  the old text-only DeepSeek scaffold/template path.
- Fixed-runtime source inspection selects a six-GPU candidate; startup,
  capacity, correctness and performance remain explicitly unverified until GPU runs.

## Tasks and ownership

1. Deployment: static runtime inspection, compose/deployment/metadata, CPU tests.
2. Orchestration: DAG, native role hooks, checklist/audit contracts, CPU tests.
3. Integration: lifecycle, verification commands, documentation, progress log.
4. Review all interfaces, run focused CPU/regression checks, create draft PR with
   remaining GPU gates and reproducible handoff instructions.

## Acceptance

CPU tests exercise actual DAG effort propagation, fixed Requirement effort,
multimodal forwarding, direct-route bypass, PASS/repair/exhaustion, strict role
hook matching, deployment inventory, lifecycle isolation and measurement
provenance. No test or status claims that model quality or six-GPU serving has
been verified. Existing examples and framework behavior stay unchanged.

## GPU handoff

After hardware is free, attest the exact image/model/config, validate DeepSeek
alone without speculation, then Qwen co-residency. Exercise text/images/tools,
streaming/cancellation, explicit/default efforts, all five routes, Requirement
high, restart and concurrency; record TTFT/TPOT/token throughput and capacity.
Only investigate DSpark after the base configuration passes. Unsupported
topology or memory failures require a new evidence-backed L1 decision.
